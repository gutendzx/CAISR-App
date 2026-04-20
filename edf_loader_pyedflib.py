import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys
from tqdm import tqdm
import pandas as pd
import pyedflib
import ast
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Union, Tuple, Optional

def load_rename_rules(csv_path: str) -> Dict[str, List[str]]:
    """
    Loads channel aliases from a CSV file and prepares the renaming rules.
    The CSV should have a 'Channel_Names' column containing string representations 
    of tuples or lists of aliases (e.g., "('C3-M2', 'C3-A2')").

    Returns:
        Dict[str, List[str]]: {Standardized Name (first alias): All Aliases}
    """
    rename_rules = {}
    try:
        channel_table = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Channel table file not found at {csv_path}")
        return rename_rules
    
    if 'Channel_Names' not in channel_table.columns:
        print("Error: CSV file must contain a 'Channel_Names' column.")
        return rename_rules

    for _, row in channel_table.iterrows():
        alias_str_raw = row['Channel_Names']
        
        if pd.isna(alias_str_raw):
            continue

        try:
            # Parse the string representation into a list or tuple
            alias_list = [a.strip().replace("'", "").replace('"', "") 
                for a in str(alias_str_raw).split(';')]
            
            # Remove empty strings
            alias_list = [a for a in alias_list if a]
            
            if alias_list:
                # Use the first alias as the standardized name
                key = alias_list[0].lower() 
                # Store all aliases in lower case for matching
                rename_rules[key] = [str(a) for a in alias_list]

        except (ValueError, SyntaxError, TypeError) as e:
            # print(f"Skipping row due to parsing error: {e} in raw string: {alias_str_raw}")
            continue
            
    return rename_rules


# --- helper function for cleaning channel names ---
def _get_cleaned_name(channel_name: str) -> str:
    """
    Standardizes channel names by converting to lower case and removing common suffixes/separators.
    """
    # convert to lower case
    cleaned = channel_name.lower()
    
    # remove common suffixes
    cleaned = cleaned.replace('_pds', '').replace('_eg', '')
    
    # remove common separators
    cleaned = cleaned.replace(':', '-')
    
    # remove leading/trailing whitespace
    cleaned = cleaned.strip()
    
    return cleaned

# --- Main mapping function ---

def map_valid_channels_rename_only(columns_original: List[str], rename_rules: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Finds the first match for each standard name from the rename_rules in the 
    list of original channel names.

    Args:
        columns_original: The raw list of channel labels found in the file.
        rename_rules: Dictionary mapping {Standard Name: [Alias1, Alias2, ...]}

    Returns:
        Dict[str, str]: {Standard Name: Matched Original Channel Name}
    """
    channel_map = {}
    
    # Step 1: Create a map from cleaned column names to original names
    # {cleaned_name: original_name}
    cleaned_to_original_map = {_get_cleaned_name(col): col for col in columns_original}
    
    # Step 2: Iterate through standard names and aliases to find a match
    for std_name, aliases in rename_rules.items():
        for alias in aliases:
            # Clean the alias for comparison
            alias_cleaned = _get_cleaned_name(alias)
            
            if alias_cleaned in cleaned_to_original_map:
                # Found a match! Get the original, uncleaned name
                orig_col = cleaned_to_original_map[alias_cleaned]
                
                # Use the standard name as the key, and the original name as the value
                channel_map[std_name] = orig_col 
                
                # Stop searching for this standard name once the best match is found
                break  
    
    # Note: If needed, you might implement more complex mirror map logic here,
    # but based on your original simplified code, we skip it for now.

    return channel_map

# --- Main standardization function ---

def standardize_channel_names_rename_only(
    columns_original: List[str], 
    rename_rules: Dict[str, List[str]]
) -> Tuple[Dict[str, str], List[str]]:
    """ 
    Standardizes channel names based on rules, identifies duplicates to drop, 
    and handles pulse/pr renaming.

    Args:
        columns_original: The raw list of channel labels (strings) from the file.
        rename_rules: Dictionary mapping {Standard Name: [Alias1, Alias2, ...]}

    Returns:
        Tuple[Dict[str, str], List[str]]: 
            - rename_map: {Original Raw Name: New Standard Name}
            - cols_to_drop: List of original raw names to be dropped.
    """
    
    # Step 1: Find the desired standard name for each matching raw channel
    # Output: {Standard Name: Matched Original Raw Name}
    channel_map = map_valid_channels_rename_only(columns_original, rename_rules)
    
    # Step 2: Reverse map (Raw Name -> Standard Name) for the final rename operation
    # Output: {Original Raw Name: Standard Name}
    rename_map = {orig_raw: std_name for std_name, orig_raw in channel_map.items()}

    # Step 3: Detect and collect duplicate aliases to drop
    cols_to_drop = []

    # Map cleaned names to their original names for quick lookup
    cleaned_to_original_map = {_get_cleaned_name(col): col for col in columns_original}
    
    for std_name, matched_raw in channel_map.items():
        # Get all cleaned aliases corresponding to this standard name
        aliases_cleaned = {_get_cleaned_name(a) for a in rename_rules.get(std_name, [])}

        # Check all existing columns in the file
        for raw_col in columns_original:
            cleaned_col = _get_cleaned_name(raw_col)
            
            # If a column's cleaned name is one of the standard name's aliases
            if cleaned_col in aliases_cleaned:
                
                # AND this column is NOT the one chosen to be kept
                # (We keep 'matched_raw' and drop others that map to the same 'std_name')
                if raw_col != matched_raw:
                    cols_to_drop.append(raw_col)
    
    # Remove duplicates from the drop list
    cols_to_drop = sorted(set(cols_to_drop))

    # Step 4: Handle pulse/pr → hr rename (if not already handled by rename_rules)
    pulse_map = {"pulse": "hr", "pr": "hr"}
    
    # Iterate through the original channels to find 'pulse' or 'pr'
    for orig_ch_raw in columns_original:
        orig_ch_cleaned = _get_cleaned_name(orig_ch_raw)
        
        for orig_ch_alias, new_ch_standard in pulse_map.items():
            if orig_ch_cleaned == orig_ch_alias:
                # Check if this raw channel has NOT been mapped to a standard name yet (to avoid overwriting EEG channel names)
                if orig_ch_raw not in rename_map:
                    # Check if 'hr' is already a standard name in the map values (to avoid duplicate 'hr' output)
                    if new_ch_standard not in rename_map.values():
                        # Add this rename directly to the final map
                        rename_map[orig_ch_raw] = new_ch_standard
    
    return rename_map, cols_to_drop


def derive_bipolar_signal(
    ch_a_signal: np.ndarray, 
    ref_signal: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], 
) -> Optional[np.ndarray]:
    """
    Derives a new bipolar EEG channel by subtracting a reference signal 
    from a primary signal (A - Reference).

    Args:
        ch_a_signal: The primary signal (e.g., C4). Must be in physical units.
        ref_signal: The reference signal(s). Can be a single Series (M1) 
                    or a tuple of two Series (M1, M2) for average referencing.
        scaling_factor: Factor applied to the reference. Use 0.5 for average 
                        mastoid reference (A - 0.5 * (B + C)).

    Returns:
        np.ndarray: The derived bipolar signal, or None if input formats are invalid.
    """
    # Make sure inputs are numPy arrays
    try:
        if isinstance(ref_signal, tuple):
            # Average Reference: A - (B + C) / 2
            sig_b, sig_c = ref_signal
            return ch_a_signal - 0.5 * (sig_b + sig_c)
        else:
            # Simple Bipolar: A - B
            return ch_a_signal - ref_signal
    except Exception as e:
        print(f"Bipolar derivation error: {e}")
        return None

def load_edf_to_nparrays(edf_path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    channel_dict: Dict[str, np.ndarray] = {}
    fs_dict: Dict[str, float] = {}

    f = None
    try:
        f = pyedflib.EdfReader(edf_path)
        n = f.signals_in_file
        for i in range(n):
            label = f.getLabel(i).lower().strip()
            fs = float(f.getSampleFrequency(i))
            sig = f.readSignal(i).astype(np.float64, copy=False)

            # Convert microvolts (uV) to volts (V)
            phys_dim = f.getPhysicalDimension(i).strip().lower()
            if phys_dim == 'uv':
                sig = sig / 1e6
            elif phys_dim == 'mv':
                sig /= 1e3

            # Remove Possible Repitition    
            key = label
            k = 2
            while key in channel_dict:
                key = f"{label}__{k}"
                k += 1

            channel_dict[key] = sig
            fs_dict[key] = fs
    finally:
        if f is not None:
            try:
                f.close()
            except AttributeError:
                f._close()

    return channel_dict, fs_dict


def load_edf_for_caisr(physiological_data_file):
    physiological_data, physiological_fs = load_edf_to_nparrays(physiological_data_file)

    original_labels = list(physiological_data.keys())

    # Step 1: Load rules and standardize names
    # Note: Use script-relative path or absolute path for robustness
    rename_rules = load_rename_rules(os.path.abspath(DEFAULT_CSV_PATH))
    rename_map, cols_to_drop = standardize_channel_names_rename_only(original_labels, rename_rules)

    # Step 2: Apply renaming to BOTH signals and their corresponding FS
    processed_channels = {}
    processed_fs = {}
    for old_label, data in physiological_data.items():
        if old_label in cols_to_drop:
            continue
        new_label = rename_map.get(old_label, old_label.lower())
        processed_channels[new_label] = data
        # Mapping the sampling rate to the new label
        processed_fs[new_label] = physiological_fs.get(old_label, 200.0) # Default to 200 if missing
    
    if 'physiological_data' in locals(): del physiological_data

    # Step 3: Construct Bipolar Derivations
    bipolar_configs = [
        ('f3-m2', 'f3', ['m2']), ('f4-m1', 'f4', ['m1']),
        ('c3-m2', 'c3', ['m2']), ('c4-m1', 'c4', ['m1']),
        ('o1-m2', 'o1', ['m2']), ('o2-m1', 'o2', ['m1']),
        ('e1-m2', 'e1', ['m2']), ('e2-m1', 'e2', ['m1']),
        ('chin1-chin2', 'chin 1', ['chin 2']),
        ('lat', 'lleg+', ['lleg-']), ('rat', 'rleg+', ['rleg-'])
    ]

    for target, pos, neg_list in bipolar_configs:
        # 1. Skip if target already exists or pos channel missing
        if target in processed_channels or pos not in processed_channels:
            continue
        
        # 2. Check all neg channels exist
        if not all(n in processed_channels for n in neg_list):
            continue

        # 3. Check sampling rate consistency
        all_involved = [pos] + neg_list
        fs_values = [processed_fs[ch] for ch in all_involved]
        
        if len(set(fs_values)) > 1:
            raise ValueError(f"Sampling rate mismatch for {target}: {dict(zip(all_involved, fs_values))}")

        # 4. Derive bipolar signal
        ref_sig = processed_channels[neg_list[0]] if len(neg_list) == 1 else tuple(processed_channels[n] for n in neg_list)
        
        derived = derive_bipolar_signal(processed_channels[pos], ref_sig)
        
        if derived is not None:
            processed_channels[target] = derived
            processed_fs[target] = processed_fs[pos]

    return processed_channels, processed_fs


# Get the absolute directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the absolute path to the CSV file relative to the script location
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, 'channel_table.csv')


# # Example usage of the helper functions
# if __name__ == "__main__":
#     edf_data_path = '/labs/collab/CAISR/prepared_data/HSP/extracted/test_set/physiological_data/I0007/'
    
#     # example edf file
#     physiological_data_file = os.path.join(edf_data_path, os.listdir(edf_data_path)[5])

#     processed_channels, processed_fs = load_edf_for_caisr(physiological_data_file)

