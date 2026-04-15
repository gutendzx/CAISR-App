"""
CAISR Resp: Respiratory Event Detection
Author: Thijs Nassi, PhD
Cleaned/Refactored for Public Repository

Description:
    Runs rule-based respiratory event detection (Apnea, Hypopnea, RERA).
    Uses Ray for parallel processing.
"""

import os
import ray
import time
import logging
import argparse
import glob
import numpy as np
import pandas as pd
from typing import List, Tuple

# --- Local Imports ---
# Ensure 'resp' folder exists
from resp.rule_based_functions.SpO2_drop_analysis import *
from resp.rule_based_functions.Ventilation_drop_analysis import *
from resp.rule_based_functions.Hypopnea_analysis import *
from resp.rule_based_functions.RERA_analysis import *
from resp.rule_based_functions.Resp_event_hierarchy import *
from resp.rule_based_functions.Post_processing import *
from resp.utils_functions.Data_loaders import *
from resp.utils_functions.Data_writers import *

def timer(tag: str) -> None:
    print(tag)
    for i in range(1, len(tag) + 1):
        print('.' * i + '     ', end='\r')
        time.sleep(1.5 / len(tag))
    print()

def extract_run_parameters(param_csv: str) -> Tuple[bool, bool]:
    if not os.path.exists(param_csv):
        return False, False # Default values
    params = pd.read_csv(param_csv)
    multiprocess = params['multiprocess'].values[0]
    overwrite = params['overwrite'].values[0]
    return multiprocess, overwrite

def set_output_paths(input_paths: List[str], csv_folder: str, raw_folder: str, overwrite: bool) -> Tuple[List[str], List[str], List[str]]:
    # Create directories
    resp_csv_dir = os.path.join(csv_folder, 'resp')
    os.makedirs(resp_csv_dir, exist_ok=True)
    os.makedirs(raw_folder, exist_ok=True)

    IDs = [p.split(os.sep)[-1].split('.')[0] for p in input_paths]
    csv_paths = [os.path.join(resp_csv_dir, f'{ID}_resp.csv') for ID in IDs]
    raw_paths = [os.path.join(raw_folder, f'{ID}.hf5') for ID in IDs]

    input_paths, csv_paths, raw_paths = filter_already_processed_files(input_paths, csv_paths, raw_paths, overwrite)
    return input_paths, csv_paths, raw_paths

def filter_already_processed_files(input_paths, csv_paths, raw_paths, overwrite):
    total = len(input_paths)
    todo_indices = [i for i, path in enumerate(csv_paths) if not os.path.exists(path)]

    if not overwrite:
        input_paths = np.array(input_paths)[todo_indices].tolist()
        csv_paths = np.array(csv_paths)[todo_indices].tolist()
        raw_paths = np.array(raw_paths)[todo_indices].tolist()

    tag = '(overwrite) ' if overwrite else ''
    print(f'>> {total - len(todo_indices)}/{total} files already processed\n>> {len(input_paths)} to go.. {tag}\n')
    return input_paths, csv_paths, raw_paths

def filter_todo_files(input_paths: List[str], csv_paths: List[str], raw_paths: List[str], keep_indices: List[int]) -> Tuple[List[str], List[str], List[str]]:
    """
    Filters input, CSV, and raw paths based on indices of files to process.
    
    Args:
    - input_paths (List[str]): List of input file paths.
    - csv_paths (List[str]): List of CSV output file paths.
    - raw_paths (List[str]): List of raw output file paths.
    - keep_indices (List[int]): Indices of files to keep for processing.

    Returns:
    - Tuple[List[str], List[str], List[str]]: Filtered lists of input, CSV, and raw paths.
    """

    input_paths = np.array(input_paths)[keep_indices].tolist()
    csv_paths = np.array(csv_paths)[keep_indices].tolist()
    raw_paths = np.array(raw_paths)[keep_indices].tolist()

    return input_paths, csv_paths, raw_paths

def format_dataframe(data: pd.DataFrame, hdr: dict, ID: str) -> Tuple[pd.DataFrame, dict]:
    """
    Formats the input dataframe by renaming columns and validating the breathing trace.
    
    Args:
    - data (pd.DataFrame): The input data.
    - hdr (dict): Header information.
    - ID (str): The patient ID.

    Returns:
    - Tuple[pd.DataFrame, dict]: The formatted dataframe and updated header.
    
    Raises:
    - ValueError: If no valid/usable breathing trace can be found.
    """

    # FIX: The data loader has already performed robust breathing trace selection.
    # Instead of re-implementing the logic, we validate its output.
    # This prevents flat signals from being passed to the core algorithm.
    if 'breathing_trace' not in data.columns or np.nanstd(data['breathing_trace']) < 1e-6:
        # Provide a clear error message indicating why the file cannot be processed.
        raise ValueError(f"No usable breathing trace found for {ID}. All available channels (ptaf, cflow, abd, chest) might be flat or have insufficient signal.")

    # Rename specific columns for consistency
    old_names = ['resp', 'stage', 'arousal', 'abd', 'chest', 'spo2']
    new_names = ['Apnea', 'Stage', 'EEG_arousals', 'ABD', 'CHEST', 'SpO2']

    for old_name, new_name in zip(old_names, new_names):
        if old_name in data.columns:
            data = data.rename(columns={old_name: new_name})

    # Add a combined ventilation column
    data['Ventilation_combined'] = data['breathing_trace']
    hdr['patient_ID'] = ID

    check_cols = ['ABD', 'CHEST', 'SpO2', 'Stage', 'EEG_arousals', 'breathing_trace', 'Ventilation_combined']
    if 'Apnea' in data.columns:
        check_cols.insert(0, 'Apnea')
    
    for col in check_cols:
        assert col in data.columns, f'ERROR: {col} not in dataframe'

    return data, hdr
    
# save functions
def save_raw_output(data: pd.DataFrame, hdr: dict, out_file: str) -> None:
    """
    Saves processed data, including breathing, effort belts, SpO2, and algorithmic predictions to an HDF5 file.

    Args:
    - data (pd.DataFrame): DataFrame containing the input signals and predictions.
    - hdr (dict): Dictionary containing metadata or header information.
    - out_file (str): The file path to save the output data.
    """
    # Create an empty DataFrame to store data
    df = pd.DataFrame()

    # Fill the DataFrame with relevant data columns
    df['breathing'] = data['breathing_trace']
    if 'ptaf' in data.columns:
        df['ptaf'] = data['ptaf']
    if 'cflow' in data.columns:
        df['cflow'] = data['cflow']
    df['abd'] = data['ABD']
    df['chest'] = data['CHEST']
    df['spo2'] = data['SpO2']
    df['sleep_stages'] = data['Stage']
    df['y_CAISR'] = data['algo_apneas']
    df['EEG_arousals'] = data['EEG_arousals']

    # Add 'airflow' if it exists in the DataFrame
    if 'airflow' in data.columns:
        df['airflow'] = data['airflow']
    
    # Add 'Apnea' (original labels) if it exists in the DataFrame
    if 'Apnea' in data.columns:
        df['y_original'] = data['Apnea']

    # Add additional header information to the DataFrame (except patient ID)
    for key, value in hdr.items():
        if key != 'patient_ID':
            df[key] = value

    # Save data to HDF5 using the custom writer function
    from Data_writers import write_to_hdf5_file
    write_to_hdf5_file(df, out_file, overwrite=True)

def export_to_csv(pred: np.ndarray, csv_path: str, verbose: int = 5, Fs: int = 10, originalFs: int = 200) -> None:
    """
    Exports the predicted respiration events to a CSV file, resampled to 1Hz.

    Args:
    - pred (np.ndarray): Array of predicted respiration events.
    - csv_path (str): Path to save the CSV file.
    - verbose (int): Verbosity level for logging.
    - Fs (int): The current sampling frequency in Hz.
    - originalFs (int): The original sampling frequency of the signal.
    """
    # Create an empty DataFrame for the predictions
    data = pd.DataFrame(columns=['start_idx', 'end_idx', 'resp'])

    # Resample predictions to 1Hz (from Fs) and insert into the DataFrame
    data['resp'] = pred[::Fs]

    # Calculate start and end indices based on the original sampling frequency
    factor = originalFs // 1  # Resampling factor for converting to 1Hz
    ind0 = np.arange(0, len(data)) * factor
    ind1 = np.concatenate([ind0[1:], [ind0[-1] + factor]])

    # Add the calculated indices to the DataFrame
    data['start_idx'] = ind0
    data['end_idx'] = ind1

    # Save the DataFrame to a CSV file
    data.to_csv(csv_path, index=False, mode='w+')

    # Verify the saved CSV if verbosity is enabled
    if verbose > 0:
        print(f'Saved to "{csv_path}"')

# RUN ALGO
@ray.remote
def set_multiprocess_run(p: int, path: str, csv_out: str, raw_out: str, csv_folder: str, data_loader, total_num: int, verbose: int=5) -> bool:
    """
    Function for running the algorithm in parallel using Ray.
    
    Args:
    - p (int): Current index for progress tracking.
    - path (str): Input file path.
    - csv_out (str): CSV output file path.
    - raw_out (str): HDF5 output file path.
    - csv_folder (str): Folder path to store the CSV.
    - data_loader: Function to load data.
    - total_num (int): Total number of files to process.
    
    Returns:
    - bool: Whether the processing was successful.
    """

    try:
        success = set_individual_run(p, path, csv_out, raw_out, csv_folder, data_loader, total_num, verbose=verbose)
        return success
    except Exception:
        return False

def set_individual_run(p: int, path: str, csv_out: str, raw_out: str, csv_folder: str, data_loader, total_num: int, verbose: int) -> bool:
    """
    Function for running the algorithm on a single file.
    
    Args:
    - p (int): Current index for progress tracking.
    - path (str): Input file path.
    - csv_out (str): CSV output file path.
    - raw_out (str): HDF5 output file path.
    - csv_folder (str): Folder path to store the CSV.
    - data_loader: Function to load data.
    - total_num (int): Total number of files to process.
    - verbose (int): Verbosity level (0: silent, >0: detailed logging).
    
    Returns:
    - bool: Whether the processing was successful.
    """

    # Print progress
    patient_ID = path.split('/')[-1].split('.')[0]
    ratio = f'# {p + 1}/{total_num}'
    tag = patient_ID if len(patient_ID)<21 else patient_ID[:20]+'..'
    print(f'(# {p + 1}/{total_num}) Processing "{tag}"')

    try:
        # Load data (note: csv_folder used for side-loading other features like stage/arousal if needed)
        data, hdr = data_loader(path, csv_folder, add_CAISR=True, verbose=verbose)
        data, hdr = format_dataframe(data, hdr, patient_ID)
    except Exception as error:
        if verbose > 1: print(f'ERROR loading ({patient_ID}): {error}')
        return False

    try:
        run_algorithm_pipeline(data, hdr, csv_out, raw_out, verbose=verbose)
    except Exception as error:
        if verbose > 1: print(f'ERROR algorithm ({patient_ID}): {error}')
        return False

    return True
    
def run_algorithm_pipeline(data: pd.DataFrame, hdr: dict, csv_out: str, raw_out: str, save_raw: bool = False, verbose: bool = True) -> None:
    """
    The main pipeline to run the algorithm on data, save results as CSV and HDF5.
    
    Args:
    - data (pd.DataFrame): Input data for the algorithm.
    - hdr (dict): Metadata or header information.
    - csv_out (str): Path to save the CSV file.
    - raw_out (str): Path to save the HDF5 file.
    - save_raw (bool): Whether to save raw output to HDF5.
    - verbose (bool): Whether to print logs.
    """

    # Compute desaturation drops and merge events
    data = compute_saturation_drops(data, hdr, sat_drop=3)
    data = merge_connecting_saturation_drops(data, hdr)

    # Identify flow reductions and match potential hypopneas
    # import pdb; pdb.set_trace()
    data = find_flow_reductions(data, hdr)
    data = match_saturation_and_ventilation_drops(data, hdr)
    data = match_EEG_with_ventilation_drops(data, hdr)

    # Detect RERAs (Respiratory Effort-Related Arousals)
    data = RERA_detection(data, hdr)

    # Combine desaturation-hypopneas with EEG-hypopneas
    data['algo_hypopneas_three'] = (data['accepted_saturation_hypopneas'] + data['accepted_EEG_hypopneas']) * 4
    data.loc[data['algo_hypopneas_three'] == 8, 'algo_hypopneas_three'] = 4

    # Rank events based on AASM hierarchy
    data = do_apnea_multiclassification(data, hdr)

    # Post-processing: merge small detections, split large events
    data = post_processing(data, hdr)

    # Save output data to CSV and HDF5 if needed
    export_to_csv(data['algo_apneas'].values, csv_out, verbose=verbose)
    if save_raw:
        raw_folder = './resp/raw_outputs/'
        os.makedirs(raw_folder, exist_ok=True)
        save_raw_output(data, hdr, raw_out)

def CAISR_resp(in_paths: List[str], csv_folder: str, raw_output_dir: str, multiprocess: bool = True, overwrite: bool = False, verbose: int = 5) -> None:
    timer('* Starting "caisr_resp" (created by Thijs Nassi, PhD)')
    data_loader = load_breathing_signals_from_prepared_data
    
    in_paths, csv_paths, raw_paths = set_output_paths(in_paths, csv_folder, raw_output_dir, overwrite)
    finished = [False] * len(in_paths)
    total = len(in_paths)
    
    if multiprocess:
        try:
            if verbose > 0: print(f'Parallel processing {total} files..')
            # Initialize Ray (adjust num_cpus as needed)
            ray.init(num_cpus=min(20, os.cpu_count()), logging_level=logging.CRITICAL)
            
            futures = [set_multiprocess_run.remote(p, pi, pc, pr, csv_folder, data_loader, total, verbose)
                       for p, (pi, pc, pr) in enumerate(zip(in_paths, csv_paths, raw_paths))]
            
            finished = ray.get(futures)
            ray.shutdown()
            
            if verbose > 0: print(f'  Successful for {sum(finished)}/{total} recordings\n')
        except Exception as e:
            if verbose > 0: print(f'Parallel processing failed or interrupted: {e}')
            if ray.is_initialized(): ray.shutdown()

    # Retry failed or process sequentially if MP disabled
    todo_indices = np.where([not f for f in finished])[0]
    if len(todo_indices) > 0:
        print("Processing remaining files sequentially...")
        for idx in todo_indices:
            finished[idx] = set_individual_run(idx, in_paths[idx], csv_paths[idx], raw_paths[idx], csv_folder, data_loader, total, verbose)

    print(f'{sum(finished)}/{total} recordings were successfully processed.')
    timer('* Finishing "caisr_resp"')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CAISR respiratory event detection.")
    parser.add_argument("--input_data_dir", type=str, required=True, help="Folder containing the prepared .h5 data files.")
    parser.add_argument("--output_csv_dir", type=str, required=True, help="Folder where CAISR's output CSV files will be stored.")
    parser.add_argument("--param_dir", type=str, required=True, help="Folder containing the run parameters file.")
    args = parser.parse_args()
        
    input_files = glob.glob(os.path.join(args.input_data_dir, '*.h5'))
    param_file = os.path.join(args.param_dir, 'resp.csv')
    
    if not os.path.exists(param_file):
        os.makedirs(args.param_dir, exist_ok=True)
        pd.DataFrame({'overwrite': [False], 'multiprocess': [True]}).to_csv(param_file, index=False)
        
    multiprocess, overwrite = extract_run_parameters(param_file)
    
    # Define raw output dir based on CSV output dir
    raw_out = os.path.join(args.output_csv_dir, 'resp', 'raw_outputs')
    
    CAISR_resp(input_files, args.output_csv_dir, raw_out, multiprocess=multiprocess, overwrite=overwrite, verbose=5)
    