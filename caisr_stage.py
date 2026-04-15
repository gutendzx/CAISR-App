"""
CAISR Stage: Sleep Staging Module
Author: Samaneh Nasiri, PhD
Cleaned/Refactored for Public Repository

Description:
    Runs the sleep staging model (GraphSleepNet/ProductGraphSleepNet) on .h5 input files.
    Supports both sequential and multiprocessing execution.

Requirements:
    - Input data must be in .h5 format.
    - Parameter file (stage.csv) must exist.
    - Pre-trained model weights must be available in `model_dir`.
"""

import sys
import os
import time
import glob
import warnings
import argparse
import logging
import multiprocessing
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from tensorflow import keras

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)
tf.get_logger().setLevel(logging.ERROR)

# --- Local Imports ---
# NOTE: Ensure the folders 'stage' and 'stage/graphsleepnet' are present in the root directory.
sys.path.insert(1, 'stage/graphsleepnet')
sys.path.insert(1, 'stage')

try:
    from DE_PSD import *
    from gcn_features import *
    from utils_docker import *
    from ProductGraphSleepNet import *
    from utils_model import *
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Ensure 'stage' and 'stage/graphsleepnet' directories exist and contain required scripts.")
    sys.exit(1)


def timer(tag: str) -> None:
    print(tag)
    for i in range(1, len(tag) + 1):
        print('.' * i + '     ', end='\r')
        time.sleep(1.5 / len(tag))
    print()


def extract_run_parameters(param_csv: str) -> 'tuple[bool, bool]':
    """Extracts run parameters from a CSV file."""
    if not os.path.exists(param_csv):
        raise FileNotFoundError(f'Run parameter file not found at {param_csv}.')
        
    params = pd.read_csv(param_csv)
    overwrite = params['overwrite'].values[0]
    
    # Get the multiprocess flag.
    multiprocess_val = params.get('multiprocess', pd.Series([False])).values[0]
    multiprocess = str(multiprocess_val).strip().lower() == 'true'
    
    return overwrite, multiprocess


def set_output_paths(input_paths: 'list[str]', csv_folder: str, overwrite: bool) -> 'tuple[list[str], list[str]]':
    IDs = [p.split(os.sep)[-1].split('.')[0] for p in input_paths]
    
    # Ensure output directory exists
    stage_out_dir = os.path.join(csv_folder, 'stage')
    os.makedirs(stage_out_dir, exist_ok=True)
    
    csv_paths = [os.path.join(stage_out_dir, f'{ID}_stage.csv') for ID in IDs]
    
    assert len(input_paths) == len(csv_paths), 'SETUP ERROR: Number of input and CSV files mismatch.'
    input_paths, csv_paths = filter_already_processed_files(input_paths, csv_paths, overwrite)
    return input_paths, csv_paths


def filter_already_processed_files(input_paths: 'list[str]', csv_paths: 'list[str]', overwrite: bool) -> 'tuple[list[str], list[str]]':
    total = len(input_paths)
    if not overwrite:
        todo_indices = [i for i, path in enumerate(csv_paths) if not os.path.exists(path)]
        input_paths = np.array(input_paths)[todo_indices].tolist()
        csv_paths = np.array(csv_paths)[todo_indices].tolist()
        processed_count = total - len(todo_indices)
    else:
        processed_count = 0 
    
    tag = '(overwrite)' if overwrite else ''
    print(f'>> {processed_count}/{total} files already processed')
    print(f'>> {len(input_paths)} files to process {tag}\n')
    return input_paths, csv_paths


def process_single_file(path: str, save_path: str, model_path: str, model_params: dict, file_info: 'tuple[int, int]'):
    """
    Processes a single input file for sleep staging.
    Loads its own model instance to ensure thread/process safety.
    """
    num, total = file_info
    
    # Unpack parameters
    opt = model_params['optimizer']
    regularizer = model_params['regularizer']
    w, h, context = model_params['w'], model_params['h'], model_params['context']
    sample_shape = (context, w, h)
    
    # Build model
    model = build_ProductGraphSleepNet(
        model_params['cheb_k'], model_params['num_of_chev_filters'], model_params['num_of_time_filters'], 
        model_params['time_conv_strides'], model_params['cheb_polynomials'],
        model_params['time_conv_kernel'], sample_shape, model_params['num_block'], opt, 
        model_params['conf_adj'] == 'GL', model_params['GLalpha'], regularizer,
        model_params['GRU_Cell'], model_params['attn_heads'], model_params['dropout']
    )
    
    weights_file = os.path.join(model_path, 'weights_fold_3.h5')
    if not os.path.exists(weights_file):
        print(f"Error: Weights file not found at {weights_file}")
        return

    model.load_weights(weights_file)
    
    the_id = path.split(os.sep)[-1].split('.')[0]
    tag = the_id if len(the_id) < 21 else the_id[:20] + '..'
    print(f'(# {num + 1}/{total}) Processing "{tag}" [PID:{os.getpid()}]')

    try:
        signals, Fs, sig_tages, length_data = select_signals_cohort(path)
    except Exception as e:
        print(f'Error loading signals for {tag}: {e}', flush=True)
        return

    window = 30
    fs_input = 200
    
    try:
        segs = segment_data_unseen(signals)
        MYpsd, MYde = graph_feat_extraction_unseen_docker(segs, sig_tages, Fs, window)
        image = AddContext(MYde, context)
        image = np.squeeze(np.array(image))
        
        # Prediction
        prediction = model.predict(image, verbose=0) 
        pred_per_subject = prediction.argmax(axis=1) + 1
        
        # Padding predictions to match input length (context window offset)
        pred_per_subject = np.concatenate([[np.nan] * 3, pred_per_subject, [np.nan] * 3])
        pred_per_subject = np.repeat(pred_per_subject, window, axis=0) # Expand to seconds/samples if needed
        
        nan_row = np.empty((1, prediction.shape[1])); nan_row[:] = np.nan
        for index in [0, 1, 2]:
            prediction = np.insert(prediction, index, nan_row, axis=0)
        for i in range(3):
            prediction = np.vstack((prediction, nan_row))
        prediction = np.repeat(prediction, window, axis=0)

        # Create DataFrame
        t1 = (np.arange(len(prediction)) * fs_input).astype(int)
        t2 = (np.arange(len(prediction)) + 1) * fs_input
        
        df = pd.DataFrame({
            'start_idx': t1, 'end_idx': t2, 'stage': pred_per_subject,
            'prob_n3': prediction[:, 0], 'prob_n2': prediction[:, 1],
            'prob_n1': prediction[:, 2], 'prob_r': prediction[:, 3],
            'prob_w': prediction[:, 4]
        })
        
        # Ensure output length matches original data length
        t1_full = (np.arange(length_data / fs_input) * fs_input).astype(int)
        t2_full = ((np.arange(length_data / fs_input) + 1) * fs_input).astype(int)
        
        df_matched = pd.DataFrame({
            'start_idx': t1_full, 'end_idx': t2_full, 'stage': np.nan,
            'prob_n3': np.nan, 'prob_n2': np.nan, 'prob_n1': np.nan, 'prob_r': np.nan, 'prob_w': np.nan
        })
        
        # Copy valid predictions into matched dataframe
        limit = min(len(df), len(df_matched))
        df_matched.iloc[0:limit] = df.iloc[0:limit]
        
        df_matched.to_csv(save_path, index=False)
        
    except Exception as error:
        print(f'({num}) Failure during feature extraction/prediction for {tag}: {error}')


def CAISR_stage(in_paths: 'list[str]', save_paths: 'list[str]', model_path: str, multiprocess: bool):
    timer('* Starting "caisr_stage" (created by Samaneh Nasiri, PhD)')
    
    # --- Define Model Parameters ---
    try:
        # Check TF version for optimizer compatibility
        if keras.optimizers.Adam(learning_rate=0.0001).get_config()['name'] == "adam": 
             opt = keras.optimizers.Adam(learning_rate=0.0001, decay=0.0, clipnorm=1)
        else:
             opt = keras.optimizers.Adam(lr=0.0001, decay=0.0, clipnorm=1)
    except:
        opt = keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1)

    model_params = {
        'optimizer': opt,
        'regularizer': keras.regularizers.l1_l2(l1=0.001, l2=0.001),
        'w': 7, 'h': 9, 'context': 7,
        'conf_adj': 'GL', 'GLalpha': 0.0,
        'num_of_chev_filters': 128, 'num_of_time_filters': 128,
        'time_conv_strides': 1, 'time_conv_kernel': 3,
        'num_block': 1, 'cheb_k': 3,
        'cheb_polynomials': None, 'dropout': 0.60,
        'GRU_Cell': 256, 'attn_heads': 40,
    }

    # --- Dispatch jobs ---
    if multiprocess and len(in_paths) > 1:
        # Use available CPUs, cap at 24 to prevent OOM if model is heavy
        num_workers = min(24, multiprocessing.cpu_count())
        print(f">> Multiprocessing enabled. Starting parallel processing with {num_workers} workers...")
        
        tasks = [
            (path, save_path, model_path, model_params, (num, len(in_paths)))
            for num, (path, save_path) in enumerate(zip(in_paths, save_paths))
        ]
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.starmap(process_single_file, tasks)

    else:
        print(">> Starting sequential processing...")
        for num, (path, save_path) in enumerate(zip(in_paths, save_paths)):
            process_single_file(path, save_path, model_path, model_params, (num, len(in_paths)))
    
    timer('* Finishing "caisr_stage"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CAISR sleep staging.")
    parser.add_argument("--input_data_dir", type=str, required=True, help="Path to the input data folder (containing .h5 files)")
    parser.add_argument("--output_csv_dir", type=str, required=True, help="Path to save the output features")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the pre-trained model directory")
    parser.add_argument("--param_dir", type=str, required=True, help="Folder containing the run parameters file (stage.csv)")
    args = parser.parse_args()

    # GPU Check
    if tf.test.is_built_with_cuda():
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f">> GPU detected: {len(gpus)} device(s)")
        else:
            print(">> TensorFlow built with CUDA, but no GPU detected at runtime.")
    else:
        print(">> TensorFlow is CPU only.")

    input_files = glob.glob(os.path.join(args.input_data_dir, '*.h5'))
    
    # Load parameters
    param_file = os.path.join(args.param_dir, 'stage.csv')
    if not os.path.exists(param_file):
         # Create default if missing for user convenience
         print(f">> Parameter file not found at {param_file}. Creating default.")
         os.makedirs(args.param_dir, exist_ok=True)
         pd.DataFrame({'overwrite': [False], 'multiprocess': [False]}).to_csv(param_file, index=False)

    overwrite, multiprocess = extract_run_parameters(param_file)
    
    in_paths, save_paths = set_output_paths(input_files, args.output_csv_dir, overwrite)
    
    if in_paths:
        CAISR_stage(in_paths, save_paths, args.model_dir, multiprocess)
    else:
        print(">> No files to process.")
        