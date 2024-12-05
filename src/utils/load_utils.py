import numpy as np
import pandas as pd 
import pickle
import os 
import glob
import pathlib
from pathlib import Path 
import re
import sys 
import shutil

from pynwb import NWBHDF5IO
from joblib import Memory
from datetime import datetime
from typing import Union, List, Optional

def load_nwb_files(base_dir: str, process_data: bool = True, cachedir: str = '/root/capsule/scratch/') -> Union[dict, List]:
    """ 
    Load NWB files from base directory from prior loading in OC capsule (behavior_** subdirecories)

    Params: 
    base_dir (str): Base directory containing behavior sub folders and NWBs 
    ignore_dirs (List[str]) (optional: directories to ignore)
    process_data (bool) (optional: True -> create lick, trial, events DataFrames for each NWB 
    cachedir (str): Cache directory to store CSV files 

    Returns: 
    Union[dict, List]: True -> Dictionary with NWB objects as values and paths as keys 
                       False -> List of paths to NWB files

    """ 

    # Convert to path object
    base_path = Path(base_dir)
    cache_path = Path(cachedir)

    cache_path.mkdir(parents=True, exist_ok = True)

    # Find all directories with regex behavior_** 
    behavior_dirs = list(base_path.glob('behavior_*'))

    if not behavior_dirs:
        raise ValueError(f'No behavior files found in {base_dir}')

    # Find all NWB files in subdirectories 
    nwb_files = []
    for behavior_dir in behavior_dirs:
        nwb_path = behavior_dir / 'nwb'
        if nwb_path.exists():
            nwb_files.extend(list(nwb_path.glob('*.nwb')))

    if not nwb_files:
        raise ValueError(f'No .nwb files found in behavior folders')

    if not process_data:
        return [str(path) for path in nwb_files]

    # Load / process each NWB file
    processed_nwbs = {}
    for nwb_file in nwb_files:
        try:
            # Load NWB file 
            nwb = nu.load_nwb_from_filename(str(nwb_file))

            # Process DataFrames
            nwb.df_trials = nu.create_df_trials(nwb)

            # Extract filenames
            filename = nwb_file.name
            match = re.search(r'behavior_(\d+)_(\d{4}-\d{2}-\d{2})', filename)

            if match:
                subject_id, session_date = match.groups()

                # Save trials to CSVs
                csv_filename = f'{subject_id}_{session_date}_fip_trials.csv'
                csv_path = cache_path / csv_filename
                nwb.df_trials.to_csv(csv_path, index=False)

            processed_nwbs[str(csv_filename)] = nwb.df_trials

        except Exception as e:
            print(f'Error processing {nwb_file}: {str(e)}')
            continue

    return processed_nwbs


def load_csvs(dir: str) -> dict:
    """
    Load local CSV files into a dictionary.

    Params:
    dir (str): Directory containing local CSV files

    Returns:
    dict: Dictionary with CSV filenames as keys and DataFrames as values
    """

    # Convert to Path object
    local_path = Path(dir)

    # Regex pattern to match specific CSV filename format
    pattern = r'(\d+)_(\d{4}-\d{2}-\d{2})_fip_trials\.csv'

    processed_csvs = {}

    # Find and load matching CSV files
    for csv_file in local_path.glob('*.csv'):
        match = re.match(pattern, csv_file.name)
        if match:
            try:
                # Read CSV into DataFrame
                df = pd.read_csv(csv_file)
                
                # Use filename as key
                processed_csvs[csv_file.name] = df
                
            except Exception as e:
                print(f"Error processing {csv_file.name}: {str(e)}")

    if not processed_csvs:
        print(f"No matching CSV files found in {dir}")

    return processed_csvs

def process_single_dataset(pkl_file_path): 
    """ 
    Process single pkl dataset and extract values

    Params:
    pkl_file_path (str or Path): Path to the pkl file 
    """ 
    try:
        # Load pkl and extract values
        with open(pkl_file_path, 'rb') as file:
            df = pickle.load(file) 

        rewc_values = {}
        unrc_values = {}

        for trial_back in range(1,16):
            rewc_values[trial_back] = df[('RewC', trial_back)].iloc[0]
            unrc_values[trial_back] = df[('UnrC', trial_back)].iloc[0]

        results_df = pd.DataFrame({
            'trial_back': range(1,16),
            'RewC': [rewc_values[i] for i in range(1,16)],
            'UnrC': [unrc_values[i] for i in range(1,16)]
        })

        # Get subject_id and session_date
        parts = pkl_file_path.stem.split('_')
        subject_id = parts[0]
        session_date = parts[1]

        # Save extracted values to output directory
        output_file = Path('/root/capsule/scratch') / f'{subject_id}_{session_date}.csv'
        results_df.to_csv(output_file, index=False)
        print(f'Successfully processed: {pkl_file_path.name}')

    except (KeyError, IndexError) as e:
        print(f'Error processing: {pkl_file_path.name}: {str(e)}')

def delete_from_scratch(folder):
    """
    Delete files from cache directory (scratch folder)

    Params:
    folder (str): folder in cache directory
    """
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
