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
from typing import Union, List, Dict, Optional

def load_nwb_files(
    base_dir: str, 
    process_data: bool = True, 
    cachedir: str = '/root/capsule/scratch/', 
    filename_pattern: str = r'(\d+)_(\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}\.nwb'
) -> Union[Dict[str, pd.DataFrame], List[str]]:
    """ 
    Load nwbs from base directory from foraging_nwb_bonsai data asset.

    Params: 
    base_dir (str): Base directory containing nwb files
    process_data (bool, optional): Whether to process and extract trials data 
    cachedir (str, optional): Directory to store processed CSV files
    filename_pattern (str, optional): Regex pattern to extract subject_id and date from filename.

        Group 1 should capture subject_id, Group 2 should capture session_date

    Returns: 
    Union[Dict[str, DataFrame], List[str]]: 
        - Dictionary with processed DataFrames
    """
    # Convert to path objects
    base_path = Path(base_dir)
    cache_path = Path(cachedir)

    # Create cache directory
    cache_path.mkdir(parents=True, exist_ok=True)

    # Find all NWB files (including in subdirectories)
    nwb_files = list(base_path.rglob('*.nwb'))

    if not nwb_files:
        raise ValueError(f'No .nwb files found in {base_dir}')

    if not process_data:
        return [str(path) for path in nwb_files]

    # Load / process each NWB file
    processed_nwbs = {}
    for nwb_file in nwb_files:
        try:
            # Extract subject_id and date from filename
            filename = nwb_file.name
            match = re.search(filename_pattern, filename)
            
            if not match:
                print(f'Skipping {filename}: Does not match expected pattern')
                continue
            
            subject_id, session_date = match.groups()

            # Load NWB file using nu.load_nwb_from_filename 
            nwb = nu.load_nwb_from_filename(str(nwb_file))

            # Process DataFrames 
            nwb.df_trials = nu.create_df_trials(nwb)

            # Add subject_id and session_date columns
            nwb.df_trials['subject_id'] = subject_id
            nwb.df_trials['session_date'] = session_date

            # Save trials to CSVs
            csv_filename = f'{subject_id}_{session_date}.csv'
            csv_path = cache_path / csv_filename
            nwb.df_trials.to_csv(csv_path, index=False)

            processed_nwbs[csv_filename] = nwb.df_trials

        except Exception as e:
            print(f'Error processing {nwb_file}: {str(e)}')
            continue

    return processed_nwbs


def load_processed_csvs(cachedir: str) -> Dict[str, pd.DataFrame]:
    """
    Load all previously processed CSV files from the cache directory.

    Params:
    cachedir (str): Directory containing processed CSV files

    Returns:
    Dict[str, DataFrame]: Dictionary of DataFrames with filenames as keys
    """
    cache_path = Path(cachedir)
    csv_files = list(cache_path.glob('*.csv'))
    
    processed_csvs = {}
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            processed_csvs[csv_file.name] = df
        except Exception as e:
            print(f'Error loading {csv_file}: {str(e)}')
    
    return processed_csvs

def extract_metadata_from_filename(file_path): 
    """
    Extract metadata from CSV filename with multiple pattern support.
    
    Supported patterns:
    1. 685983_2023-10-25.csv
    2. 685983_2023-10-25_11-52-06.csv
    
    Params:
        file_path (str): Full path to the CSV file
    
    Returns:
        Tuple[Optional[str], Optional[str]]: (subject_id, session_date)
    """
    filename = os.path.basename(file_path)
    
    # Define patterns
    patterns = [
        r'(\d{6})_(\d{4}-\d{2}-\d{2})\.csv$', 
        r'(\d{6})_(\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}\.csv$'  
    ]
    
    # Try each pattern
    for pattern in patterns:
        match = re.match(pattern, filename)
        if match:
            return match.group(1), match.group(2)
    
    return None, None

def process_csv_files(csv_dir):
    """
    Process CSV files from a directory, extracting metadata and combining into a master DataFrame.
    
    Params: 
        csv_dir (str): Directory containing CSV files
    
    Returns:
        pd.DataFrame: Combined DataFrame with added metadata columns
    """
    # Find all CSV files
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
    
    dataframes = []
    
    for file in csv_files:
        try:
            # Extract subject_id and session_date
            subject_id, session_date = extract_metadata_from_filename(file)
            
            # Skip files without valid metadata
            if subject_id is None or session_date is None:
                print(f'Could not extract metadata from {os.path.basename(file)}')
                continue
            
            # Read CSV
            df = pd.read_csv(file)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)
            
            # Add metadata columns
            df['subject_id'] = subject_id
            df['session_date'] = session_date
            
            dataframes.append(df)
        
        except Exception as e:
            print(f'Error processing {file}: {e}')
    
    # Combine DataFrames
    if not dataframes:
        raise ValueError("No valid CSV files found or processed")
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Convert subject_id to numeric
    combined_df['subject_id'] = pd.to_numeric(combined_df['subject_id'], errors='coerce')
    
    return combined_df

def download_dataframe_to_csv(df, filepath=None, filename=None, overwrite=False):
    """
    Download a pandas DataFrame to a CSV file.

    Params: 
    df : DataFrame: DataFrame to be saved as a CSV.
    
    filepath : str, optional: directory path where the CSV will be saved. 
    
    filename : str, optional: name of the CSV file. 
    
    overwrite : bool -- True -- will overwrite, False -- will append a new filename 

    Returns:
    str - The full path to the saved CSV file.
    """
    # Use current working directory if no filepath is provided
    if filepath is None:
        filepath = os.getcwd()
    
    # Validate filepath
    if not os.path.isdir(filepath):
        raise ValueError(f"The provided filepath '{filepath}' is not a valid directory.")
    
    # Generate filename if not provided
    if filename is None:
        filename = f"dataframe_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Ensure filename ends with .csv
    if not filename.lower().endswith('.csv'):
        filename += '.csv'
    
    # Full path to the file
    full_path = os.path.join(filepath, filename)
    
    # Handle potential file name conflicts
    if not overwrite:
        counter = 1
        original_filename = filename
        while os.path.exists(full_path):
            # If file exists, modify filename
            filename = f"{os.path.splitext(original_filename)[0]}_{counter}.csv"
            full_path = os.path.join(filepath, filename)
            counter += 1
    
    # Save the DataFrame to CSV
    df.to_csv(full_path, index=False)
    
    return full_path

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
        output_file = Path('/root/capsule/scratch') / f'{subject_id}_{session_date}_rc.csv'
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
