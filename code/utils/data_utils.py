'''
Utility functions for accessing, processing, and analyzing neural and behavioral data
in the dynamic foraging task

'''

import pandas as pd
import os
from codeocean.data_asset import DataAssetAttachParams

def filter_fiber_probes(df):
    """
    Filters the dataframe to keep rows where the 'fiber_probes' column is not empty (i.e., not '[]' or NaN).
    This is to ensure that the filtered dataframe only contains sessions in which fiber photometry data
    was collected.

    Parameters:
    df (pd.DataFrame): The input dataframe to filter.

    Returns:
    pd.DataFrame: A filtered dataframe where 'fiber_probes' is not '[]' or NaN.
    """
    # Convert 'fiber probe' to string to avoid issues with non-string values (if any)
    df['fiber_probes'] = df['fiber_probes'].astype(str)
    
    # Filter out rows where 'fiber probe' is '[]' or NaN (NaNs are automatically excluded with dropna)
    filtered_df = df[(df['fiber_probes'] != '[]') & (df['fiber_probes'].notna())]
    
    return filtered_df

def get_processed_CO_dataID_for_stage(df, stage):
    """
    Returns a list of processed_CO_dataID values associated with the given current_stage_actual.
    
    Parameters:
    df (pd.DataFrame): The input dataframe containing the 'processed_CO_dataID' and 'current_stage_actual' columns.
    stage (str): The specific value of 'current_stage_actual' to filter by.
    
    Returns:
    list: A list of processed_CO_dataID values for the specified stage.
    """
    # Filter the dataframe for the specified stage
    filtered_df = df[df['current_stage_actual'] == stage]
    
    # Extract the processed_CO_dataID column values and return them as a list
    return filtered_df['processed_CO_dataID'].tolist()

def generate_data_asset_params(data_asset_IDs, mount_point=None):
    """
    Generates the code for attaching data assets using DataAssetAttachParams.
    
    Parameters:
    data_asset_IDs (list): A list of data_asset_ID strings.
    mount_point (str, optional): The mount point for each data asset. If None, it is left out.

    Returns:
    list: A list of DataAssetAttachParams objects.
    """
    data_assets = []
    
    # Iterate over each data_asset_ID and create a DataAssetAttachParams object
    for asset_id in data_asset_IDs:
        if mount_point:
            # Attach with specified mount point
            data_assets.append(DataAssetAttachParams(id=asset_id, mount=mount_point))
        else:
            # Attach without specifying a mount point
            data_assets.append(DataAssetAttachParams(id=asset_id))
    
    return data_assets

def get_nwb_suffix_for_stage(df, stage):
    """
    Returns a list of nwb_suffix values associated with the given current_stage_actual.
    
    Parameters:
    df (pd.DataFrame): The input dataframe containing the 'nwb_suffix' and 'current_stage_actual' columns.
    stage (str): The specific value of 'current_stage_actual' to filter by.
    
    Returns:
    list: A list of nwb_suffix values for the specified stage.
    """
    # Filter the dataframe for the specified stage
    filtered_df = df[df['current_stage_actual'] == stage]
    
    # Extract the nwb_suffix column values and return them as a list
    return filtered_df['nwb_suffix'].tolist()

def get_processed_session_name_for_stage(df, stage):
    """
    Returns a list of processed_session_name values associated with the given current_stage_actual.
    
    Parameters:
    df (pd.DataFrame): The input dataframe containing the 'processed_session_name' and 'current_stage_actual' columns.
    stage (str): The specific value of 'current_stage_actual' to filter by.
    
    Returns:
    list: A list of processed_session_name values for the specified stage.
    """
    # Filter the dataframe for the specified stage
    filtered_df = df[df['current_stage_actual'] == stage]
    
    # Extract the processed_session_name column values and return them as a list
    return filtered_df['processed_session_name'].tolist()

def format_nwb_suffixes(nwb_suffixes):
    """
    Converts a list of nwb_suffix integers into strings in the format XX-XX-XX.
    If a number has only 5 digits, it will be padded with a leading zero.

    Parameters:
    nwb_suffixes (list of int): The list of nwb_suffix integers.

    Returns:
    list: A list of formatted nwb_suffix strings in the form XX-XX-XX.
    """
    formatted_suffixes = []
    
    for suffix in nwb_suffixes:
        # Convert the suffix to a string and pad with leading zeroes if necessary
        suffix_str = str(suffix).zfill(6)
        
        # Format the string in the form XX-XX-XX
        formatted_suffix = f"{suffix_str[:2]}-{suffix_str[2:4]}-{suffix_str[4:]}"
        
        # Append the formatted string to the result list
        formatted_suffixes.append(formatted_suffix)
    
    return formatted_suffixes

def find_nwb_files_with_suffixes(nwb_folder, formatted_suffixes):
    """
    Searches for nwb files in the specified folder that have a suffix matching
    one of the formatted suffixes in the list.

    Parameters:
    nwb_folder (str): The folder path where the nwb files are located.
    formatted_suffixes (list): The list of formatted suffixes (XX-XX-XX).

    Returns:
    list: A list of nwb file names that match the suffixes.
    """
    matching_files = []
    
    # Iterate through all files in the nwb folder
    for file_name in os.listdir(nwb_folder):
        if file_name.endswith(".nwb"):
            # Split the file name to extract the suffix part (XX-XX-XX)
            try:
                # Example file name: subjectID_year-month-day_XX-XX-XX.nwb
                suffix = file_name.split('_')[-1].replace('.nwb', '')
                
                # Check if the suffix matches any in the formatted_suffixes list
                if suffix in formatted_suffixes:
                    matching_files.append(file_name)
            except IndexError:
                # If the file name format is unexpected, we skip it
                continue
    
    return matching_files
