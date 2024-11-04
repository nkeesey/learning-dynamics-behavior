import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
from scipy import stats
import seaborn as sns

def filter_by_group_size(df, group_col='subject_id', max_rows=None, category_col=None, category_limits=None, category_max=None):
    """ 
    Filter DataFrame by number of rows globally or by specified categorical column: 

    Params:
    df (DataFrame): Input DataFrame
    group_col (str): column to group by 
    max_rows (int) (optional): Global maximum number of rows allowed per group 
    category_col (str) (optional): column to filter by 
    category_limits (dict) (optional): mapping category values to their maximum allowed rows 
    category_max (int) (optional): maximum number of allowed rows for any given category in column 

    Returns: 
    df DataFrame: Filtered DataFrame only containing groups with rows <= max_rows

    """ 
    if category_col is not None:
        # Get counts per subject/category 
        group_sizes = df.groupby([group_col, category_col]).size().unstack(fill_value=0)
        
        invalid_groups = set()
        
        # Check category-specific limits
        if category_limits is not None:
            for category, limit in category_limits.items():
                if category in group_sizes.columns:
                    # Find groups that exceed limit
                    over_limit = group_sizes[group_sizes[category] > limit].index
                    invalid_groups.update(over_limit)
        
        # Check global category max limit 
        if category_max is not None:
            # For categories not in category_limits, apply max limit
            for category in group_sizes.columns:
                if category_limits is None or category not in category_limits:
                    over_limit = group_sizes[group_sizes[category] > category_max].index
                    invalid_groups.update(over_limit)
        
        # If neither category_limits nor category_max provided, return original df
        if category_limits is None and category_max is None:
            return df
            
        # Keep only valid groups 
        valid_groups = set(group_sizes.index) - invalid_groups
        
    else:
        # If no categories set, apply global limit or return original DataFrame
        if max_rows is None:
            return df
        
        group_sizes = df[group_col].value_counts()
        valid_groups = group_sizes[group_sizes <= max_rows].index
    
    # Filter the dataframe to keep only valid groups
    filtered_df = df[df[group_col].isin(valid_groups)]
    
    return filtered_df

def filter_by_column(df: pd.DataFrame, column: str) -> dict:
    """ 
    Input a DataFrame and output a dictionary containing dataframes containing unique values of 
    selected column

    Params: 
    df (DataFrame): Input Dataframe
    column (str): column to group by and seperate DataFrame

    Returns:
    split_dfs (dict): Dictionary of dataframes sorted by given column with keys=column values
    """

    if column not in df.columns:
        raise ValueError(f'Column: {column} not in DataFrame')

    # Get unique values in column 

    unique_values = df[column].unique()

    # Create dictionary with splits = unique values in selected column 
    split_dfs = {value: df[df[column] == value].copy() for value in unique_values}

    return split_dfs

