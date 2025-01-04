import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
from scipy import stats
import seaborn as sns
from typing import Union, List, Dict, Optional, Tuple

def clean_dataframe(df, threshold=0.8, verbose=False):
    """
    Clean DataFrame by removing rows containing too many np.inf, -np.inf, or np.nan values

    Params:
    df (DataFrame): Input DataFrame to clean
    threshold (float): Between 0 and 1. Rows with more than this fraction of NaN values will be removed
    verbose (bool): If True, prints diagnostic information

    Returns:
    DataFrame: Cleaned DataFrame with rows containing too many null values removed
    """
    # Store original shape
    original_shape = df.shape
    
    # Replace inf values with nan
    cleaned_df = df.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False)
    
    # Calculate the fraction of NaN values in each row
    nan_fraction = cleaned_df.isna().sum(axis=1) / cleaned_df.shape[1]
    
    # Keep rows where the fraction of NaN values is below the threshold
    cleaned_df = cleaned_df[nan_fraction <= threshold]
    
    if verbose:
        print(f"Removed {original_shape[0] - cleaned_df.shape[0]} rows with more than {threshold*100}% NaN values")
        
        # Print summary of remaining NaN values
        nan_counts = cleaned_df.isna().sum()
        print("\nRemaining NaN counts per column:")
        print(nan_counts[nan_counts > 0].sort_values(ascending=False))

    print(f"\nOriginal DataFrame shape: {original_shape}")
    print(f"Cleaned DataFrame shape: {cleaned_df.shape}")

    return cleaned_df

def remove_columns(df, columns_to_remove):
    """
    Remove specified columns from DataFrame

    Params:
    df (DataFrame): Input DataFrame
    columns_to_remove (str or list): Column name(s) to remove from DataFrame

    Returns:
    DataFrame: DataFrame with specified columns removed
    """
    # Convert single column to list
    if isinstance(columns_to_remove, str):
        columns_to_remove = [columns_to_remove]
        
    # Get list of columns that actually exist in df
    columns_to_remove = [col for col in columns_to_remove if col in df.columns]
    
    # Remove columns and return
    return df.drop(columns=columns_to_remove)


def filter_by_group_size(df, 
                         group_col='subject_id', 
                         max_rows=None, 
                         category_col=None, 
                         category_limits=None, 
                         category_max=None):
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

def filter_by_column(df, column, filter_value=None, value_range=None, reset_sessions=False):
    """
    Filter dataframe by column value or range
    
    Args:
        df: DataFrame to filter
        column: Column name to filter on
        filter_value: Exact value to filter for (optional)
        value_range: Tuple of (min, max) values to filter between (optional)
        reset_sessions: If True, reset session numbers after filtering (default False)
    """
    if value_range is not None:
        min_val, max_val = value_range
        filtered_df = df[(df[column] >= min_val) & (df[column] <= max_val)].copy()
    elif filter_value is not None:
        if filter_value not in df[column].unique():
            raise ValueError(f'Value {filter_value} not found in column {column}')
        filtered_df = df[df[column] == filter_value].copy()
    else:
        raise ValueError("Must provide either filter_value or value_range")
        
    if reset_sessions and 'session' in filtered_df.columns:
        filtered_df['session'] = filtered_df.groupby('subject_id').cumcount() + 1
        
    return filtered_df

def merge_dataframes_subject_id(df1, df2, column='session_date'):
    """ 
    Merge two DataFrames on subject_id and n column

    Params:

    df1, df2 (DataFrame): DataFrames to merge

    Returns:

    merged_df (DataFrame): Merged DataFrame
    """

    df1[column] = df1[column].astype('datetime64[ns]')
    df2[column] = df2[column].astype('datetime64[ns]')

    merged_df = pd.merge(df1, df2, on=['subject_id', column], how='inner')

    return merged_df

def analyze_session_distribution(df, task_col=None, bins=20, bins_task=40, y_max=None, y_max_task=None):
    """ 
    Create DataFrame showing the number of sessions per subject per stage and/or task
    with proper density-based distribution visualization
    
    Params: 

    df: DataFrame
        Foraging DataFrame with metric, stage features, task features
    task_col = optional for task DataFrame

    bins (task) = set number of bins for either DataFrame
    ymax (task) = set manual maximum y axis value, if None default will be used
        
    Returns: 
    tuple : (DataFrame, DataFrame)
        session_counts: Individual counts per subject
        summary_stats: Summary statistics of the distribution
    """

    stage_order = df['current_stage_actual'].unique()

    # Group by subject and stage, and task
    if task_col:
        session_counts = (df.groupby(['subject_id', 'current_stage_actual', task_col], observed=True)
                         .agg({'session': 'count'})
                         .reset_index()
                         .rename(columns={'session': 'num_sessions'}))
    else:
        session_counts = (df.groupby(['subject_id', 'current_stage_actual'], observed=True)
                         .agg({'session': 'count'})
                         .reset_index()
                         .rename(columns={'session': 'num_sessions'}))
    
    # Create density plots
    plt.figure(figsize=(12, 6))
    
    # Create distribution plot and proper normalization
    if task_col:
        g = sns.displot(data=session_counts, 
                    x='num_sessions',
                    col='current_stage_actual',
                    row=task_col,
                    kind='hist',
                    bins=bins_task,
                    height=4,
                    aspect=1.5,
                    stat='count',
                    common_norm=False,
                    kde=True)

        # Adjust x-ticks and y-axis limits for all subplots
        for ax in g.axes.flat:
            if y_max is not None:
                ax.set_ylim(0, y_max)
    else:
        g = sns.displot(data=session_counts, 
                    x='num_sessions',
                    col='current_stage_actual',
                    kind='hist',
                    bins=bins,
                    height=4,
                    aspect=1.5,
                    stat='count',
                    common_norm=False,
                    kde=True)
                    
        # Adjust x-ticks and y-axis limits for all subplots
        for ax in g.axes.flat:
            # Calculate bin edges
            edges = [rect.get_x() for rect in ax.patches] + [ax.patches[-1].get_x() + ax.patches[-1].get_width()]
            
            # Set x-ticks to bin edges
            ax.set_xticks(edges)
            ax.tick_params(axis='x', rotation=45)
            
            if y_max is not None:
                ax.set_ylim(0, y_max)
    
    if task_col:
        plt.suptitle('Distribution of Session Counts by Stage and Task', y=1.02)
    else:
        plt.suptitle('Distribution of Session Counts by Stage', y=1.02)
    
    # Add more detailed statistics
    if task_col:
        summary_stats = (session_counts.groupby(['current_stage_actual', task_col], observed=True)
                        .agg({
                            'num_sessions': ['count', 'mean', 'std', 'min', 'max', 
                                           lambda x: x.quantile(0.25),
                                           lambda x: x.quantile(0.75)]
                        })
                        .round(2))
    else:
        summary_stats = (session_counts.groupby('current_stage_actual', observed=True)
                        .agg({
                            'num_sessions': ['count', 'mean', 'std', 'min', 'max',
                                           lambda x: x.quantile(0.25),
                                           lambda x: x.quantile(0.75)]
                        })
                        .round(2))
    
    summary_stats.columns = ['num_subjects', 'mean_sessions', 'std_sessions', 
                           'min_sessions', 'max_sessions', 'q25_sessions', 'q75_sessions']
    summary_stats = summary_stats.reset_index()
    
    return session_counts, summary_stats

def split_by_session_threshold(df, session_counts, threshold=None, task_col=None):
    """ 
    Split DataFrame based on session count Threshold in STAGE_1

    Params:

    df: DataFrame containing all data
    session counts: DataFrame containing session counts per subject_id/stage/task
    threshold: int (optional) based off of prior session count visualization
    task_col: str (optional) for task DataFrame specification 

    Returns:

    tuple: (DataFrame, DataFrame, int): 
            slow_df: DataFrame with subjects above threshold
            fast_df: DataFrame with subjects below threshold
            threshold_value: threshold (int) used for splitting 
    """ 

    # Get Stage 1 counts
    stage_n_counts = session_counts[
        session_counts['current_stage_actual'] == 'STAGE_FINAL'
    ].copy()

    # For task DataFrame average across tasks for Stage 1
    if task_col:
        stage_n_counts = (stage_n_counts.groupby('subject_id')['num_sessions'].mean().reset_index())

    # Use median if no threshold is provided
    if threshold is None:
        threshold_value = stage_n_counts['num_sessions'].median()
    else:
        threshold_value = threshold

    # Get subject_ids below and above threshold
    slow_subjects = stage_n_counts[stage_n_counts['num_sessions'] >= threshold_value]['subject_id'].unique()

    fast_subjects = stage_n_counts[stage_n_counts['num_sessions'] < threshold_value]['subject_id'].unique()

    # Split DataFrame
    slow_df = df[df['subject_id'].isin(slow_subjects)].copy()
    fast_df = df[df['subject_id'].isin(fast_subjects)].copy()

    # Summary stats
    print(f'Threshold value: {threshold_value:.2f} sessions')
    print(f'Number of slow learners {len(slow_subjects)}')
    print(f'Number of fast learners {len(fast_subjects)}')

    return slow_df, fast_df, threshold_value

def analyze_splits(df, task_col=None, threshold=None, **kwargs):
    """
    Analyze session distributions for both slow and fast session count DataFrames

    params:

    df: DataFrame in original analyze_sessions function
    task_col: str (optional) for task-specific DataFrame
    threshold: int (optional) threshold for split calculation
    **kwargs: additional args

    Returns:
    tuple: (dict, dict): DataFrames and stats for slow and fast learners
    """ 

    # First get session counts
    session_counts, stats = analyze_session_distribution(df, task_col=task_col, **kwargs)
    
    # Split the data
    slow_df, fast_df, threshold_value = split_by_session_threshold(
        df, session_counts, threshold=threshold, task_col=task_col
    )
    
    # Analyze both splits
    slow_counts, slow_stats = analyze_session_distribution(slow_df, task_col=task_col, **kwargs)
    fast_counts, fast_stats = analyze_session_distribution(fast_df, task_col=task_col, **kwargs)
    
    return {
        'slow': {'df': slow_df, 'counts': slow_counts, 'stats': slow_stats},
        'fast': {'df': fast_df, 'counts': fast_counts, 'stats': fast_stats},
        'threshold': threshold_value
    }

def add_session_column(df):
    """ 
    Add session column for each subject_id based on session_date column.

    Params: 
    df (DataFrame): Input DataFrame with subject_id and session_date columns

    Returns: 
    df (DataFrame): Modified DataFrame with 'session' column
    """ 
    # Validate required columns exist
    required_cols = ['subject_id', 'session_date']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # Ensure session_date is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df['session_date']):
        df['session_date'] = pd.to_datetime(df['session_date'])

    # Sort by subject_id, date, and any existing trial/temporal ordering
    sort_cols = ['subject_id', 'session_date']
    if 'trial' in df.columns:  # Add trial number if it exists
        sort_cols.append('trial')
    
    df_sorted = df.sort_values(sort_cols)

    # Create session column based on unique dates
    df_sorted['new_session'] = df_sorted.groupby('subject_id')['session_date'].transform(
        lambda x: pd.factorize(x)[0] + 1
    )

    return df_sorted

def analyze_column_distribution(df, column):
    """
    Analyze the distribution of values in a column of a DataFrame

    Params:
    df (DataFrame): Input DataFrame to analyze
    column (str): Name of column to analyze

    Returns:
    dict: Dictionary containing value counts and statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    value_counts = df[column].value_counts()
    percentages = df[column].value_counts(normalize=True) * 100

    most_common = (value_counts.index[0], value_counts.iloc[0])
    least_common = (value_counts.index[-1], value_counts.iloc[-1])

    return {
        'value_counts': value_counts,
        'percentages': percentages,
        'total_count': len(df),
        'unique_values': len(value_counts),
        'most_common': most_common,
        'least_common': least_common
    }
