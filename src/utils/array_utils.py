import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import scipy
import seaborn as sns
sns.set_theme()

def create_arrays(df, stage_col='current_stage_actual', subject_col='subject_id', metric='foraging_eff', task_col='task'):
    """
    Create arrays with dimensions (subject_id x sessions) filled with chosen metric seperated by either stage and/or task

    Params:
    df (DataFrame): Inpute DataFrame with stage, subject, metric and/or task columns
    stage_col (str): Column for which seperate arrays will be created 
    subject_col (str): Column for which each row will be made
    metric (str): Column for which each cell is filled
    task_col (str): Task column (optional)
    """

    stages = df[stage_col].unique()
    subjects = df[subject_col].unique()

    if task_col is None:
        tasks = [None]
    else:
        tasks = df[task_col].unique()
    
    # Initialize array dictionary 
    arrays = {}

    # Loop through stages, tasks
    for stage in stages:
        for task in tasks:
            if task is None:
                # Create stage specific arrays
                stage_df = df[df[stage_col] == stage]
                max_sessions = stage_df.groupby(subject_col).size().max()
                if pd.isna(max_sessions):
                    print(f'No data found for {stage}')
                    continue
                max_sessions = int(max_sessions)
                stage_array = np.full((len(subjects), max_sessions), np.nan)
                for i, subject in enumerate(subjects):
                    subject_data = stage_df[stage_df[subject_col] == subject][metric].values
                    stage_array[i, :len(subject_data)] = subject_data
                arrays[stage] = stage_array
            else:
                # Create stage, task arrays 
                stage_task_df= df[(df[stage_col] == stage) & (df[task_col] == task)]
                max_sessions = stage_task_df.groupby(subject_col).size().max()
                if pd.isna(max_sessions):
                    print(f'No data found for {stage} and {task}')
                    continue 
                max_sessions = int(max_sessions)
                stage_task_array = np.full((len(subjects), max_sessions), np.nan)
                for i, subject in enumerate(subjects):
                    subject_data = stage_task_df[stage_task_df[subject_col] == subject][metric].values
                    stage_task_array[i, :len(subject_data)] = subject_data
                arrays[(stage, task)] = stage_task_array

    return arrays

def remove_outliers_n(data_dict, n):
    """
    Removes n largest entries from array and shortens max rows to n-1 entry max

    Params:
    data_dict (dict): Array dictionary
    n (int): Number of outliers needed to remove 

    Returns:
    result (dict): Dictionary containing filtered arrays 
    """
    result = {}

    for key, array in data_dict.items():

        if isinstance(key, tuple):
            stage, task = key
        else:
            stage = key
            task = None

        # Count num of fe values in each row
        non_nan_counts = np.sum(~np.isnan(array), axis=1)
        
        # Sort rows by their fe counts
        sorted_indices = np.argsort(non_nan_counts)[::-1]
        sorted_counts = non_nan_counts[sorted_indices]
        
        # Find the cutoff point
        if len(sorted_counts) > n:
            cutoff = sorted_counts[n]
        else:
            cutoff = sorted_counts[-1]
        
        # Create a mask for rows to keep
        keep_mask = non_nan_counts <= cutoff
        
        # Create make for rows to trim
        trim_mask = non_nan_counts > cutoff
        
        new_array = np.full((array.shape[0], cutoff), np.nan)
        
        # Fill in the rows that are kept
        new_array[keep_mask] = array[keep_mask][:, :cutoff]
        
        # Fill in the trimmed rows
        for i in np.where(trim_mask)[0]:
            non_nan_indices = np.where(~np.isnan(array[i]))[0][:cutoff]
            new_array[i, :len(non_nan_indices)] = array[i, non_nan_indices]
        
        if task is not None:
            result[(stage, task)] = new_array
        else:
            result[stage] = new_array
    
    return result


def summary_statistics(data_dict):
    """
    Summarizes dictionary by organizing by Stage, Task, Shape, Mean, STD, and Outlier stats

    Params:  
    data_dict (dict): Dictionary containing arrays 

    Returns: 
    summary_df (DataFrame): DataFrame with feature columns 
    """
    summary_data = []

    for key, array in data_dict.items():
        non_nan_counts = np.sum(~np.isnan(array), axis=1)
        
        Q1 = np.percentile(non_nan_counts, 5)
        Q3 = np.percentile(non_nan_counts, 95)
        IQR = Q3 - Q1
        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)
        outliers = non_nan_counts[(non_nan_counts > upper_bound)]
        
        if isinstance(key, tuple):
            stage, task = key
        else:
            stage, task = key, "N/A"

        summary_data.append({
            'Stage': stage,
            'Task': task,
            'Array Shape': array.shape,
            'Mean': np.mean(non_nan_counts),
            'Std Dev': np.std(non_nan_counts),
            'Min': np.min(non_nan_counts),
            'Max': np.max(non_nan_counts),
            'Outliers': len(outliers)
        })

    # Create a summary DataFrame
    summary_df = pd.DataFrame(summary_data)

    return summary_df

def calculate_average_vectors(data_dict):
    """
    Calculates average vectors column wise in each array

    Params:
    data_dict (dict): Array dictionary 

    Returns: 
    averages (dict): Dictionary containing 1 dim vectors of averaged metric for each array
    """
    averages = {}

    if isinstance(next(iter(data_dict.keys())), tuple):
        # Stage task dictionary
        for (stage, task), array in data_dict.items():
            if stage not in averages:
                averages[stage] = {}
            averages[stage][task] = np.nanmean(array, axis=0)
    else:
        # Stage dictionary
        for stage, array in data_dict.items():
            averages[stage] = np.nanmean(array, axis=0)

    # Find the average vectors
    for stage, value in averages.items():
        print(f"\nAverage foraging efficiency for stage {stage}:")
        if isinstance(value, dict):
            for task, average_vector in value.items():
                print(f" \n Task: {task}")
                print(f" \n Average vector: {average_vector}")
                print(f" \n Shape: {average_vector.shape}")
        else:
            print(f"Average vector: {value}")
            print(f"Shape: {value.shape}")

    return averages


def normalize_stage_data(plot_df, n_bins=5):
    """Normalize each stage to same number of bins, handling cases with few sessions"""
    normalized_data = []
    
    for stage in plot_df['Stage'].unique():
        stage_data = plot_df[plot_df['Stage'] == stage].copy()
        
        # Count unique sessions in this stage
        unique_sessions = stage_data['Stage Session'].nunique()
        
        # Adjust number of bins if necessary
        actual_bins = min(n_bins, unique_sessions)
        
        try:
            bins = pd.qcut(stage_data['Stage Session'], 
                         actual_bins, 
                         labels=False, 
                         duplicates='drop')
            
            stage_data['Normalized_Session'] = bins
            
            # Calculate statistics per bin
            bin_stats = stage_data.groupby('Normalized_Session').agg({
                'Score': ['mean', 'std', 'count'],
                'Subject': 'nunique'
            }).reset_index()
            
            # Rename columns to match expected format
            bin_stats.columns = ['Normalized_Session', 'Mean', 'Std', 'Count', 'Subjects']
            bin_stats['Stage'] = stage
            
            normalized_data.append(bin_stats)
            
        except ValueError:
            # If binning fails, use session numbers directly
            simple_stats = stage_data.groupby('Stage Session').agg({
                'Score': ['mean', 'std', 'count'],
                'Subject': 'nunique'
            }).reset_index()
            
            # Rename columns to match expected format
            simple_stats.columns = ['Normalized_Session', 'Mean', 'Std', 'Count', 'Subjects']
            simple_stats['Stage'] = stage
            
            normalized_data.append(simple_stats)
    
    result = pd.concat(normalized_data, ignore_index=True)
    return result


def detect_outliers(plot_df, zscore_threshold=3):
    """Detect outliers using z-score method"""
    outliers = {}
    for stage in plot_df['Stage'].unique():
        stage_data = plot_df[plot_df['Stage'] == stage]['Score']
        z_scores = scipy.stats.zscore(stage_data)
        outlier_mask = abs(z_scores) > zscore_threshold
        outliers[stage] = stage_data[outlier_mask]
    return outliers


stage_sequence = ['STAGE_1', 'STAGE_2', 'STAGE_3', 'STAGE_4', 'STAGE_FINAL', 'GRADUATED']

def ci_plot_metric(data_dict, 
                stage_sequence=stage_sequence, 
                figsize=(16, 6),
                ylabel='Foraging Efficiency',
                verbose=False,
                min_sample_threshold=5,
                ci_level=0.95,
                zscore_threshold=3,
                normalize_stages=False):
    """ 
    Enhanced plotting function with outlier detection, sample size filtering, and stage normalization

    Parameters:
    -----------
    data_dict : dict
        Dictionary containing arrays to be concatenated 
    stage_sequence : list
        Stage sequence for order of concatenation 
    ylabel : str
        Y-axis label for metric 
    verbose : bool
        Whether to print detailed statistics
    min_sample_threshold : int
        Minimum number of samples required for statistics calculation
    ci_level : float
        Confidence interval level (0-1)
    zscore_threshold : float
        Threshold for outlier detection
    normalize_stages : bool
        Whether to add normalized stage duration plot
    """
    is_stage_task = isinstance(next(iter(data_dict.keys())), tuple)
    
    # Convert to long format DataFrame first
    all_data = []
    overall_session = 0
    
    for stage in stage_sequence:
        if is_stage_task:
            tasks = list(set(task for s, task in data_dict.keys() if s == stage))
            for task in tasks:
                if (stage, task) not in data_dict:
                    continue
                stage_data = data_dict[(stage, task)]
                task_name = task
        else:
            if stage not in data_dict:
                continue
            stage_data = data_dict[stage]
            task_name = 'All Tasks'
            
        num_sessions = stage_data.shape[1]
        num_subjects = stage_data.shape[0]
        
        for session in range(num_sessions):
            for subject in range(num_subjects):
                value = stage_data[subject, session]
                if not np.isnan(value):
                    all_data.append({
                        'Stage': stage,
                        'Overall Session': overall_session,
                        'Stage Session': session,
                        'Subject': subject,
                        'Score': value,
                        'Task': task_name
                    })
            overall_session += 1
    
    # Convert to DataFrame
    plot_df = pd.DataFrame(all_data)
    
    # Calculate statistics with data quality checks
    stats_df = plot_df.groupby('Overall Session').agg({
        'Score': ['mean', 'std', 'count'],
        'Stage': 'first'
    }).reset_index()
    stats_df.columns = ['Overall Session', 'Mean', 'Std', 'Count', 'Stage']
    
    # Add data quality checks and calculations
    stats_df['Sample_Size'] = stats_df['Count']
    z_score = scipy.stats.norm.ppf((1 + ci_level) / 2)
    
    # Filter low-n sessions and calculate statistics
    stats_df['Valid_Session'] = stats_df['Count'] >= min_sample_threshold
    stats_df['Weighted_Mean'] = np.where(
        stats_df['Valid_Session'],
        stats_df['Mean'],
        np.nan
    )
    
    stats_df['CI'] = np.where(
        stats_df['Valid_Session'],
        (stats_df['Std'] / np.sqrt(stats_df['Count'])) * z_score,
        np.nan
    )
    
    # Add subject tracking
    stats_df['Unique_Subjects'] = stats_df.apply(
        lambda x: len(plot_df[(plot_df['Overall Session'] == x['Overall Session'])]['Subject'].unique()),
        axis=1
    )
    
    # Define a color palette for stages
    stage_colors = dict(zip(stage_sequence, sns.color_palette("RdYlGn", len(stage_sequence))))
    
    # Create plots
    if normalize_stages:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1]*2))
        main_ax = ax1
    else:
        fig, main_ax = plt.subplots(1, 1, figsize=figsize)
    
    # Plot individual points
    sns.scatterplot(data=plot_df, 
                    x='Overall Session', 
                    y='Score', 
                    alpha=0.4, 
                    color='grey', 
                    legend=False,
                    ax=main_ax)
    
    # Plot continuous mean line
    valid_data = stats_df[stats_df['Valid_Session']]
    main_ax.plot(valid_data['Overall Session'],
                valid_data['Weighted_Mean'],
                color='grey',
                alpha=0.5,
                label='Mean Score',
                zorder=3,
                linewidth=2)
    
    # Add colored confidence intervals by stage
    for stage in stage_sequence:
        if stage in stats_df['Stage'].values:
            stage_data = stats_df[stats_df['Stage'] == stage]
            stage_data = stage_data[stage_data['Valid_Session']]
            color = stage_colors[stage]
            
            # Add confidence interval with stage-specific color
            main_ax.fill_between(stage_data['Overall Session'],
                               stage_data['Weighted_Mean'] - stage_data['CI'],
                               stage_data['Weighted_Mean'] + stage_data['CI'],
                               alpha=0.6,
                               color=color,
                               label=f'{stage} CI')
    
    # Detect stage transitions
    stage_transitions = []
    prev_stage = None
    for idx, row in stats_df.iterrows():
        if prev_stage is not None and row['Stage'] != prev_stage:
            stage_transitions.append(idx)
        prev_stage = row['Stage']
    
    # Add stage boundaries and labels
    stage_boundaries = []
    for stage in stage_sequence:
        if stage in stats_df['Stage'].values:
            stage_data = stats_df[stats_df['Stage'] == stage]
            start_session = stage_data['Overall Session'].min()
            end_session = stage_data['Overall Session'].max()
            if start_session not in stage_boundaries:
                stage_boundaries.append(start_session)
            
            # Add stage label with subject count
            mid_point = (start_session + end_session) / 2
            subject_count = len(plot_df[plot_df['Stage'] == stage]['Subject'].unique())
            label = f"{stage}\n(n={subject_count})"
            main_ax.text(mid_point, main_ax.get_ylim()[1], label, 
                        horizontalalignment='center', 
                        verticalalignment='bottom')
    
    # Add vertical lines at stage transitions
    for transition in stage_transitions:
        main_ax.axvline(x=stats_df.iloc[transition]['Overall Session'] - 0.5, 
                       color='b', 
                       linestyle='--', 
                       alpha=0.5)
    
    # Customize x-axis ticks
    xticks = []
    xticklabels = []
    for stage in stage_sequence:
        if stage in stats_df['Stage'].values:
            stage_data = stats_df[stats_df['Stage'] == stage]
            stage_sessions = range(
                int(stage_data['Overall Session'].min()),
                int(stage_data['Overall Session'].max()) + 1,
                4
            )
            xticks.extend(stage_sessions)
            xticklabels.extend(range(0, len(stage_sessions) * 4, 4))
    
    main_ax.set_xticks(xticks)
    main_ax.set_xticklabels(xticklabels)
    plt.setp(main_ax.get_xticklabels(), rotation=45, ha='right')
    
    # Labels and legend
    main_ax.set_xlabel('Session (across all stages)', fontsize=12)
    main_ax.set_ylabel(ylabel, fontsize=12)
    main_ax.legend()
    
    if normalize_stages:
        # Add normalized plot
        normalized_stats = normalize_stage_data(plot_df)
        
        # Plot normalized data
        for stage in stage_sequence:
            if stage in normalized_stats['Stage'].values:
                stage_data = normalized_stats[normalized_stats['Stage'] == stage]
                color = stage_colors[stage]
                
                # Plot mean line
                ax2.plot(stage_data['Normalized_Session'],
                        stage_data['Mean'],
                        '-o',
                        color=color,
                        label=stage)
                
                # Add confidence interval with matching color
                ci = (stage_data['Std'] / np.sqrt(stage_data['Count'])) * z_score
                ax2.fill_between(stage_data['Normalized_Session'],
                               stage_data['Mean'] - ci,
                               stage_data['Mean'] + ci,
                               alpha=0.6,
                               color=color)
    
    
    plt.tight_layout()
    plt.show()
    
    if verbose:
        print("\nDetailed Statistics by Stage:")
        stage_stats = plot_df.groupby('Stage').agg({
            'Score': ['count', 'mean', 'std', lambda x: x.std()/np.sqrt(len(x)), 'min', 'max'],
            'Subject': 'nunique'
        }).round(3)
        stage_stats.columns = ['N', 'Mean', 'Std', 'SEM', 'Min', 'Max', 'Subjects']
        print(stage_stats)
        
        print("\nOutlier Analysis:")
        outliers = detect_outliers(plot_df, zscore_threshold)
        for stage, outlier_values in outliers.items():
            if len(outlier_values) > 0:
                print(f"\n{stage}:")
                print(f"Number of outliers: {len(outlier_values)}")
                print(f"Outlier values: {outlier_values.values}")
                print(f"Percentage: {(len(outlier_values)/len(plot_df[plot_df['Stage']==stage]))*100:.1f}%")
        
        print("\nPerformance Summary:")
        for stage in stage_sequence:
            if stage in stats_df['Stage'].values:
                stage_data = stats_df[stats_df['Stage'] == stage]
                print(f"\n{stage}:")
                print(f"Sessions: {len(stage_data)}")
                print(f"Mean ± SEM: {stage_data['Weighted_Mean'].mean():.3f} ± "
                      f"{stage_data['Weighted_Mean'].std()/np.sqrt(len(stage_data)):.3f}")
                print(f"Subjects: {stage_data['Unique_Subjects'].max()}")
                print(f"Sample size range: {stage_data['Count'].min()}-{stage_data['Count'].max()}")
                print(f"Sessions below threshold: {sum(~stage_data['Valid_Session'])}")
