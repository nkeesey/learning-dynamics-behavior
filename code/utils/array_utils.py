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


def plot_metric(data_dict, stage_sequence, ylabel='Foraging Efficiency'):
    """ 
    Plot metric over sessions to visualize changes over stages and/or tasks

    Params: 
    data_dict (dict): Dictionary containing arrays to be concatenated 
    stage_sequence (list): Stage sequence for order of concatenation 
    ylabel (str): Y-axis label for metric 
    """
    is_stage_task = isinstance(next(iter(data_dict.keys())), tuple)
    
    # Choose between task/stage dictionary and stage dictionary 
    if is_stage_task:
        tasks = list(set(task for _, task in data_dict.keys()))
    else:
        tasks = ['']
    
    for task in tasks:
        data = []
        subject_data = []
        overall_session = 0

        for stage in stage_sequence:
            if is_stage_task:
                if (stage, task) in data_dict:
                    stage_data = data_dict[(stage, task)]
                else:
                    continue
            else:
                if stage in data_dict:
                    stage_data = data_dict[stage]
                else:
                    continue

            num_sessions = stage_data.shape[1]
            num_subjects = stage_data.shape[0]
            
            for session in range(num_sessions):
                session_data = stage_data[:, session]
                mean = np.nanmean(session_data)
                std = np.nanstd(session_data)
                
                data.append({
                    'Stage': stage,
                    'Session': overall_session,
                    'Stage Session': session,
                    'Mean': mean,
                    'Std': std
                })
                
                for subject in range(num_subjects):
                    subject_data.append({
                        'Stage': stage,
                        'Session': overall_session,
                        'Stage Session': session,
                        'Subject': subject,
                        'Score': stage_data[subject, session]
                    })

                overall_session += 1

        stage_plot_df = pd.DataFrame(data)
        subject_plot_df = pd.DataFrame(subject_data)

        plt.figure(figsize=(20, 6))

        sns.scatterplot(x='Session', y='Score', data=subject_plot_df, alpha=0.2, color='grey', legend=False)
        sns.lineplot(x='Session', y='Mean', data=stage_plot_df)

        plt.fill_between(stage_plot_df['Session'], 
                        stage_plot_df['Mean'] - stage_plot_df['Std'], 
                        stage_plot_df['Mean'] + stage_plot_df['Std'], 
                        alpha=0.2, color='b')

        plt.xlabel('Session (across all stages)', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)

        stage_boundaries = [stage_plot_df[stage_plot_df['Stage'] == stage]['Session'].min() 
                            for stage in stage_sequence if stage in stage_plot_df['Stage'].unique()]
        for boundary in stage_boundaries[1:]:
            plt.axvline(x=boundary - 0.5, color='b', linestyle='--', alpha=0.5)

        xticks = []
        xticklabels = []
        for i, stage in enumerate(stage_sequence):
            if stage in stage_plot_df['Stage'].unique():
                stage_sessions = stage_plot_df[stage_plot_df['Stage'] == stage]['Session']
                stage_start = stage_sessions.min()
                stage_end = stage_sessions.max()
                stage_ticks = range(int(stage_start), int(stage_end) + 1, 4)
                xticks.extend(stage_ticks)
                xticklabels.extend(range(0, (len(stage_ticks) - 1) * 4 + 1, 4))

                mid_point = (stage_start + stage_end) / 2
                plt.text(mid_point, plt.ylim()[1], stage, horizontalalignment='center', verticalalignment='bottom')

        plt.xticks(xticks, xticklabels)
        plt.xticks(ha='right')

        plt.plot([], [], color='grey', alpha=0.2, linewidth=0, marker='o', markersize=10, label='Subject Scores')

        if is_stage_task:
            plt.plot([], [], color='blue', linewidth=2, label=f'{task} Mean Score')
        else:
            plt.plot([], [], color='blue', linewidth=2, label='All Task Mean Score')
        
        plt.fill_between([], [], alpha=0.2, label='Standard Deviation')
        plt.legend()

        plt.tight_layout()
        plt.show()

        print(f"\nStatistics for {'Task: ' + task if is_stage_task else 'All Tasks'}:")
        for stage in stage_sequence:
            if stage in stage_plot_df['Stage'].unique():
                stage_data = stage_plot_df[stage_plot_df['Stage'] == stage]
                print(f'\n{stage}:')
                print(f'Number of Sessions: {len(stage_data)}')
                print(f"Mean {ylabel}: {stage_data['Mean'].mean():.2f}")
                print(f"Standard deviation: {stage_data['Mean'].std():.2f}")
                print(f"Min efficiency: {stage_data['Mean'].min():.2f}")
                print(f"Max efficiency: {stage_data['Mean'].max():.2f}")
