import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
from tqdm import tqdm
import importlib
from foraging_choice import compute_foraging_choice

from aind_behavior_gym.dynamic_foraging.task import CoupledBlockTask
from aind_dynamic_foraging_models.generative_model import ForagerCollection
from aind_dynamic_foraging_models.generative_model.params.util import get_params_options
from aind_dynamic_foraging_basic_analysis import compute_foraging_efficiency

def run_task_env(parameter_sets, seed_start=42, n_sims=1000):
    """
    Run simulations for each stage task and get foraging choice values
    
    Params:
    
    parameter_sets : pd.DataFrame
        DataFrame containing parameter sets to test. Must include columns:
        'learn_rate_rew', 'learn_rate_unrew', 'forget_rate_unchosen',
        'biasL', 'softmax_inverse_temperature', 'current_stage_actual'
    seed_start : int, optional
        Starting random seed. Defaults to 42
    n_sims : int, optional
        Number of simulations per parameter set. Defaults to 100
    """
    # Define task environments
    task_environments = {
        'STAGE_1': CoupledBlockTask(
            reward_baiting=True, 
            num_trials=200,
            block_min=10,
            block_max=20,
            block_beta=5,
            p_reward_pairs=[[0.8, 0]],
            seed=42
        ),
        'STAGE_2': CoupledBlockTask(
            reward_baiting=True, 
            num_trials=200,
            block_min=10,
            block_max=40,
            block_beta=10,
            p_reward_pairs=[[0.55, 0.05]],
            seed=42
        ),
        'STAGE_3': CoupledBlockTask(
            reward_baiting=True, 
            num_trials=300,
            block_min=20,
            block_max=60,
            block_beta=20,
            p_reward_pairs=[[0.4, 0.05]],
            seed=42
        ),
        'STAGE_FINAL': CoupledBlockTask(
            reward_baiting=True, 
            num_trials=400,
            block_min=20,
            block_max=60,
            block_beta=20,
            p_reward_pairs=[
                [0.4, 0.05],
                [0.25, 0.2],
                [0.1, 0.35],
                [0.3, 0.15]
            ],
            seed=42
        ),
        'GRADUATED': CoupledBlockTask(
            reward_baiting=True, 
            num_trials=450,
            block_min=20,
            block_max=60,
            block_beta=20,
            p_reward_pairs=[
                [0.4, 0.05],
                [0.25, 0.2],
                [0.1, 0.35],
                [0.3, 0.15]
            ],
            seed=42
        )
    }
    
    results = []
    
    # Iterate through each parameter set
    for idx, params_row in parameter_sets.iterrows():
        stage = params_row['current_stage_actual']
        if stage not in task_environments:
            print(f"Warning: Stage {stage} not found in task environments. Skipping...")
            continue
            
        task = task_environments[stage]
        
        for sim in tqdm(range(n_sims), 
                       desc=f'Running {n_sims} simulations for parameter set {idx+1}/{len(parameter_sets)} ({stage})'):
            
            forager_collection = ForagerCollection()
            forager = forager_collection.get_preset_forager('Hattori2019', seed=seed_start+sim)
            
            # Create parameters dictionary from the row
            params = {
                'learn_rate_rew': params_row['learn_rate_rew'],
                'learn_rate_unrew': params_row['learn_rate_unrew'],
                'forget_rate_unchosen': params_row['forget_rate_unchosen'],
                'biasL': params_row['biasL'],
                'softmax_inverse_temperature': params_row['softmax_inverse_temperature']
            }
            
            # Update parameters
            forager.params = forager.params.model_copy(update=params)
            
            # Set new seed
            task.seed = seed_start + sim
            forager.perform(task)

            # Calculate metrics
            foraging_eff, eff_random_seed = compute_foraging_efficiency(
                baited=task.reward_baiting,
                choice_history=forager.get_choice_history(),
                reward_history=forager.get_reward_history(),
                p_reward=forager.get_p_reward(),
                random_number=task.random_numbers.T,
            )
            
            foraging_choice_local, foraging_choice_global = compute_foraging_choice(
                choice_history=forager.get_choice_history(),
                reward_history=forager.get_reward_history(),
                p_reward=forager.get_p_reward(),
                global_calc=None
            )
            
            results.append({
                'parameter_set': idx,
                'stage': stage,
                'simulation': sim,
                'foraging_efficiency': foraging_eff,
                'foraging_efficiency_random_seed': eff_random_seed,
                'foraging_choice_local': foraging_choice_local,
                'foraging_choice_global': foraging_choice_global,
                'random_seed': seed_start + sim,
                'num_trials': task.num_trials,
                'block_min': task.block_min,
                'block_max': task.block_max,
                'block_beta': task.block_beta,
                **params  # Include the parameter values in results
            })
    
    return pd.DataFrame(results)
