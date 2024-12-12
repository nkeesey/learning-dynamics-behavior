import pandas as pd
import numpy as np

from typing import List, Tuple, Union

from aind_dynamic_foraging_basic_analysis.data_model.foraging_session import ForagingSessionData

def add_block_transitions_and_stats(p_reward):
    """
    Identifies block transitions and calculates block statistics from reward probability arrays.
    
    Params:
        p_reward: List containing two NumPy arrays [reward_probabilityL, reward_probabilityR]
        
    Returns:
        Tuple containing (block_numbers, block_lengths, max_block_length, median_block_length)
    """
    # Convert probability arrays to boolean transitions
    pr_L, pr_R = p_reward
    transitions_L = np.diff(pr_L, prepend=pr_L[0]) != 0
    transitions_R = np.diff(pr_R, prepend=pr_R[0]) != 0
    
    # Combine transitions from both sides
    block_transitions = (transitions_L | transitions_R).astype(int)
    
    # First trial is always a block transition
    block_transitions[0] = 1
    
    # Create block numbers
    block_numbers = np.cumsum(block_transitions)
    
    # Calculate block lengths
    unique_blocks, block_lengths = np.unique(block_numbers, return_counts=True)
    
    # Calculate block statistics
    max_block_length = np.max(block_lengths)
    median_block_length = np.median(block_lengths)
    
    return block_numbers, block_lengths, max_block_length, median_block_length


def compute_foraging_choice(
        choice_history: Union[List, np.ndarray],
        reward_history: Union[List, np.ndarray],
        p_reward: Union[List, np.ndarray],
        autowater_offered: Union[List, np.ndarray] = None,
        global_calc: bool = None
) -> Tuple[float, float]:
    """
    Calculates foraging choice metric on a trial by trial basis. 
    
    Params:
        choice_history: Array of choices (0 for left, 1 for right, nan for ignored)
        reward_history: Array of reward outcomes
        p_reward: List of two arrays [reward_probabilityL, reward_probabilityR]
        autowater_offered: Boolean array indicating autowater trials
        global_calc: Whether to calculate metric globally or locally by block
    
    Returns:
        Tuple of (local_metric, global_metric) where one will be nan based on global_calc
    """
    block_numbers, block_lengths, max_block_length, median_block_length = add_block_transitions_and_stats(p_reward)

    data = ForagingSessionData(
        choice_history=choice_history,
        reward_history=reward_history,
        p_reward=p_reward,
        autowater_offered = autowater_offered
    )

    # Foraging choice is calculated only on finished and non-autowater trials
    ignored = np.isnan(data.choice_history)
    valid_trials = (~ignored & autowater_offered) if autowater_offered is not None else ~ignored

    choice_history = data.choice_history[valid_trials]
    reward_history = data.reward_history[valid_trials]
    
    # Get probability arrays for valid trials
    pr_L = p_reward[0][valid_trials]
    pr_R = p_reward[1][valid_trials]
    blocks = block_numbers[valid_trials]

    # Get prob_chosen, not_chosen
    prob_chosen = np.where(choice_history == 0, pr_L, pr_R)
    prob_not_chosen = np.where(choice_history == 0, pr_R, pr_L)

    # Initialize return values
    local_fc = np.nan
    global_fc = np.nan

    # Calculate global metric if requested
    if global_calc is None or global_calc:
        prob_diff = prob_chosen - prob_not_chosen
        p_max_glob = max(prob_chosen.max(), prob_not_chosen.max())
        p_min_glob = min(prob_chosen.min(), prob_not_chosen.min())

        if p_max_glob != p_min_glob:
            global_fc = prob_diff.mean() / (p_max_glob - p_min_glob)

    # Calculate local metric if requested
    if global_calc is None or not global_calc:
        block_fcs = []
        for block in np.unique(blocks):
            block_mask = blocks == block
            
            block_chosen = prob_chosen[block_mask]
            block_not_chosen = prob_not_chosen[block_mask]
            block_diff = block_chosen - block_not_chosen

            p_max_block = max(block_chosen.max(), block_not_chosen.max())
            p_min_block = min(block_chosen.min(), block_not_chosen.min())

            if p_max_block == p_min_block:
                continue
            
            block_fc = block_diff.mean() / (p_max_block - p_min_block)
            block_fcs.append(block_fc)

        if block_fcs:
            local_fc = np.mean(block_fcs)

    return local_fc, global_fc
