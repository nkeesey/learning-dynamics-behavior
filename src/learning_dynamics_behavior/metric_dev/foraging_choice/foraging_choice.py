import pandas as pd
import numpy as np

from typing import List, Tuple, Union

from aind_dynamic_foraging_basic_analysis import ForagingSessionData

def add_block_transitions_and_stats(p_reward):
    """
    Identifies block transitions and calculates block statistics from reward probability arrays.
    
    Params:
        p_reward: List containing two NumPy arrays [reward_probabilityL, reward_probabilityR]
        
    Returns:
        Tuple containing (block_numbers, block_lengths, max_block_length, median_block_length)
    """
    # Convert probability arrays to boolean transitions
    prob_L, prob_R = p_reward
    transitions_L = np.diff(prob_L, prepend=prob_L[0]) != 0
    transitions_R = np.diff(prob_R, prepend=prob_R[0]) != 0
    
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


def foraging_choice(
        choice_history: Union[List, np.ndarray],
        p_reward: Union[List, np.ndarray]
) -> Tuple[float, float]:
    """
    Calculates the foraging choice metric.
    """
    