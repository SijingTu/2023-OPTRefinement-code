import math
import numpy as np
import copy


def sphere_uniform(n: int):
    vec = np.random.normal(size=n)
    norm = np.sqrt(np.sum(vec ** 2))
    return vec / norm


def move_size_checker(G, initial_left, new_left, k):
    """Check if the final partition is correct (with k changes)

    Args:
        G (networkx): undirected graph
        initial_left ([]): initial left part
        new_left ([]): new left part after algorithm
        k (int): parameter on number of moves

    Returns:
        True or False
    """
    
    n = G.number_of_nodes()
    
    changed_size_1 = len(np.setdiff1d(new_left, initial_left)) + len(np.setdiff1d(initial_left, new_left))
    changed_size_2 = np.absolute(n - changed_size_1)
    
    if (changed_size_1 == k) or (changed_size_2 == k):
        return True 
    else:
        return False