from .maxcut_greedy import move_nodes_back_enabler
import copy
import numpy as np
import timeout_decorator

@timeout_decorator.timeout(7200)
def local_greedy_max_cut(G, initial_left_side_nodes, k, weight = 'weight'):
    # greedy part
    H = copy.deepcopy(G)
    
    left = move_nodes_back_enabler(H, initial_left_side_nodes, np.array(H.nodes()), k, weight)
    
    return left 

