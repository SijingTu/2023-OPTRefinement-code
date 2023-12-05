"""
1, greedy algorithm;
2, Local search algorithm;
> 3, Linear Programming Based algorithm.  
"""

from .fibheap import FibonacciHeap as FibHeap
import networkx as nx
import copy
import numpy as np
import timeout_decorator

############################
## 1, Greedy algorithm 
############################

@timeout_decorator.timeout(7200)
def greedy_maxcut(Ginput, weight='weight'):
    """
    The greedy algorithm to approximately solve max-cut with 1/2 approximaiton ratio.
    Initially both left and right side are empty, greedily put nodes into left or right side to maximize cut at each step. 

    Parameters
    ----------
    Ginput: undirected, graph (networkx)
    weight: str that specify the edge attribute name of edge weight; None if the graph is unweighted

    Returns
    -------
    partition[0]: list of nodes on the left side 

    """

    H = copy.deepcopy(Ginput)
    
    partition = [[],[]] # partition[0] holds left partition, partition[1] holds right nodes
    node_dict = dict()
    
    for nd in H.nodes():
        node_dict[nd] = [list(), 0] # node_dict[nd][1] saves the cut_to_left - cut_to_right
        node_dict[nd][0].extend(H[nd]) # first include alll the neighbors. 
    
    for node_to_add in H.nodes():
        if node_dict[node_to_add][1] > 0:
            greedy_max_cut_posit_node(H, partition, node_dict, node_to_add, 'right', weight)
        else:
            greedy_max_cut_posit_node(H, partition, node_dict, node_to_add, 'left', weight)
        del node_dict[node_to_add]

    return partition[0]

def greedy_max_cut_posit_node(H, partition, node_dict, node_to_add, pos, weight):
    """A helper to put a node according to pos

    Args:
        H (networkx ): networkx undirected graph
        partition (int [2][n]): partition of nodes at current step
        node_dict (dict): a dictionary to store the neighbor of nodes outside of partitions, and a score to decide where to put the node
        node_to_add (int): the node to add to the partition at this step
        pos (str): 'left' or 'right', determine where to put node_to_add     
        weight (str): 'weight' or none
    """
    sign = -1 if pos == 'right' else 1
    idx = 1 if pos == 'right' else 0
    partition[idx].append(node_to_add)
    for neighbor in node_dict[node_to_add][0]:
        node_dict[neighbor][0].remove(node_to_add)
        node_dict[neighbor][1]= node_dict[neighbor][1] + sign * H[node_to_add][neighbor][weight] # might need to modify
        
#############################
### 2, Local search algorithm
############################
@timeout_decorator.timeout(7200)
def local_search_max_cut(Ginput, init_partition = None, weight = 'weight'):
    """Local search algorithm to solve max cut.
    Given a inital partition of nodes of the graph (if `init_partition` is none, then randomly assign the partition).
    At each time step, move a node to the other side if the weight of cut is maximized. 

    Args:
        Ginput (networkx undirected graph): undirected graph
        init_partition (int [], optional): A list of nodes on the left side of partition. Defaults to None.
        weight (srt, optional): 'weight' or None. Defaults to 'weight'.

    Returns:
        left_part: list of nodes on the left side
    """
    
    H = copy.deepcopy(Ginput)
    
    partition_idx = local_search_max_cut_init_partition(H, init_partition)
    node_dict, fib_heap = local_search_max_cut_init_heap(H, partition_idx, weight='weight')
    left_part = local_search_max_cut_optimize(H, partition_idx, node_dict, fib_heap, weight)
    
    return left_part 

def local_search_max_cut_init_partition(H, init_partition):
    """Init partition

    Args:
        H (networkx graph): undirected graph
        init_partition (list): list of nodes on left side of the partition

    Returns:
        partition_idx: a dictionary, whose key is node, and value is 0 or 1. The value indicates the side of each node. 
    """
    n = H.number_of_nodes()
    partition_idx = {}
    for nd in list(H.nodes()):
        partition_idx[nd] = 0
    
    if init_partition == None:
        tmp_partition = np.random.randint(0, 2, n) # intial partition
        cur_idx = 0
        for key in partition_idx.keys():
            partition_idx[key] = tmp_partition[cur_idx]
            cur_idx += 1
        
    else: # not efficient
        for select_node in init_partition:
            partition_idx[select_node] = 1
            
    return partition_idx 
    

def local_search_max_cut_init_heap(H, partition_idx, weight='weight'):
    """ init heap
    Parameters
    ----------
    H: undirected, graph (networkx)
    partition_idx: a dictionary, whose key is node, and value is 0 or 1. The value indicates the side of each node. 
    weight: str that specify the edge attribute name of edge weight; None if the graph is unweighted

    Returns
    ----------
    partition_idx: a 0-1 list which stores the location of each node
    node_dict: dict, node id as key, heap node as value. Here heap node is a
    pointer to the corresponding node in fibheap.
    fibheap: FibonacciHeap, support fast extraction of min degree node and value change.
    """
    node_dict = dict()
    fibheap = FibHeap()
    for node in H.nodes():
        node_dict[node] = fibheap.insert(0, node)
        for neighbor in H[node]:
            edge_w = H[node][neighbor][weight]
            
            if partition_idx[neighbor] == partition_idx[node]:
                fibheap.decrease_key(node_dict[node], node_dict[node].key - edge_w)
            else:
                fibheap.decrease_key(node_dict[node], node_dict[node].key + edge_w)

    return node_dict, fibheap

def local_search_max_cut_optimize(H, partition_idx, node_dict, fib_heap, weight = 'weight'):
    """At each time, select the node to change its side in order to maximize the cut; if there does not exist such node to increase the cut, the loop stops. 

    Args:
        H (networkx graph): undirected graph
        partition_idx (dictionary): a dictionary that stores the position of each node
        node_dict (dictionary): node is its key, heap node is its value, the heap node is a pointer to the corresponding node in the fibheap
        fib_heap (FibonacciHeap): support fast extraction of min degree node
        weight (str, optional): 'weight' or None. Defaults to 'weight'.

    Returns:
        left_part: nodes on the left part
    """
    node_list = np.array(H.nodes())
    while 1:
        to_switch = fib_heap.find_min()
        min_key = to_switch.key 
        min_node = to_switch.value 
        
        if min_key >= 0: # nothing to change anymore
            break
        else:
            old_position = partition_idx[min_node]
            for neighbor in H[min_node]:
                # update the cut value of the neighborhood
                edge_w = 1 if weight is None else H[min_node][neighbor][weight]
                if partition_idx[neighbor] == old_position: # return the weight
                    fib_heap.decrease_key(node_dict[neighbor], node_dict[neighbor].key + edge_w)
                else:
                    fib_heap.decrease_key(node_dict[neighbor], node_dict[neighbor].key - edge_w)
                    
            # update the position
            partition_idx[min_node] = 1 - old_position 
            # update the key 
            fib_heap.decrease_key(node_dict[min_node], - min_key)

    left_part = [node for node in node_list if partition_idx[node] == 0]
    return left_part 

##############################
### Move nodes back
##############################

def move_nodes_back(Ginput, initial_left, left, k, weight = 'weight'):

    H = copy.deepcopy(Ginput)

    diff_list = []
    diff_list.append(np.union1d(np.setdiff1d(initial_left, left), np.setdiff1d(left, initial_left)))
    diff_list.append(np.setdiff1d(H.nodes(), diff_list[0]))

    min_id = np.argmin([len(i) for i in diff_list])

    if len(diff_list[min_id]) < k:
        new_left = move_nodes_back_enabler(H, left, diff_list[1 - min_id], k - len(diff_list[min_id]), weight)
    elif len(diff_list[min_id]) > k:
        new_left = move_nodes_back_enabler(H, left, diff_list[min_id], len(diff_list[min_id]) - k, weight)
    else:
        new_left = left

    #print("current cut is", nx.cut_size(G, new_left))
    
    return new_left

def move_nodes_back_enabler(H, init_partition, over_select_nodes, num_nodes_move_back, weight='weight'):
    """Move the over_selected_nodes back to the graph

    Args:
        H (networkx): undirected graph
        init_partition(list): list of nodes on left side of the partition
        over_select_nodes (list): choose from those nodes to move back
        num_nodes_move_back (int): number of nodes that need to change side
        weight (str, optional): 'weight' or None. Defaults to 'weight'.
    """
     
    # init the priority queue
    partition_idx = move_nodes_back_init_partition(H, init_partition) 
    node_dict, fib_heap = move_nodes_back_init_heap(H, partition_idx, over_select_nodes, weight)
    left_part = move_nodes_back_optimize(H, partition_idx, num_nodes_move_back, over_select_nodes, node_dict, fib_heap, weight)
    
    return left_part 

def move_nodes_back_init_partition(H, init_partition):
    """Init partition

    Args:
        H (networkx graph): undirected graph
        init_partition (list): list of nodes on left side of the partition

    Returns:
        partition_idx: a dictionary, whose key is node, and value is 0 or 1. The value indicates the side of each node. 
    """
    
    partition_idx = {}
    for nd in list(H.nodes()):
        partition_idx[nd] = 0
        
    for select_node in init_partition:
        partition_idx[select_node] = 1
            
    return partition_idx 
    
def move_nodes_back_init_heap(H, partition_idx, over_select_nodes, weight='weight'):
    """ init heap
    Parameters
    ----------
    H: undirected, graph (networkx)
    partition_idx: a dictionary, whose key is node, and value is 0 or 1. The value indicates the side of each node. 
    over_select_nodes: a list of nodes that have been overly selected, and needs to move back, 
    weight: str that specify the edge attribute name of edge weight; None if the graph is unweighted

    Returns
    ----------
    partition_idx: a 0-1 list which stores the location of each node
    node_dict: dict on over_select nodes, node id as key, heap node as value. Here heap node is a pointer to the corresponding node in fibheap.
    fibheap: FibonacciHeap, support fast extraction of min degree node and value change.
    """
    node_dict = dict()
    fibheap = FibHeap()
    for node in over_select_nodes:
        node_dict[node] = fibheap.insert(0, node)
        for neighbor in H[node]:
            edge_w = H[node][neighbor][weight]
            
            if partition_idx[neighbor] == partition_idx[node]:
                fibheap.decrease_key(node_dict[node], node_dict[node].key - edge_w)
            else:
                fibheap.decrease_key(node_dict[node], node_dict[node].key + edge_w)

    return node_dict, fibheap   

def move_nodes_back_optimize(H, partition_idx, num_nodes_move_back, over_select_nodes, node_dict, fib_heap, weight = 'weight'):
    """At each time, select the node to change its side in order to maximize the cut; if there does not exist such node to increase the cut, the loop stops. 

    Args:
        H (networkx graph): undirected graph
        partition_idx (dictionary): a dictionary that stores the position of each node
        num_nodes_move_back (int): the number of nodes that need to be moved back 
        node_dict (dictionary): node is its key, heap node is its value, the heap node is a pointer to the corresponding node in the fibheap
        fib_heap (FibonacciHeap): support fast extraction of min degree node
        weight (str, optional): 'weight' or None. Defaults to 'weight'.

    Returns:
        left_part: nodes on the left part
    """
    node_list = np.array(H.nodes())
    
    over_select_nodes_dict = {} # a reference of remaining over selected nodes
    for node in node_list:
        over_select_nodes_dict[node] = 0
    for node in over_select_nodes:
        over_select_nodes_dict[node] = 1 
    
    for _ in range(num_nodes_move_back):
        to_switch = fib_heap.extract_min()
        min_node = to_switch.value 
        
        old_position = partition_idx[min_node]
        for neighbor in H[min_node]:
            if over_select_nodes_dict[neighbor] == 1: # if the neighbor is in the over_select_nodes_dict 
                # update the cut value of the neighborhood
                edge_w = 1 if weight is None else H[min_node][neighbor][weight]
                if partition_idx[neighbor] == old_position: # return the weight
                    fib_heap.decrease_key(node_dict[neighbor], node_dict[neighbor].key + edge_w)
                else:
                    fib_heap.decrease_key(node_dict[neighbor], node_dict[neighbor].key - edge_w)
                    
        # remove from over_select_nodes_dict 
        over_select_nodes_dict[min_node] = 0 
        partition_idx[min_node] = 1 - partition_idx[min_node]
        
        del node_dict[min_node]

    left_part = [node for node in node_list if partition_idx[node] == 0]
    
    return left_part 