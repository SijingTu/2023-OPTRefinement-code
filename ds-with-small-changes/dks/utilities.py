import math
import numpy as np
import copy


def rho_subgraph(G, S, weight='weight'):
    ''' degree density e(S)/|S| '''
    H = G.subgraph(S)
    return H.size(weight) / float(H.number_of_nodes())

def rho(G, weight='weight'):
    return rho_subgraph(G, G.nodes(), weight)

def sphere_uniform(n: int):
    vec = np.random.normal(size=n)
    norm = np.sqrt(np.sum(vec ** 2))
    return vec / norm

def setdiff(A, B):
    return np.union1d(np.setdiff1d(A, B), np.setdiff1d(B, A))

def random_choice(G, com, k, I = 10):
    """Return score of random choices
    """

    tmp_score = 0
    for _ in range(I):
        selected_nodes = np.random.choice(G.nodes(), k, replace=False)
        union_nodes = np.union1d(com, selected_nodes)
        inter_nodes = np.intersect1d(com, selected_nodes)
        final_nodes = np.setdiff1d(union_nodes, inter_nodes)
        
        tmp_score += rho_subgraph(G, final_nodes)
    
    return tmp_score / I 


def merge_nodes(Ginput, nodes_in_subgraph, weight = 'weight'):
    """
    Merges the selected `nodes` of the graph G into one `new_node`.
    
    The edge weight gets updated. 
    """
    G = copy.deepcopy(Ginput)
    origin_edge_list = list(G.edges(data=True))
    
    new_node = np.amax(G.nodes) + 1 # fix the new node to be the next node
    G.add_node(new_node) # Add the 'merged' node

    nodes_checker = np.zeros(new_node) # initialize 
    
    for x in nodes_in_subgraph:
        nodes_checker[x] = 1
    
    for n1,n2,data in origin_edge_list:
        # For all edges related to one of the nodes to merge,
        # make an edge going to or coming from the `new gene`.
        if (nodes_checker[n1] == 1) and (nodes_checker[n2] == 1):
            continue
        if nodes_checker[n1] == 1:
            if  G.has_edge(new_node, n2):
                G[new_node][n2][weight] += G[n1][n2][weight]
            else:
                G.add_edge(new_node,n2, weight = data[weight])
        elif nodes_checker[n2] == 1:
            if G.has_edge(new_node, n1):
                G[new_node][n1][weight] += G[n2][n1][weight]
            else:
                G.add_edge(new_node,n1, weight = data[weight])
    
    for any_node in nodes_in_subgraph: # remove the merged nodes
        G.remove_node(any_node)
        
    return G


def remove_nodes(Ginput, nodes_in_subgraph):
    """
    remove the selected `nodes` of the graph G.
    
    The edge weight gets updated. 
    """
    G = copy.deepcopy(Ginput)
    
    for any_node in nodes_in_subgraph:
        G.remove_node(any_node)
    
    return G