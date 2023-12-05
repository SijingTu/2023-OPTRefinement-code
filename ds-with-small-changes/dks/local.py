""" A implmentation of computing the densest subgraph with a give community and an allowance on k changes
1, SDP based algorithm
2, local search based algorithm
"""

import copy
import numpy as np
from mosek.fusion import * 
import networkx as nx
from .utilities import * 
from .peeling import greedy_charikar_fix_nodes, init_heap_from_graph_given_set_of_nodes
import sys
import timeout_decorator

def local_sdp_sum_weights(G, sub_com_nodes, k, I = 100, weight = 'weight'):
    """Given a graph G, a subgraph com, and a integer k. 
    Choose k nodes to change their positions inside or outside of the subgraph, to make the total edge
    weights of the subgraph maximized.  

    Args:
        G (NetworkX undirected graph): undirected graph
        sub_com_nodes ([int]): list of nodes in the given subgraph 
        k (int): cardinality parameter k
        I (int): The number of rounds the algorithm repeats
        weight (string) : None or 'weight'
        
    Return:
        final_nodes ([int]):  The nodes in the final subgraph 
        density (float): The density of the subgraph 
        opt' : The optimal result before rounding procedure, divided by (k + len(sub_com_nodes))
    """
    
    n, node_list, W, Wtilde, x0 = local_sdp_sum_weights_param(G, sub_com_nodes, weight)
    opt, X = local_sdp_sum_weights_mosek(n, x0, W, Wtilde, k)
    final_nodes, density = local_sdp_sum_weights_rounding(X, x0[1:], opt, n, node_list, G, k, I, weight)
    
    return final_nodes, density, opt / (0.1*(k + len(sub_com_nodes)))

def local_sdp_sum_weights_param(G, sub_com_nodes, weight):
    """Based on Feige et al.'s algorithm to formulate the densest k subgraph through semidefinite relaxation. 
    Construct the Parameters for the solver. 

    Args:
        G (NetworkX undirected graph): undirected graph
        sub_com_nodes ([int]): list of nodes in the given subgraph
        weight (string): None or 'weight'
    
    Returns:
        n (int): number of nodes 
        nodes (numpy int [n]): list of nodes with the same order as G.nodes()
        W (numpy matrix n X n) : adjacency matrix of G
        Wtilde (numpy matrix (n+1) X (n+1)): W_0j = sum_i W_ij, W_i0 = sum_j W_ij, W_00 = 0
        x0 (int [n + 1]): the first entry equals to 0, the i+1th entry equals to 1 if nodes[i] in the subgraph, otherwise equals to -1 
    """    
    n = G.number_of_nodes() 
    node_list = np.array(G.nodes()) # store the node id and its name
    W = nx.adjacency_matrix(G, nodelist=node_list, weight = weight).todense()

    Wtilde = np.matrix(np.zeros((n+1, n+1)))
    Wtilde[1:, 1:] = W
    Wtilde[1:, 0] = np.sum(W, 1)
    Wtilde[0, 1:] = np.sum(W, 0)

    x0 = np.array([0] + [1 if i in sub_com_nodes else -1 for i in node_list])
    
    return n, node_list, W, Wtilde, x0 


def local_sdp_sum_weights_mosek(n, x0, W, Wtilde, k):
    """Based on Feige et al.'s algorithm to formulate the densest k subgraph through semidefinite relaxation. 
    Use the Mosek sovler. 

    Args:
        n (int): number of nodes
        A (float[n+1][n+1]): The first column of the matrix is n - 2k, otherwise all ones.
        W (float[n][n]): Adjacency matrix of undirected graph G
        Wtilde (float[n+1][n+1]): W_0j = sum_i W_ij, W_i0 = sum_j W_ij, W_00 = 0
        k (int): The paprameter k
    
    Returns:
        obj : the relaxed objective value
        sol : The semidefinite matrix of size (n+1, n+1)
    """

    with Model("Densest Subgraph with small changes") as M:
        # M.setLogHandler(sys.stdout)

        # Create variable 'X' of size (n+1 * n+1)
        X = M.variable("X", Domain.inPSDCone(n+1))
        x = X.diag()
        
        A = np.outer(x0, x0)
        B = np.zeros((n+1, n+1))
        B[:, 0] = x0

        # Create constraints
        M.constraint("c1", Expr.dot(A, X), Domain.equalsTo((n - 2*k)**2))
        M.constraint("c2", Expr.dot(B, X), Domain.equalsTo(n - 2*k))
        M.constraint("c3", x, Domain.equalsTo(1.))

        # # Set the objective function to (c^t * x)
        M.objective("obj", ObjectiveSense.Maximize, Expr.dot(Wtilde, X))

        # # Solve the problem
        M.solve()

        # Get the solution values
        sol = X.level().reshape(n+1, n+1)
        obj = (M.primalObjValue() + np.sum(W))/8

        #print("objective is ", obj)
    return obj, sol


def local_sdp_sum_weights_rounding(X, x0, optvalue, n, node_list, G, k, I = 100, weight = 'weight'):
    """
    Cutting plane rounding technique.
    Choose theta1 = 0.65, theta2 = 3.87, the parameters can change according to the choice of k.

    Args:
        X (float[n+1][n+1]): A solution for the sdp, in np.array format
        x0 (int[n]): A -1 1 vector shows originally a node belongs to 
        optvalue (float): the optimal objective value for sdp
        n (int): number of nodes
        nodes (numpy array int[n]) : G.nodes()
        G (networkx.graph): undirected graph
        k (int): parameter k
        I (int): number of rounds
        weight (string, optional): None or 'weight'. Defaults to None.

    Returns:
        Unew: The nodes inside the final subgraph
        rhostar: Density of the final subgraph
    """
    
    #  start = time.time()
    
    L = np.linalg.cholesky(X)
    AR = AfterRounding(n)
    rhostar = 0
    
    for _ in range(I):
        size_of_switch_nodes, sub_nodes, switch_nodes, switch_inside_sub_nodes, switch_outside_sub_nodes = rounding_one_step(L, n, x0)
     
        # choose the best, theoretically should update with choosen parameters. 
        AR.rounding_update_storage(0, size_of_switch_nodes, sub_nodes, switch_nodes, switch_inside_sub_nodes, switch_outside_sub_nodes)
        Unew_temp = adjust_size_of_swich_nodes(G, AR, n, k, node_list, weight)
        if rho_subgraph(G, Unew_temp, weight) > rhostar:
            rhostar = rho_subgraph(G, Unew_temp, weight)
            Unew = Unew_temp 
    
    return Unew, rhostar


class AfterRounding: #stands for after rounding 
    def __init__(self, n):
        self.size_of_switch_nodes = 0
        self.z = 0
        self.sub_nodes = np.zeros(n)
        self.switch_nodes = np.zeros(n)
        self.switch_inside_sub_nodes =  np.zeros(n)
        self.switch_outside_sub_nodes =  np.zeros(n)
        
    def rounding_update_storage(self, z, size_of_switch_nodes, sub_nodes, switch_nodes, switch_inside_sub_nodes, switch_outside_sub_nodes):
        self.z = z
        self.size_of_switch_nodes = size_of_switch_nodes
        self.sub_nodes = sub_nodes
        self.switch_nodes = switch_nodes
        self.switch_inside_sub_nodes = switch_inside_sub_nodes
        self.switch_outside_sub_nodes = switch_outside_sub_nodes

def rounding_one_step(L, n, x0):
    r = sphere_uniform(n+1)
    x = np.sign(np.dot(L[1:, :], r) * np.dot(L[0, :],r))
    
    sub_nodes = np.where(x == 1)[0] # select the nodes that in the subgraph
    #print(sub_nodes)
    switch_nodes = np.where(x != np.array(x0))[0] # select the nodes that have changed positions
    #print(switch_nodes)

    switch_inside_sub_nodes = np.intersect1d(switch_nodes, sub_nodes) # Obtain nodes that both in C and U
    switch_outside_sub_nodes = np.setdiff1d(switch_nodes, sub_nodes) # Obtain nodes that only in C

    size_of_switch_nodes = np.size(switch_nodes)

    return size_of_switch_nodes, sub_nodes, switch_nodes, switch_inside_sub_nodes, switch_outside_sub_nodes


# def rounding_update_threshold(G, size_of_switch_nodes, sub_nodes, nodes, theta1, theta2, n, k, optvalue, weight):
#     if len(sub_nodes) == 0:
#         print(" === The size of the obtained subgraph is zero. This can happen when initial subgraph is too small, and k is too large. ===")
#         z = 0
#     else:
#         rho_init = rho_subgraph(G, nodes[sub_nodes], weight=weight)
#         z = rho_init * np.size(sub_nodes) / optvalue + theta1 * (n - len(sub_nodes))/ (n - k) + theta2 * (len(sub_nodes)) * (2*k - len(sub_nodes))/(n * n) # this one might need to be modified
    
#     return z

def adjust_size_of_swich_nodes(G, AR, n, k, node_list, weight):
    
    #print("the size of switched nodes is :", AR.size_of_switch_nodes)
    if AR.size_of_switch_nodes > k:
        # We need to select the subgraph, and shrink the size of nodes that have been switched
        # First include the nodes not in new_sub_nodes
        if np.size(AR.switch_inside_sub_nodes) <= k: # if it is enough to move nodes back from outside 
            #print(1)
            id_of_additional_nodes = np.union1d(AR.sub_nodes, AR.switch_outside_sub_nodes[:AR.size_of_switch_nodes - k])
            Unew = list(node_list[id_of_additional_nodes]) # add extra
            #print(np.size(np.setdiff1d(com, Unew)) + np.size(np.setdiff1d(Unew, com)))
            
        else: # not enough
            #print(2)
            AR.sub_nodes = np.union1d(AR.sub_nodes, AR.switch_outside_sub_nodes) # first add all nodes 
            g = G.subgraph(node_list[AR.sub_nodes])
            move_inside_final = greedy_charikar_fix_nodes(g, node_list[AR.switch_inside_sub_nodes], k, weight=weight)
            not_to_move_inside_final = np.setdiff1d(node_list[AR.switch_inside_sub_nodes], move_inside_final)
            Unew = list(np.setdiff1d(node_list[AR.sub_nodes], not_to_move_inside_final))
            #print(np.size(np.setdiff1d(com, Unew)) + np.size(np.setdiff1d(Unew, com)))
            
        
    elif AR.size_of_switch_nodes < k:
        #print(3)
        # Uniformly randomly adding k - num ber of vertices
        pool = np.delete(np.arange(n), np.union1d(AR.sub_nodes, AR.switch_nodes)) # select the nodes that xtilde != 1
        newNodes = np.random.choice(pool, size = k - AR.size_of_switch_nodes, replace = False)
        Unew = list(node_list[np.concatenate([AR.sub_nodes, newNodes])])
        #print(np.size(np.setdiff1d(com, Unew)) + np.size(np.setdiff1d(Unew, com)))
    
    elif AR.size_of_switch_nodes == k:
        #print(4)
        Unew = list(node_list[AR.sub_nodes])
        #print(np.size(np.setdiff1d(com, Unew)) + np.size(np.setdiff1d(Unew, com)))
    
    return Unew

########################################################
## Now we start the local search algorithm ##
@timeout_decorator.timeout(7200)
def local_search(k, G, com, weight = "weight", I = 100):
    """First choose k nodes to swap with com, then apply a local search method

    Args:
        k (int): parameter
        G (networkx graph): input graph
        com (list of selected nodes): list of selected nodes
    """
    g_1 = G.subgraph(com)
    g_2 = G.subgraph(com).copy()
    out_nodes = np.setdiff1d(G.nodes(), com)
    
    g_2.add_nodes_from(out_nodes)
    for node in out_nodes:
        for node_all in g_1.nodes():
            if G.has_edge(node, node_all):
                g_2.add_edge(node, node_all, weight = -G[node][node_all][weight])
    
    #out_nodes = np.intersect1d(g_2.nodes(), out_nodes) # remove nodes that do not have any edge

    node_dict_1, fib_heap_1 = init_heap_from_graph_given_set_of_nodes(g_1, com) 
    node_dict_2, fib_heap_2 = init_heap_from_graph_given_set_of_nodes(g_2, out_nodes)
    cur_size = len(com)
    cur_score = g_1.size(weight) # total edge weight of the given community 
    select_nodes = [] # store all the selected nodes 
    
    for _ in range(k):
        # find min node from graph (remove from heap)
        to_remove_1 = fib_heap_1.find_min()
        to_remove_2 = fib_heap_2.find_min() 
        
        if type(to_remove_1) == type(None): 
            to_remove = moveToInside(fib_heap_1, fib_heap_2, select_nodes, node_dict_1, node_dict_2, G)
            
        else: 
            if (cur_score - to_remove_1.key)/(cur_size - 1) > (cur_score - to_remove_2.key)/(cur_size + 1):
                to_remove = moveToOutside(fib_heap_1, fib_heap_2, select_nodes, node_dict_1, node_dict_2, G)
            else:
                to_remove = moveToInside(fib_heap_1, fib_heap_2, select_nodes, node_dict_1, node_dict_2, G)
        
        cur_score = cur_score - to_remove.key 
    
    sanity_test(G, select_nodes, com, cur_score, weight = "weight")
    
    # starting from here, we implement the hill-climbing, and repeat a thousand times. Let "com" be the original community  
    #final_com = hillClimbing(G, select_nodes, com, I = 100)
    
    final_com = np.union1d(np.setdiff1d(select_nodes, com), np.setdiff1d(com, select_nodes))
        
    return final_com


def moveToOutside(fib_heap_1, fib_heap_2, select_nodes, node_dict_1, node_dict_2, G, weight = 'weight'):
    # if we need to remove a node from the current community
    to_remove = fib_heap_1.extract_min()
    node_to_remove = to_remove.value
    
    #print("The next one before implementation is ", fib_heap_1.find_min().value)
        
    select_nodes.append(node_to_remove)
    
    # for every neighbor node this min node have inside com
    for neighbor in node_dict_1[node_to_remove][0]:
        edge_w = 1 if weight is None else G[node_to_remove][neighbor][weight]
        fib_heap_1.decrease_key(node_dict_1[neighbor][1], node_dict_1[neighbor][1].key - edge_w)
        node_dict_1[neighbor][0].remove(node_to_remove)

    del node_dict_1[node_to_remove]
    
    # for every neighbor node this mind node have outside com 
    for node in node_dict_2.keys():
        if G.has_edge(node, node_to_remove):
            edge_w = 1 if weight is None else G[node_to_remove][node][weight]
            fib_heap_2.decrease_key(node_dict_2[node][1], node_dict_2[node][1].key + edge_w)
            node_dict_2[node][0].remove(node_to_remove)
    
    return to_remove 

def moveToInside(fib_heap_1, fib_heap_2, select_nodes, node_dict_1, node_dict_2, G, weight = 'weight'):
    # if we need to include a node from outside 
    to_remove = fib_heap_2.extract_min()
    node_to_remove = to_remove.value 
    
    #print("The next one before implementation is ", fib_heap_2.find_min().value)
    
    select_nodes.append(node_to_remove)
    
    for neighbor in node_dict_2[node_to_remove][0]:
        edge_w = 1 if weight is None else G[node_to_remove][neighbor][weight] 
        fib_heap_1.decrease_key(node_dict_1[neighbor][1], node_dict_1[neighbor][1].key + edge_w)
    
    del node_dict_2[node_to_remove]
    
    for node in node_dict_2.keys():
        if G.has_edge(node, node_to_remove):
            edge_w = 1 if weight is None else G[node_to_remove][node][weight]
            fib_heap_2.decrease_key(node_dict_2[node][1], node_dict_2[node][1].key - edge_w)  
            
    return to_remove

def hillClimbing(G, select_nodes, com, I = 100):
    final_com = np.union1d(np.setdiff1d(select_nodes, com), np.setdiff1d(com, select_nodes))
    print("Before hill-climbing", rho_subgraph(G, final_com))
    
    un_select_nodes = np.setdiff1d(G.nodes(), select_nodes)
    list_n1 = np.random.choice(select_nodes, I)
    list_n2 = np.random.choice(un_select_nodes, I)
    
    counter = 0
    for i in range(I): # repeat the process
        n1 = list_n1[i]
        n2 = list_n2[i]
        tmp_final = copy.deepcopy(final_com)

        # if the picked node is inside final_com, and the unpicked node is also inside final_com 
        if (n1 in final_com) and (n2 in final_com):
            tmp_final = np.setdiff1d(tmp_final, [n1, n2])
        
        # if the picked node is inside final_com, and the unpicked node is outside of final_com
        elif (n1 in final_com) and (n2 not in final_com):
            tmp_final = np.setdiff1d(tmp_final, [n1])
            tmp_final = np.union1d(tmp_final, [n2])
            
        # if the picked node is outside of final_com, and the unpicked node is inside final_com
        elif (n1 not in final_com) and (n2 in final_com):
            tmp_final = np.setdiff1d(tmp_final, [n2])
            tmp_final = np.union1d(tmp_final, [n1])
            
        # if the picked node is outside of final_com, and the unpicked node is outside of final_com 
        elif (n1 not in final_com) and (n2 in final_com):
            tmp_final = np.union1d(tmp_final, [n1, n2])
        
        if rho_subgraph(G, tmp_final) > rho_subgraph(G, final_com):
            counter += 1
            
            select_nodes = np.setdiff1d(select_nodes, [n1])
            un_select_nodes = np.union1d(un_select_nodes, [n2])
            
            list_n1 = np.random.choice(select_nodes, I)
            list_n2 = np.random.choice(un_select_nodes, I)
            
            final_com = tmp_final

    print(counter *1.0 / I * 100, "% changes are made in hill climbing process.")
    return final_com 

def sanity_test(G, select_nodes, com, cur_score, weight = "weight"):
    final_com = np.union1d(np.setdiff1d(select_nodes, com), np.setdiff1d(com, select_nodes))
    if G.subgraph(final_com).size(weight) != cur_score:
        sys.exit("error somewhere, check if the graph contains a self loop \n")

