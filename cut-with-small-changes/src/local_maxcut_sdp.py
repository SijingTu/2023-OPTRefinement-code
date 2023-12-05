from .maxcut_utilities import sphere_uniform
from .maxcut_greedy import move_nodes_back
import numpy as np
import networkx as nx
from mosek.fusion import *

#sys.path.append(os.path.abspath(os.path.join('..', 'algorithm')))


def local_sdp_max_cut(G, initial_left_side_nodes, k, I = 100, weight = 'weight'):
    """A combination of sdp-solver and a simple rounding algorithm

    Args:
        G (networkx): undirected graph
        I (int, optional): number of rounds. Defaults to 100.
        weight (str, optional): 'weight' or none. Defaults to 'weight'.

    Returns:
        _type_: _description_
    """

    n, node_list, Ltilde, x0 = local_sdp_max_cut_param(G, initial_left_side_nodes, weight) # obtain parameters of max cut algorithm 
    opt, X = local_sdp_max_cut_mosek_solver(n, x0, Ltilde, k) # get the optimal solution
    final_nodes, final_cut = local_sdp_max_cut_rounding(X, initial_left_side_nodes, n, node_list, G, k, I, weight)

    print("the optimal solution for sdp is ", opt, "after rounding is", final_cut, "ratio is ", final_cut/opt)
    
    return final_nodes, final_cut, opt

def local_sdp_max_cut_param(G, sub_com_nodes, weight):
    n = G.number_of_nodes()
    node_list = np.array(G.nodes())
    L = nx.laplacian_matrix(G, nodelist=node_list, weight = weight).todense() #obtain graph laplacian 
    
    Ltilde = np.matrix(np.zeros((n+1, n+1)))
    Ltilde[1:, 1:] = L 
    
    x0 = np.array([0] + [1 if i in sub_com_nodes else -1 for i in node_list])
    return n, node_list, Ltilde, x0
    

def local_sdp_max_cut_mosek_solver(n, x0, Ltilde, k):
    """
    Use Mosek to solve the quadratic function without constraints. 
    """
    with Model("Max Cut with small changes") as M:
        #M.setLogHandler(sys.stdout)
        X = M.variable("X", Domain.inPSDCone(n+1))
        x = X.diag() 
        
        A = np.outer(x0, x0)
        B = np.zeros((n+1, n+1))
        B[:, 0] = x0
        
        #constraints
        M.constraint("c1", Expr.dot(A, X), Domain.equalsTo((n - 2*k)**2))
        M.constraint("c2", Expr.dot(B, X), Domain.equalsTo(n - 2*k))
        M.constraint("c3", x, Domain.equalsTo(1.))
        
        #objective
        M.objective("obj", ObjectiveSense.Maximize, Expr.dot(np.array(Ltilde), X))
       
        #solve the problem  
        M.solve()
    
        #Get the solution
        obj = M.primalObjValue() / 4.0
        
        print("obj is ", obj)
        sol = X.level().reshape(n+1, n+1)
    return obj, sol

def local_sdp_max_cut_rounding(X, initial_left_side_nodes, n, node_list, G, k, I = 100, weight = 'weight'):
    # Use the dks parameters to bound, might need to update 2022-09-27
    
    lower_triangle = np.linalg.cholesky(X)
    current_largest_cut = 0
    new_left_return = np.zeros(n)
    
    for _ in range(I):
        r = sphere_uniform(n+1)
        x = np.sign(np.dot(lower_triangle[1:, :], r) * np.dot(lower_triangle[0, :], r)) # if x_i is positive, then i is in the pasitive side, otherwise, i is in the negative side
        
        left_side_nodes = node_list[np.where(x==1)[0]] # the left side nodes 
        
        new_left = move_nodes_back(G, initial_left_side_nodes, left_side_nodes, k, weight)
        
        if nx.cut_size(G, new_left) > current_largest_cut:
            current_largest_cut = nx.cut_size(G, new_left)
            new_left_return = new_left
        
    return new_left_return, current_largest_cut