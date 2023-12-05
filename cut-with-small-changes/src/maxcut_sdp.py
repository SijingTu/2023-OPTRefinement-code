from .maxcut_utilities import sphere_uniform
import numpy as np
import networkx as nx
from mosek.fusion import *
#from pyrsistent import T

#sys.path.append(os.path.abspath(os.path.join('..', 'algorithm')))


def sdp_max_cut(G, I = 100, weight = 'weight'):
    """A combination of sdp-solver and a simple rounding algorithm

    Args:
        G (networkx): undirected graph
        I (int, optional): number of rounds. Defaults to 100.
        weight (str, optional): 'weight' or none. Defaults to 'weight'.

    Returns:
        _type_: _description_
    """

    n, node_list, L = sdp_max_cut_param(G, weight) # obtain parameters of max cut algorithm 
    opt, X = sdp_max_cut_mosek_solver(L, n) # get the optimal solution
    obj, left_side_nodes = sdp_max_cut_rounding(X, L, n, node_list, I, weight)
    print("the optimal solution for sdp is ", opt, "after rounding is", obj, "ratio is ", obj/opt)

    return left_side_nodes

def sdp_max_cut_param(G, weight):
    n = G.number_of_nodes()
    node_list = np.array(G.nodes())
    L = nx.laplacian_matrix(G, nodelist=node_list, weight = weight).todense() #obtain graph laplacian 
    return n, node_list, L
    

def sdp_max_cut_mosek_solver(L, n):
    """
    Use Mosek to solve the quadratic function without constraints. 
    """
    with Model("Max Cut") as M:
        #M.setLogHandler(sys.stdout)
        X = M.variable("X", Domain.inPSDCone(n))
        x = X.diag() 
        
        #constraints
        M.constraint("c1", x, Domain.equalsTo(1.))
        
        #objective
        M.objective("obj", ObjectiveSense.Maximize, Expr.dot(np.array(L), X))
       
        #solve the problem  
        M.solve()
    
        #Get the solution

        obj = M.primalObjValue() / 4.0
        sol = X.level().reshape(n, n)

    return obj, sol


def sdp_max_cut_rounding(X, L, n, node_list, I = 100, weight = 'weight'):
    lower_triangle = np.linalg.cholesky(X)
    obj = 0
    out_put_x = np.zeros(n)
    
    for _ in range(I):
        r = sphere_uniform(n)
        x = np.sign(np.dot(lower_triangle, r))
        
        tmp_obj = np.dot(np.dot(x, L), x) / 4
        if tmp_obj > obj:
            obj = tmp_obj 
            out_put_x = x 
    
    left_side_nodes = [node_list[i] for i in range(n) if out_put_x[i] == -1] # put the nodes with final sign -1 to the left side
    
    return obj, left_side_nodes
            