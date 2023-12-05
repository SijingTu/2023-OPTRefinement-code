""" The following part is an implementation for the densest k subgraph 

Approximation Algorithms for Maximization Problems arising in Graph Partitioning 
by Feige and Langberg
"""
import copy
import numpy as np
from mosek.fusion import * 
import networkx as nx
from .utilities import * 
from .peeling import greedy_charikar

def sdp_dks(k, G, I = 100, weight = 'weight'):
    """A combination of sdp-solver, and a simple randomized rounding technique.

    Args:
        k (int): cardinality parameter k
        G (network x graph): undirected graph
        I : rounding time
        weight : weighted graph
        
    Return:
        select_nodes: 
        density:
        opt / k:
    """
    
    n = G.number_of_nodes() 
    nodelist = np.array(G.nodes()) # store the node id and its name
    W = nx.adjacency_matrix(G, nodelist=nodelist).todense()

    Wtilde = np.matrix(np.zeros((n+1, n+1)))
    Wtilde[1:, 1:] = W
    Wtilde[1:, 0] = np.sum(W, 1)
    Wtilde[0, 1:] = np.sum(W, 0)

    ###########
    # Based on Feige et al
    # Ai = np.ones(n+1) 
    # Ai[0] = n-2*k

    # A = np.zeros((n+1, n+1))
    # for i in range(n+1):
    #     A[i :] = Ai
    ##########
    
    # update with new constraints
    x0 = np.zeros(n+1)
    x0[1:] = np.ones(n)
    A = np.outer(x0, x0)
    B = np.zeros((n+1, n+1))
    B[:, 0] = x0

    opt, X = sdp_mosek_changed_constraints(n, A, B, k, W, Wtilde)
    select_nodes, density = simple_rounding(X, n, nodelist, G, k)
    
    return select_nodes, density, opt / k

def sdp_mosek_changed_constraints(n, A, B, k, W, Wtilde):
    """Calculate the densest k subgraph through semidefinite relaxation, replace Feige et al.'s constraints

    Args:
        n (int): number of nodes
        A (float[n+1][n+1]): all ones except the first row and columns are zeros
        B (float [n+1]): all zeros except the first column, which is all-ones except the first row
        k int: parameter
        W (float[n][n]): Adjacency matrix of undirected graph G
        Wtilde (float[n+1][n+1]): W_0j = sum_i W_ij, W_i0 = sum_j W_ij, W_00 = 0
    
    Returns:
        obj : the relaxed objective value
        sol : The semidefinite matrix of size (n+1, n+1)
    """

    with Model("Densest k Subgraph") as M:

        #M.setLogHandler(sys.stdout)

        # Create variable 'X' of size (n+1 * n+1)
        X = M.variable("X", Domain.inPSDCone(n+1))
        x = X.diag()

        # Create constraints
        M.constraint("c1", Expr.dot(A, X), Domain.equalsTo((n - 2*k)**2))
        M.constraint("c2", Expr.dot(B, X), Domain.equalsTo(-n + 2*k))
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

def sdp_mosek_fiege(n, A, W, Wtilde):
    """Calculate the densest k subgraph through semidefinite relaxation, based on Feige et al.'s algorithm.

    Args:
        n (int): number of nodes
        A (float[n+1][n+1]): The first column of the matrix is n - 2k, otherwise all ones.
        W (float[n][n]): Adjacency matrix of undirected graph G
        Wtilde (float[n+1][n+1]): W_0j = sum_i W_ij, W_i0 = sum_j W_ij, W_00 = 0
    
    Returns:
        obj : the relaxed objective value
        sol : The semidefinite matrix of size (n+1, n+1)
    """

    with Model("Densest k Subgraph") as M:

        #M.setLogHandler(sys.stdout)

        # Create variable 'X' of size (n+1 * n+1)
        X = M.variable("X", Domain.inPSDCone(n+1))
        x = X.diag()

        # Create constraints
        M.constraint("c1", Expr.mulDiag(A, X), Domain.equalsTo(0.))
        M.constraint("c2", x, Domain.equalsTo(1.))

        # # Set the objective function to (c^t * x)
        M.objective("obj", ObjectiveSense.Maximize, Expr.dot(Wtilde, X))

        # # Solve the problem
        M.solve()

        # Get the solution values
        sol = X.level().reshape(n+1, n+1)
        obj = (M.primalObjValue() + np.sum(W))/8

        #print("objective is ", obj)
        
    return obj, sol

def simple_rounding(X, n, nodelist, G, k, I = 100, weight = 'weight'):
    """
    Cutting plane rounding technique.
    Choose theta1 = 0.65, theta2 = 3.87, the parameters can change according to the choice of k.

    Args:
        X (float[n+1][n+1]): A solution for the sdp, in np.array format
        optvalue (float): the optimal objective value for sdp
        n (int): number of nodes
        nodelist : a list of nodes
        G (networkx.graph): undirected graph
        k (int): parameter k
        I (int): number of rounds
        weight (string, optional): None or 'weight'. Defaults to None.

    Returns:
        Unew: The selected nodes
        rhostar: Density of the selected nodes.
    """
    
    size_select = 0
    rhostar_largest = 0
    
    L = np.linalg.cholesky(X)
    
    for _ in range(I):
        r = sphere_uniform(n + 1)
        x = np.sign(np.dot(L[1:, :], r) * np.dot(L[0, :],r))
        
        Utilde = np.where(x==1)[0] # modified
        #print("size of Utilde is ", len(Utilde))
        size_select = len(Utilde)
        
        if len(Utilde) == 0:
            continue
        
        #print("before greedy, rhostar is:, ", rho_init)
        
        if size_select > k:
            # We need to select the subgraph, and shrink the size
            g = G.subgraph(nodelist[Utilde])
            Unew, rhostar = greedy_charikar(g, k, weight=weight)
            
        elif size_select < k:
            # Uniformly randomly adding k - number of vertices
            pool = np.delete(np.arange(n), Utilde) # select the nodes that xtilde != 1
            newNodes = np.random.choice(pool, size = k - size_select, replace = False)
            Unew = list(nodelist[np.concatenate([Utilde, newNodes])])
            rhostar = rho_subgraph(G, Unew, weight=weight)
        
        elif size_select == k:
            Unew = list(nodelist[Utilde])
            rhostar = rho_subgraph(G, Unew, weight = weight)
            
        if rhostar > rhostar_largest:
            rhostar_largest = rhostar
            Unew_largest = Unew
    
    return Unew_largest, rhostar_largest