from .fibheap import FibonacciHeap as FibHeap
import networkx as nx
import copy
import numpy as np
import timeout_decorator

def check_undirected_graph(G):
    if isinstance(G, nx.Graph):
        return

    raise NotImplementedError('Only accept networkx undirected graph (nx.Graph).')


# @staticmethod
@timeout_decorator.timeout(172800)
def greedy_charikar(Ginput, k, weight='weight'):
    """
    
    MODIFIED: FIX k. 
    
    Charikar's 1/2 greedy algorithm

    This function greedily removes nodes, and ouputs a set of nodes that optimizes
    the objective rho(S) = e[S]/|S|


    Parameters
    ----------
    Ginput: undirected, graph (networkx)
    weight: str that specify the edge attribute name of edge weight; None if the graph is unweighted

    Returns
    -------
    H: list, subset of nodes  corresponding to densest subgraph
    rho: float, density of H induced subgraph

    """

    H = copy.deepcopy(Ginput)

    node_dict, fibheap, total_degree = init_heap_flowless_from_graph(H, weight=weight)
    Hstar, rhostar = greedy_helper_from_graph(H, node_dict, fibheap, total_degree, k, weight=weight)

    return Hstar, rhostar


def init_heap_flowless_from_graph(G, weight='weight'):
    """
    Parameters
    ----------
    G: undirected, graph (networkx)
    weight: str that specify the edge attribute name of edge weight; None if the graph is unweighted

    Returns
    ----------
    node_dict: dict, node id as key, tuple (neighbor list, heap node) as value. Here heap node is a
    pointer to the corresponding node in fibheap.
    fibheap: FibonacciHeap, support fast extraction of min degree node and value change.
    total_weight: edge weight sum.
    """
    node_dict = dict()
    fibheap = FibHeap()
    total_degree = 0
    for node in G.nodes():
        node_dict[node] = (list(), fibheap.insert(0, node))
        for neighbor in G[node]:
            if weight is None:
                edge_w = 1
            else:
                edge_w = G[node][neighbor][weight]
            fibheap.decrease_key(node_dict[node][1], node_dict[node][1].key + edge_w)
            node_dict[node][0].append(neighbor)
            total_degree += edge_w
    total_weight = total_degree / 2

    return node_dict, fibheap, total_weight


def greedy_helper_from_graph(G, node_dict, fib_heap, total_degree, k, weight='weight'):
    """
    UPDATE: modified the greedy algorithm to fix k
    
    Greedy peeling algorithm. Peel nodes iteratively based on their current degree.

    Parameters
    ----------
    G: undirected, graph (networkx)
    node_dict: dict, node id as key, tuple (neighbor list, heap node) as value. Here heap node is a
    pointer to the corresponding node in fibheap.
    fibheap: FibonacciHeap, support fast extraction of min degree node and value change.
    total_weight: edge weight sum.
    weight: str that specify the edge attribute name of edge weight; None if the graph is unweighted.

    Returns
    ----------
    H: list, subset of nodes corresponding to densest subgraph.
    max_avg: float, density of H induced subgraph.
    new_loads: dict, new loads for nodes, only used for the flowless algorithm when T>1.
    """
    n = G.number_of_nodes()
    avg_degree = total_degree / n
    H = list(G)
    max_avg = avg_degree

    for i in range(n - 1):
        # find min node from graph (remove from heap)
        to_remove = fib_heap.extract_min()
        node_to_remove = to_remove.value

        # for every neighbor node this min node have
        for neighbor in node_dict[node_to_remove][0]:
            edge_w = 1 if weight is None else G[node_to_remove][neighbor][weight]

            # here the key can be actually increased
            if neighbor != node_to_remove:
                fib_heap.decrease_key(node_dict[neighbor][1], node_dict[neighbor][1].key - edge_w)
                node_dict[neighbor][0].remove(node_to_remove)
            total_degree -= edge_w

        del node_dict[node_to_remove]
        
        # EXTRA CODE
        if i == n - k - 1:
            max_avg = total_degree / (n - i - 1)
            H = list(node_dict.keys())
            #print("size is ", len(H))
            break 

    return H, max_avg 


def greedy_charikar_fix_nodes(Ginput, set_of_nodes, k, weight='weight'):
    """
    
    MODIFIED: FIX k. 
    
    Charikar's 1/2 greedy algorithm

    This function greedily removes nodes, and ouputs a set of nodes that optimizes
    the objective rho(S) = e[S]/|S|

    Parameters
    ----------
    Ginput: undirected, graph (networkx)
    weight: str that specify the edge attribute name of edge weight; None if the graph is unweighted

    Returns
    -------
    H: list, subset of nodes  corresponding to densest subgraph
    rho: float, density of H induced subgraph

    """
    H = copy.deepcopy(Ginput)

    node_dict, fibheap = init_heap_from_graph_given_set_of_nodes(H, set_of_nodes, weight=weight)
    Hstar = greedy_helper_from_graph_given_set_of_nodes(H, set_of_nodes, node_dict, fibheap, k, weight=weight)

    return Hstar

def init_heap_from_graph_given_set_of_nodes(G, set_of_nodes, weight='weight'):
    """
    Parameters
    ----------
    G: undirected, graph (networkx)

    weight: str that specify the edge attribute name of edge weight; None if the graph is unweighted

    Returns
    ----------
    node_dict: dict, node id as key, tuple (neighbor list, heap node) as value. Here heap node is a
    pointer to the corresponding node in fibheap.
    fibheap: FibonacciHeap, support fast extraction of min degree node and value change.
    total_weight: edge weight sum.
    """
    node_dict = dict()
    fibheap = FibHeap()
    for node in set_of_nodes:
        node_dict[node] = (list(), fibheap.insert(0, node))
        for neighbor in G[node]:
            if weight is None:
                edge_w = 1
            else:
                edge_w = G[node][neighbor][weight]
            fibheap.decrease_key(node_dict[node][1], node_dict[node][1].key + edge_w)
            node_dict[node][0].append(neighbor)

    return node_dict, fibheap

def greedy_helper_from_graph_given_set_of_nodes(G, set_of_nodes, node_dict, fib_heap, k, weight=None):
    """
    UPDATE: modified the greedy algorithm to fix k
    
    Greedy peeling algorithm. Peel nodes iteratively based on their current degree, remove n-k nodes.

    Parameters
    ----------
    G: undirected, graph (networkx)
    node_dict: dict, node id as key, tuple (neighbor list, heap node) as value. Here heap node is a
    pointer to the corresponding node in fibheap.
    fibheap: FibonacciHeap, support fast extraction of min degree node and value change.
    total_weight: edge weight sum.
    weight: str that specify the edge attribute name of edge weight; None if the graph is unweighted.

    Returns
    ----------
    H: list, subset of nodes corresponding to densest subgraph.
    max_avg: float, density of H induced subgraph.
    new_loads: dict, new loads for nodes, only used for the flowless algorithm when T>1.
    """
    n = np.size(set_of_nodes)
    
    refer_list = np.zeros(np.amax(G.nodes()) + 1) # create a list to quickly select the nodes in the set_of_nodes
    for x in set_of_nodes:
        refer_list[x] = 1
    

    for i in range(n - 1):
        # find min node from graph (remove from heap)
        to_remove = fib_heap.extract_min()
        node_to_remove = to_remove.value

        # for every neighbor node this min node have
        for neighbor in node_dict[node_to_remove][0]:
            edge_w = 1 if weight is None else G[node_to_remove][neighbor][weight]

            # here the key can be actually increased, ensure that a node is in the choosen set
            if (neighbor != node_to_remove) and (refer_list[neighbor] == 1):
                fib_heap.decrease_key(node_dict[neighbor][1], node_dict[neighbor][1].key - edge_w)
                node_dict[neighbor][0].remove(node_to_remove)

        del node_dict[node_to_remove]
        
        # EXTRA CODE
        if i == n - k - 1:
            H = list(node_dict.keys())
            #print("size is ", len(H))
            break 

    return H


def exact_densest(hyper_list: list):
    """
    Goldberg's exact max flow algorithm.
    Parameters
    ----------
    G: undirected, graph (networkx).
    Returns
    -------
    Sstar: list, subset of nodes corresponding to densest subgraph.
    opt: float, density of Sstar induced subgraph.
    """
    # self.check_undirected_graph(G)

    if isinstance(hyper_list, nx.Graph):
        return exact_densest_from_graph(hyper_list)

    weight = len(hyper_list[0])
    node_w = dict()
    for e in hyper_list:
        for node in e:
            node_w[node] = 1.0 if node not in node_w else node_w[node]+1

    inf = len(hyper_list)*10.0

    m = float(len(hyper_list))
    n = float(len(node_w))
    minD = m / n  # rho^* >= m/n since V is a feasible solution
    maxD = m
    # a tighter upper bound
    # degree_seq = [d for n,d in G.degree()]
    # maxD = (max(degree_seq)-1 )/2

    opt = minD
    Sstar = node_w.keys()
    # print(maxD, minD)
    if minD == maxD:
        return Sstar, maxD
    while maxD - minD > 1 / n ** 2:  # binary search
        query = (maxD + minD) / 2
        # print('Query value is ',query )
        H = create_flow_network_from_list(hyper_list, query, node_w, inf, weight)
        solution = nx.minimum_cut(H, 's', 't', capacity='capacity')  # unspecified behavior
        
        cut = solution[1][0]
        # print(solution[0], cut)
        if cut == {'s'}:
            maxD = query  # this means there is no subgraph S such that the degree density is at least query
        else:
            #             print('Found denser subgraph!')
            minD = query
            Sstar = cut
            opt = query
    Sstar = list(set(Sstar)&set(node_w.keys()))
    return Sstar, opt

# @staticmethod
def exact_densest_from_graph(G: nx.Graph):
    """
    Goldberg's exact max flow algorithm.
    Parameters
    ----------
    G: undirected, graph (networkx).
    Returns
    -------
    Sstar: list, subset of nodes corresponding to densest subgraph.
    opt: float, density of Sstar induced subgraph.
    """
    check_undirected_graph(G)

    m = G.number_of_edges()
    n = G.number_of_nodes()
    minD = m / n  # rho^* >= m/n since V is a feasible solution
    maxD = (n - 1) / 2
    # a tighter upper bound
    # degree_seq = [d for n,d in G.degree()]
    # maxD = (max(degree_seq)-1 )/2

    opt = 0
    Sstar = G.nodes()
    if minD == maxD:
        return Sstar, maxD
    while maxD - minD > 1 / n ** 2:  # binary search
        query = (maxD + minD) / 2
        # print('Query value is ',query )
        H = create_flow_network_from_graph(G, query)
        solution = nx.minimum_cut(H, 's', 't', capacity='capacity')  # unspecified behavior
        #         print(solution[0])
        cut = solution[1][0]
        #         print(cut)
        if cut == {'s'}:
            maxD = query  # this means there is no subgraph S such that the degree density is at least query
        else:
            #             print('Found denser subgraph!')
            minD = query
            Sstar = cut
            opt = query
    Sstar = list(set(Sstar)&set(G.nodes()))
    return Sstar, opt

def create_flow_network_from_graph(G, query):
    m = G.number_of_edges()
    G = nx.DiGraph(G)
    H = G.copy()
    H.add_node('s')
    H.add_node('t')
    for e in G.edges():
        H.add_edge(e[0], e[1], capacity=1)

    for v in G.nodes():
        H.add_edge('s', v, capacity=m)
        H.add_edge(v, 't', capacity=m + 2 * query - G.in_degree(v))
    return H


def create_flow_network_from_list(hyper_list, query, node_w, inf, weight):

    # m = G.number_of_edges()
    # G = nx.DiGraph(G)
    H = nx.DiGraph()
    H.add_node('s')
    H.add_node('t')
    for e in hyper_list:
        for node in e:
            H.add_edge(node, tuple(e), capacity=1.0/weight)
            H.add_edge(tuple(e), node, capacity=inf)

    for v in node_w:
        H.add_edge('s', v, capacity=node_w[v]/weight)
        H.add_edge(v, 't', capacity=query)
    return 