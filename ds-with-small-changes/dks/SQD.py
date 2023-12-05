import sys
from .Goldberg import Densest_Subgraph
import networkx as nx
import numpy as np
import itertools
import random
import math
import time
import timeout_decorator
from memory_profiler import profile


def construct_graph_with_unselected_nodes(G, sub_com_nodes, weight="weight"):
    """Given a graph G, a subgraph com. 
    Construct a subgraph with nodes other than the sub_com_nodes.
    Reassign the node weights, w(v) = sum_(u in Neighbor(v), u in sub_com_nodes) w(u).

    Args:
        G (NetworkX undirected graph): undirected graph
        sub_com_nodes ([int]): list of nodes in the given subgraph 
        weight (string) : None or 'weight'
        
    Return:
        temp_graph_nodes: a list of nodes
        temp_graph_edges: a list of edges
        temp_graph_node_weight: node_weights
        temp_graph_num_of_edges(int): number of edges
        ind_to_node
    """ 
    temp_graph_nodes = np.setdiff1d(list(G), sub_com_nodes) # get new node list

    #print("temp_graph_nodes is ", temp_graph_nodes)
    
    node_to_ind = {} # rename all the nodes starting from 0
    ind_to_node = {} # index to node
    
    temp_graph_edges = []
    temp_graph_num_of_edges = 0
    idx = 0
    
    for new_node in temp_graph_nodes:
        node_to_ind[new_node] = idx
        ind_to_node[idx] = new_node
        idx += 1
        
    temp_graph_node_weight = [0]*idx # init all the node weights to be 1
        
    for e in G.edges():
        if ((e[0] not in sub_com_nodes) and (e[1] not in sub_com_nodes)):
            temp_graph_edges.append((node_to_ind[e[0]], node_to_ind[e[1]]))
            temp_graph_num_of_edges += 1
        if ((e[0] in sub_com_nodes) and (e[1] not in sub_com_nodes)):
            temp_graph_node_weight[node_to_ind[e[1]]] += 1
        if ((e[0] not in sub_com_nodes) and (e[1] in sub_com_nodes)):
            temp_graph_node_weight[node_to_ind[e[0]]] += 1
    
    return temp_graph_nodes, np.array(temp_graph_edges), temp_graph_node_weight, temp_graph_num_of_edges, ind_to_node


def is_independent_to_all(node, adjacencies,nodes):

    for othernode in nodes:
        if node in adjacencies[othernode]:
            return False
    return True

def is_adjacent_to_all(node, adjacencies, nodes):
    for othernode in nodes:
        if node not in adjacencies[othernode]:
            return False
    return True

def find_min_red_edges_node(redcount, max_possible):
    minimum=max_possible
    min_node=-1
    for node in redcount.keys():
        if redcount[node]<=minimum:
            minimum=redcount[node]
            min_node=node
            
    return minimum,min_node
        
def keep_subgraph(edgelist, which_ones):
    new_edgelist=[]
    for e in edgelist:
        if e[0] in which_ones and e[1] in which_ones:
            new_edgelist.append(e)
    return np.array(new_edgelist)




def join_step(itemsets, adjacencies):
    """
    Join k length itemsets into k + 1 length itemsets.

    This algorithm assumes that the list of itemsets are sorted, and that the
    itemsets themselves are sorted tuples. Instead of always enumerating all
    n^2 combinations, the algorithm only has n^2 runtime for each block of
    itemsets with the first k - 1 items equal.

    Parameters
    ----------
    itemsets : list of itemsets
        A list of itemsets of length k, to be joined to k + 1 length
        itemsets.

    Examples
    --------
    >>> # This is an example from the 1994 paper by Agrawal et al.
    >>> itemsets = [(1, 2, 3), (1, 2, 4), (1, 3, 4), (1, 3, 5), (2, 3, 4)]
    >>> list(join_step(itemsets))
    [(1, 2, 3, 4), (1, 3, 4, 5)]
    """
    i = 0
    # Iterate over every itemset in the itemsets
    while i < len(itemsets):

        # The number of rows to skip in the while-loop, intially set to 1
        skip = 1

        # Get all but the last item in the itemset, and the last item
#        itemset_first= itemsets[i][:-1]
#        itemset_last=itemsets[i][-1]
        *itemset_first, itemset_last = itemsets[i]

        # temp_graph_edgese now iterate over every itemset following this one, stopping
        # if the first k - 1 items are not equal. If we're at (1, 2, 3),
        # we'll consider (1, 2, 4) and (1, 2, 7), but not (1, 3, 1)

        # Keep a list of all last elements, i.e. tail elements, to perform
        # 2-combinations on later on
        tail_items = [itemset_last]
        tail_items_append = tail_items.append  # Micro-optimization

        # Iterate over ever itemset following this itemset
        for j in range(i + 1, len(itemsets)):

            # Get all but the last item in the itemset, and the last item
            *itemset_n_first, itemset_n_last = itemsets[j]
#            itemset_n_first= itemsets[j][:-1]
#            itemset_n_last=itemsets[j][-1]

            # If it's the same, append and skip this itemset in while-loop
            if itemset_first == itemset_n_first:

                # Micro-optimization
                tail_items_append(itemset_n_last)
                skip += 1

            # If it's not the same, break out of the for-loop
            else:
                break

        # For every 2-combination in the tail items, yield a new candidate
        # itemset, which is sorted.
        itemset_first_tuple = tuple(itemset_first)
        for a, b in sorted(itertools.combinations(tail_items, 2)):
            if is_independent_to_all(a,adjacencies,itemset_first_tuple) and is_independent_to_all(b,adjacencies,itemset_first_tuple) and a not in adjacencies[b]:
                yield itemset_first_tuple + (a,) + (b,)

        # Increment the while-loop counter
        i += skip


def prune_step(
    itemsets, possible_itemsets):
    """
    Prune possible itemsets whose subsets are not in the list of itemsets.

    Parameters
    ----------
    itemsets : list of itemsets
        A list of itemsets of length k.
    possible_itemsets : list of itemsets
        A list of possible itemsets of length k + 1 to be pruned.

    Examples
    -------
    >>> itemsets = [('a', 'b', 'c'), ('a', 'b', 'd'),
    ...             ('b', 'c', 'd'), ('a', 'c', 'd')]
    >>> possible_itemsets = list(join_step(itemsets))
    >>> list(prune_step(itemsets, possible_itemsets))
    [('a', 'b', 'c', 'd')]
    """

    # For faster lookups
    itemsets = set(itemsets)

    # Go through every possible itemset
    for possible_itemset in possible_itemsets:

        # Remove 1 from the combination, same as k-1 combinations
        # The itemsets created by removing the last two items in the possible
        # itemsets must be part of the itemsets by definition,
        # due to the way the `join_step` function merges the sorted itemsets

        for i in range(len(possible_itemset) - 2):
            removed = possible_itemset[:i] + possible_itemset[i + 1 :]

            # If every k combination exists in the set of itemsets,
            # yield the possible itemset. If it does not exist, then it's
            # support cannot be large enough, since supp(A) >= supp(AB) for
            # all B, and if supp(S) is large enough, then supp(s) must be large
            # enough for every s which is a subset of S.
            # This is the downward-closure property of the support function.
            if removed not in itemsets:
                break

        # If we have not breaked yet
        else:
            yield possible_itemset


def apriori_gen(itemsets, adjacencies):

    possible_extensions = join_step(itemsets, adjacencies)
    yield from prune_step(itemsets, possible_extensions)

def construct_subsets(generated_sets, neighs, adjacencies, sigma):
#    k=2
#    for pair in itertools.combinations(neighs,2):
#        #the two nodes are not adjacent, so this tuple can be a candidate
#        if pair[0] not in adjacencies[pair[1]]:
#            tuples.append(pair)
    
    
        # STEP 2 - Build up the size of the itemsets
    # ------------------------------------------
    prev_itemsets=[(i,) for i in neighs]
    # While there are itemsets of the previous size
    k = 2
    while (k<=sigma+1):
#        print(" Counting itemsets of length {}.".format(k))

        # STEP 2a) - Build up candidate of larger itemsets

        # Generate candidates of length k + 1 by joning, prune, and copy as set
        C_k = list(apriori_gen(prev_itemsets, adjacencies))

        # If no candidate itemsets were found, break out of the loop
        if not C_k:
            return
            

        prev_itemsets=C_k       

        k += 1

        
    for itemset in prev_itemsets:
        if itemset not in generated_sets:
            generated_sets.add(itemset)


    return

    
            

def construct_bipartite_graph(adjacencies, selected_nodes, sigma):
    Bipartite=nx.Graph()    
    cou=max(selected_nodes)+1
    generated_sets=set()
            
    #create subsets based on neighbourhood of every node
    for node in selected_nodes:
        neighs=adjacencies[node]
        Bipartite.add_node(node)
        construct_subsets(generated_sets, neighs, adjacencies, sigma)  
        
    how_many_subsets=0
    A={}
    
    for subset in generated_sets:
        how_many_subsets+=1
        A[cou]=subset
        for node in selected_nodes:
            if is_adjacent_to_all(node, adjacencies, subset):
                Bipartite.add_edge(node,cou, color='red')
            if node in subset:
                Bipartite.add_edge(node,cou, color='black')
            
        cou+=1
#    print("About to print bipartite edges!")
#    print(Bipartite.edges())
#    for e in Bipartite.edges():
#        print(e)
#        print ( str( (e[0],A[e[1]])) +Bipartite[e[0]][e[1]]['color'])
    
    return Bipartite,how_many_subsets
    
    

def compute_qeo(subgraph, selected_nodes, sigma):
    #transform edgelist to adjacency list
    adjacencies={}
    for e in subgraph:
        v1=e[0]
        v2=e[1]
        if v1 not in adjacencies:
            adjacencies[v1]=[]
        if v2 not in adjacencies:
            adjacencies[v2]=[]
        adjacencies[v1].append(v2)
        adjacencies[v2].append(v1)

    Bipartite,how_many_subsets=construct_bipartite_graph(adjacencies, selected_nodes, sigma)    

#    print( 'ellapsed seconds to build bipartite graph {}'.format(time.clock() - start_time))
    qeo={}
    redcount={}
    #count how many red edges are adjacent to each node
    for nodeB in selected_nodes:
        if nodeB not in redcount:
            redcount[nodeB]=0
        for nodeA in Bipartite.neighbors(nodeB):
            if Bipartite[nodeB][nodeA]['color']=='red':
                redcount[nodeB]+=1
                
#    print (redcount)
    #start removing nodes
    for i in range(len(selected_nodes)):
        num_red_edges,min_red_edges_node=find_min_red_edges_node(redcount, how_many_subsets)
        if num_red_edges>0:
            print ("There does not exist a qeo with sigma: {}".format(sigma))
            return qeo,[] # will retrun a list, leads to an error. 
        #remove all neighbors
        neighs=list(Bipartite.neighbors(min_red_edges_node))
        for node in neighs:
            #update red edges counter for all deleted red edges adjacent to node from set A that will be deleted
            for nodeB in Bipartite.neighbors(node):
                if Bipartite[nodeB][node]['color']=='red':
                    redcount[nodeB]-=1
            #remove node from set A
            Bipartite.remove_node(node)
        #remove the node
        Bipartite.remove_node(min_red_edges_node)
        del redcount[min_red_edges_node]
        qeo[min_red_edges_node]=i
        
        
    #process adjacency list to convert it to predecessor list in the qeo
    
    for i in selected_nodes:
        order=qeo[i]
        adjacencies[i]=[j for j in adjacencies[i] if qeo[j]<qeo[i]]
                
    
    print ("Found a qeo with sigma: {}".format(sigma), flush=True)
    return qeo, adjacencies


def count_edges(temp_graph_edges, d_vF, nodes):
    new_edges=0
    for e in temp_graph_edges:
        if e[0] in nodes and e[1] in nodes:
            new_edges+=1
#    print("new edges from weak edges: "+str(new_edges))
    for node in nodes:
        new_edges+=d_vF[node]
        
    return new_edges
    
@profile 
def remove_nodes(k, subgraph,selected_nodes, sigma, d_vF):
    try:
        qeo,predecessor_list=compute_qeo(subgraph,selected_nodes,sigma)
    except:
        raise MemoryError(f"Memory usage exceeds the limit")
    #print(qeo)
    #print(predecessor_list)
    thresh=math.floor(k/2.0)
    dense_subset=[]
    #check if there is a node with a large predecessor set
    print(thresh, flush=True)
    for node in predecessor_list.keys():
        if len(predecessor_list[node])>=thresh:
            dense_subset=predecessor_list[node]
            print("There is a node with a large predecessor set")
            break
            
    if len(dense_subset)>0:
        feasible_subset=random.sample(dense_subset,k=thresh)
#        print(feasible_subset)
        sorted_indices=np.argsort(d_vF)[::-1]
        for node in sorted_indices:
            if node not in feasible_subset and len(feasible_subset)<k:
                feasible_subset.append(node)

#    print(qeo)
    return feasible_subset


def is_tuple_eq(tup1, tup2):
    if tup1[0]==tup2[0] and tup1[1]==tup2[1]:
        return True
    return False



def take_subgraph(edgelist, nodes):
    """Get the largest connected component subgraph from nodes.

    Args:
        edgelist (numpy list with tuples): []
        nodes (list): []

    Returns:
        largest connected component subgraph edge list, nodes.
    """
    subgraph = []
    for e in edgelist:
        if (e[0] in nodes) and (e[1] in nodes):
            subgraph.append(e)
    
    # Create a graph from the edge list
    G = nx.Graph(subgraph)
    
    # Find the largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    
    # Filter the subgraph to include only edges from the largest connected component
    largest_cc_subgraph = [(u, v) for u, v in subgraph if u in largest_cc and v in largest_cc]
    
    return np.array(largest_cc_subgraph), list(largest_cc)



# def take_subgraph(edgelist, nodes):
#     """Get subgraph from nodes, also, remove the singletons in nodes. 

#     Args:
#         edgelist (numpy list with tuples): []
#         nodes (list): []

#     Returns:
#         subgraph edge list, nodes.
#     """
#     subgraph=[]
#     subgraph_nodes = set()
#     for e in edgelist:
#         if (e[0] in nodes) and (e[1] in nodes):
#             subgraph.append(e)
#             subgraph_nodes.add(e[0])
#             subgraph_nodes.add(e[1])
    
#     if len(list(subgraph_nodes)) != len(nodes):
#         print("There are some singletons")
            
#     return np.array(subgraph), list(subgraph_nodes)

    
def iterative_calls(temp_graph_edges,no_edges,node_weights, k):
    removed_edges=[]
    selected_nodes=[]
    subgraph=[]
    num_removed_nodes=0
    iters=0
#    print ("in the first iteration node_weights has this many elements: {}".format(len(node_weights) ))
    while num_removed_nodes<k/2.0:
        iters+=1
#        print("Iterative call: "+str(iters))
#        print("selected nodes: "+str(selected_nodes))
#        print("node weights: "+str(set(zip(range(len(node_weights)),node_weights))))
#        print(temp_graph_edges)
        new_selected_nodes=Densest_Subgraph(temp_graph_edges,no_edges,node_weights, selected_nodes)
#        print ("selected to remove: {}".format(new_selected_nodes))
            
        #remove selected nodes and update weight of their neighbours
        cou=0
        indices_to_remove=[]
        for tupl in temp_graph_edges:
            i=tupl[0]
            j=tupl[1]
            if i in new_selected_nodes:
                node_weights[i]=0
                if j in new_selected_nodes:
                    node_weights[j]=0
                else:
                    node_weights[j]+=1
                indices_to_remove.append(cou)
                removed_edges.append(temp_graph_edges[cou])
            elif j in new_selected_nodes:
                #j is in removed nodes by i is not
                node_weights[j]=0
                node_weights[i]+=1
                indices_to_remove.append(cou)
                removed_edges.append(temp_graph_edges[cou])
            cou+=1
 #       print("len of temp_graph_edges before removal: "+str(len(temp_graph_edges)))
        temp_graph_edges=np.delete(temp_graph_edges,indices_to_remove,axis=0)
#        print("len of temp_graph_edges after removal: "+str(len(temp_graph_edges)))
        no_edges=no_edges-len(indices_to_remove)
        
        selected_nodes=selected_nodes+new_selected_nodes
#        print("selected so far: {}".format(selected_nodes))
        num_removed_nodes=len(selected_nodes)
#        print("I have removed so far {} nodes".format(num_removed_nodes))
    
    return selected_nodes
    
@timeout_decorator.timeout(36000)
def run_SQD(G, sub_com_nodes, k, sigma):
    """run_SDQ, note if selected nodes >= 700, then directly return with "LARGE_DENSE_GRAPH".
    """
    # LARGE_DENSE_GRAPH = 600

    _, temp_graph_edges, node_weights, no_edges, ind_to_node = construct_graph_with_unselected_nodes(G, sub_com_nodes)
    d_vF=node_weights.copy()
    
    #iterative calls
    selected_nodes=iterative_calls(temp_graph_edges.copy(),no_edges,node_weights, k)
       
    # print("temp_graph_edges size is after iterative calls, ", len(temp_graph_edges))
    
    # if size_of_dense_subgraph > LARGE_DENSE_GRAPH:
    #     return "LARGE_DENSE_GRAPH", size_of_dense_subgraph

    try: 
        subgraph, subgraph_nodes =take_subgraph(temp_graph_edges,selected_nodes)
        size_of_dense_subgraph = len(subgraph_nodes)
    except: 
        return "EMPTY_SUBGRAPH", 0
    
    print("the total node weights of the subgraph is, ", sum([d_vF[node] for node in subgraph_nodes])) # the edges from S to F
    print("the total edges of subgraph is, ", len(subgraph))
    
    print("The iterative calls gave a solution of size: "+str(len(subgraph_nodes)), flush=True)
#    sigma=4
    if len(subgraph_nodes) > k: 
        #we need to remove some nodes
        print ("removing...", flush=True)
        try: 
            solution=remove_nodes(k, subgraph, subgraph_nodes, sigma, d_vF)
        except:
            return "SMALL_SIGMA", size_of_dense_subgraph 
    else: 
        print("no need to remove nodes", flush=True)
        sorted_indices=np.argsort(d_vF)[::-1]
        solution=list(subgraph_nodes)
        for node in sorted_indices:
            if node not in solution and len(solution)<k:
                solution.append(node)
    
    subgraph_nodes = [ind_to_node[ind] for ind in solution]

    return "SMALL_DENSE_GRAPH", size_of_dense_subgraph, np.array(subgraph_nodes)