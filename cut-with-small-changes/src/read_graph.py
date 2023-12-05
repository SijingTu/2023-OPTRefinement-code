import networkx as nx 
import numpy as np

########## 
# Type 1, compute through the LFR model
##########
def generate_with_LFR_model(N = 1000):
    """
    Generate LFR model using networkx.
    Fix parameters, but very the graph size.
    Set edge weights to be 1.

    Args:
       N (int) grpah size

    Returns:
        G: networkx graph
        com: community 
        n: size of the graph
    """
    G = nx.LFR_benchmark_graph(n = N, tau1 = 3, tau2 = 1.5, mu = 0.2, average_degree = 10, min_community = 50, seed = 10)
    nx.set_edge_attributes(G, values = 1, name = 'weight')
    G.remove_edges_from(nx.selfloop_edges(G))
    com = list(G.nodes[0]['community'])
    n = G.number_of_nodes()
    
    for (_, d) in G.nodes(data=True): # remove community labels
        del d['community']

    return G, com, n

##########
# Type 2, compute through the stochastic block model 
##########
def generate_with_sb_model_balanced(N = 1000):
    """
    Generate stochastic block model using networkx (balanced).
    Vary graph size. 
    Fix four subgraphs. p = 0.3; q = 0.1

    Args:
        N (int): grpah size 

    Returns:
        G: networkx graph
        com: community 
        n: size of the graph
    """
    sub_size = int(np.floor(N / 4))
    
    ns = [sub_size, sub_size, sub_size, sub_size] # size of clusters
    ps = [[0.3, 0.1, 0.1, 0.1], [0.1, 0.3, 0.1, 0.1], [0.1, 0.1, 0.3, 0.1], [0.1, 0.1, 0.1, 0.3]] # probability of edge
    G = nx.stochastic_block_model(ns, ps)

    nx.set_edge_attributes(G, values = 1, name = 'weight')
    G.remove_edges_from(nx.selfloop_edges(G))
    com = [i for i in list(G.nodes()) if G.nodes()[i]['block'] == 0]
    n = G.number_of_nodes()
    
    for (_, d) in G.nodes(data=True): # remove community labels
        del d['block']

    return G, com, n

def generate_with_sb_model_unbalanced(N = 1000, type = "dense"):
    """
    Generate stochastic block model using networkx (un balanced).
    Vary graph size. 
    Fix four subgraphs. p_{0, 0} = 0.8 ; q = 0.1; p_{i, i} = 0.2, i =\= 0

    Args:
        N (int): grpah size 
        type (string): If type == "dense", select the densest community as the subgraph; 
        if type == "sparse", select the less dense community as the subgraph.

    Returns:
        G: networkx graph
        com: community 
        n: size of the graph
    """
    sub_size = int(np.floor(N / 4))
    
    ns = [sub_size, sub_size, sub_size, sub_size] # size of clusters
    ps = [[0.8, 0.1, 0.1, 0.1], [0.1, 0.2, 0.1, 0.1], [0.1, 0.1, 0.2, 0.1], [0.1, 0.1, 0.1, 0.2]] # probability of edge
    G = nx.stochastic_block_model(ns, ps)

    nx.set_edge_attributes(G, values = 1, name = 'weight')
    G.remove_edges_from(nx.selfloop_edges(G))
    if type == "dense":
        com = [i for i in list(G.nodes()) if G.nodes()[i]['block'] == 0]
    elif type == "sparse":
        com = [i for i in list(G.nodes()) if G.nodes()[i]['block'] == 1]
    n = G.number_of_nodes()
    
    for (_, d) in G.nodes(data=True): # remove community labels
        del d['block']

    return G, com, n

##########
# Type 3, inside one country, one of the two parties
#########
def party_creater(country):
    country_identifier = 0
    if country == "GB":
        country_identifier = 1
    elif country == "DE":
        country_identifier = 2
    elif country == "AT":
        country_identifier = 3
    elif country == "ES":
        country_identifier = 4
    elif country == "US":
        country_identifier = 5
    
    in_power_com, opposit_com = get_party_member_with_identifier(country_identifier)
    
    country_com = np.union1d(in_power_com, opposit_com)
    
    G, n = construct_country_graph(country_com)
    
    new_in_power_com = np.intersect1d(G.nodes(), in_power_com)
    
    return G, new_in_power_com, n
    
def get_party_member_with_identifier(country_identifier):
    in_power_com = []
    opposit_com = []
    
    with open("../data/new_wikipoli/vtxToPage.txt") as vtxToPage:
        for line in vtxToPage:
            line_info = line.strip("\n").split("#")
            tmp_id = int(line_info[0])
            tmp_identifier = int(line_info[2])
            
            if tmp_identifier == country_identifier:
                in_power_com.append(tmp_id)
            elif tmp_identifier == -country_identifier:
                opposit_com.append(tmp_id)
    return in_power_com, opposit_com 

def construct_country_graph(country_com):
    G = nx.Graph() #initialize country graph
    member_checker = np.zeros(np.max(country_com) + 1)
    largest_checker = np.max(country_com)
    
    for entry in country_com:
        member_checker[entry] = 1
    G.add_nodes_from(country_com)
    with open("../data/new_wikipoli/edges.txt") as edges:
        for line in edges:
            edge = line.strip("\n").split("#")
            e0 = int(edge[0])
            e1 = int(edge[1])
            
            if (e0 > largest_checker) or (e1 > largest_checker):
                continue
            
            if (member_checker[e0] == 1) and (member_checker[e1] == 1):
                G.add_edge(e0, e1)
    
    nx.set_edge_attributes(G, values = 1, name = 'weight')
    G.remove_edges_from(nx.selfloop_edges(G))

    #Gcc = sorted(nx.connected_components(G), key=len, reverse=True) # obtain connected component
    #G = nx.Graph(G.subgraph(Gcc[0])) # obtain only the largest connected component 
    
    n = G.number_of_nodes()
    
    return G, n

##########
# Type 4, over several countries, one of the two parties in one country 
#########

def party_creater_over_world(country):
    country_identifier = 0
    if country == "GB":
        country_identifier = 1
    elif country == "DE":
        country_identifier = 2
    elif country == "AT":
        country_identifier = 3
    elif country == "ES":
        country_identifier = 4
    elif country == "US":
        country_identifier = 5
    
    in_power_com, _ = get_party_member_with_identifier(country_identifier)
    
    G, n = construct_world_leader_graph()
    
    new_in_power_com = np.intersect1d(G.nodes(), in_power_com)
    
    return G, new_in_power_com, n

def construct_world_leader_graph():
    G = nx.Graph() #initialize country graph
    
    with open("../data/new_wikipoli/edges.txt") as edges:
        for line in edges:
            edge = line.strip("\n").split("#")
            e0 = int(edge[0])
            e1 = int(edge[1])
            G.add_edge(e0, e1)
    
    nx.set_edge_attributes(G, values = 1, name = 'weight')
    G.remove_edges_from(nx.selfloop_edges(G))

    #Gcc = sorted(nx.connected_components(G), key=len, reverse=True) # obtain connected component
    #G = nx.Graph(G.subgraph(Gcc[0])) # obtain only the largest connected component 

    n = G.number_of_nodes()
    
    return G, n

############################
# SNAP graph 
############################

def read_snap_graph(dataset):
    path = "../data/" + dataset + "/" + "com-" + dataset + ".ungraph.txt"
    comments = '#'
    delimiter = '\t'
    create_using = nx.Graph
    nodetype = int 
    
    G = nx.read_edgelist(path = path, comments = comments, delimiter = delimiter, create_using = create_using, nodetype = nodetype)
    nx.set_edge_attributes(G, values = 1, name = 'weight')
    G.remove_edges_from(nx.selfloop_edges(G))
    
    return G

def construct_snap_graph(dataset):
    G = read_snap_graph(dataset)

    return G, G.number_of_nodes()