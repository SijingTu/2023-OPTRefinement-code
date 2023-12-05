import networkx as nx
from src import *
import time, sys
from mosek.fusion import *
import json

####################################
# Set up the parameters
# 1, we choose the LFR model; 2, we choose the stochastic block model
k = int(sys.argv[1])
graph_type = sys.argv[2]

# choose the type of graph 
if graph_type == "sb_model_balanced":
    G, _, n = generate_with_sb_model_balanced() 
elif graph_type == "sb_model_dense":
    G, _, n = generate_with_sb_model_unbalanced(1000, 'dense')
elif graph_type == "wiki_gb":
    G, _, n = party_creater("GB")
elif graph_type == "wiki_de":
    G, _, n = party_creater("DE")
elif graph_type == "wiki_es":
    G, _, n = party_creater("ES")
elif graph_type == "wiki_us":
    G, _, n = party_creater("US")

#################################s


initial_left_cut = list(nx.algorithms.approximation.maxcut.randomized_partitioning(G)[1][0])
init_score = nx.cut_size(G, initial_left_cut)
######################################

out_dict = {"n":n, "k":k, "init_score": init_score, \
    "black_box_greedy_cut":0, "black_box_greedy_time":0,\
        "black_box_local_search_without_greedy_cut":0, "black_box_local_search_without_greedy_time":0,  \
            "black_box_local_search_with_greedy_cut":0, "black_box_local_search_with_greedy_time":0, \
                "black_box_sdp_cut":0, "black_box_sdp_time":0, \
                    "local_greedy_cut":0, "local_greedy_time":0, \
                        "local_sdp_cut":0, "local_sdp_time":0}
##########
# The codes for the greedy based black box solver
##########

start = time.time()
black_box_greedy_maxcut_left = greedy_maxcut(G)
black_box_greedy_left = move_nodes_back(G, initial_left_cut, black_box_greedy_maxcut_left, k)

black_box_greedy_cut = nx.cut_size(G, black_box_greedy_left)
black_box_greedy_time = time.time() - start 

out_dict['black_box_greedy_cut'] = black_box_greedy_cut
out_dict['black_box_greedy_time'] = black_box_greedy_time 

##########
# The codes for the local search based black box solver
##########

# without greedy 

start = time.time()
black_box_local_search_without_greedy_maxcut_left = local_search_max_cut(G, init_partition = None, weight = 'weight')
black_box_local_search_without_greedy_left = move_nodes_back(G, initial_left_cut, black_box_local_search_without_greedy_maxcut_left, k)
black_box_local_search_without_greedy_cut = nx.cut_size(G, black_box_local_search_without_greedy_left)
black_box_local_search_without_greedy_time = time.time() - start

out_dict['black_box_local_search_without_greedy_time'] = black_box_local_search_without_greedy_time
out_dict['black_box_local_search_without_greedy_cut'] = black_box_local_search_without_greedy_cut 

# with greedy 

start = time.time()
black_box_local_search_with_greedy_maxcut_left = local_search_max_cut(G, init_partition = black_box_greedy_maxcut_left, weight = 'weight')
black_box_local_search_with_greedy_left = move_nodes_back(G, initial_left_cut, black_box_local_search_with_greedy_maxcut_left, k)
black_box_local_search_with_greedy_cut = nx.cut_size(G, black_box_local_search_with_greedy_left)
black_box_local_search_with_greedy_time = time.time() - start

out_dict['black_box_local_search_with_greedy_time'] = black_box_local_search_with_greedy_time + black_box_greedy_time
out_dict['black_box_local_search_with_greedy_cut'] = black_box_local_search_with_greedy_cut 

###########
# The codes for the sdp based black box solver 
###########

start = time.time()
black_box_sdp_maxcut_left =sdp_max_cut(G, I = 100, weight = 'weight')
black_box_sdp_left = move_nodes_back(G, initial_left_cut, black_box_sdp_maxcut_left, k)
black_box_sdp_cut = nx.cut_size(G, black_box_sdp_left)
black_box_sdp_time = time.time() - start 

out_dict['black_box_sdp_time'] = black_box_sdp_time
out_dict['black_box_sdp_cut'] = black_box_sdp_cut

##########
# The codes for the greedy-based algorithm
#########

start = time.time()
local_greedy_left = local_greedy_max_cut(G, initial_left_cut, k)
local_greedy_cut = nx.cut_size(G, local_greedy_left)
local_greedy_time = time.time() - start 

out_dict['local_greedy_time'] = local_greedy_time
out_dict['local_greedy_cut'] = local_greedy_cut

##############
# The codes for the sdp_based algorithm
##############

start = time.time()
local_sdp_left, final_cut, opt = local_sdp_max_cut(G, initial_left_cut, k, I = 100, weight = 'weight')
local_sdp_cut = nx.cut_size(G, local_sdp_left)
local_sdp_time = time.time() - start 

out_dict['local_sdp_time'] = local_sdp_time 
out_dict['local_sdp_cut'] = local_sdp_cut


###
# Write to a file
###
print(out_dict)
out_lfr = "out/" + graph_type + "_results.json" # add the number of nodes get changed
with open(out_lfr, "a+") as outfile:
    json.dump(out_dict, outfile)
    outfile.write("\n")



