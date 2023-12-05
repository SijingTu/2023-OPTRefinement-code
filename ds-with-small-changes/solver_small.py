import networkx as nx
import numpy as np
from dks import *
import time, sys
from mosek.fusion import *
import json
import timeout_decorator

####################################
# Set up the parameters
# 1, we choose the LFR model; 2, we choose the stochastic block model
choice = sys.argv[2]
graph_type = sys.argv[3]

sigma = 5

# choose the type of graph 
if graph_type == "lfr_model":
    G, init_com, n = generate_with_LFR_model()
elif graph_type == "sb_model_balanced":
    G, init_com, n = generate_with_sb_model_balanced() 
elif graph_type == "sb_model_dense":
    G, init_com, n = generate_with_sb_model_unbalanced(1000, 'dense')
elif graph_type == "sb_model_sparse":
    G, init_com, n = generate_with_sb_model_unbalanced(1000, 'sparse')
    sigma = int(sys.argv[5])
elif graph_type == "wiki_gb":
    G, init_com, n = party_creater("GB")
    sigma = int(sys.argv[5])
elif graph_type == "wiki_de":
    G, init_com, n = party_creater("DE")
elif graph_type == "wiki_es":
    G, init_com, n = party_creater("ES")
elif graph_type == "wiki_us":
    G, init_com, n = party_creater("US")

init_score = rho_subgraph(G, init_com)
print("before removing nodes, the density is ", init_score)

ratio = 0
if sys.argv[4] == "k":
    k = int(sys.argv[1])
if sys.argv[4] == "ratio":
    ratio = int(sys.argv[1])
    k = int(ratio / 100.0 * len(init_com))

print("===================Successfully Load Graph====================", flush=True)

######################################
# add ratio: ratio * 0.01 is the correct ratio

out_dict = {"n":n, "k":k, "ratio": ratio, "size_init_com": len(init_com),"choice":choice, "init_score": init_score, "init_score_after_remove_nodes": 0, "init_select_nodes": np.zeros(k+1).tolist(),\
    "greedy_merge_time":0, "greedy_merge_score":0, "greedy_merge_k_density":0, "greedy_merge_nodes":np.zeros(k+1).tolist(), \
        "sdp_merge_time":0, "sdp_merge_score":0, "sdp_merge_k_density":0, "sdp_merge_k_upp":0, "sdp_merge_nodes": np.zeros(k+1).tolist(), \
            "local_search_time":0, "local_search_score":0,  "local_search_nodes": np.zeros(k+1).tolist(), \
                "local_sdp_time":0, "local_sdp_score": 0, "local_sdp_upp": 0, "local_sdp_nodes": np.zeros(k+1).tolist(), \
                    "SQD_time":0, "SQD_score":0, "SQD_dense_subgraph_size":0, "SQD_nodes": np.zeros(k+1).tolist(), "SQD_success":"False", \
                        "random_time":0, "random_score": 0}


# choose if remove nodes from original subgraph 
if choice == "move_out":
    nodes_to_remove = np.random.choice(init_com, k, replace = False) # repeat the procedure 
    com = list(np.setdiff1d(init_com, nodes_to_remove))
    init_score_after_remove_nodes = rho_subgraph(G, com)

    out_dict["init_score_after_remove_nodes"] = init_score_after_remove_nodes
    out_dict["init_select_nodes"] = nodes_to_remove.tolist()
    
    print("after removing nodes, the density is ", init_score_after_remove_nodes)
elif choice == "not_move_out":
    init_score_after_remove_nodes = 0
    com = list(init_com) 


print("===================Start Algo : Peeling with black box solver====================", flush=True)

##########
# The codes for the peeling based black box solver
##########

# Pre-prosessing
G_merge = merge_nodes(G, com)
new_node = np.amax(G.nodes) + 1

# 1.1 merge
start = time.time()
greedy_merge = greedy_charikar(G_merge, k+1)
try:
    greedy_merge[0].remove(new_node)
except:
    print("does not contain the new node")
    # just remove the first node
    first_node = greedy_merge[0][0]
    greedy_merge[0].remove(first_node)
    
greedy_merge_nodes = com + greedy_merge[0]
greedy_merge_score = rho_subgraph(G, greedy_merge_nodes)
greedy_merge_time = time.time() - start 

out_dict['greedy_merge_time'] = greedy_merge_time
out_dict['greedy_merge_score'] = greedy_merge_score
out_dict['greedy_merge_k_density'] = greedy_merge[1] # the density on the merged graph
out_dict['greedy_merge_nodes'] = [int(i) for i in greedy_merge[0]] # selected nodes

print("===================Start Algo : SDP with black box solver====================", flush=True)
#########
# The codes for the sdp based black box solver
#########

# 2.1 merge
start = time.time()
sdp_merge = sdp_dks(k+1, G_merge)
try:
    sdp_merge[0].remove(new_node)
except:
    print("does not contain the new node")
sdp_merge_nodes = com + sdp_merge[0]
sdp_merge_score = rho_subgraph(G, sdp_merge_nodes)
sdp_merge_time = time.time() - start 

out_dict['sdp_merge_time'] = sdp_merge_time 
out_dict['sdp_merge_score'] = sdp_merge_score 
out_dict['sdp_merge_k_density'] =sdp_merge[1]
out_dict['sdp_merge_k_upp'] = sdp_merge[2] 
out_dict['sdp_merge_nodes'] = [int(i) for i in sdp_merge[0]]

print("===================Start Algo : SDP ====================", flush=True)
#########
# The codes for the sdp-based method 
########
start = time.time()
local_sdp_sol = local_sdp_sum_weights(G, com, k, weight = "weight")
local_sdp_time = time.time() - start 

out_dict['local_sdp_time'] = local_sdp_time 
out_dict['local_sdp_upp'] = local_sdp_sol[2] # NOTE, this is the upper bound of adding k extra nodes, note necessarily the optimal density of "changing" k nodes. 
out_dict['local_sdp_score'] = rho_subgraph(G, local_sdp_sol[0])
out_dict['local_sdp_nodes'] = setdiff(local_sdp_sol[0], com).tolist()

print("===================Start Algo : Greedy ====================", flush=True)
#######
# The codes for the greedy-based method 
#######
start = time.time()
local_search_nodes = local_search(k, G, com, weight = "weight")
local_search_score = rho_subgraph(G, local_search_nodes)
local_search_time = time.time() - start 

out_dict["local_search_time"] = local_search_time
out_dict["local_search_score"] = local_search_score
out_dict['local_search_nodes'] = setdiff(local_search_nodes, com).tolist()

print("===================Start Algo : Random====================", flush=True)
#####
# The codes for randomly selecting nodes
####
start = time.time()
random_score = random_choice(G, com, k)
random_time = time.time() - start

out_dict["random_time"] = random_time
out_dict["random_score"] = random_score 

# print("===================Start Algo : SQD====================", flush=True)
# ###
# # The codes for SQD 
# ###

# start = time.time()
# try: 
#     SQD_results = run_SQD(G, com, k, sigma) # we may only run on two small datasets
#     if SQD_results[0] == "SMALL_SIGMA":
#         out_dict["SQD_success"] = "SMALL_SIGMA"
#         out_dict["SQD_dense_subgraph_size"] = SQD_results[1]
#     elif SQD_results[0] == "EMPTY_SUBGRAPH":
#         out_dict["SQD_success"] = "EMPTY_SUBGRAPH"
#         out_dict["SQD_dense_subgraph_size"] = SQD_results[1]
#     else:
#         SQD_select_nodes = SQD_results[2]
#         SQD_score = rho_subgraph(G, np.union1d(SQD_select_nodes, com))
#         SQD_time = time.time() - start 
        
#         out_dict["SQD_success"] = "True"
#         out_dict["SQD_dense_subgraph_size"] = SQD_results[1]
#         out_dict["SQD_time"] = SQD_time
#         out_dict["SQD_score"] = SQD_score
#         out_dict['SQD_nodes'] = SQD_select_nodes.tolist()
# except timeout_decorator.timeout_decorator.TimeoutError:
#     out_dict["SQD_success"] = "TIME_OUT_LARGE"
#     out_dict["SQD_dense_subgraph_size"] = SQD_results[1]
# except:
#     out_dict["SQD_success"] = "Other_Error"
#     out_dict["SQD_dense_subgraph_size"] = SQD_results[1]

###
# Write to a file
###
print(out_dict)
out_lfr = "out/" + graph_type + "_results.json" # add the number of nodes get changed
with open(out_lfr, "a+") as outfile:
    json.dump(out_dict, outfile)
    outfile.write("\n")