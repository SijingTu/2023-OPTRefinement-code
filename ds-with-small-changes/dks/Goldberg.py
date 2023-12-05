import maxflow
import itertools
import numpy as np       


def Find_Densest_Subgraph(temp_graph_edges,no_edges,node_weights,removed_nodes):
        ''' This function performs the binary search of the density of subgraph and finds the densest subgraph.'''
        #construct full graph, but make sure that removed nodes have zero capacity edges
        number_of_nodes = len(node_weights)-len(removed_nodes)
 #       print "there are this many nodes: ", number_of_nodes-len(removed_nodes)
        number_of_edges = no_edges
            
        min_degree = 0
        max_degree = number_of_edges
        subgraph = []
        difference = 1.0/( (number_of_nodes) * (number_of_nodes - len(removed_nodes) - 1) )
#        print difference, " diff"
        while(max_degree - min_degree >= difference):
                #print "max - min = ", max_degree - min_degree
                least_density = (max_degree + min_degree)/2.0
                #print "ld--->", least_density 
                source_segment = make_graph(temp_graph_edges, node_weights, number_of_nodes, number_of_edges, least_density, removed_nodes)
                if(source_segment == []):
                        max_degree = least_density
                else:
                        min_degree = least_density
                        subgraph = source_segment
#        print("subgraph: "+str(subgraph))
        return subgraph


def make_graph(temp_graph_edges, node_weights, number_of_nodes, number_of_edges, least_density, removed_nodes):
        ''' Constructs the network as per the specifications given by Goldberg'''
        ind={}
        cou_to_ind={}
        cou=0
        for i in range(len(node_weights)):
            if i not in removed_nodes:
                ind[i]=cou
                cou_to_ind[cou]=i
                cou+=1
  
        graph = maxflow.Graph[float](number_of_nodes, number_of_edges)
        nodes = graph.add_nodes(number_of_nodes)
        degrees = {}
        #print degrees   
        max_weight=max(node_weights)
        for (i,j) in temp_graph_edges:
            from_node=ind[i]
            to_node=ind[j]
            #print edge.split()
            graph.add_edge(nodes[from_node], nodes[to_node], 1,1)
            if from_node in degrees:
                    degrees[from_node] += 1
            else:
                    degrees[from_node] = 1
            if to_node in degrees:
                    degrees[to_node] += 1
            else:
                    degrees[to_node] = 1
        for i in range(number_of_nodes):
                if i not in degrees:
                    degrees[i] = 0
#                if i in removed_nodes:
#                    print ("iterating removed node: "+str(i)+ " with degree: "+str(degrees[i]))
#                    graph.add_tedge(nodes[i],0,0)
#                if i not in removed_nodes:
#                graph.add_tedge(nodes[i], number_of_edges, number_of_edges + 2*least_density - degrees[i])
                graph.add_tedge(nodes[i], number_of_edges + 2*max_weight, number_of_edges + 2*max_weight + 2*least_density - degrees[i] - 2*node_weights[cou_to_ind[i]])
                #print "s -- ",number_of_edges,"-->", nodes[i], "--",number_of_edges + 2*least_density - degrees[str(i)], "-->t\n"  
        source_segment = []
        '''Computes the max-flow in the graph'''
        flow=graph.maxflow()
#        print("there is this much flow: "+str(flow))
        '''The following section of code finds which node belongs to which cutset.'''
        for i in nodes:
                #print nodes[i] ,"--->", graph.get_segment(nodes[i])
                if(graph.get_segment(nodes[i]) == 0):
                        source_segment.append(cou_to_ind[nodes[i]])
        #print degrees
        return source_segment


def Densest_Subgraph(temp_graph_edges,no_edges,node_weights, removed_nodes):

    subgraph = Find_Densest_Subgraph(temp_graph_edges,no_edges,node_weights,removed_nodes)
#    print (time.clock() - start_time)
#    print answer
#    Find_Density(answer)
    return subgraph
