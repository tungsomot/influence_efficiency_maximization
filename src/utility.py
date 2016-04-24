#coding = utf-8
import networkx as nx
import numpy as np
import time
import random
import math
from collections import deque

# Load an unweighted graph file @file_name.
# return a directed graph @DG
def load_unweighted_graph(file_name):
    dg = nx.DiGraph()
    try:
        with open(file_name,'r') as graph_file:
            for line in graph_file:
                # An edge from @u to @v with weight @w
                u,v = line.split()
                dg.add_edge(u,v,weight=1)
    except Exception as e:
        raise
    return dg

# Load a weighted graph file @file_name.
# return a directed graph @DG
def load_weighted_graph(file_name):
    dg = nx.DiGraph()
    try:
        with open(file_name,'r') as graph_file:
            for line in graph_file:
                # An edge from @u to @v with weight @w
                u,v,w = line.split()
                dg.add_edge(u,v,weight=w)
    except Exception as e:
        raise
    return dg

# To generate a subgraph based on source graph @g with a probability @p
# UIC means it is a uniform independent cascade (UIC)
def generate_subgraph_uic(g,p):
    g_copy = g.copy()
    for edge in g_copy.edges():
        flip = random.random()
        if flip > p:
            g_copy.remove_edge(edge[0],edge[1])
    return g_copy

# To generate a subgraph based on source graph @G with a probability p(u,v)=1/dv
# dv is the indegree of node v
# WIC means it is a weighted independent cascade (WIC)
def generate_subgraph_wic(g, indegree_dict):
    g_copy = g.copy()
    for edge in g_copy.edges():
        flip = random.random()
        if flip > float(1.0/indegree_dict[edge[1]]):
            g.remove_edge(edge[0],edge[1])
    return g

# Estimation of the influence efficiency with seed set @S in graph @G,
# return the float @influence_efficiency
def estimation(g,s):
    # To label the vertex which is influenced
    influenced_dict ={}
    # distance dic from seed set @S
    dist_dict = {}
    # Initialize
    for node in g.nodes():
        influenced_dict[node] = False
        dist_dict[node] = np.inf
    # Add seed set @S
    seed_deque = deque([])
    for seed in s:
        dist_dict[seed] = 1
        seed_deque.append(seed)
    # @flag represents the iteration is still going on
    # flag = True
    # Influence cascades
    while len(seed_deque)>0:
        # popleft of the seed_deque, the node has minimum distance value
        node = seed_deque.popleft()
        influenced_dict[node] = True
        for v in g.neighbors(node):
            update_distance = dist_dict[node]+float(g.edge[node][v]['weight'])
            if not influenced_dict[v] and update_distance<dist_dict[v]:
                dist_dict[v] = update_distance
                seed_deque.append(v)
    influence_efficiency = float(0)
    for k in dist_dict:
        influence_efficiency += 1/dist_dict[k]
    return influence_efficiency



if __name__ == '__main__':
    # file_name = 'data/simpleGraph1'
    # file_name = 'data/cit-HepPh.txt'
    # start = time.clock()
    # # print start
    # DG = LoadUnweightedGraph(file_name)
    # # S = ['5']
    # S = ['1113','3323','8273','2171','8298']
    # inf_eff = Estimation(DG,S)
    # print 'The influence_efficiency using seed set %s is %f.' % (S,inf_eff)
    # # sp = Dijkstra(DG,'3033')
    # # for k,v in sp.iteritems():
    # #     print '%s: %f' % (k,v)
    # stop = time.clock()
    # # print stop
    # time_cost = float(stop-start)
    # print 'Estimation cost: %f' % time_cost
    # # print DG.edge['0']['1']['weight']
    # # print DG.nodes()
    # # print DG.neighbors('1')
    # # p = 0.01
    #
    # # start = time.clock()
    # # # g = GenerateSubgraphUIC(DG,p)
    # # g = GenerateSubgraphWIC(DG)
    # # stop = time.clock()
    # # time_cost = float(stop-start)
    # # print 'DG has %i edges, g has %i edges' % (DG.number_of_edges(),g.number_of_edges())
    # # # print 'g edges: %s' % g.edges()
    # # print 'GenerateSubgraph cost: %f' % time_cost
    #
    # for i in xrange(100):
    #     start = time.clock()
    #     dist_dict = GenerateRRSet(DG)
    #     stop = time.clock()
    #     time_cost = float(stop-start)
    #     # print dist_dict
    #     for k in dist_dict.keys():
    #         if dist_dict[k]!=np.inf:
    #             print dist_dict[k]
    #     print 'GenerateRRSet cost: %f' % time_cost

    # test simpleGraph1
    # file_name_simple = 'data/simpleGraph2'
    # simpleG = LoadUnweightedGraph(file_name_simple)
    # n = simpleG.number_of_nodes()
    # r = int(n*math.log(n))
    # rr_sets = []
    # for i in xrange(r):
    #     rr_set = GenerateRRSet(simpleG)
    #     rr_sets.append(rr_set)
    # k = 1
    # S = RES(rr_sets,k,simpleG)
    # print S

    # test greedy algorithm
    file_name_simple = 'data/simpleGraph2'
    simpleG = LoadUnweightedGraph(file_name_simple)
    print simpleG.edges()
    n = simpleG.number_of_nodes()
    # r = int(n*math.log(n))
    r = 10000
    k = 2
    start = time.clock()
    S = RES(simpleG,k,r)
    stop = time.clock()
    print "%f" % float(stop-start)
    print S
