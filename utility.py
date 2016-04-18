#coidng = utf-8
import networkx as nx
import numpy as np
import time
import random
import math
from collections import deque

# Load an unweighted graph file @file_name.
# return a directed graph @DG
def LoadUnweightedGraph(file_name):
    DG = nx.DiGraph()
    try:
        with open(file_name,'r') as graph_file:
            for line in graph_file:
                # An edge from @u to @v with weight @w
                u,v = line.split()
                DG.add_edge(u,v,weight=1)
    except Exception as e:
        raise
    return DG

# Load a weighted graph file @file_name.
# return a directed graph @DG
def LoadWeightedGraph(file_name):
    DG = nx.DiGraph()
    try:
        with open(file_name,'r') as graph_file:
            for line in graph_file:
                # An edge from @u to @v with weight @w
                u,v,w = line.split()
                DG.add_edge(u,v,weight=w)
    except Exception as e:
        raise
    return DG

# Dijkstra algorithm for a single vertex.
# @G: the graph, class networkx
# @s: the source node
# @dist_dict: distance dict from @s
def Dijkstra(G,s):
    # A utility function to find the vertex with minimum distance value,
    # from the set of vertices not yet included in shortest path tree.
    # @min_distance_vertex is the node with min_distance
    def CalculateMinDistance(dist_dict,spt_dict):
        min_distance = np.inf
        min_distance_vertex = '-1'
        for node in G.nodes():
            if not spt_dict[node] and dist_dict[node] < min_distance:
                min_distance = dist_dict[node]
                min_distance_vertex = node
        return min_distance_vertex

    vertex_count = len(G.nodes())
    # @spt_dict is the shortest path tree
    spt_dict = {}
    dist_dict = {}
    for node in G.nodes():
        spt_dict[node] = False
        dist_dict[node] = np.inf
    # Add source node @s
    dist_dict[s] = 1
    for i in xrange(vertex_count-1):
        u = CalculateMinDistance(dist_dict,spt_dict)
        if u == '-1':
            break
        spt_dict[u] = True
        for v in G.neighbors(u):
            update_distance = dist_dict[u]+float(G.edge[u][v]['weight'])
            if not spt_dict[v] and update_distance<dist_dict[v]:
                dist_dict[v] = update_distance
    return dist_dict

# Estimation of the influence efficiency with seed set @S in graph @G,
# return the float @influence_efficiency
def Estimation(G,S):
    # To label the vertex which is influenced
    influenced_dict ={}
    # distance dic from seed set @S
    dist_dict = {}
    # Initialize
    for node in G.nodes():
        influenced_dict[node] = False
        dist_dict[node] = np.inf
    # Add seed set @S
    seed_deque = deque([])
    for seed in S:
        dist_dict[seed] = 1
        seed_deque.append(seed)
    # @flag represents the iteration is still going on
    # flag = True
    # Influence cascades
    while len(seed_deque)>0:
        # popleft of the seed_deque, the node has minimum distance value
        node = seed_deque.popleft()
        influenced_dict[node] = True
        for v in G.neighbors(node):
            update_distance = dist_dict[node]+float(G.edge[node][v]['weight'])
            if not influenced_dict[v] and update_distance<dist_dict[v]:
                dist_dict[v] = update_distance
                seed_deque.append(v)
    influence_efficiency = float(0)
    for k in dist_dict:
        influence_efficiency += 1/dist_dict[k]
    return influence_efficiency

# To generate a subgraph based on source graph @G with a probability @p
# UIC means it is a uniform independent cascade (UIC)
def GenerateSubgraphUIC(G,p):
    g = G.copy()
    for edge in g.edges():
        flip = random.random()
        if flip > p:
            g.remove_edge(edge[0],edge[1])
    return g

# To generate a subgraph based on source graph @G with a probability p(u,v)=1/dv
# dv is the indegree of node v
# WIC means it is a weighted independent cascade (WIC)
def GenerateSubgraphWIC(G):
    g = G.copy()
    indegree_dict = g.in_degree()
    for edge in g.edges():
        flip = random.random()
        if flip > float(1.0/indegree_dict[edge[1]]):
            g.remove_edge(edge[0],edge[1])
    return g

# To generate a RR set, which is used for Reverse Influence Sampling (RIS) method
# This method uses WIC model.
# @g is a subgraph of @G
# return random node @v and corresponding distance dictory @dist_dict
# Only record the node from which is reachable
def GenerateRRSet(g):
    # To represents whether node is searched
    searched_dict = {}
    # To record the distance of nodes to a certain node @v.
    dist_dict = {}
    # To initialize all the nodes
    for node in g.nodes():
        searched_dict[node] = False
        # dist_dict[node] = np.inf
    # To select a node @v uniformly
    n = len(g.nodes())
    v = g.nodes()[int(random.random()*n)]
    # print v
    # To get the indegree @indegree_dict
    indegree_dict = g.in_degree()
    # Seed set of @v which is used for RIS
    seed_deque = deque([v])
    dist_dict[v] = 1.0
    # print g.in_edges(v)
    while len(seed_deque)>0:
        node = seed_deque.popleft()
        searched_dict[node] = True
        for in_edge in g.in_edges(node):
            flip = random.random()
            if flip < float(1.0/indegree_dict[node]):
                u = in_edge[0]
                update_distance = dist_dict[node] + float(g.edge[u][node]['weight'])
                if not searched_dict[u]:
                    if dist_dict.has_key(u) and update_distance<dist_dict[u] or not dist_dict.has_key(u):
                        dist_dict[u] = update_distance
                        seed_deque.append(u)
    return dist_dict

# Reverse efficiency sampling (RES) algorithm:
# A greedy algorithm based on RR sets
# @G is the original graph
# @k is the number of @S
# @r is the number of simulations of return of function @GenerateRRSet
# @S is the seed set
def RES(G,k,r):
    """
    :type @G: networkx.DiGraph
    :type @k: int
    :type @r: int
    """
    # Initialize seed set @S
    S = []
    # dist_dict represents the minimum distance from @S to node i
    dist_dict = {}
    # To generate rr_sets
    rr_sets = []
    for i in xrange(r):
        rr_sets.append(GenerateRRSet(G))
        # Attention! Indices here is from 0 to len(rr_sets)-1
        # It is different from the node index (e.g. '267') in the graph G
        # It is the sampling node's index
        # Set the distance to node i infinite
        dist_dict[i] = np.inf
    # print rr_sets
    # To find @S
    for i in xrange(k):
        eff = {}
        for j in xrange(r):
            for u in rr_sets[j].keys():
                if rr_sets[j][u]<dist_dict[j]:
                    if eff.has_key(u):
                        eff[u] += float(1.0/rr_sets[j][u] - 1.0/dist_dict[j])
                    else:
                        eff[u] = float(1.0/rr_sets[j][u] - 1.0/dist_dict[j])
        # To calculate the max efficient node in ith iteration
        # print eff
        max_eff = 0
        max_node = '-1'
        for u in eff.keys():
            if eff[u]>max_eff:
                max_node = u
                max_eff = eff[u]
        # Add max_node to @S
        S.append(max_node)
        # print S
        # Update @dist_dict and @rr_sets
        j = 0
        while j < r:
            if rr_sets[j].has_key(max_node):
                dist_dict[j] = rr_sets[j][max_node]
            min_distance = dist_dict[j]
            min_distance_node = max_node
            for u in rr_sets[j].keys():
                if rr_sets[j][u]<min_distance:
                    min_distance = rr_sets[j][u]
                    min_distance_node = u
            # remove the rr_set if max_node is already in @S
            # this will reduce time-comsuming
            if min_distance_node == max_node:
                rr_sets.remove(rr_sets[j])
                r -= 1
            j += 1
    return S

def greedy(G,k,r):
    """
    :type @G: networkx.DiGraph
    :type @k: int
    :type @r: int
    """
    # Generate subgraphs
    subgraphs = []
    for i in xrange(r):
        subgraphs.append(GenerateSubgraphWIC(G))
    # Initialize @S
    S = []
    # dist_dict = {}
    # for v in G.nodes():
    #     dist_dict[v] = np.inf
    for i in xrange(k):
        max_eff = float(0)
        max_node = '-1'
        for node in G.nodes():
            if node not in S:
                eff_sum = float(0)
                # dist_dict_tmp = {}
                # for v in G.nodes():
                #     dist_dict_tmp[v] = np.inf
                for subgraph in subgraphs:
                    eff_sum += Estimation(subgraph,S+[node])
                if eff_sum > max_eff:
                    max_eff = eff_sum
                    max_node = node
        S.append(max_node)
    return S

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
    k = 1
    S = RES(simpleG,k,r)
    print S
