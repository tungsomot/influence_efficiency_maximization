#coidng = utf-8
import networkx as nx
import numpy as np
import time
from collections import deque

# Load an unweighted graph file @file_name.
# return a directed graph @DG
def LoadUnweightedGraph(file_name):
    DG = nx.DiGraph()
    try:
        with open(file_name,'r') as graph_file:
            for line in graph_file:
                # An edge from @u to @v with weight @w
                v,u = line.split()
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



if __name__ == '__main__':
    # file_name = 'data/simpleGraph1'
    file_name = 'data/cit-HepPh.txt'
    start = time.clock()
    # print start
    DG = LoadUnweightedGraph(file_name)
    # S = ['5']
    S = ['1113','3323','8273','2171','8298']
    inf_eff = Estimation(DG,S)
    print 'The influence_efficiency using seed set %s is %f.' % (S,inf_eff)
    # sp = Dijkstra(DG,'3033')
    # for k,v in sp.iteritems():
    #     print '%s: %f' % (k,v)
    stop = time.clock()
    # print stop
    time_cost = float(stop-start)
    print 'Estimation cost: %f' % time_cost
    # print DG.edge['0']['1']['weight']
    # print DG.nodes()
    # print DG.neighbors('1')
