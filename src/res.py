# coding = utf-8
import networkx as nx
import numpy as np
import random
import math
import time
import sys
import utility
from collections import deque
# This function is used to generate reverse reachable (RR) dictory.
# Weighted Influence Cascade (WIC) model is used in this function.
# @g is the graph
# @vertex_set is the vertex set of @g
# @n is the size of @vertex_set
# @indegree_dict is the indegree dictory
# @rr_dict is distance dictory
def generate_rr_dict(g, vertex_set, n, indegree_dict):
    """
    :type @g: networkx.DiGraph
    :type @vertex_set: dictory {str : int}
    :type @n: int
    :type @indegree_dict: {str : int}
    :type @rr_dict: {str : float}
    """
    # To record whether the vertex is searched
    searched_dict = {}
    # To record the distance of nodes to a certain node @v.
    rr_dict = {}
    # To select a node @v uniformly
    index = int(random.random()*n)
    v = vertex_set[index]
    # seed deque of @v
    s_deque = deque([v])
    rr_dict[v] = 1.0
    while len(s_deque) > 0:
        vertex = s_deque.popleft()
        searched_dict[vertex] = True
        for in_edge in g.in_edges(vertex):
            flip = random.random()
            if flip < 1.0/indegree_dict[vertex]:
                u = in_edge[0]
                if not searched_dict.has_key(u):
                    distance = rr_dict[vertex] + g.edge[u][vertex]['weight']
                    if not rr_dict.has_key(u) or rr_dict.has_key(u) and distance < rr_dict[u]:
                        rr_dict[u] = distance
                        s_deque.append(u)
    return rr_dict

# Reverse efficiency sampling (RES) algorithm:
# A greedy algorithm based on RR dictory
# @g is the original graph
# @k is the number of @s
# @r is the number of @rr_dict
# @s is the seed set to return
def res(g, k, r):
    """
    :type @G: networkx.DiGraph
    :type @k: int
    :type @r: int
    :type @s: [str]
    """
    # Initialize seed set @S
    s = []
    # dist_dict represents the minimum distance from @s to node i
    dist_dict = {}
    # To generate rr_dicts
    rr_dicts = []
    indegree_dict = g.in_degree()
    vertex_set = g.nodes()
    n = len(vertex_set)
    for i in xrange(r):
        # Attention! Indices here is from 0 to len(rr_sets)-1
        # It is different from the node index (e.g. '267') in the graph G
        # It is the sampling node's index
        rr_dicts.append(generate_rr_dict(g, vertex_set, n, indegree_dict))
        # Set the distance to node i infinite
        dist_dict[i] = np.inf
    # To find @s
    for i in xrange(k):
        print r
        eff = {}
        for j in xrange(r):
            for u in rr_dicts[j].keys():
                if rr_dicts[j][u] < dist_dict[j]:
                    if eff.has_key(u):
                        eff[u] += 1.0/rr_dicts[j][u] - 1.0/dist_dict[j]
                    else:
                        eff[u] = 1.0/rr_dicts[j][u] - 1.0/dist_dict[j]
        # To calculate the max efficient node in ith iteration
        max_eff = 0
        max_node = '-1'
        for u in eff.keys():
            if eff[u] > max_eff:
                max_node = u
                max_eff = eff[u]
        # Add max_node to @S
        s.append(max_node)
        # Update @dist_dict and @rr_dicts
        j = 0
        while j < r:
            if rr_dicts[j].has_key(max_node):
                dist_dict[j] = rr_dicts[j][max_node]
                rr_dicts[j].pop(max_node)
            # If the maxnode has min distance in the jth RR dictory,
            # delete the jth RR dictory, because it won't get marginal gain any more
            min_distance = dist_dict[j]
            min_distance_node = max_node
            for u in rr_dicts[j].keys():
                if rr_dicts[j][u] < min_distance:
                    min_distance = rr_dicts[j][u]
                    min_distance_node = u
            # remove the rr_dict if max_node is already in @s
            # this will reduce time-comsuming
            if min_distance_node == max_node:
                rr_dicts.remove(rr_dicts[j])
                r -= 1
            # else:
                # rr_dicts[j].pop(max_node)
            j += 1
    return s

if __name__ == '__main__':
    file_name = sys.argv[1]
    g = utility.LoadUnweightedGraph(file_name)
    indegree_dict = g.in_degree()
    vertex_set = g.nodes()
    n = len(vertex_set)
    k = 2
    r = int(n*math.log(n))
    # r = 10000
    start = time.clock()
    s = res(g, k, r)
    stop = time.clock()
    print "%f" % float(stop-start)
    print s
