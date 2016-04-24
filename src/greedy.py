# coding = utf-8
import utility
import sys
import time
import math

def greedy(g,k,r):
    """
    :type @g: networkx.DiGraph
    :type @k: int
    :type @r: int
    """
    indegree_dict = g.in_degree()
    # Generate subgraphs
    subgraphs = []
    for i in xrange(r):
        subgraphs.append(utility.generate_subgraph_wic(g, indegree_dict))
    # Initialize @S
    s = []
    # dist_dict = {}
    # for v in G.nodes():
    #     dist_dict[v] = np.inf
    for i in xrange(k):
        max_eff = float(0)
        max_node = '-1'
        for node in g.nodes():
            if node not in s:
                eff_sum = float(0)
                # dist_dict_tmp = {}
                # for v in G.nodes():
                #     dist_dict_tmp[v] = np.inf
                for subgraph in subgraphs:
                    eff_sum += utility.estimation(subgraph,s+[node])
                if eff_sum > max_eff:
                    max_eff = eff_sum
                    max_node = node
        s.append(max_node)
    return s

if __name__ == '__main__':
    file_name = sys.argv[1]
    g = utility.load_unweighted_graph(file_name)
    vertex_set = g.nodes()
    n = len(vertex_set)
    k = 2
    # r = 10000
    r = int(n*math.log(n))
    start = time.clock()
    s = greedy(g, k, r)
    stop = time.clock()
    print "%f" % float(stop-start)
    print s
