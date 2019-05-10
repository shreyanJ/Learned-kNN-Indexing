import networkx as nx
from networkx.algorithms import community
import pickle

# import the knn graph
G = nx.read_gpickle('pickles/knn_graph.gpickle')

# we use a hierarchical partitioning approach to generate m balanced partitions of the graph
# specifically, we choose m to be a power of 2 and repeatedly use Kernighan-Lin to bisect the 
# graph into 2 approximately equal parts
def graph_partition(graph, level):
	if level == 0:
		return [set(graph.nodes())]
	else:
		part1, part2 = community.kernighan_lin_bisection(graph, weight='weight')
		print("done with iteration at level: {}".format(level))
		return graph_partition(graph.subgraph(part1), level - 1) + graph_partition(graph.subgraph(part2), level - 1)

# First, partition the knn graph formed by the training set
num_levels = 4 # this gives M = 16
M = 2 ** num_levels
partition = graph_partition(G, num_levels)

# check the partition was correct
union = []
for part in partition:
	union += list(part)
assert set(union) == set(G.nodes())

pickle.dump(partition, open("pickles/graph_partitions.pickle", "wb"))
