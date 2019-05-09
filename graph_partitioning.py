import networkx as nx
from networkx.algorithms import community

# import the knn graph
G = nx.read_gpickle('graph.gpickle')

# we use a hierarchical partitioning approach to generate m balanced partitions of the graph
# specifically, we choose m to be a power of 2 and repeatedly use Kernighan-Lin to bisect the 
# graph into 2 approximately equal parts
num_levels = 3 # this gives m = 2^3 = 8 parts

def graph_partition(graph, level):
	if level == num_levels:
		return [graph]
	else:
		part1, part2 = community.kernighan_lin_bisection(graph, weight='weight')
		return graph_partition(graph.subgraph(part1), level + 1) + graph_partition(graph.subgraph(part2), level + 1)

partition = graph_partition(G, 0)
for part in partition:
	print(part.nodes())

# check the partition was correct
union = []
for part in partition:
	union += list(part.nodes())
assert set(union) == set(G.nodes())
