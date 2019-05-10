import pickle
import numpy as np
import networkx as nx

N = 1000 # 1000 data points in our training set
k = 10 # Each node in the graph will have edges to its 10 nearest neighbors
S = 50 # need S-nn graph for soft label empirical distribution over S nearest neighbors, S >> k

# Load the dataset of points
points = pickle.load(open('pickles/points.pickle', 'rb'))
assert len(points) == N
for point in points:
	# every point is a collison event with 3 readings (angle, Y, energy) for some number of particles
	assert point.shape[1] == 3

# Build the k-nearest neighbor graph from these nodes
knn_graph = nx.Graph()
for i, point in enumerate(points):
	knn_graph.add_node(i, reading=point)

# Load the pairwise distances
distances = pickle.load(open('pickles/CERN_emds.p', 'rb'))
assert distances.shape == (N, N)

# Compute the k and S nearest neighbors by sorting the distances
S_nearest_neighbors = {}
for i in range(N):
	distances_for_point = list(zip(range(N), distances[i]))
	distances_for_point.sort(key=lambda x: x[1])

	# Add an edge to each of the closest neighbors, skipping the node itself
	for j in range(1, k + 1):
		if not knn_graph.has_edge(i, distances_for_point[j][0]):
			knn_graph.add_edge(i, distances_for_point[j][0], weight=distances_for_point[j][0])
	
	# Keep track of the exactly S nearest neighbors - need this for soft labels
	S_nearest_neighbors[i] = [i]
	for j in range(1, S):
		S_nearest_neighbors[i].append(distances_for_point[j][0])

nx.write_gpickle(knn_graph, 'pickles/knn_graph.gpickle')
pickle.dump(S_nearest_neighbors, open("pickles/nearest_neighbors.pickle", "wb"))
