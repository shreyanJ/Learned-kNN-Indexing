import networkx as nx
import pickle
import numpy as np

def get_training_data():
	# import the knn graph and the partitions
	G = nx.read_gpickle('pickles/knn_graph.gpickle')
	partitions = nx.read_gpickle('pickles/graph_partitions.pickle')
	N = len(G.nodes())
	M = len(partitions)

	# load the training data
	points = pickle.load(open('pickles/points.pickle', 'rb'))
	assert len(points) == N

	# flatten each training point's feature matrix into a single feature vector
	# note that not every collision event may have the same number of particles
	num_particles = 0
	for point in points:
		num_particles = max(num_particles, point.shape[0])
	num_readings = points[0].shape[1] # every particle should have the same number (3) of readigns

	X = []
	for point in points:
		feature = np.copy(point)
		feature.resize((num_readings * num_particles,))
		X.append(feature)
	X = np.array(X)

	# create labels for each node
	labels = {}
	for i, part in enumerate(partitions):
		for node in part:
			labels[node] = i

	# turn the labels into soft labels
	# for this, the label becomes the empirical distribution of the part that each node's S nearest neighbors belong to
	nns = nx.read_gpickle('pickles/nearest_neighbors.pickle')
	Y = []
	for i in range(N):
		distribution = np.zeros(M)
		for n in nns[i]:
			distribution[labels[n]] += 1
		distribution = np.divide(distribution, np.sum(distribution))
		Y.append(distribution)
	Y = np.array(Y)

	return X, Y
