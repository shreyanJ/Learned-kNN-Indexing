import ot
import sklearn
import pickle

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from tqdm import tqdm
from energyflow.emd import emd, emds
from energyflow.datasets import qg_jets

N = 50
M = 30
d = 2
k = 10

def dist(x, y):
	return np.linalg.norm(x-y)

cov1 = np.array([[1, 0], [0, 1]])
cov2 = np.array([[0.866, 0.5], [0.5, 0.866]])
means = np.random.uniform(0.0, 1.0, size=(2*N, d))

data = []
for i in range(N):
	x = np.random.multivariate_normal(means[i, :], cov1, size=M)
	data.append(x)

for i in range(N, 2*N):
	x = np.random.multivariate_normal(means[i, :], cov2, size=M)
	data.append(x)


knn = []
for i in tqdm(range(2*N)):
	distances = []
	for j in range(2*N):
		mu_i, mu_j = data[i], data[j]
		C = np.zeros((M, M))
		for x in range(M):
			for y in range(M):
				C[x, y] = dist(mu_i[x], mu_j[y])
		distances.append((j, ot.emd2(np.ones(M), np.ones(M), C)))
	distances.sort(key=lambda x: x[1])
	for v in distances[:k]:
		knn.append((i, v[0], {'weight': v[1]}))

G = nx.Graph()
G.add_edges_from(knn)
M = nx.adjacency_matrix(G)
print(M.shape)

nx.write_gpickle(G, 'graph.gpickle')