import ot
import sklearn
import pickle

import numpy as np
import networkx as nx

from tqdm import tqdm
from energyflow.emd import emd, emds
from energyflow.datasets import qg_jets

N = 1000
M = 200
c = 4
d = 2
k = 10
S = 20 # need S-nn graph for soft label empirical distribution over S nearest neighbors, S >> k

a = 2.5

def dist(x, y):
	return np.linalg.norm(x-y)

cov1 = np.array([[1, 0], [0, 1]])
centres = np.array([[a, a], [-a, a], [a, -a], [-a, -a]])
means = np.random.uniform(-0.5, 0.5, size=(c*N, d))

data = []
for j in range(c):
	for i in range(N):
		x = np.random.multivariate_normal(centres[j, :] + means[i, :], cov1, size=M)
		data.append(x)
        
data = np.array(data)
print(data.shape)
xs = data.reshape(-1, 2)
pickle.dump(data, open("pickles/bigger_toy_points.pickle", "wb"))
knn = []
all_distances = np.zeros((c*N, c*N))
from multiprocessing import Pool
pool = Pool(processes=16)
for i in tqdm(range(c*N)):
    distances = []
    mu_i = data[i]
    
    def f(mu_j):
    	C = np.zeros((M, M))
        for x in range(M):
            for y in range(M):
                C[x, y] = dist(mu_i[x], mu_j[y])
        d = ot.emd2([], [], C)
        return (j,d)
   	
    distances = pool.map(f, data)
    for j in range(len(distances)):
	    all_distances[i, distances[j][0]] = distances[j][1]

    distances.sort(key=lambda x: x[1])
    for v in distances[:k]:
        knn.append((i, v[0], {'weight': v[1]}))

pickle.dump(all_distances, open("pickles/bigger_toy_distances.pickle", "wb"))        

G = nx.Graph()
G.add_edges_from(knn)
M = nx.adjacency_matrix(G)
print(M.shape)

nx.write_gpickle(G, 'bigger_toy_graph.gpickle')
