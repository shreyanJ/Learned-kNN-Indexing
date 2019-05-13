import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py as hd

from tqdm import tqdm
from scipy.stats import entropy

fn =  '../data/anomaly.h5'
N = 10000
p = 700

df = pd.read_hdf(fn,stop=N)
print(df.shape)
print("Memory in GB:",sum(df.memory_usage(deep=True)) / (1024**3))

y = df.iloc[:, -1].values
X = df.iloc[:,:-1].values
X = np.reshape(X, (N, p, 3))
Bs, Ss = [], []
for arr,events in [(Bs, X[y==0]), (Ss, X[y==1])]:
    for i,x in enumerate(events):
        # ignore padded particles and removed particle id information
        x = x[x[:,0] != 0]

        # center jet according to pt-centroid
        yphi_avg = np.average(x[:,1:3], weights=x[:,0], axis=0)
        x[:,1:3] -= yphi_avg

        # mask out any particles farther than R=0.4 away from center (rare)
        # add to list
        x = sorted(x, key=lambda a: a[0], reverse=True)
        x = np.array(x[:100])
        arr.append(x)

# choose interesting events
from energyflow.emd import emd, emds
import matplotlib.pyplot as plt

ev0, ev1 = Ss[0], Ss[15]

# calculate the EMD and the optimal transport flow
R = 5
emdval, B = emd(ev0, ev1, R=R, return_flow=True)

# # plot the two events
# colors = ['red', 'blue']
# labels = ['Gluon Jet 1', 'Gluon Jet 2']
# for i,ev in enumerate([ev0, ev1]):
#     pts, ys, phis = ev[:,0], ev[:,1], ev[:,2]
#     plt.scatter(ys, phis, marker='o', s=2*pts, color=colors[i], lw=0, zorder=10, label=labels[i])
    
# # plot the flow
# mx = B.max()
# xs, xt = ev0[:,1:3], ev1[:,1:3]
# for i in range(xs.shape[0]):
#     for j in range(xt.shape[0]):
#         if B[i, j] > 0:
#             plt.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]],
#                      alpha=B[i, j]/mx, lw=1.25, color='black')

# # plot settings
# plt.xlim(-R, R); plt.ylim(-R, R)
# plt.xlabel('Rapidity'); plt.ylabel('Azimuthal Angle')
# plt.xticks(np.linspace(-R, R, 5)); plt.yticks(np.linspace(-R, R, 5))

# plt.text(0.6, 0.03, 'EMD: {:.1f} GeV'.format(emdval), fontsize=10, transform=plt.gca().transAxes)
# plt.legend(loc=(0.1, 1.0), frameon=False, ncol=2, handletextpad=0)

# plt.show()

data = Bs + Ss
#emds = emds(data, R=R, norm=True, verbose=1, n_jobs=1, print_every=10000)

def kls(data):
    N = len(data)
    out = np.zeros((N, N))
    for i in tqdm(range(len(data))):
        for j in range(len(data)):
            mu_i, mu_j = [x[0] for x in data[i]], [x[0] for x in data[j]]
            mu_i = mu_i + [0 for _ in range(100-len(mu_i))]
            mu_j = mu_j + [0 for _ in range(100-len(mu_j))]
            kl = entropy(mu_i, mu_j)
            out[i][j] = kl 
    return out

KLs = kls(data)

import pickle
pickle.dump(KLs, open("CERN_kls.p", "wb"))
