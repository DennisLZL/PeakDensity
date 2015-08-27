__author__ = 'dennis'

import pandas as pd
from ggplot import *
import numpy as np

rawData = pd.read_csv('./Data/flame.txt', names=['x', 'y', 'cluster'], sep='\t')

# print ggplot(rawData, aes(x='x', y='y', color='cluster')) + geom_point()

# compute distance between every two points

data = rawData[['x', 'y']]

n = data.shape[0]
dist = np.zeros((n, n))

for i in range(n):
    for j in range(i + 1, n):
        dist[i, j] = np.sqrt(np.sum(np.array(data.iloc[i, :] - data.iloc[j, :]) ** 2))
        dist[j, i] = dist[i, j]
distList = dist[np.triu_indices(n, 1)]
per = 2.0
distList.sort()
dc = distList[int(n * per / 100)]


# local density using Gaussian Kernel
rho = np.sum(np.exp(-(dist/dc)**2), 1) - 1

# compute delta
maxd = np.max(distList)
delta = np.zeros(n)
neigh = np.zeros(n)

ids = np.argsort(rho)[::-1]
rho_sorted = rho[ids]
delta[ids[0]] = -1

for i in range(1, n):
    delta[ids[i]] = maxd
    for j in range(0, i):
        if dist[ids[i], ids[j]] < delta[ids[i]]:
            delta[ids[i]] = dist[ids[i], ids[j]]
delta[ids[0]] = np.max(delta)
