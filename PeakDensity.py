__author__ = 'dennis'

import pandas as pd
from ggplot import *
import numpy as np
from sklearn import manifold


def local_density(dist, dc, gaussian=False):
    if gaussian:
        rho = np.sum(np.exp(-(dist/dc)**2), 0) - 1
    else:
        rho = np.sum(dist < dc, 0) - 1
    return rho


def distance_peak(dist, rho):
    n = len(rho)
    delta = np.zeros(n)
    for i in range(n):
        peaks = dist[rho > rho[i], i]
        if len(peaks) == 0:
            delta[i] = np.max(dist[:, i])
        else:
            delta[i] = np.min(peaks)
    return delta


def estimate_dc(dist, low=0.01, high=0.02):
    dc = np.min(dist)
    dc_mod = np.median(dist[np.triu_indices(len(dist), 1)]) * 0.01
    n = len(dist)

    while True:
        neigh_rate = np.mean((np.sum(dist < dc, 0) - 1) / float(n))
        if low < neigh_rate < high:
            break
        else:
            if neigh_rate > high:
                dc -= dc_mod
                dc_mod /= 2
            dc += dc_mod
    return dc

def density_cluster(dist, gaussian=False):
    dc = estimate_dc(dist)
    rho = local_density(dist, dc, gaussian=gaussian)
    delta = distance_peak(dist, rho)
    info = pd.DataFrame({'rho': pd.Series(rho), 'delta': pd.Series(delta)})
    return info


def find_clusters(dist, info, rho, delta):
    peaks = np.where([x and y for x, y in zip(info['rho'] > rho, info['delta'] > delta)])[0]
    run_order = list(np.argsort(info.rho)[::-1])
    cluster = np.zeros(len(dist))
    for run in run_order:
        if run in peaks:
            cluster[run] = list(peaks).index(run)
        else:
            higher = list(np.where(info.rho > info.rho[run])[0])
            cluster[run] = cluster[higher[np.argmin(dist[run, higher])]]
    return cluster


if __name__ == '__main__':
    raw_data = pd.read_csv('./Data/iris.dat')
    data = raw_data.iloc[:, 0:4]
    n = data.shape[0]
    distance = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distance[i, j] = np.sqrt(np.sum((np.array(data.iloc[i, :]) - np.array(data.iloc[j, :])) ** 2))
            distance[j, i] = distance[i, j]
    res = density_cluster(distance, gaussian=False)


    print ggplot(aes(x='rho', y='delta'), data=res) + geom_point() + ggtitle('decision graph')

    cls = find_clusters(distance, res, np.percentile(res.rho, 80), np.percentile(res.delta, 80))

    mds = manifold.MDS(dissimilarity="precomputed")
    coord = mds.fit(distance).embedding_
    points = pd.DataFrame({'x': pd.Series(coord[:, 0]), 'y': pd.Series(coord[:, 1]), 'cluster': pd.Series(cls)})

    print ggplot(aes(x='x', y='y', color='cluster'), data=points) + geom_point()

