__author__ = 'dennis'

import pandas as pd
import numpy as np
from sklearn import manifold
from ggplot import *


def peak_density_cluster(dist, method, gaussian=False):
    low = 0.01
    high = 0.02
    n = len(dist)
    dc = np.min(dist)
    dc_mod = np.median(dist[np.triu_indices(n, 1)]) * 0.01

    # compute dc
    while True:
        neigh_rate = np.mean((np.sum(dist < dc, 0) - 1) / float(n))
        if low < neigh_rate < high:
            break
        else:
            if neigh_rate > high:
                dc -= dc_mod
                dc_mod /= 2
            dc += dc_mod

    # compute local density
    if gaussian:
        rho = np.sum(np.exp(-(dist/dc) ** 2), 0) - 1
    else:
        rho = np.sum(dist < dc, 0) - 1

    # compute distance to peak
    delta = np.zeros(n)
    for i in range(n):
        peaks = dist[rho > rho[i], i]
        if len(peaks) == 0:
            delta[i] = np.max(dist[:, i])
        else:
            delta[i] = np.min(peaks)

    # find clusters
    if method.keys() == ['per']:
        per = method.values()[0]
        rho_low = np.min(rho) + (np.max(rho) - np.min(rho)) * per
        delta_low = np.min(delta) + (np.max(delta) - np.min(delta)) * per
    elif method.keys() == ['K']:
        K = method.values()[0]



    peaks = list(np.where([x and y for x, y in zip(rho > rho_low, delta > delta_low)])[0])

    # assign to clusters
    run_orders = list(np.argsort(rho)[::-1])
    clusters = np.zeros(n)
    for run in run_orders:
        if run in peaks:
            clusters[run] = peaks.index(run)
        else:
            higher = list(np.where(rho > rho[run])[0])
            clusters[run] = clusters[higher[np.argmin(dist[run, higher])]]

    is_peak = [i in peaks for i in range(n)]

    print "number of clusters: ", len(peaks)

    return rho, delta, is_peak, clusters

if __name__ == '__main__':
    raw_data = pd.read_csv('./Data/iris.dat')
    data = raw_data.iloc[:, 0:4]
    n = data.shape[0]
    distance = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            distance[i, j] = np.sqrt(np.sum((np.array(data.iloc[i, :]) - np.array(data.iloc[j, :])) ** 2))
            distance[j, i] = distance[i, j]

    mds = manifold.MDS(dissimilarity="precomputed")
    coord = mds.fit(distance).embedding_

    r, d, pk, cls = peak_density_cluster(distance, gaussian=True)

    results = pd.DataFrame({'rho': pd.Series(r), 'delta': pd.Series(d), 'peak': pd.Series(pk),
                            'cluster': pd.Series(cls), 'x': pd.Series(coord[:, 0]), 'y': pd.Series(coord[:, 1])})

    g1 = ggplot(aes(x='rho', y='delta', color='peak'), data=results)+geom_point()
    g1.draw()
    g2 = ggplot(aes(x='x', y='y', color='cluster'), data=results)+geom_point()
    print g2


