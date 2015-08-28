__author__ = 'dennis'

import pandas as pd
from ggplot import *
import numpy as np
from sklearn import manifold


def localDensity(dist, dc, gaussian=False):
    if gaussian:
        rho = np.sum(np.exp(-(dist/dc)**2), 0) - 1
    else:
        rho = np.sum(dist < dc, 0) - 1
    return rho


def distanceToPeak(dist, rho):
    n = len(rho)
    delta = np.zeros(n)
    for i in range(n):
        peaks = dist[rho > rho[i], i]
        if len(peaks) == 0:
            delta[i] = np.max(dist[:, i])
        else:
            delta[i] = np.min(peaks)
    return delta


def estimateDc(dist, low=0.01, high=0.02):
    dc = np.min(dist)
    dcMod = np.median(dist[np.triu_indices(len(dist), 1)]) * 0.01
    n = len(dist)

    while True:
        neighRate = np.mean((np.sum(dist < dc, 0) - 1) / float(n))
        if low < neighRate < high:
            break
        else:
            if neighRate > high:
                dc -= dcMod
                dcMod /= 2
            dc += dcMod
    return dc

def densityCluster(dist, gaussian=False):
    dc = estimateDc(dist)
    rho = localDensity(dist, dc, gaussian=gaussian)
    delta = distanceToPeak(dist, rho)
    densityDelta = pd.DataFrame({'rho': pd.Series(rho), 'delta': pd.Series(delta)})
    return densityDelta

def Kclusters(dist, densityDelta, K):
    pass

def findClusters(dist, densityDelta, rho, delta):
    peaks = np.where([x and y for x, y in zip(densityDelta['rho'] > rho, densityDelta['delta'] > delta)])[0]
    n = len(dist)
    runorder = np.argsort(densityDelta.rho)[::-1]
    cluster = np.zeros(n)
    for i in runorder:
        if i in peaks:
            cluster[i] = list(peaks).index(i)
        else:
            higher = list(np.where(densityDelta.rho > densityDelta.rho[i])[0])
            cluster[i] = cluster[higher[np.argmin(dist[i, higher])]]
    return cluster


if __name__ == '__main__':
    rawData = pd.read_csv('./Data/iris.dat')
    data = rawData.iloc[:, 0:4]
    n = data.shape[0]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = np.sqrt(np.sum((np.array(data.iloc[i, :]) - np.array(data.iloc[j, :])) ** 2))
            dist[j, i] = dist[i, j]
    res = densityCluster(dist, gaussian=False)
    print ggplot(aes(x='rho', y='delta'), data=res) + geom_point() + ggtitle('decision graph')
    cluster = findClusters(dist, res, np.percentile(res.rho, 80), np.percentile(res.delta, 80))
    mds = manifold.MDS(dissimilarity="precomputed")
    coords = mds.fit(dist).embedding_
    points = pd.DataFrame({'x': pd.Series(coords[:, 0]), 'y': pd.Series(coords[:, 1]), 'cluster': pd.Series(cluster)})
    print ggplot(aes(x='x', y='y', color='cluster'), data=points) + geom_point()
