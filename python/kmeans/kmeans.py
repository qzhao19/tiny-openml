# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt

class KMeans(object):
    """K-Means clustering

    Args:
        n_clusters: int. The number of clusters to form as well as the number of centroids to generate.

    """
    def __init__(self, n_clusters=8):
        self.n_clusters = n_clusters

    def _calc_euclidean_dist(self, x, y):
        """Calculate euclidien distance between vector1 and vector2
        """
        return np.sqrt(sum(np.power(x - y, 2)))

    def _init_random_centroid(self, data):
        """
        """
        n_samples = data.shape[0]
        centroid = np.zeros((self.n_clusters, data.shape[1]), dtype=float)
        index = np.random.randint(0, n_samples-1, self.n_clusters)
        centroid = data[index, :]
        return centroid

    def fit(data):
        """
        """
        n_samples = data.shape[0]
        cluster_assment = np.zeros((n_samples,2), dtype=float)     ###creer une matrice pour assigner les points da dataset
        centroid = self._init_random_centroid(data)
        cluster_changed = True
        while cluster_changed:
            cluster_changed = False
            for i in range(n_samples):       ###assigner chaque point pour le plus proche centroid
                min_dist = np.inf; min_index = -1
                for j in range(self.n_clusters):
                    dist_eclid = self._calc_euclidean_dist(centroid[j,:], data[i,:])     ###calculer la distance entre chaque point de centroids et celui de dataset
                    if dist_eclid < min_index:                               ###mise a jour centroids
                        min_dist=dist_eclid; min_index=j
                if cluster_assment[i,0] != min_index:
                    cluster_changed=True
                cluster_assment[i,:] = min_index, min_dist**2
            print(centroid)
            for cent in range(self.n_clusters):           ###recalculer les centroids
                pts_in_cluster = dataset[np.nonzero(cluster_assment[:,0] == cent)[0]]
                pts_in_cluster[cent,:] = mean(pts_in_cluster,axis=0)
        return centroid, cluster_assment

