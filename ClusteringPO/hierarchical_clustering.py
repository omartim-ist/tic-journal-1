import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt


class HierarchicalClustering:
    """
    Hierarchical clustering using average linkage starting from
    a correlation matrix.
    """

    def __init__(self, R, verbose=1):
        """
        Parameters
        ----------
        R : ndarray (N x N)
            Correlation matrix with ones on the diagonal.
        verbose : int
            If >= 1, dendrograms are plotted.
        """
        self.R = R
        self.verbose = verbose

    def get_dendrogram(self):
        """
        Builds the hierarchical clustering linkage matrix using
        average linkage and optionally plots the dendrogram.

        The correlation-to-distance transformation used is:
            D_ij = sqrt(0.5 * (1 - R_ij))
        """
        D = np.sqrt(0.5 * (1 - self.R))
        condensed = squareform(D, checks=False)
        self.Z = linkage(condensed, method="average")

        if self.verbose >= 1:
            dendrogram(self.Z)
            plt.show()

    def get_clusters(self, K):
        """
        Cuts the dendrogram to obtain a fixed number of clusters.

        Parameters
        ----------
        K : int
            Number of desired clusters.

        Returns
        -------
        clusters : ndarray
            Cluster label (1..K) assigned to each observation.
        """
        clusters = fcluster(self.Z, t=K, criterion="maxclust")

        if self.verbose >= 1:
            plt.figure()
            dendrogram(self.Z)
            plt.axhline(y=self.Z[-(K - 1), 2], linestyle="--")
            plt.show()

        return clusters
