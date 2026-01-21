import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import math


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
        
        D = np.sqrt(0.5 * (1 - self.R))
        self.D = D
        
        np.fill_diagonal(self.D, 0.0)
        condensed = squareform(self.D, checks=False)
        self.Z = linkage(condensed, method="average")
        
    def get_dendrogram(self):
        """
        Builds the hierarchical clustering linkage matrix using
        average linkage and optionally plots the dendrogram.

        The correlation-to-distance transformation used is:
            D_ij = sqrt(0.5 * (1 - R_ij))
        """
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
    
    def choose_k_silhouette(self, k_min=2, k_max=None):
        """
        Selects the number of clusters K by maximizing the silhouette score,
        using the *precomputed* distance matrix from the correlation matrix.

        Parameters
        ----------
        k_min : int
            Minimum K to test (must be >= 2).
        k_max : int or None
            Maximum K to test. If None, uses min(N-1, 25).

        Returns
        -------
        SilhouetteResult
            best_k, best_score, and dict of scores per K.
        """
        if not hasattr(self, "Z") or not hasattr(self, "D"):
            self.get_dendrogram()

        N = self.R.shape[0]
        if k_max is None:
            k_max = 1 + math.floor(2 * np.sqrt(N))
        k_min = max(2, int(k_min))
        k_max = int(k_max)

        if k_min > k_max:
            raise ValueError(f"k_min ({k_min}) must be <= k_max ({k_max}).")

        scores = {}
        best_k = None
        best_score = -np.inf

        for k in range(k_min, k_max + 1):
            labels = fcluster(self.Z, t=k, criterion="maxclust")

            # Silhouette requires at least 2 clusters and not all singletons.
            n_unique = np.unique(labels).size
            if n_unique < 2 or n_unique >= N:
                continue

            s = silhouette_score(self.D, labels, metric="precomputed")
            scores[k] = float(s)

            if s > best_score:
                best_score = s
                best_k = k


        if self.verbose >= 1:
            ks = sorted(scores.keys())
            vals = [scores[k] for k in ks]
            plt.figure()
            plt.plot(ks, vals, marker="o")
            plt.axvline(best_k, linestyle="--")
            plt.xlabel("K (number of clusters)")
            plt.ylabel("Silhouette score")
            plt.show()

        return best_k, self.get_clusters(best_k)
