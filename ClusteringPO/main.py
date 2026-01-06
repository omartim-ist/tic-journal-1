### 1) Import packages
import numpy as np
import yfinance as yf


### 2) Import dependencies
from config import SECURITIES, DATE_INIT, DATE_END, N_BOOT, L, T
from hierarchical_clustering import HierarchicalClustering
from functions import get_ICP, RiskParity
from politis_romano import PolitisRomanoBootstrap


### 3) Download data
SECURITIES = sorted(SECURITIES)
data = yf.download(SECURITIES, start=DATE_INIT, end=DATE_END, auto_adjust=True)['Close'] 
data_np = data.to_numpy().T
n_sec = len(SECURITIES)


### 4) Hierarchical Clustering
lrets = np.diff(np.log(data_np))
R = np.corrcoef(lrets)
HC = HierarchicalClustering(R)
HC.get_dendrogram()

while True:
    K = int(input("Select nº of clusters."))
    clusters = HC.get_clusters(K)
    done = input("Write 'Done' if you're done else 'False'.")
    if done == 'Done':
        break


### 5) Intra-Cluster Portfolios
Var = np.cov(lrets)
Mu = np.mean(lrets, axis=1)
asset_weights = {}
for cluster, assets_index in clusters.items():
    w_cluster = get_ICP(assets_index, Mu)

    for idx, w in zip(assets_index, w_cluster):
        asset_weights[idx] = (cluster, w)


### 6) Inter-Clustering Portfolio
