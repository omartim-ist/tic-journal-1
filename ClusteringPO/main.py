### 1) Import packages
import numpy as np
import yfinance as yf
import sys
import os


### 2) Import dependencies
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from config import SECURITIES, DATE_INIT, DATE_END, N_BOOT, L, T
from hierarchical_clustering import HierarchicalClustering
from functions import get_ICP, build_CVar, build_asset_weights, RiskParity
from politis_romano import PolitisRomanoBootstrap
from validation import Validation


### 3) Download data
SECURITIES = sorted(SECURITIES)
data = yf.download(SECURITIES, start=DATE_INIT, end=DATE_END, auto_adjust=True)['Close'] 
data = data.ffill()
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
for cluster in np.unique(clusters):
    assets_index = np.where(clusters == cluster)[0]
    w_cluster = get_ICP(assets_index, Mu, Var)

    for idx, w in zip(assets_index, w_cluster):
        asset_weights[idx] = (cluster, w)
CVar = build_CVar(asset_weights, Var) # Clusters Covariance Matrix


### 6) Inter-Clustering Portfolio
x_clusters, sigma_rpp = RiskParity(CVar)
x_assets = build_asset_weights(asset_weights, x_clusters)


### 7) Validation via Politis-Romano
prb = PolitisRomanoBootstrap(serie=data_np, n_boot=N_BOOT, l=L, T=T)

val = Validation(prb, x_assets)
val._plot_statistics()
val._plot_paths()
