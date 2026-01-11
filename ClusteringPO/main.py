### 1) Import packages
import numpy as np
import pandas as pd
import sys
import os
import time


### 2) Import dependencies
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from config import OFFSET, SHARPE_MIN, SECURITIES, N_CLUSTERS, N_BOOT, L, T, WEIGHT_CUTOFF
from hierarchical_clustering import HierarchicalClustering
from functions import sharpe, get_ICP, build_CVar, build_asset_weights, RiskParity
from politis_romano import PolitisRomanoBootstrap
from validation import Validation


### 3) Download data
SECURITIES = sorted(SECURITIES)
data = pd.read_csv(os.path.join(ROOT, 'data', 'historical_data.csv')).drop(columns=['Date'])
data = data.reindex(columns=SECURITIES)


### 4) Assets filtration
data_ = data.ffill()
data_ = data_.loc[3:,:].reset_index().drop(columns=['index'])
data_ = data_.loc[:, ~data_.iloc[0].isna()]

const_cols = data_.columns[data_.apply(lambda s: np.ptp(s.to_numpy()) <= 1e-12)]
data_ = data_.drop(columns=const_cols)

data_np = data_.to_numpy().T

lrets = np.diff(np.log(data_np))
sharpes = sharpe(lrets, OFFSET)
mask = sharpes > SHARPE_MIN

data_np, data_ = data_np[mask], data_.loc[:,mask]
lrets = np.diff(np.log(data_np))


### 5) Hierarchical Clustering
R = np.corrcoef(lrets)
N = R.shape[0]
HC = HierarchicalClustering(R)
clusters = HC.get_clusters(N_CLUSTERS)


### 6) Intra-Cluster Portfolios
Var = np.cov(lrets)
Mu = np.mean(lrets, axis=1)
asset_weights = {}
for cluster in np.unique(clusters):
    assets_index = np.where(clusters == cluster)[0]
    w_cluster = get_ICP(assets_index, Mu, Var)
    
    keep = w_cluster >= WEIGHT_CUTOFF
    
    assets_index = assets_index[keep]
    w_cluster = w_cluster[keep]
    w_cluster = w_cluster / w_cluster.sum()
    
    for idx, w in zip(assets_index, w_cluster):
        asset_weights[idx] = (cluster, w)
CVar = build_CVar(asset_weights, Var) # Clusters Covariance Matrix


### 7) Inter-Clustering Portfolio
x_clusters, sigma_rpp = RiskParity(CVar)
x_assets = build_asset_weights(asset_weights, x_clusters, lrets.shape[0])

x_assets = np.where(x_assets < WEIGHT_CUTOFF, 0.0, x_assets)
x_assets = x_assets / x_assets.sum()

df_weights = pd.DataFrame({"asset": data_.columns,"weight": x_assets, 'cluster': clusters})
df_weights.to_csv("asset_weights.csv")


### 8) Validation via Politis-Romano
mask = x_assets > 0
x_assets_cut = x_assets[mask]
data_np_cut = data_np[mask,:]

prb = PolitisRomanoBootstrap(serie=data_np_cut, n_boot=N_BOOT, l=L, T=T)

ti = time.time()
validation = Validation(prb, x_assets_cut, OFFSET)
print('total time:', time.time() - ti)

validation._plot_statistics()
validation._plot_paths()

