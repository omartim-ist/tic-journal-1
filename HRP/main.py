### 1) Import packages
import numpy as np
import pandas as pd
import sys
import os
import time


### 2) Import dependencies
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from config import OFFSET, SHARPE_MIN, N_SPLITS, TRAIN_BLOCKS_INIT, TEST_BLOCKS, SECURITIES, N_CLUSTERS, ALPHA, N_BOOT, L, T, WEIGHT_CUTOFF
from walk_forward_split import WalkForwardSplit
from HRP.hierarchical_clustering import HierarchicalClustering
from HRP.functions import sharpe, get_ICP, build_CCov, build_asset_weights, RiskParity
from politis_romano import PolitisRomanoBootstrap
from HRP.validation import Validation


### 3) Download data
SECURITIES = sorted(SECURITIES)
data = pd.read_csv(os.path.join(ROOT, 'data', 'historical_data.csv')).drop(columns=['Date'])
data = data.reindex(columns=SECURITIES)


### 4) Remove assets without full data
data_ = data.ffill()
data_ = data_.loc[3:,:].reset_index().drop(columns=['index'])
data_ = data_.loc[:, ~data_.iloc[0].isna()]

const_cols = data_.columns[data_.apply(lambda s: np.ptp(s.to_numpy()) <= 1e-12)]
data_ = data_.drop(columns=const_cols)

data_np = data_.to_numpy()


### 5) Walk Forward Splits
WFS = WalkForwardSplit(N_SPLITS, TRAIN_BLOCKS_INIT, TEST_BLOCKS)
splits = WFS.expanding(data_np)


### 6) Apply Walk-Forward
def _get_portfolio(lrets):

    ### A) Hierarchical Clustering
    R = np.corrcoef(lrets, rowvar=False)
    HC = HierarchicalClustering(R)
    clusters = HC.get_clusters(N_CLUSTERS)


    ### B) Intra-Cluster Portfolios
    Cov = np.cov(lrets, rowvar=False)
    Sigma = np.std(lrets, axis=0)
    Mu = np.mean(lrets, axis=0)
    asset_weights = {}
    for cluster in np.unique(clusters):
        assets_index = np.where(clusters == cluster)[0]
        w_cluster = get_ICP(assets_index, Mu, Sigma, ALPHA)
        
        keep = w_cluster >= WEIGHT_CUTOFF
        
        assets_index = assets_index[keep]
        w_cluster = w_cluster[keep]
        w_cluster = w_cluster / w_cluster.sum()
        
        for idx, w in zip(assets_index, w_cluster):
            asset_weights[idx] = (cluster, w)
    CCov = build_CCov(asset_weights, Cov) # Clusters Covariance Matrix
    
    
    ### C) Inter-Clustering Portfolio
    x_clusters, _ = RiskParity(CCov)
    x_assets = build_asset_weights(asset_weights, x_clusters, lrets.shape[1])
    
    x_assets = np.where(x_assets < WEIGHT_CUTOFF, 0.0, x_assets)
    return x_assets / x_assets.sum()


def _stats_portfolio(lrets, x):
    p_lrets = lrets @ x
    
    ann_ret = np.expm1(np.mean(p_lrets) * 365)
    ann_vol = np.std(p_lrets) * np.sqrt(365)
    
    return np.array([ann_ret, ann_vol, (ann_ret - OFFSET) / ann_vol])
    
    
wf_stats = (np.empty([len(splits), 3]), np.empty([len(splits), 3]))
for split in range(len(splits)):

    train_lrets = np.diff(np.log(splits[split][0]), axis=0)
    test_lrets = np.diff(np.log(splits[split][1]), axis=0)
    
    train_sharpes = sharpe(train_lrets, OFFSET)
    mask = train_sharpes > SHARPE_MIN
    
    x_valid = _get_portfolio(train_lrets[:,mask])
    X = np.zeros(train_lrets.shape[1])
    X[mask] = x_valid
    
    wf_stats[0][split,:] = _stats_portfolio(train_lrets, X)
    wf_stats[1][split,:] = _stats_portfolio(test_lrets, X)

print(wf_stats[0])
print()
print(wf_stats[1])
breakpoint()    
 
    

#df_weights = pd.DataFrame({"asset": data_.columns,"weight": x_assets, 'cluster': clusters})
#df_weights.to_csv("asset_weights.csv")








### 8) Validation via Politis-Romano
prb = PolitisRomanoBootstrap(serie=p_value, n_boot=N_BOOT, l=L, T=T)

ti = time.time()
validation = Validation(prb, OFFSET)
print('total time:', time.time() - ti)

validation._plot_statistics()
validation._plot_paths()

