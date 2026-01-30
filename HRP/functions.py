import numpy as np
import pandas as pd
from scipy.optimize import minimize
from joblib import Parallel, delayed
import os
import math

from .config import RETURN_MIN, WEIGHT_CUTOFF, OFFSET
from .hierarchical_clustering import HierarchicalClustering

''' Basics '''

def portfolio_metrics(log_prices):
    lrets = np.diff(log_prices)
    
    mu = np.mean(lrets)
    sigma = np.std(lrets)
    
    mu_a = np.expm1(mu * 365)
    sigma_a = sigma * np.sqrt(365)
    
    ath = np.maximum.accumulate(log_prices)
    max_dd = 1 - np.exp(np.min(log_prices - ath))
    return [mu_a, sigma_a, (mu_a - OFFSET) / sigma_a, max_dd]
   
 
''' Clusters Covariance Matrix '''
def build_CCov(asset_weights, Cov):
    """
    Build the cluster-level covariance matrix from the asset-level covariance matrix.

    Parameters
    ----------
    asset_weights : dict
        Dictionary mapping asset index -> (cluster_id, intra-cluster weight).
        Each asset belongs to exactly one cluster, and weights within each cluster
        are assumed to sum to 1.
    Cov : np.ndarray
        Asset-level covariance matrix of shape (N, N).

    Returns
    -------
    np.ndarray
        Cluster-level covariance matrix of shape (K, K), where K is the number of clusters.
    """

    # Number of assets
    N = Cov.shape[0]

    # Unique cluster identifiers (sorted for deterministic ordering)
    cluster_ids = np.array(sorted({c for (c, _) in asset_weights.values()}))
    K = len(cluster_ids)

    # Map each cluster id to a column index in the cluster weight matrix
    c2j = {c: j for j, c in enumerate(cluster_ids)}

    # Weight matrix W of shape (N, K)
    # Column j contains the portfolio weights of cluster j over all assets
    W = np.zeros((N, K), dtype=float)
    for i, (c, w) in asset_weights.items():
        W[i, c2j[c]] = w

    # Aggregate asset-level covariance to cluster-level covariance
    # Var_cluster = W^T * Var * W
    return W.T @ Cov @ W


def build_asset_weights(asset_weights, x_clusters, N_total):
    """
    Build final asset-level weights from inter-cluster and intra-cluster weights.

    Parameters
    ----------
    asset_weights : dict
        asset_weights[i] = (cluster_id, intra_cluster_weight)
    x_clusters : np.ndarray
        Risk parity weights at the cluster level (ordered by sorted cluster ids)

    Returns
    -------
    np.ndarray
        Final asset weights vector of shape (N,)
    """
    x = np.zeros(N_total, dtype=float)

    # cluster ids in the same order used to build CCov
    cluster_ids = np.array(sorted({c for (c, _) in asset_weights.values()}))
    c2j = {c: j for j, c in enumerate(cluster_ids)}

    for i, (c, w_ic) in asset_weights.items():
        x[i] = x_clusters[c2j[c]] * w_ic

    return x



    
''' Risk Parity '''

def RC(x, Cov):
    '''
    Computes the risk contribution of each asset to the portfolio volatility.
    '''
    sigma = np.sqrt(x.T @ Cov @ x)
    return x*(Cov @ x) / sigma

def objective(x, Cov):
    '''
    Objective function for risk parity.
    It penalizes differences between pairwise risk contributions, so that
    all assets end up contributing equally to total portfolio risk.
    '''
    Rc = RC(x, Cov)
    S = Rc.sum()
    S_2 = (Rc**2).sum()
    return (len(Rc) * S_2 - S**2) * 10_000

def RiskParity(Cov):  
    '''
    Solves the long-only risk parity optimization problem.
    Returns the optimal risk-parity weights and the resulting
    portfolio volatility.
    '''
    # initial equal-weight portfolio
    x_0 = np.array([1/len(Cov) for i in range(len(Cov))])
    
    # fully invested, long-only portfolio
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = [(0, None) for _ in range(len(Cov))]

    # constrained optimization
    res = minimize(objective, x_0, args=(Cov,), constraints=cons, method='SLSQP', bounds=bounds)
    return res.x





''' Strategies '''

def HRP(lrets, n_clusters):
    
    ### B) Hierarchical Clustering
    R = np.corrcoef(lrets, rowvar=False)
    HC = HierarchicalClustering(R)
    clusters = HC.get_clusters(n_clusters)
    
    
    ### C) Intra-Cluster Portfolios
    Cov = np.cov(lrets, rowvar=False)
    asset_weights = {}
    for cluster in np.unique(clusters):
        assets_index = np.where(clusters == cluster)[0]
        
        w_cluster = RiskParity(Cov[np.ix_(assets_index, assets_index)])
        
        keep = w_cluster >= WEIGHT_CUTOFF
        
        assets_index = assets_index[keep]
        w_cluster = w_cluster[keep]
        w_cluster = w_cluster / w_cluster.sum()
        
        for idx, w in zip(assets_index, w_cluster):
            asset_weights[idx] = (cluster, w)
    CCov = build_CCov(asset_weights, Cov) # Clusters Covariance Matrix
    
    
    ### D) Inter-Clustering Portfolio
    x_clusters = RiskParity(CCov)
    x_assets = build_asset_weights(asset_weights, x_clusters, lrets.shape[1])
    
    x_assets = np.where(x_assets < WEIGHT_CUTOFF, 0.0, x_assets)  
    return x_assets / x_assets.sum()

    





''' Walk Forward '''

def _get_portfolio(lrets, strategy, n_clusters):
    
    ### A) Filter by minimum in-sample returns
    Mu = np.mean(lrets, axis=0)
    mask = np.expm1(Mu*365) > RETURN_MIN

    if strategy == 'HRP':
        x_filt = HRP(lrets[:,mask], n_clusters)
        x = np.zeros(lrets.shape[1])
        x[mask] = x_filt
        return x
    
    elif strategy == 'RP':
        x_filt = HRP(lrets[:,mask], 1)
        x = np.zeros(lrets.shape[1])
        x[mask] = x_filt
        return x
    
    elif strategy == 'EWP':
        return np.full(lrets.shape[1], 1 / lrets.shape[1])

def _stats_portfolio(lrets, x):
    p_lrets = np.log(np.expm1(lrets) @ x + 1)

    try:
        ann_ret = np.expm1(np.mean(p_lrets) * 365)
        ann_vol = np.std(p_lrets) * np.sqrt(365)   
        return np.array([ann_ret, ann_vol, (ann_ret - OFFSET) / ann_vol]), p_lrets
    except FloatingPointError:
        return None, []
       

def _one_split(split_idx, split_data, strategy, n_clusters):
    
    train_prices, test_prices = split_data
    train_prices, test_prices = train_prices.to_numpy(), test_prices.to_numpy()

    train_lrets = np.diff(np.log(train_prices), axis=0)
    test_lrets = np.diff(np.log(test_prices), axis=0)

    x = _get_portfolio(train_lrets, strategy, n_clusters)

    train_row, _ = _stats_portfolio(train_lrets, x)
    test_row, p_lrets = _stats_portfolio(test_lrets, x)

    p_lrets_list = np.asarray(p_lrets).ravel().tolist()

    return split_idx, np.asarray(train_row), np.asarray(test_row), p_lrets_list


### ------ Paralel execution ------ ###
def _exe_backtest(strategy, splits, n_clusters=None):
    n_splits = len(splits)
    
    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_one_split)(i, splits[i], strategy, n_clusters,)
        for i in range(n_splits)
    )
    
    # ordenar por índice do split (ordem garantida)
    results.sort(key=lambda t: t[0])
    
    wf_train = np.empty((n_splits, 3), dtype=float)
    wf_test = np.empty((n_splits, 3), dtype=float)
    all_p_lrets = []
    
    for i, train_row, test_row, p_lrets_list in results:
        wf_train[i, :] = train_row
        wf_test[i, :]  = test_row
        all_p_lrets.extend(p_lrets_list)
        
    return np.cumsum(all_p_lrets)



''' SP500 '''
def get_sp500(dates, ROOT):
    
    df = pd.DataFrame(index=dates)
    
    sp500 = pd.read_csv(os.path.join(ROOT, "data", "sp500_historical_data.csv"))
    sp500['Date'] = pd.to_datetime(sp500["Date"])
    sp500 = sp500.set_index("Date")
    
    df = df.join(sp500, how="left")
    df = df.ffill().bfill()
    
    return np.log(df['SPY'] / df['SPY'].iloc[0])



''' Get last weights '''
def _get_weights(prices, n_clusters):
    
    lrets = np.diff(np.log(prices.to_numpy()), axis=0)

    ### A) Filter by minimum in-sample returns
    Mu = np.mean(lrets, axis=0)
    mask = np.expm1(Mu*365) > RETURN_MIN

    x_filt = HRP(lrets[:,mask], n_clusters)
    x = np.zeros(lrets.shape[1])
    x[mask] = x_filt
    
    return pd.Series(x, index=prices.columns, name="weight")
    

