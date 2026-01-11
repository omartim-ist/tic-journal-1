import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt



''' Basics '''

def sharpe(lrets, offset):
    mu = np.mean(lrets, axis=1)
    sigma = np.std(lrets, axis=1)
    
    mu_a = np.expm1(mu * 365)
    sigma_a = sigma * np.sqrt(365)
    return (mu_a - offset) / sigma_a
    
''' Intra-Clustering Portfolio '''

def get_ICP(assets_index, Mu, Var):
    """
    Computes intra-cluster portfolio weights adjusted for the return–volatility relationship.

    - If the cluster has 1 asset: assigns 100% weight to that asset.
    - If the cluster has 2 assets: uses Sharpe-like weights (max(mu, 0) / sigma).
    - If the cluster has 3 or more assets: removes the structural return–volatility
      component via a cross-sectional regression and allocates weights based on
      residual volatility (risk not explained by return).

    Parameters
    ----------
    assets_index : list[int]
        Global indices of the assets in the cluster
    Mu : array-like
        Expected returns of all assets
    Var : array-like
        Assets Covariance matrix

    Returns
    -------
    w : np.ndarray
        Normalized intra-cluster weights (sum to 1)
    """

    if len(assets_index) == 1:
        # Single-asset cluster: full allocation
        return np.array([1.0])
        
    # Asset volatilities within the cluster
    Sigma = np.sqrt([Var[index, index] for index in assets_index])
    Mu = [Mu[index] for index in assets_index]
    
    if len(assets_index) == 2:
        # Small cluster: Sharpe-like allocation with non-negative returns
        w = np.maximum(Mu, 0.0) / Sigma
        w /= w.sum()
        return w
        
    # For larger clusters, use return magnitude
    Mu = np.abs(Mu)

    # Cross-sectional regression: Sigma = a + b * |Mu|
    X = np.column_stack([np.ones(len(Mu)), Mu])
    a, b = np.linalg.lstsq(X, Sigma, rcond=None)[0]

    # Residual volatility (excess risk given return)
    Sigma_adj = Sigma - b * Mu

    # Allocate inversely to residual risk
    w = 1.0 / Sigma_adj
    w /= w.sum()
    
    return w


def build_CVar(asset_weights, Var):  # Clusters Covariance Matrix
    """
    Build the cluster-level covariance matrix from the asset-level covariance matrix.

    Parameters
    ----------
    asset_weights : dict
        Dictionary mapping asset index -> (cluster_id, intra-cluster weight).
        Each asset belongs to exactly one cluster, and weights within each cluster
        are assumed to sum to 1.
    Var : np.ndarray
        Asset-level covariance matrix of shape (N, N).

    Returns
    -------
    np.ndarray
        Cluster-level covariance matrix of shape (K, K), where K is the number of clusters.
    """

    # Number of assets
    N = Var.shape[0]

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
    return W.T @ Var @ W


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

    # cluster ids in the same order used to build CVar
    cluster_ids = np.array(sorted({c for (c, _) in asset_weights.values()}))
    c2j = {c: j for j, c in enumerate(cluster_ids)}

    for i, (c, w_ic) in asset_weights.items():
        x[i] = x_clusters[c2j[c]] * w_ic

    return x



    
''' Inter-Clustering Portfolio: Risk Parity '''

def RC(x, Var):
    '''
    Computes the risk contribution of each asset to the portfolio volatility.
    '''
    sigma = np.sqrt(x.T @ Var @ x)
    return x*(Var @ x) / sigma

def objective(x, Var):
    '''
    Objective function for risk parity.
    It penalizes differences between pairwise risk contributions, so that
    all assets end up contributing equally to total portfolio risk.
    '''
    Rc = RC(x, Var)
    S = Rc.sum()
    S_2 = (Rc**2).sum()
    return (len(Rc) * S_2 - S**2) * 10_000

def RiskParity(Var):  
    '''
    Solves the long-only risk parity optimization problem.
    Returns the optimal risk-parity weights and the resulting
    portfolio volatility.
    '''
    # initial equal-weight portfolio
    x_0 = np.array([1/len(Var) for i in range(len(Var))])
    
    # fully invested, long-only portfolio
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = [(0, None) for _ in range(len(Var))]

    # constrained optimization
    res = minimize(objective, x_0, args=(Var,), constraints=cons, method='SLSQP', bounds=bounds)

    # final risk-parity weights and portfolio volatility
    x_rp = res.x
    sigma_rp = np.sqrt(x_rp.T @ Var @ x_rp)
    return x_rp, sigma_rp
