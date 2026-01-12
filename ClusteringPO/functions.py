import numpy as np
from scipy.optimize import minimize

''' Basics '''

def sharpe(lrets, offset):
    mu = np.mean(lrets, axis=1)
    sigma = np.std(lrets, axis=1)
    
    mu_a = np.expm1(mu * 365)
    sigma_a = sigma * np.sqrt(365)
    return (mu_a - offset) / sigma_a
   
 
''' Intra-Clustering Portfolio '''

def get_ICP(assets_index, Mu, Sigma, alpha=2.0):
    """
    Compute intra-cluster weights using a return-to-risk rule.

    Weights are proportional to μ / σ^alpha, where μ is the expected return
    and σ the asset volatility. The parameter alpha controls how strongly
    risk is penalized. Negative returns are set to zero (long-only).
    """
    
    Sigma = np.array([Sigma[index] for index in assets_index], dtype=float)
    Mu = np.array([max(Mu[index], 0) for index in assets_index], dtype=float)

    w = Mu / Sigma**alpha
    w /= w.sum()
    return w


def build_CCov(asset_weights, Cov):  # Clusters Covariance Matrix
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



    
''' Inter-Clustering Portfolio: Risk Parity '''

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

    # final risk-parity weights and portfolio volatility
    x_rp = res.x
    sigma_rp = np.sqrt(x_rp.T @ Cov @ x_rp)
    return x_rp, sigma_rp

