import numpy as np
from scipy.optimize import minimize



''' Intra-Clustering Portfolio '''

def get_ICP(assets_index, Mu):
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



''' Risk Parity '''

def RC(x, Sigma):
    sigma = np.sqrt(x.T @ Sigma @ x)
    return x*(Sigma @ x)/sigma

def objective(x,Sigma):
    Rc = RC(x,Sigma)
    return sum([(Rc[i] - Rc[j])**2 for i in range(len(Rc)) for j in range(len(Rc))])

def RiskParity(Sigma):    
    x_0 = np.array([1/len(Sigma) for i in range(len(Sigma))])
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = [(0, None) for _ in range(len(Sigma))]
    res = minimize(objective, x_0, args=(Sigma,), constraints=cons, method='SLSQP', bounds=bounds)
    return res

