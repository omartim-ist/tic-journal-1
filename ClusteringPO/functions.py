import numpy as np

# Intra-Clustering Portfolio
def get_ICP(assets_index, Mu):
    
    if len(assets_index) == 1:
        return np.array([1.0])
        
    Sigma = np.sqrt([Var[index, index] for index in assets_index])
    Mu = [Mu[index] for index in assets_index]
    
    if len(assets_index) == 2:
        w = np.maximum(Mu, 0.0) / Sigma
        w /= w.sum()
        return w
        
    Mu = np.abs(Mu)
    # Linear regression
    X = np.column_stack([np.ones(len(Mu)), Mu])
    beta = np.linalg.lstsq(X, Sigma, rcond=None)[0]
    a, b = beta

    Sigma_adj = Sigma - b * Mu

    # pesos intra-cluster
    w = 1.0 / Sigma_adj
    w /= w.sum()
    
    return w
