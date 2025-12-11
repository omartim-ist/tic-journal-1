import numpy as np

def variance(w, Sigma):
    return w.T @ Sigma @ w
  
def sharpe(w, mu, Sigma, rf):
    ret = w @ mu
    vol = np.sqrt(variance(w, Sigma))
    return (ret - r_f) / vol

def get_moments(set):
    d_set = np.diff(np.log(set), axis=1) # log-returns
    mu = np.mean(d_set, axis=1)
    Sigma = np.cov(d_set)
    return mu, Sigma
        
    
    


    
