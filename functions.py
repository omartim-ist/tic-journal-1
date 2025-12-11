import numpy as np

def get_moments(set_):
    d_set = np.diff(np.log(set_), axis=1) # log-returns
    mu = np.mean(d_set, axis=1)
    Sigma = np.cov(d_set)
    return mu, Sigma

def variance(w, Sigma):
    return w.T @ Sigma @ w
  
def sharpe(w, mu, Sigma, rf):
    ret = w @ mu
    vol = np.sqrt(variance(w, Sigma))
    return (ret - r_f) / vol


def OBJ_(w, moments_l, metric, rf=None):
    list_ = []
    for (mu, Sigma) in moments_l:
        
        if metric == 'sharpe':
            val = sharpe(w, mu, Sigma, rf)
        elif metric == 'variance':
            val = variance(w, Sigma)
            
        list_.append(val)

    if metric == 'sharpe':
        return - np.percentile(list_, 50) / (np.percentile(list_, 75) - np.percentile(list_, 10))
    elif metric == 'variance':
        return np.percentile(list_, 90)
    


        
    
    


    
