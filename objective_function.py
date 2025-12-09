import numpy as np


def variance(w, Sigma):
    return w.T @ Sigma @ w
  
def sharpe(w, mu, Sigma, rf):
    ret = w @ mu
    vol = np.sqrt(variance(w, Sigma))
    return (ret - r_f) / vol

def get_moments(data):
    


    
