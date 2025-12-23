import numpy as np
import yfinance as yf

def logrets(time_series): 
    series_np = time_series.to_numpy()
    lrets = np.log(series_np)
    return np.array([[lrets[i+1][j]-lrets[i][j] for i in range(len(lrets)-1)] for j in range(0,len(time_series.columns))])

def corr_dist(lrets):
    rho_matrix = np.corrcoef(lrets)
    return np.sqrt(0.5*(1-rho_matrix))
    
def intra_port(cluster):
    sigma = np.sqrt(np.diag(np.cov(lrets)))
    return (1/sigma)/(sum(1/sigma))
