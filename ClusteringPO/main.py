### 1) Import packages
import numpy as np
import yfinance as yf
from joblib import Parallel, delayed
from scipy.optimize import minimize
import time # for debug


### 2) Import dependencies
from config import SECURITIES, DATE_INIT, DATE_END, N_BOOT, L, T, OPT_METHOD, METRIC, RF
from hierarchical_clustering import HierarchicalClustering
from politis_romano import PolitisRomanoBootstrap


### 3) Download data
SECURITIES = sorted(SECURITIES)
data = yf.download(SECURITIES, start=DATE_INIT, end=DATE_END, auto_adjust=True)['Close'] 
data_np = data.to_numpy().T
n_sec = len(SECURITIES)


### 4) Hierarchical Clustering
R = np.corrcoef(data_np)
HC = HierarchicalClustering(R)
HC.get_dendrogram()

while True:
    K = int(input("Select nº of clusters."))
    clusters = HC.get_clusters(K)
    done = input("Write 'Done' if you're done else 'False'.")
    if done == 'Done':
        break

### 5)  



