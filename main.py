### 1) Import packages
import numpy as np
import yfinance as yf
from joblib import Parallel, delayed
from scipy.optimize import minimize
import time # for debug


### 2) Import dependencies
from config import SECURITIES, DATE_INIT, DATE_END, N_BOOT, L, T, OPT_METHOD, METRIC, RF
from politis_romano import PolitisRomanoBootstrap
from functions import get_moments, _OBJ


### 3) Download data
SECURITIES = sorted(SECURITIES)
data = yf.download(SECURITIES, start=DATE_INIT, end=DATE_END, auto_adjust=True)['Close'] 
data_np = data.to_numpy().T
n_sec = len(SECURITIES)


### 4) Data Generation via Politis-Romano Bootstrap
prb = PolitisRomanoBootstrap(serie=data_np, n_boot=N_BOOT, l=L, T=T)
prb_data = prb.generate()

# get first and second order moments for each path
moments_l = Parallel(n_jobs=-1)(
    delayed(get_moments)(set_) for set_ in prb_data
)


### 5) Optimization
def _optimize():
    res = minimize(
      _OBJ, # objective function
      w0, # initialization
      args=(moments_l, METRIC, RF,), # extra arguments for objective function
      method=OPT_METHOD, # optimization method
      bounds= [(0, 1)] * n_sec, # no short positions avoided
      constraints= {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}, # sum of weights equals 1.0 (no leverage) 
    )
    return res.x

w0 = np.ones(n_sec) / n_sec  # Initialization (equal weighted portfolio)
ti = time.time()
w_opt = _optimize()
print(time.time() - ti)


### 6) Validation

