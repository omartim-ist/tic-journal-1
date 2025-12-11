### 1) Import packages
import numpy as np
import yfinance as yf
from joblib import Parallel, delayed
from scipy.optimize import minimize


### 2) Import dependencies
from config import SECURITIES, DATE_INIT, N_BOOT, L, T, OBJ_, OPT_METHOD
from politis_romano import PolitisRomanoBootstrap
from functions import get_moments
from optimizer import Optimizer


### 3) Download data
SECURITIES = sorted(SECURITIES)
data = yf.download(SECURITIES, start=DATE_INIT, auto_adjust=adj_close)['Close'] 
data_np = data.to_numpy().T
n_sec = len(SECURITIES)


### 4) Data Generation via Politis-Romano Bootstrap
prb = PolitisRomanoBootstrap(serie=data_np, n_boot=N_BOOT, l=L, T=T)
prb_data = prb.generate()


### 5) Optimization
moments_l = Parallel(n_jobs=-1)(
    delayed(get_moments)(set_) for set_ in prb_data
)

obj_ = OBJ_
def _optimize():
    res = minimize(
      obj_, # objetive function
      w0, # initialization
      args=(moments_l), # extra arguments for objective function
      method=method, # optimization method
      bounds= [(0, 1)] * n_sec # no short positions avoided
      constraints= {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}, # sum of weights equals 1.0 (no leverage) 
    )
    return res.x

w0 = np.ones(n_sec) / n_sec  # Initialization (equal weighted portfolio)
w_opt = _optimize()


### 6) Validation
