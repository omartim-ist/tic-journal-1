### 1) Import packages
import numpy as np
import yfinance as yf

### 2) Import dependencies
from config import SECURITIES, CONSTRAINTS, DATE_INIT, N_BOOT, L, T
from politis_romano import PolitisRomanoBootstrap
from optimizer import Optimizer

### 3) Download data
SECURITIES = sorted(SECURITIES)
data = yf.download(SECURITIES, start=DATE_INIT, auto_adjust=adj_close)['Close'] 
data_np = data.to_numpy().T

### 4) Data Generation via Politis-Romano Bootstrap
prb = PolitisRomanoBootstrap(serie=data_np, n_boot=N_BOOT, l=L, T=T)
prb_data = prb.generate()

### 5) Otimization


### 6) Validation
