### 1) Import packages
import numpy as np
import yfinance as yf

### 2) Import dependencies
from config import SECURITIES, CONSTRAINTS, DATE_INIT, N_BOOT, L, T
from bootstrap.politis_romano import PolitisRomanoBootstrap

### 3) Download data
SECURITIES = sorted(SECURITIES)
data = yf.download(SECURITIES, start=DATE_INIT, auto_adjust=adj_close)['Close'] 

### 4) Data Generation via Politis-Romano Bootstrap
prb = PolitisRomanoBootstrap(serie=serie, n_boot=N_BOOT, l=L, T=T)
prb_data = prb.generate()

### 5) Otimization


### 6) Validation
