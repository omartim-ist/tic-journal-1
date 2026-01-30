### 1) Import packages
import numpy as np
import pandas as pd
import sys
import os
import time
from joblib import Parallel, delayed



### 2) Import dependencies
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from DrawdownOptimization.config import OFFSET, N_SPLITS, TRAIN_BLOCKS_INIT, TEST_BLOCKS, WF_METHOD, L2_NORM, WEIGHT_CUTOFF, SECURITIES
from walk_forward_split import WalkForwardSplit
from DrawdownOptimization.functions import _min_cum_sortino as _OBJ



### 3) Download data
SECURITIES = sorted(SECURITIES)
data = pd.read_csv(os.path.join(ROOT, 'data', 'historical_data.csv')).drop(columns=['Date'])
data = data.reindex(columns=SECURITIES)



### 4) Remove assets without full data
data_ = data.ffill()
data_ = data_.loc[3:,:].reset_index().drop(columns=['index'])
data_ = data_.loc[:, ~data_.iloc[0].isna()]

const_cols = data_.columns[data_.apply(lambda s: np.ptp(s.to_numpy()) <= 1e-12)]
data_ = data_.drop(columns=const_cols)
data_ = data_.drop(data_.columns[330], axis=1)

data_np = data_.to_numpy()



### 5) Walk Forward Splits
WFS = WalkForwardSplit(N_SPLITS, TRAIN_BLOCKS_INIT, TEST_BLOCKS)

if WF_METHOD == 'rolling':
    splits = WFS.rolling(data_np)
elif WF_METHOD == 'expanding':
    splits = WFS.expanding(data_np)
else:
    raise ValueError



### 6) Apply Walk-Forward
from scipy.optimize import minimize, NonlinearConstraint
def _get_portfolio(lrets):
    
    n_assets = lrets.shape[1]
    x0 = np.full(n_assets, 1.0/n_assets)
    
    '''
    window = 10
    lrets_cum = np.full_like(lrets, np.nan)
    for t in range(window - 1, lrets.shape[0]):
        lrets_cum[t] = lrets[t - window + 1 : t + 1].sum(axis=0)
    lrets_cum = lrets_cum[window - 1:]
    Cov = np.cov(lrets_cum, rowvar=False)

    res = minimize(
      _OBJ, # objective function
      x0, # initialization
      args=(lrets, 5,), # extra arguments for objective function
      method='trust-constr', # 'SLSQP' # optimization method
      bounds = [(-0.3, 0.3)] * n_assets,
      constraints = [
          {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
          {'type': 'ineq', 'fun': lambda x: L2_NORM**2 - np.sum(x**2)}
          ],
      options={"maxiter": 200, 'disp': True},
    )
    return res.x
    '''
    
    def cons_sum1(x):
        return np.sum(x)

    def cons_l2sq(x):
        return np.sum(x**2)

    nlc_sum1 = NonlinearConstraint(cons_sum1, 1.0, 1.0)          # == 1
    nlc_l2   = NonlinearConstraint(cons_l2sq, 0.0, L2_NORM**2) # <= L2_NORM^2
    
    res = minimize(
        _OBJ,
        x0,
        args=(lrets, 5),
        method="trust-constr",
        bounds=[(-0.3, 0.3)] * n_assets,
        constraints=[nlc_sum1, nlc_l2],
        options={"maxiter": 200, "disp": True},
    )
    return res.x


def _stats_portfolio(lrets, x):
    p_lrets = np.log(np.exp(lrets) @ x)

    try:
        ann_ret = np.expm1(np.mean(p_lrets) * 365)
        ann_vol = np.std(p_lrets) * np.sqrt(365)   
        return np.array([ann_ret, ann_vol, (ann_ret - OFFSET) / ann_vol]), p_lrets
    except FloatingPointError:
        return None, []
        

def _one_split(split_idx, split_data):
    print(f"[split {split_idx}] start", flush=True)
    
    train_prices, test_prices = split_data

    train_lrets = np.diff(np.log(train_prices), axis=0)
    test_lrets = np.diff(np.log(test_prices), axis=0)

    x = _get_portfolio(train_lrets)

    train_row, _ = _stats_portfolio(train_lrets, x)
    test_row, p_lrets = _stats_portfolio(test_lrets, x)

    p_lrets_list = np.asarray(p_lrets).ravel().tolist()

    print(f"[split {split_idx}] end", flush=True)
    return split_idx, np.asarray(train_row), np.asarray(test_row), p_lrets_list


# ---- paralel ----
n_splits = len(splits)

results = Parallel(n_jobs=-1, prefer="processes")(
    delayed(_one_split)(i, splits[i])
    for i in range(n_splits)
)


# ordenar por índice do split (ordem garantida)
results.sort(key=lambda t: t[0])

wf_train = np.empty((n_splits, 3), dtype=float)
wf_test  = np.empty((n_splits, 3), dtype=float)
cum_p_lrets = []

for i, train_row, test_row, p_lrets_list in results:
    wf_train[i, :] = train_row
    wf_test[i, :]  = test_row
    cum_p_lrets.extend(p_lrets_list)

wf_stats = (wf_train, wf_test)

print(wf_stats[0])
print()
print(wf_stats[1])

import matplotlib.pyplot as plt
plt.plot((np.cumsum(cum_p_lrets)))


