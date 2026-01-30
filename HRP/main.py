''' 1) Import packages '''
import numpy as np
import pandas as pd
import sys
import os
import time
import matplotlib.pyplot as plt
from pathlib import Path
import pickle



''' 2) Import dependencies '''
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from HRP.config import DATE_INIT, OFFSET, N_SPLITS, TRAIN_BLOCKS_INIT, TEST_BLOCKS, WF_METHOD, SECURITIES, N_CLUSTERS, N_BOOT, L, T, crypto, commodities
from walk_forward_split import WalkForwardSplit
from HRP.functions import portfolio_metrics, _exe_backtest, get_sp500, _get_weights
from politis_romano import PolitisRomanoBootstrap
from HRP.validation import Validation



''' 3) Download data '''
data = pd.DataFrame(columns=['Date'])
for category in SECURITIES:
    df = pd.read_csv(os.path.join(ROOT, 'data', f'{category}_historical_data.csv'))
    df['Date'] = pd.to_datetime(df["Date"], errors="coerce")
    data = (data.merge(df, on="Date", how="outer").sort_values("Date"))
data = data.set_index('Date')



''' 4) Walk Forward Splits '''
pkl_path = Path(__file__).resolve().parent.parent / "data" / "sp500_constituents_by_time.pkl"
with open(pkl_path, "rb") as f:
    sp500_comp = pickle.load(f)
    
assets = {}
for key, value in sp500_comp.items():
    assets[key] = value + crypto + commodities

WFS = WalkForwardSplit(N_SPLITS, TRAIN_BLOCKS_INIT, TEST_BLOCKS, DATE_INIT, assets)

if WF_METHOD == 'rolling':
    splits = WFS.rolling(data)
elif WF_METHOD == 'expanding':
    splits = WFS.expanding(data)
else:
    raise ValueError
    
    
    
''' 5) Walk Forward backtesting '''
log_p_hrp = _exe_backtest('HRP', splits, N_CLUSTERS)
print(portfolio_metrics(log_p_hrp))
breakpoint()

log_p_rp = _exe_backtest('RP', splits)
log_p_ewp = _exe_backtest('EWP', splits)



''' 6) Out of Sample Evaluation & Benchmark Comparison '''

days = pd.DatetimeIndex([])
for _, df in splits:
    days = days.union(df.index)
days = days.sort_values()

df_eval = pd.DataFrame(index=days[1:])
df_eval = (df_eval
           .assign(
               HRP = log_p_hrp,
               RP = log_p_rp,
               EWP = log_p_ewp,
               SP500 = get_sp500(df_eval.index, ROOT))
           )

metrics = {}
plt.figure(figsize=(20, 12))
for col in df_eval.columns:
    plt.plot((df_eval[col]), label=col, linewidth=1.3, alpha=0.6)
    metrics[col] = portfolio_metrics(df_eval[col].to_numpy())
plt.legend()
plt.show()




''' 7) Validation via Politis-Romano '''
prb = PolitisRomanoBootstrap(serie=np.exp(log_p_hrp), n_boot=N_BOOT, l=L, T=T)

ti = time.time()
validation = Validation(prb, OFFSET)
print('total time:', time.time() - ti)

validation._plot_statistics()
validation._plot_paths()


''' 8) Get current weights '''
last_data = (pd.concat([splits[-1][0], splits[-1][1]], axis=0).sort_index())
weights = _get_weights(last_data, N_CLUSTERS)
weights.to_csv("asset_weights.csv")
