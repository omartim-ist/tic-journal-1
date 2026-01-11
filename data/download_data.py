### 1) Import packages
import sys
import os
import pandas as pd
import yfinance as yf


### 2) Import dependencies
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from config import SECURITIES, DATE_INIT, DATE_END


### 3) Download Data
data = yf.download(SECURITIES, start=DATE_INIT, end=DATE_END, auto_adjust=True)['Close']
'''
data = pd.read_csv('historical_data.csv')
data = (data
        .assign(Date = lambda x: pd.to_datetime(x['Date']))
        .set_index('Date')
        )
'''

SUFFIX_TO_CURRENCY = {
    ".L":  "GBPUSD=X",
    ".DE": "EURUSD=X",
    ".F":  "EURUSD=X",
    ".PA": "EURUSD=X",
    ".MI": "EURUSD=X",
    ".MC": "EURUSD=X",
    ".AS": "EURUSD=X",
    ".ST": "SEKUSD=X",
    ".CO": "DKKUSD=X",
    ".HE": "EURUSD=X",
    ".OL": "NOKUSD=X",
    ".SW": "CHFUSD=X",
    ".T":  "JPYUSD=X",
    ".HK": "HKDUSD=X",
    ".TO": "CADUSD=X",
    ".AX": "AUDUSD=X",
    ".SI": "SGDUSD=X",
}

def infer_currencies(tickers):
    out = {}
    for tk in tickers:
        if "." not in tk:
            out[tk] = "USD"
        else:
            suffix = "." + tk.split(".")[-1]
            out[tk] = SUFFIX_TO_CURRENCY.get(suffix, "USD")
    return out


currencies_map = infer_currencies(SECURITIES)
currencies = list(set(list(currencies_map.values())))
currencies.remove('USD')

data_currencies = yf.download(currencies, start=DATE_INIT, end=DATE_END, auto_adjust=True)['Close'] 


df = data.join(data_currencies, how="outer")


for asset in data.columns:
    cur = currencies_map[asset]
    if cur != 'USD': data[asset] = data[asset] * df[cur]

data.to_csv('historical_data.csv')

