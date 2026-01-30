import pandas as pd
from datetime import datetime, timedelta, date

def _get_sp500_composition(date):
    
    while True:
        url = (
        "https://raw.githubusercontent.com/"
        "riazarbi/sp500-scraper/main/"
        f"ishares/sp500/csv/{date}.csv"
    )
        try:
            tickers = pd.read_csv(url, usecols=[0]).iloc[:, 0].tolist()
            print(date)
            return tickers
        except Exception:
            date = datetime.strptime(date, "%Y%m%d")
            prev = date - timedelta(days=1)
            date = prev.strftime("%Y%m%d")


def _get_quarterly_dates(year, end_date="20260101"):

    months = ("01", "04", "07", "10")
    dates = {}

    for y in range(year, int(end_date[:4]) + 1):
        for m in months:
            d = f"{y}{m}01"
            if d <= end_date:
                dates[d] = _get_sp500_composition(d)

    return dates

data = _get_quarterly_dates(2015)
union = sorted(list(set().union(*data.values())))



date_end = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
commodities = ["USO", "BNO", "UNG", "GLD", "SLV", "PPLT", "PALL", "DBA", "DBC", "GSG"]
crypto = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'BNB-USD', 'SOL-USD', 'TRX-USD', 'DOGE-USD', 'ADA-USD']


import yfinance as yf
data_stock = yf.download(union, start='2015-01-01', end=date_end, auto_adjust=True)['Close']
spy = yf.download('SPY', start='2015-01-01', end=date_end, auto_adjust=True)['Close']
data_commodities = yf.download(commodities, start='2015-01-01', end=date_end, auto_adjust=True)['Close']
data_crypto = yf.download(crypto, start='2015-01-01', end=date_end, auto_adjust=True)['Close']


data_stock.to_csv('sp500_constituents_historical_data.csv')
spy.to_csv('sp500_historical_data.csv')
data_commodities.to_csv('commodities_historical_data.csv')
data_crypto.to_csv('crypto_historical_data.csv')





data = {
    pd.to_datetime(str(k), format="%Y%m%d"): v
    for k, v in data.items()
}
import pickle
with open("sp500_constituents_by_time.pkl", "wb") as f:
    pickle.dump(data, f)

