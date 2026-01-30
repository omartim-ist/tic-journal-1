import pickle

ALIAS = {
    # Media / Tech
    "GOOG": "GOOGL",
    "FB": "META",
    "DISCA": "WBD",
    "DISCK": "WBD",
    "SNI": "WBD",
    "TWX": "WBD",
    "VIAB": "VIAC",
    "VIAC": "PARA",

    # Consumer / Retail
    "WFM": "AMZN",         # aquisição (opcional map)
    "KORS": "CPRI",
    "LB": "BBWI",

    # Industrials / Chemicals
    "DD": "DWDP",
    "DWDP": ["DD", "DOW"], # split
    "DOW": "DOW",
    "APC": "COP",          # aquisição
    "BHI": "BHGE",
    "BHGE": "BKR",

    # Healthcare
    "ANTM": "ELV",
    "AGN": "ABBV",
    "ABC": "COR",
    "BCR": "BDX",
    "HCN": "WELL",
    "HCP": "PEAK",
    "STJ": "ABT",
    "ALXN": "AZN",

    # Financials
    "CTL": "LUMN",
    "ETFC": "MS",
    "PBCT": "MTB",
    "BBT": "TFC",
    "STI": "TFC",
    "WCG": "CNC",

    # Tech / Semis
    "XLNX": "AMD",
    "MXIM": "ADI",
    "LLTC": "ADI",
    "SYMC": "GEN",
    "CA": "BROADCOM",      # legado (opcional)

    # Travel / Leisure
    "PCLN": "BKNG",
    "WYNN": "WYN",

    # Misc
    "INFO": "ICE",
    "FISV": "FI",
}


def _apply_aliases(tickers, alias_map):
    """
    - Substitui tickers com base em alias_map.
    - Se alias_map[t] == None: remove o ticker.
    - Se alias_map[t] == list: expande (ex: DWDP -> ['DD','DOW']).
    - Remove duplicados preservando ordem.
    """
    out = []
    for t in tickers:
        if t in alias_map:
            repl = alias_map[t]
            if repl is None:
                continue
            if isinstance(repl, (list, tuple)):
                out.extend(repl)
            else:
                out.append(repl)
        else:
            out.append(t)

    # dedup preservando ordem
    seen = set()
    out_dedup = []
    for t in out:
        if t not in seen:
            seen.add(t)
            out_dedup.append(t)
    return out_dedup

with open("sp500_constituents_by_time.pkl", "rb") as f:
    sp500_comp = pickle.load(f)

sp500_comp_updated = {}
for dt, tickers in sp500_comp.items():
    sp500_comp_updated[dt] = _apply_aliases(tickers, ALIAS)

with open("sp500_constituents_by_time_updated.pkl", "wb") as f:
    pickle.dump(sp500_comp_updated, f)
