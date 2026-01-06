# Optimization Universe
SECURITIES = ['XLE', 'XLF', 'XLRE', 'XLI', 'XLB', 'XLK', 'XLU', 'XLC', 'XLY', 'XLP', 'XLV', 'BTC-USD']

# Dates
DATE_INIT = '2018-07-02'
from datetime import date, timedelta
DATE_END = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

# Bootstrap
N_BOOT = 10_000 # nº of series to generate
L = 365/12 * 4 # average length of blocks
T = 365*5 # length of each serie

