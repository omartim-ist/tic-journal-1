# Optimization Universe
SECURITIES = ['XLE', 'XLF', 'XLRE', 'XLI', 'XLB', 'XLK', 'XLU', 'XLC', 'XLY', 'XLP', 'XLV']

# Dates
DATE_INIT = '2018-07-01'
from datetime import date, timedelta
DATE_END = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

# Bootstrap
N_BOOT = 1_000 # nº of series to generate
L = 252/12 * 4 # average length of blocks
T = 252*5 # length of each serie
