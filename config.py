# Optimization Universe
SECURITIES = []

# Dates
DATE_INIT =
DATE_END =

# Bootstrap
N_BOOT = # nº of series to generate
L = # average length of blocks
T = # length of each serie

# Optimization
OPT_METHOD = 'SLSQP' # SLSQP
METRIC = 'variance' # variance or sharpe

import numpy as np
RF =  0.02 # offset for Sharpe Ratio calculation; 0.02 equals 2% annualized return
RF = np.log((1 + RF)**(1/252))
