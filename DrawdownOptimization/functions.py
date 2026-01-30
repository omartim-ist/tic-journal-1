import numpy as np

def _min_cum_drawdown(x, lrets):
    p_rets = np.expm1(lrets) @ x
    p_value = np.cumprod(p_rets+1)
    if np.any(p_value <= 0) or np.any(np.isinf(p_value)):
        return 1e12
    
    p_ath = np.maximum.accumulate(p_value)
    
    
    if np.any(np.isinf(p_ath)):
        print(np.max(p_ath))
    dd = 1 - p_value / p_ath
    return np.sum(dd)

def _min_devs(x, lrets, qsi):
    p_value = np.cumprod((np.expm1(lrets) @ x) + 1)
    if np.any(p_value <= 0) or np.any(np.isinf(p_value)):
        return 1e12
    
    w = 2*qsi + 1
    kernel = np.ones(w, dtype=float) / w
    m = np.convolve(p_value, kernel, mode="same")  # média móvel

    valid = slice(qsi, len(p_value) - qsi)
    devs = (1.0 - m[valid] / p_value[valid])**2
    return float(np.sum(devs))

def _MVP(x, Cov):
    return x @ Cov @ x

def _min_cum_sortino(x, lrets, qsi):
    p_value = np.cumprod((np.expm1(lrets) @ x) + 1)
    if np.any(p_value <= 0) or np.any(np.isinf(p_value)):
        return 1e3
    
    p_l_value = np.log(p_value)
    p_lrets = p_l_value[qsi:] - p_l_value[:-qsi]
    return np.sum((np.minimum(0, p_lrets))**2)