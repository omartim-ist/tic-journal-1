import numpy as np
from scipy.optimize import minimize

def RC(x, Sigma):
    sigma = np.sqrt(x.T @ Sigma @ x)
    return x*(Sigma @ x)/sigma

def objective(x,Sigma):
    Rc = RC(x,Sigma)
    return sum([(Rc[i] - Rc[j])**2 for i in range(len(Rc)) for j in range(len(Rc))])

def RiskParity(Sigma):    
    x_0 = np.array([1/len(Sigma) for i in range(len(Sigma))])
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = [(0, None) for _ in range(len(Sigma))]
    res = minimize(objective, x_0, args=(Sigma,), constraints=cons, method='SLSQP', bounds=bounds)
    return res
