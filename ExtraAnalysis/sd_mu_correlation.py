import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from ClusteringPO.main import data_np

lrets = np.diff(np.log(data_np), axis=1)

mu = np.mean(lrets, axis=1)
sigma = np.std(lrets, axis=1)

Mu = np.expm1(mu * 365)
Sigma = sigma * np.sqrt(365)

plt.scatter(Mu / Sigma**2, Mu / Sigma, s=3)


plt.scatter(sigma, mu, s=3)
np.corrcoef(Sigma, Mu)




plt.hist(Mu**(1/5), bins=30)
plt.hist(Sigma**(1/5), bins=30)

X = Mu**(1/5)
Y = Sigma**(1/5)

plt.scatter(X, Y, s=3)
np.corrcoef(X, Y)

model = LinearRegression(fit_intercept=True)
model.fit(X.reshape(-1, 1), Y)

x_line = np.linspace(X.min(), X.max(), 100)
y_line = (model.intercept_ + x_line * model.coef_[0])

plt.figure()
plt.scatter(X**5 * 100, Y**5 * 100, s=3, color='blue')
plt.plot(x_line**5 * 100, y_line**5 * 100, color='red')
plt.xlabel('Annualized Return %')
plt.ylabel('Annualized Volatility %')
plt.title('Return vs Volatility Correlation')
plt.show()

np.std(Y**5 - model.predict(X.reshape(-1,1))**5)
np.std(Y**5)



