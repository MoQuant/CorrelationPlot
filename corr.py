import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def nonlin(x, y):
    X = np.array([[1, i] for i in x])
    Y = np.array(y)
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
    xx = list(sorted(x))
    yy = [beta[0] + beta[1]*i for i in xx]
    return xx, yy

def fx(x, y, rho):
    return np.exp(-(x**2 - 2*x*y*rho + y**2)/(2.0*(1 - rho**2)))/(2.0*np.pi*np.sqrt(1 - rho**2))

def JGrid(rho):
    a = np.arange(-3.5, 3.5 + 0.1, 0.1)
    x, y = np.meshgrid(a, a)
    z = fx(x, y, rho)
    return x, y, z

def dist(x, bins=50):
    x0, x1 = np.min(x), np.max(x)
    dx = (x1 - x0) / bins
    ux, uy = [], []
    for i in range(bins):
        v0 = x0 + i*dx
        v1 = x0 + (i+1)*dx
        count = 0
        for ror in x:
            if ror >= v0 and ror < v1:
                count += 1
        ux.append((v0 + v1)/2.0)
        uy.append(count)
    return ux, uy

def correlation(x, y):
    cov = np.cov(x, y)
    sd = np.sqrt(np.diag(cov))
    sd = np.array([[i] for i in sd])
    corr = cov/(sd.dot(sd.T))
    return corr[0][1]

tickers = ['SPY','AMZN','MSFT','NVDA','TSLA']

close = np.array([pd.read_csv(f'{tick}.csv')[::-1]['adjClose'].values.tolist() for tick in tickers]).T

ror = close[1:]/close[:-1] - 1.0

fig = plt.figure(figsize=(10, 7))
N = len(tickers)

ax = []
k = 1
for i in range(N):
    tax = []
    for j in range(N):
        tax.append(fig.add_subplot(N, N, k))
        k += 1
    ax.append(tax)

for t in range(N):
    rate = ror[:, t]
    hx, hy = dist(rate)
    ax[t][t].bar(hx, hy, 0.01, 0.01, color='blue')
    ax[0][t].set_title(tickers[t])
    ax[t][0].set_ylabel(tickers[t])
    for u in range(N):
        if t <= u and t != u:
            rho = correlation(ror[:, t], ror[:, u])
            bx, by, bz = JGrid(rho)
            ax[t][u].contourf(bx, by, bz, cmap='hsv')
        if t > u and t != u:
            ax[t][u].scatter(ror[:, t], ror[:, u], color='red', s=8)
            lx, ly = nonlin(ror[:, t], ror[:, u])
            ax[t][u].plot(lx, ly, color='black', linewidth=0.8)

plt.show()

