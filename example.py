#%%
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ---------------------------------------------
# 下载数据
tickers = ["AAPL", "GOOG", "MSFT", "AMZN", "ORCL"]
data = yf.download(tickers, period="5y")
data = data['Close']
#%%
# ---------------------------------------------
# 计算收益
returns = data.pct_change().dropna()
mean_returns = returns.mean() * 252
cov_matrix  = returns.cov() * 252
risk_free_rate = 0.02

# ---------------------------------------------
# 投资组合绩效函数
def portfolio_performance(weights, mean_returns, cov_matrix):
    p_ret = np.sum(mean_returns * weights)
    p_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return p_ret, p_std

# 目标函数（-Sharpe）
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_ret, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_std

# 最小方差目标
def min_variance(weights, mean_returns, cov_matrix, risk_free_rate=None):
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]

# 优化函数
def optimize_portfolio(mean_returns, cov_matrix, objective_func):
    num_assets = len(mean_returns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for _ in range(num_assets))
    initial_weights = num_assets * [1./num_assets]
    
    result = minimize(objective_func, initial_weights,
                      args=(mean_returns, cov_matrix, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

#%%
# ---------------------------------------------
# 最大 Sharpe 比率组合
max_sharpe_weights = optimize_portfolio(mean_returns, cov_matrix, negative_sharpe_ratio)
max_sharpe_ret, max_sharpe_std = portfolio_performance(max_sharpe_weights, mean_returns, cov_matrix)
max_sharpe_ratio = (max_sharpe_ret - risk_free_rate) / max_sharpe_std
max_sharpe_ratio
#%%
# 最小波动率组合
min_vol_weights = optimize_portfolio(mean_returns, cov_matrix, min_variance)
min_vol_ret, min_vol_std = portfolio_performance(min_vol_weights, mean_returns, cov_matrix)

#%%
# ---------------------------------------------
# 有效前沿
def efficient_frontier(mean_returns, cov_matrix, num_points=1000):
    frontier = []
    target_returns = np.linspace(min_vol_ret, max_sharpe_ret, num_points)
    for tr in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: portfolio_performance(x, mean_returns, cov_matrix)[0] - tr}
        )
        initial_weights = len(mean_returns) * [1./len(mean_returns)]
        result = minimize(min_variance, initial_weights, args=(mean_returns,cov_matrix),
                          method='SLSQP', bounds=tuple((0,1) for _ in range(len(mean_returns))),
                          constraints=constraints)
        std = portfolio_performance(result.x, mean_returns, cov_matrix)[1]
        frontier.append((std, tr))
    return np.array(frontier)

frontier = efficient_frontier(mean_returns, cov_matrix)

# ---------------------------------------------
# # 绘图
plt.figure()
plt.scatter(frontier[:,0], frontier[:,1], label='Efficient Frontier')
plt.scatter(max_sharpe_std, max_sharpe_ret, marker='*', s=200, label='Max Sharpe')
plt.scatter(min_vol_std,   min_vol_ret,   marker='o', s=100, label='Min Volatility')
plt.show()
#%%
# Monte Carlo 模拟有效前沿（1000个组合）
def simulate_random_portfolios(mean_returns, cov_matrix, num_portfolios=100000):
    results = []
    num_assets = len(mean_returns)
    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        ret, std = portfolio_performance(weights, mean_returns, cov_matrix)
        results.append((std, ret))
    return np.array(results)

frontier = simulate_random_portfolios(mean_returns, cov_matrix, num_portfolios=1000)

# ---------------------------------------------
# 绘图
plt.style.use("ggplot")
plt.figure()
# 有效前沿点
plt.scatter(frontier[:,0], frontier[:,1],marker='x',alpha=0.5 ,label='Random Portfolios')
# 最大 Sharpe、最小波动点
plt.scatter(max_sharpe_std, max_sharpe_ret, marker='*', label='Max Sharpe')
plt.scatter(min_vol_std,   min_vol_ret,   marker='o', label='Min Volatility')

#%%
# CAL
x_vals = np.linspace(0, frontier[:,0].max(), 100)
cal    = risk_free_rate + max_sharpe_ratio * x_vals
plt.plot(x_vals, cal, linestyle='--', label=f'CAL (Sharpe={max_sharpe_ratio:.2f})')

plt.xlabel('Volatility (Std)')
plt.ylabel('Expected Return')
plt.legend()
plt.title('Efficient Frontier / Max Sharpe / Min Volatility / CAL')
plt.show()

#%%
# ---------------------------------------------
# 输出权重
print("=== Max Sharpe Weights ===")
print(pd.Series(max_sharpe_weights, index=mean_returns.index))
print("\n=== Min Volatility Weights ===")
print(pd.Series(min_vol_weights, index=mean_returns.index))