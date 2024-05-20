import numpy as np
import yfinance as data

import os , sys
current_dir = os.path.dirname(os.path.abspath(__file__))
path = '/home/tho/Downloads/geo'
path = r'C:\Users\boonh\Downloads\code\geo'
sys.path.append(path)

from fin_bot.portfolio_optimization.markowitz import efficient_frontier , efficient_frontier_detect

if __name__ == '__main__':
    '''
    import data > function > check it.
    https://www.machinelearningplus.com/machine-learning/portfolio-optimization-python-example/
    '''

    risk_free_rate = 0.02
    start='2015-01-01'
    end='2019-12-31'
    epoch = 10000
    ticker_symbol = ['AAPL', 'NKE', 'GOOGL', 'AMZN']
    df = data.download(ticker_symbol, start=start, end=end)
    df = df['Adj Close']

    funds_covariance = df.pct_change().apply(lambda x:np.log(1+x)).cov()
    funds_returns = df.pct_change().apply(lambda x:np.log(1+x)).mean()
    funds_volatility = df.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))

    # funds_weight = {'AAPL': 0.1, 'NKE': 0.2, 'GOOGL': 0.5, 'AMZN': 0.2}
    funds_weight = np.array([0.1,0.2,0.5,0.2])

    funds_returns,funds_volatility,funds_covariance = funds_returns.to_numpy(),funds_volatility.to_numpy(),funds_covariance.to_numpy()

    portf = efficient_frontier(funds_returns,funds_volatility,funds_covariance,funds_weight,epoch)
    returns , volatility = portf.estimate_portfolio()

    portfolio = portf.simulation_portfolio()
    portf.simulation_plot(portfolio['returns'],portfolio['volatility'])

    ef = efficient_frontier_detect(portfolio,risk_free_rate)
    ef.query_sharpe_ratio()