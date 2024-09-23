from my_utils import daily_portfolio_value, cumulative_returns, compute_daily_returns, compute_daily_returns_portfolio
import pandas as pd
import matplotlib.pyplot as plt
from util import plot_data

def run():
    dates = pd.date_range(start='2009-01-01', end='2011-12-31')
    symbols = ['SPY', 'XOM', 'GOOG', 'GLD']
    allocs = [0.4, 0.4, 0.1, 0.1]

    value = daily_portfolio_value(symbols=symbols, date_range=dates, start_val=1000000, allocs=allocs)

    daily_returns = compute_daily_returns_portfolio( value)

    cum = cumulative_returns(value)

    print(cum)
    plot_data(value)




if __name__ == '__main__':
    run()
