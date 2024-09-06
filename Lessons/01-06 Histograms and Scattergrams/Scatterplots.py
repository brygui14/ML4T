import pandas as pd
import matplotlib.pyplot as plt
from util import get_data, plot_data
from my_utils import compute_daily_returns
import numpy as np

def run():
    dates = pd.date_range(start='2009-01-01', end='2012-12-31')
    symbols = ['SPY', 'XOM', 'GLD']
    df = get_data(symbols, dates)
    plot_data(df)

    daily_returns = compute_daily_returns(df)
    plot_data(daily_returns, title="Daily Returns", ylabel="Daily Returns")

    daily_returns.plot(kind="scatter", x='SPY', y='XOM')
    beta_XOM, alpha_XOM = np.polyfit(daily_returns['SPY'], daily_returns['XOM'], 1)
    plt.plot(daily_returns['SPY'], beta_XOM * daily_returns['SPY'] + alpha_XOM, '-', color='r')

    print(f"Beta XOM: {beta_XOM}")
    print(f"Alpha XOM: {alpha_XOM}")

    plt.show()

    daily_returns.plot(kind="scatter", x="SPY", y="GLD")
    beta_GLD, alpha_GLD = np.polyfit(daily_returns['SPY'], daily_returns['GLD'], 1)
    plt.plot(daily_returns['SPY'], beta_GLD * daily_returns['SPY'] + alpha_GLD, '-', color='r')

    print(f"Beta GLD: {beta_GLD}")
    print(f"Alpha GLD: {alpha_GLD}")

    plt.show()

    corr = daily_returns.corr(method='pearson')
    print(corr)


if __name__ == "__main__":
    run()