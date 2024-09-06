import pandas as pd
import matplotlib.pyplot as plt

from util import get_data, plot_data

def compute_daily_returns(df):
    daily_returns = df.copy()

    daily_returns = (df / df.shift(1)) - 1
    daily_returns.iloc[0, :] = 0

    return  daily_returns

def run():
    dates = pd.date_range(start='2009-01-01', end='2012-12-31')
    df = get_data(symbols=['SPY'], dates=dates)
    daily_returns = compute_daily_returns(df)
    plot_data(df)
    plot_data(daily_returns, title="Daily Return", ylabel='Daily Returns')

    daily_returns.hist()
    daily_returns.hist(bins=20)

    mean = daily_returns['SPY'].mean()
    print(f"Mean: {mean}")
    std = daily_returns['SPY'].std()
    print(f"Std: {std}")

    plt.axvline(mean, color='w', linestyle='dashed', linewidth=2)
    plt.axvline(std, color='r', linestyle='dashed', linewidth=2)
    plt.axvline(-std, color='r', linestyle='dashed', linewidth=2)
    plt.show()

    print(f"Kurtosis: {daily_returns.kurtosis()}")


if __name__ == "__main__":
    run()