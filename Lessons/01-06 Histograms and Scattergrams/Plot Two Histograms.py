import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import label

from util import get_data, plot_data

def compute_daily_returns(df):
    daily_returns = df.copy()

    daily_returns = (df / df.shift(1)) - 1
    daily_returns.iloc[0,:] = 0

    return daily_returns

def run():
    dates = pd.date_range(start='2009-01-01', end='2012-12-31')
    df = get_data(symbols=['SPY', 'XOM'], dates=dates)

    plot_data(df)

    daily_returns = compute_daily_returns(df)

    daily_returns['SPY'].hist(bins=20, label='SPY')
    daily_returns['XOM'].hist(bins=20, label='SPY')
    plt.legend(loc='upper right')
    plt.show()



if __name__ == "__main__":
    run()