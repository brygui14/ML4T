import pandas as np
from util import get_data


def compute_daily_returns(df):
    """Get Daily Returns of a Stock Multi-Dimension Array"""
    daily_returns = df.copy()

    daily_returns = (df / df.shift(1)) - 1
    daily_returns.iloc[0, :] = 0

    return daily_returns

def compute_daily_returns_portfolio(df):
    """Get Daily Returns of a Stock 1D Array"""
    daily_returns = df.copy()

    daily_returns = (df / df.shift(1)) - 1
    daily_returns.iloc[0] = 0

    return daily_returns

def normalize(df):
    """Normalize stock price from first day"""
    return df/df.iloc[0]

def daily_portfolio_value(start_val, date_range, symbols, allocs):
    """Returns the portfolio value of each day for a distrubution of stocks with a overall value"""
    df = get_data(symbols=symbols, dates=date_range)

    norm = normalize(df)

    alloced = norm * allocs

    position_values = alloced * start_val

    portfolio_values = position_values.sum(axis=1)

    return portfolio_values

def cumulative_returns(df):
    return (df[-1] / df[0]) - 1

def avg_daily_returns(df):
    return df.mean()

def std_daily_returns(df):
    return df.std()

def sharpe_ratio():
    """Compute the risk adjusted return"""


