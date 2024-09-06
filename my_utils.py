import pandas as np


def compute_daily_returns(df):
    daily_returns = df.copy()

    daily_returns = (df / df.shift(1)) - 1
    daily_returns.iloc[0, :] = 0

    return daily_returns
