import pandas as pd
from util import plot_data, get_data

def run():
    dates = pd.date_range(start='2005-12-31', end='2014-12-07')
    df = get_data(symbols=['JAVA', 'FAKE1', 'FAKE2'], dates=dates)


    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    plot_data(df)

if __name__ == "__main__":
    run()