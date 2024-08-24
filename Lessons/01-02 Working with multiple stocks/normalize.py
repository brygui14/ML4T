import pandas as pd
import matplotlib.pyplot as plt
import util

def normalize(df):
    return df/df.iloc[0]

def run():
    dates = pd.date_range('2010-01-01', '2010-12-31')

    symbols = ['GOOG', 'IBM', 'GLD']

    df = util.get_data(symbols=symbols, dates=dates)

    df1 = df['2010-01-01':'2010-12-31']
    # print(df1)

    df2 = df[['GOOG','SPY']]

    # print(df2)

    df3 = df['2010-01-01':'2010-01-31'][['SPY','GOOG']]

    # print(df3)
    tmp = normalize(df1)
    util.plot_data(tmp)



if __name__ == '__main__':
    run()