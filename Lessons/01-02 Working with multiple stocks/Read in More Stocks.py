from operator import index

import pandas as pd
import matplotlib.pyplot as plt

def run():
    # Define data range
    start_date = '2010-01-22'
    end_date = '2010-01-26';
    dates = pd.date_range(start_date, end_date)

    # Create an empty dataframe
    df1 = pd.DataFrame(index=dates)  # define empty dataframe with these dates as index

    # Read SPY data into temporary dataframe
    dfSPY = pd.read_csv("data/SPY.csv", index_col="Date",
                        parse_dates=True, usecols=['Date', 'Adj Close'],
                        na_values=['nan'])

    dfSPY = dfSPY.rename(columns={'Adj Close':'SPY'})

    # Join the two dataframes using DataFram.join()
    df1 = df1.join(dfSPY, how="inner")

    symbols = ['GOOG', 'IBM', 'GLD']

    for symbol in symbols:
        dftmp = pd.read_csv(f"data/{symbol}.csv", index_col="Date",
                            parse_dates=True, usecols=['Date', 'Adj Close'],
                            na_values=['nan'])

        dftmp = dftmp.rename(columns={'Adj Close':f'{symbol}'})


        df1 = df1.join(dftmp, how="inner")

    print(df1)



if __name__ == "__main__":
    run()