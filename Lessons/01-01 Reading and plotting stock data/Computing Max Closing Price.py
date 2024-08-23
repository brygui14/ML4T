import pandas as pd 

def get_max_close(symbol):
    """Return the maximum closing value for stock indicated by symbol"""

    df = pd.read_csv(f"data/{symbol}.csv")
    return df['Close'].max()

def test_run():
    for symbol in ['AAPL', 'IBM']:
        print("Max Close")
        print(symbol, get_max_close(symbol=symbol))

if __name__ == '__main__':
    test_run()