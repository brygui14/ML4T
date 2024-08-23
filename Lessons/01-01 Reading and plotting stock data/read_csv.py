import pandas as pd
import sys
print(sys.version)

def test_data():
    df = pd.read_csv('data/AAPL.csv')
    print(df.head())


if __name__ == '__main__':
    test_data()