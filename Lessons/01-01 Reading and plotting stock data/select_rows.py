import pandas as pd
import sys
print(sys.version)

def test_data():
    df = pd.read_csv('data/AAPL.csv')
    print(df[10:21])


if __name__ == '__main__':
    test_data()