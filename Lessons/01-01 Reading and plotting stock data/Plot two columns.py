import pandas as pd
import matplotlib.pyplot as plt

def run():
    df = pd.read_csv("data/AAPL.csv")
    df[['Close', 'Adj Close']].plot()
    plt.show()

if __name__ == "__main__":
    run()