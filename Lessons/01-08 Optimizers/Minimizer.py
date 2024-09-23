import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo

def f(X):
    Y = (X - 1.5)**2 + 0.5
    print(f"X: {X}\nY: {Y}")
    return Y

def run():
    Xguess = 2.0
    min_result = spo.minimize(f, Xguess, method="SLSQP", options={'disp':True})
    print(f"X = {min_result.x}, Y = {min_result.fun}")

    XPlot = np.linspace(0.5,2.5,21)
    YPlot = f(XPlot)
    plt.plot(XPlot, YPlot)
    plt.plot(min_result.x,min_result.fun, 'ro')
    plt.show()

if __name__ == "__main__":
    run()