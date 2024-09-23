"""Minimize an objective function using SciPy: 3D"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo


def error_poly(C, data):  # error function
    """Compute error between given polynomial and observed data.

    Parameters
    ----------
    C: numpy.poly1d object or equivalent array representing polynomial coefficients
    data: 2D array where each row is a point (x, y)

    Returns error as a single real value.
    """
    # Metric: Sum of squared Y-axis differences
    err = np.sum((data[:, 1] - np.polyval(C, data[:, 0])) ** 2)
    return err


def fit_poly(data, error_func, degree=3):
    """Fit a polynomial to given data, using a supplied error function.

    Parameters
    ----------
    data: 2D array where each row is a point (X0, Y)
    error_func: function that computes the error between a polynomial and observed data

    Returns polynomial that minimizes the error function.
    """
    # Generate initial guess for line model (all coeffs = 1)
    Cguess = np.poly1d(np.ones(degree + 1, dtype=np.float32))
    print(Cguess)

    # Plot initial guess (optional)
    x = np.linspace(-5, 5, 2001)
    plt.plot(x, np.polyval(Cguess, x), 'm--', linewidth=2.0, label="Initial guess")

    # Call optimizer to minimize error function
    result = spo.minimize(error_func, Cguess, args=(data,), method='SLSQP', options={'disp': True})
    return np.poly1d(result.x)  # convert optimal result into a poly1d object and return


def run():
    fig = plt.figure()
    ax = fig.gca()


# Define original line
    p_orig = np.float32([3,2,1,0])
    print("Original polynomial: C0 = {}, C1 = {}, C2 = {}, C3 = {}".format(p_orig[0], p_orig[1], p_orig[2], p_orig[3]))
    Xorig = np.linspace(-20,30,2001)
    Yorig = (p_orig[0] * pow(Xorig, 3)) + (p_orig[1] * pow(Xorig, 2)) + (p_orig[2] * Xorig) + p_orig[3]
    plt.plot(Xorig, Yorig, 'b--', linewidth=2.0, label="Original line")

    # Generate noisy data points


    noise_sigma = 1000.0
    noise = np.random.normal(0, noise_sigma, Yorig.shape)
    data = np.asarray([Xorig, Yorig + noise]).T
    plt.plot(data[:, 0], data[:, 1], 'go', label="Data points")


# Try to fit a line to this data
    p_fit = fit_poly(data, error_poly)
    print("Fitted line: C0 = {}, C1 = {}, C2 = {}, C3 = {}".format(p_fit[0], p_fit[1], p_fit[2], p_fit[3]))

    values = (p_fit[0] * pow(Xorig, 3)) + (p_fit[1] * pow(Xorig, 2)) + (p_fit[2] * Xorig) + p_fit[3]
    plt.plot(Xorig, values , 'r--', linewidth=2.0, label="Fitted Line")
# Add a legend and show plot
    print(values)
    ax.set_xlim([-20,30])
    ax.set_ylim([-25000, 90000])
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    run()