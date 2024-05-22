import numpy as np
from sklearn.metrics import mean_squared_error as mse
from scipy.optimize import curve_fit

def poly2_regression(x, a, b, c):
    return a * x + b * x**2 + c

def poly3_regression(x, a, b, c, d):
    return a * x + b * x ** 2 + c * x ** 3 + dx

def fit(data_x, data_y):
    popt, _ = curve_fit(poly2_regression, data_x, data_y)
    a, b, c = popt
    dist_space = np.linspace(min(data_x), max(data_x), 100)
    fit_y = poly2_regression(dist_space, a, b, c)
    error = mse(data_y, poly2_regression(data_x, a, b, c))
    return dist_space, fit_y, error