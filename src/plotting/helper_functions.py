import numpy as np


"""This script contains a set of helper functions that are used in the plotting scripts for the pub DOI_FROM_PUB"""


#helper functions
def mean_absolute_percentage_error(y_true, y_pred):
    # calculates the mean absolute percentage error of a set of predictions
    eps = 1e-15 #minimum value to avoid underflow and allow handling of divzero
    y_true, y_pred = np.array(y_true + eps), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def MSE(y_true, y_pred):
    # calculates the mean squared error of a set of predictions
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def calc_coef_of_det(y_true, y_pred):
    # calculates the coefficient of determination of a model
    y_true = np.array(y_true)
    np.array(y_pred)
    ssres = np.sum((y_true - y_pred) ** 2)
    sstot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ssres / sstot)

