import numpy as np

"""This script contains a set of helper functions that are used
 in the plotting scripts for the pub DOI_FROM_PUB"""


# helper functions
def mean_absolute_percentage_error(y_true, y_pred):
    # calculates the mean absolute percentage error of a set of predictions
    eps = 1e-15  # minimum value to avoid underflow and allow handling of divzero
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


def calculate_fpr_threshold(fpr, thresholds):
    """uses interpolation to calculate the thershold required to
    achieve a 1% false positive rate. Expects a vector of false
    positive rates and thresholds"""
    upper_index = list(fpr).index(min([x for x in fpr if x > 0.01])) + 1
    lower_index = list(fpr).index(max([x for x in fpr if x <= 0.01]))
    x_0 = fpr[lower_index]
    x_1 = fpr[upper_index]
    y_0 = thresholds[lower_index]
    y_1 = thresholds[upper_index]
    out_fpr = y_0 + (0.01 - x_0) * (y_1 - y_0) / (x_1 - x_0)
    return out_fpr
