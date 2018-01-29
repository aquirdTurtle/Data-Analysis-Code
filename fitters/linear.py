import numpy as np


def center():
    return None


def f(x, a, b):
    """
    The normal function call for this function. Performs checks on valid arguments, then calls the "raw" function.
    :return:
    """
    return f_raw(x, a, b)


def args():
    return 'Slope', 'Offset'


def f_raw(x, a, b):
    """
    The raw function call, performs no checks on valid parameters..
    :return:
    """
    return a * x + b


def f_unc(x, a, b):
    """
    similar to the raw function call, but uses unp instead of np for uncertainties calculations.
    :return:
    """
    return f_raw(x, a, b)


def guess(key, values):
    """
    Returns guess values for the parameters of this function class based on the input. Used for fitting using this
    class.
    :param key:
    :param values:
    :return:
    """
    return [(max(values) - min(values))/(key[np.argmax(values)] - key[np.argmin(values)]), min(values)]
