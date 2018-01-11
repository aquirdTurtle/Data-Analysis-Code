import numpy as np


def center():
    return 3


def f(x, a, b, x0):
    """
    The normal function call for this function. Performs checks on valid arguments, then calls the "raw" function.
    :return: a + b*(x-x0)**2 , b < 0
    """
    if a < 0:
        return 10 ** 10 * np.ones(len(x))
    if b < 0:
        return 10 ** 10 * np.ones(len(x))
    return f_raw(x, a, b, x0)


def f_raw(x, a, b, x0):
    """
    This assumes downward facing. Best to write another function for upward facing if need be, I think.
    :return: a + b*(x-x0)**2 , b < 0
    """
    return a + b * (x - x0) ** 2


def f_unc(x, a, b, x0):
    """
    similar to the raw function call, but uses unp instead of np for uncertainties calculations.
    :return: a + b*(x-x0)**2 , b < 0
    """
    return f_raw(x, a, b, x0)


def guess(key, values):
    """
    Returns guess values for the parameters of this function class based on the input. Used for fitting using this
    class.
    :param key:
    :param values:
    :return:
    """