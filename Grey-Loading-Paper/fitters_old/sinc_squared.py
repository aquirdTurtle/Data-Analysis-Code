import numpy as np
import uncertainties.unumpy as unp


def center():
    return 2 # or the arg-number of the center.


def f(x, A, c, scale, offset):
    """
    The normal function call for this function. Performs checks on valid arguments, then calls the "raw" function.
    :return:
    """
    if offset < 0:
        return x * 10**10
    if A < 0:
        return x * 10**10
    return f_raw(x, A, c, scale, offset)


def f_raw(x, A, c, scale, offset):
    """
    The raw function call, performs no checks on valid parameters..
    :return:
    """
    return A * np.sinc((x - c)/scale)**2 + offset


def f_unc(x, A, c, scale, offset):
    """
    similar to the raw function call, but uses unp instead of np for uncertainties calculations.
    :return:
    """
    return A * unp.sinc((x - c)/scale)**2 + offset


def guess(key, values):
    """
    Returns guess values for the parameters of this function class based on the input. Used for fitting using this
    class.
    :param key:
    :param values:
    :return:
    """

