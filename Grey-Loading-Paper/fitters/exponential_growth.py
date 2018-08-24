import numpy as np
import uncertainties.unumpy as unp


def center():
    return None


def f(x, A, tau):
    """
    The normal function call for this function. Performs checks on valid arguments, then calls the "raw" function.
    :return:
    """
    # decay!
    if A < 0:
        return np.ones(len(x)) * 1e10
    return f_raw(x, A, tau)


def f_raw(x, A, tau):
    """
    The raw function call, performs no checks on valid parameters..
    :return:
    """
    return A * np.exp(x/tau)


def f_unc(x, A, tau):
    """
    similar to the raw function call, but uses unp instead of np for uncertainties calculations.
    :return:
    """
    return A * unp.exp(x/tau)


def guess(key, values):
    """
    Returns guess values for the parameters of this function class based on the input. Used for fitting using this
    class.
    :param key:
    :param values:
    :return:
    """