import numpy as np
import uncertainties.unumpy as unp


def center():
    return None  # or the arg-number of the center.


def f(x, k, weight):
    """
    The normal function call for this function. Performs checks on valid arguments, then calls the "raw" function.
    This function calculates p_k{x} = weight * e^(-k) * k^x / x!.
    :param x: argument of the Poisson distribution
    :param k: order or (approximate) mean of the Poisson distribution.
    :param weight: a weight factor, related to the maximum data this is supposed to be fitted to, but typically over-
    weighted for the purposes of this function.
    :return: the Poisson distribution evaluated at x given the parameters.
    """
    return f_raw(x, k, weight)


def f_raw(x, k, weight):
    """
    The raw function call, performs no checks on valid parameters..
    This function calculates p_k{x} = weight * e^(-k) * k^x / x!.
    :param x: argument of the Poisson distribution
    :param k: order or (approximate) mean of the Poisson distribution.
    :param weight: a weight factor, related to the maximum data this is supposed to be fitted to, but typically over-
    weighted for the purposes of this function.
    :return: the Poisson distribution evaluated at x given the parameters.
    """
    term = 1
    if x == 0:
        return np.exp(-k)
    # calculate the term k^x / x!. Can't do this directly, x! is too large.
    for n in range(0, int(x)):
        term *= k / (x - n) * np.exp(-k/int(x))
    return term * weight


def f_unc(x, k, weight):
    """
    similar to the raw function call, but uses unp instead of np for uncertainties calculations.
    :return:
    """
    term = 1
    # calculate the term k^x / x!. Can't do this directly, x! is too large.
    for n in range(0, int(x)):
        term *= k / (x - n) * unp.exp(-k/int(x))
    return term * weight


def guess(key, values):
    """
    Returns guess values for the parameters of this function class based on the input. Used for fitting using this
    class.
    :param key:
    :param values:
    :return:
    """

