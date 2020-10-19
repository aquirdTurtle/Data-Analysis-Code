import numpy as np
import uncertainties.unumpy as unp
from fitters.Gaussian import gaussian


def center():
    return None


def args():
    return ['Polynomial Coefficients']


def f(xpts, *coefficients):
    """
    The normal function call for this function. Performs checks on valid arguments, then calls the "raw" function.
    :return:
    """
    # limit the angle to a small range to prevent unncecessary flips of the axes. The 2D gaussian has two axes of
    # symmetry, so only a quarter of the 2pi is needed.
    return f_raw(xpts, *coefficients).ravel()


def f_raw(xpts, *coefficients):
    """
    The raw function call, performs no checks on valid parameters..
    :return:
    """
    res = 0.0
    for i, p in enumerate(coefficients):
        res += p*xpts**i
    return res


def f_unc(xpts, offset, *params):
    """
    similar to the raw function call, but uses unp instead of np for uncertainties calculations.
    :return:
    """
    res = 0
    for i, p in enumerate(coefficients):
        res += p*xpts**i
    return res


def guess(key, values):
    """
    Returns guess values for the parameters of this function class based on the input. Used for fitting using this
    class.
    :param key:
    :param values:
    :return:
    """
    # need to know a number of gaussians in order to give a sensible guess. 
    return None

