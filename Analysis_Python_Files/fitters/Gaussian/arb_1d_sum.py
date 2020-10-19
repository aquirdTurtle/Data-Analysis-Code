import numpy as np
import uncertainties.unumpy as unp
from . import gaussian


def center():
    return None


def args():
    return gaussian_2d.args()


def f(xpts, *gaussParams):
    """
    The normal function call for this function. Performs checks on valid arguments, then calls the "raw" function.
    :return:
    """
    # limit the angle to a small range to prevent unncecessary flips of the axes. The 2D gaussian has two axes of
    # symmetry, so only a quarter of the 2pi is needed.
    return f_raw(xpts, *gaussParams).ravel()


def f_raw(xpts, offset, *params):
    """
    The raw function call, performs no checks on valid parameters..
    :return:
    """
    if len(params) % 3 != 0:
        print(len(params), params)
        raise ValueError("Error: invlaid number of arguments passed to arb 2d gaussian sum. must be multiple of 5.")
    gaussParams = np.reshape(params, (int(len(params)/3), 3))
    res = 0
    for p in gaussParams:
        res += gaussian.f(xpts, *p, 0)
    res += offset
    return res


def f_unc(xpts, offset, *params):
    """
    similar to the raw function call, but uses unp instead of np for uncertainties calculations.
    :return:
    """
    if len(params) % 3 != 0:
        raise ValueError("Error: invlaid number of arguments passed to arb 1d gaussian sum! Number was " + str(len(params)) + ", but must be multiple of 3.")
    gaussParams = np.reshape(params, (int(len(params)/3), 3))
    res = 0
    for p in gaussParams:
        res += gaussian.f_unc(xpts, *p, 0)
    res += offset
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

