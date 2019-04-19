import numpy as np
import uncertainties.unumpy as unp
from fitters.Gaussian import gaussian_2d


def center():
    return None


def args():
    return gaussian_2d.args()


def f(coordinate, *gaussParams):
    """
    The normal function call for this function. Performs checks on valid arguments, then calls the "raw" function.
    :return:
    """
    # limit the angle to a small range to prevent unncecessary flips of the axes. The 2D gaussian has two axes of
    # symmetry, so only a quarter of the 2pi is needed.
    return f_raw(coordinate, *gaussParams).ravel()


def f_raw(coordinate, offset, *params):
    """
    The raw function call, performs no checks on valid parameters..
    :return:
    """
    if len(params) % 5 != 0:
        raise ValueError("Error: invlaid number of arguments passed to arb 2d gaussian sum. must be multiple of 5.")
    gaussParams = np.reshape(params, (int(len(params)/5), 5))
    res = 0
    #for p in gaussParams:
    #    if p[-1] > 1.6 or p[-2] > 1.6:
    #        res += 1e6
    for p in gaussParams:
        res += gaussian_2d.f_noravel(coordinate, *p, 0, 0)
    res += offset
    return res


def f_raw2(coordinate, packedParams):
    """
    The raw function call, performs no checks on valid parameters..
    :return:
    """
    gaussParams = packedParams['pack']
    res = 0
    for p in gaussParams:
        res += gaussian_2d.f_noravel(coordinate, *p)
    return res


def f_unc(coordinate, gaussParams):
    """
    similar to the raw function call, but uses unp instead of np for uncertainties calculations.
    :return:
    """
    pass


def guess(key, values):
    """
    Returns guess values for the parameters of this function class based on the input. Used for fitting using this
    class.
    :param key:
    :param values:
    :return:
    """

