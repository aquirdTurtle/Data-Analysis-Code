import numpy as np
import uncertainties.unumpy as unp
from . import gaussian_2d


def center():
    return None


def args():
    return gaussian_2d.args()


def f(coordinate, *gaussParams):
    """
    The normal function call for this function. Performs checks on valid arguments, then calls the "raw" function.
    coordinate: an (x,y) pair, usually created using np.meshgrid
    *gaussparams: first the offset, followed by a list of parameter inputs to the gaussian_2d function 
        not including the offset and angle. 
        e.g. [offset,  amp_1, xo_1, yo_1, sigma_x_1, sigma_y_1, amp_2, xo_2, yo_2, sigma_x_2, sigma_y_2, etc. ]
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
        raise ValueError("Error: invlaid number of arguments passed to arb 2d gaussian sum. must be multiple of 5. Number was " + str(len(params)))
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

