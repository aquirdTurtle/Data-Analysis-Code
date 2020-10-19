import numpy as np
import uncertainties.unumpy as unp
from . import sinc_sq

def center():
    return None # or the arg-number of the center.


def getCenter(args):
    # return the average
    return (args[1] + args[4] + args[7])/3


def f(x, offset, *params):
    """
    The normal function call for this function. Performs checks on valid arguments, then calls the "raw" function.
    :return:
    """
#    if offset < 0:
#        return x * 10**10
#    if A < 0:
#        return x * 10**10
    return f_raw(x, offset, *params)


def args():
    # can't use the arb sum like this, args is supposed to not take any arguments so the module can't figure out how many args there were.
    return None


def f_raw(xpts, offset, *params):
    """
    The raw function call, performs no checks on valid parameters..
    :return:
    """
    if len(params) % 3 != 0:
        raise ValueError("Error: invlaid number of arguments passed to arb sinc squared sum. must be multiple of 3 plus the offset.")
    sincParams = np.reshape(params, (int(len(params)/3), 3))
    res = 0
    for p in sincParams:
        res += sinc_sq.f(xpts, *p, 0)
    res += offset
    return res


def f_unc(x,offset, *params):
    if len(params) % 3 != 0:
        raise ValueError("Error: invlaid number of arguments passed to arb 2d gaussian sum. must be multiple of 5.")
    sincParams = np.reshape(params, (int(len(params)/3), 3))
    res = 0
    for p in sincParams:
        res += sinc_sq.f_unc(x, *p, 0)
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
    return None

