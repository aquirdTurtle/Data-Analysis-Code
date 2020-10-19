import numpy as np
import uncertainties.unumpy as unp


def center():
    return None  # or the arg-number of the center.


def f(x, A1, x01, sig1, A2, x02, sig2, A3, x03, sig3, offset):
    """
    The normal function call for this function. Performs checks on valid arguments, then calls the "raw" function.
    :return:
    """
    if A1 < 0 or A2 < 0 or A3 < 0:
        # Penalize negative fits.
        return 10**10
    if offset < 0:
        return 10**10
    return f_raw(x, A1, x01, sig1, A2, x02, sig2, A3, x03, sig3, offset)


def f_raw(x, A1, x01, sig1, A2, x02, sig2, A3, x03, sig3, offset):
    """
    The raw function call, performs no checks on valid parameters..
    :return:
    """
    return (offset + A1 * np.exp(-(x-x01)**2/(2*sig1**2)) + A2 * np.exp(-(x-x02)**2/(2*sig2**2))
            + A3 * np.exp(-(x-x03)**2/(2*sig3**2)))


def f_unc(x, A1, x01, sig1, A2, x02, sig2, A3, x03, sig3, offset):
    """
    similar to the raw function call, but uses unp instead of np for uncertainties calculations.
    :return:
    """
    return (offset  + A1 * unp.exp(-(x-x01)**2/(2*sig1**2))
                    + A2 * unp.exp(-(x-x02)**2/(2*sig2**2))
                    + A3 * unp.exp(-(x-x03)**2/(2*sig3**2)))


def guess(key, values):
    """
    Returns guess values for the parameters of this function class based on the input. Used for fitting using this
    class.
    :param key:
    :param values:
    :return:
    """
