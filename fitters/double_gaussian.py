import numpy as np
import uncertainties.unumpy as unp


def center():
    return None  # or the arg-number of the center.


def f(x, A1, x01, sig1, A2, x02, sig2, offset):
    """
    The normal function call for this function. Performs checks on valid arguments, then calls the "raw" function.
    :return:
    """
    penalty = 10**10 * np.ones(len(x))
    if A1 < 0 or A2 < 0:
        # Penalize negative fits.
        return penalty
    if offset < 0:
        return penalty
    if not (min(x) < x01 < max(x) and min(x) < x02 < max(x)):
        return penalty
    # assume that there's at least a little peak
    if A1 < 3 or A2 < 3:
        return penalty
    # sometimes if low signal or lossy, the second peak can be very spread out. 
    # The fitting of the second gaussian then sometimes assumes it's even broader than it is to make it an effective offset.
    r = max(x) - min(x)
    if sig1 > r/5 or sig2 > r/5:
        return penalty
    return f_raw(x, A1, x01, sig1, A2, x02, sig2, offset)


def f_raw(x, A1, x01, sig1, A2, x02, sig2, offset):
    """
    The raw function call, performs no checks on valid parameters..
    :return:
    """
    return offset + A1 * np.exp(-(x-x01)**2/(2*sig1**2)) + A2 * np.exp(-(x-x02)**2/(2*sig2**2))


def f_unc(x, A1, x01, sig1, A2, x02, sig2, offset):
    """
    similar to the raw function call, but uses unp instead of np for uncertainties calculations.
    :return:
    """
    return offset + A1 * unp.exp(-(x-x01)**2/(2*sig1**2)) + A2 * unp.exp(-(x-x02)**2/(2*sig2**2))


def guess(key, values):
    """
    Returns guess values for the parameters of this function class based on the input. Used for fitting using this
    class.
    :param key:
    :param values:
    :return:
    """

def areas(A1, x01, sig1, A2, x02, sig2):
    return np.array([A1*sig1,A2*sig2])*np.sqrt(2*np.pi)