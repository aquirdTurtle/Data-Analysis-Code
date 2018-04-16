
import numpy as np
import uncertainties.unumpy as unp


def center():
    return None  # or the arg-number of the center.


def args():
    return ['Omega', 'resonance']


def f(o, O, o_0):
    """
    The normal function call for this function. Performs checks on valid arguments, then calls the "raw" function.
    :return:
    """
    return f_raw(o, O, o_0)


def f_raw(o, O, o_0):
    """
    The raw function call, performs no checks on valid parameters..
    :return:
    """
    t = 0.045e-3
    t1 = O**2 / ((o-o_0)**2 + O**2)
    sin_arg = t / 2 * np.sqrt((o-o_0)**2 + O**2)
    return t1 * np.sin(sin_arg)**2


def f_unc(o, O, o_0):
    """
    similar to the raw function call, but uses unp instead of np for uncertainties calculations.
    :return:
    """
    t = 0.045e-3
    t1 = O**2 / ((o-o_0)**2 + O**2)
    sin_arg = t / 2 * unp.sqrt((o-o_0)**2 + O**2)
    return t1 * unp.sin(sin_arg)**2


def guess(key, values):
    """
    Returns guess values for the parameters of this function class based on the input. Used for fitting using this
    class.
    :param key:
    :param values:
    :return:
    """
    return [100000, 6.840930e9]
