import numpy as np
import uncertainties.unumpy as unp


def center():
    return None


def fitCharacter(vals):
    return vals[1]

def getFitCharacterString():
    return r'$\tau$'


def f(x, a, tau, c):
    """
    The normal function call for this function. Performs checks on valid arguments, then calls the "raw" function.
    :return:
    """
    # saturation!
    if a > 0:
        return np.ones(len(x)) * 1e10
    return f_raw(x, a, tau, c)


def args():
    return ['Amplitude', 'tau', 'Offset']


def f_raw(x, a, tau, c):
    """
    The raw function call, performs no checks on valid parameters..
    :return:
    """
    return a * np.exp(- x / tau) + c


def f_unc(x, a, tau, c):
    """
    similar to the raw function call, but uses unp instead of np for uncertainties calculations.
    :return:
    """
    return a * unp.exp(-x/tau) + c


def guess(key, values):
    """
    Returns guess values for the parameters of this function class based on the input. Used for fitting using this
    class.
    :param key:
    :param values:
    :return:
    """
    return [min(values)-max(values), (max(key)-min(key))/3, min(values)]