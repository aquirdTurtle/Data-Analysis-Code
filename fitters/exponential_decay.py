import numpy as np
import uncertainties.unumpy as unp


def center():
    return None  # or the arg-number of the center.


def args():
    return ["A", r"$\tau$", "Offset"]


def fitCharacter(params):
    return params[1]

def getFitCharacterString():
    return r'$\tau$'

def f(t, A, tau, offset):
    """
    The normal function call for this function. Performs checks on valid arguments, then calls the "raw" function.
    :return:
    """
    return f_raw(t, A, tau, offset)


def f_raw(t, A, tau, offset):
    """
    The raw function call, performs no checks on valid parameters..
    :return:
    """
    return A * np.exp(-t/tau) + offset


def f_unc(t, A, tau, offset):
    """
    similar to the raw function call, but uses unp instead of np for uncertainties calculations.
    :return:
    """
    return A * unp.exp(-t/tau) + offset


def guess(key, values):
    """
    Returns guess values for the parameters of this function class based on the input. Used for fitting using this
    class.
    :param key:
    :param values:
    :return:
    """
    return [max(values)-min(values), 0.5*(max(key)-min(key)), min(values)]

