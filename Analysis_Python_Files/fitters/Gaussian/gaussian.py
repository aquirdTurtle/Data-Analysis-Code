import numpy as np
import uncertainties.unumpy as unp
# the main gaussian function. the other gaussian functions in this submodule call this one.


def getFitCharacterString():
    return 'Center'

def fitCharacter(fitv):
    return fitv[1]

def fitCharacterErr(fitv, fite):
    return fite[1]

def args():
    return 'Amp', 'Center', r'$\sigma$', 'offset'


def f(x, A1, x01, sig1, offset):
    """
    The normal function call for this function. Performs checks on valid arguments, then calls the "raw" function.
    :return:
    """
    return f_raw(x, A1, x01, sig1, offset)


def f_raw(x, A1, x01, sig1, offset):
    """
    The raw function call, performs no checks on valid parameters..
    :return: offset + A1 * np.exp(-(x - x01) ** 2 / (2 * sig1 ** 2))
    """
    if sig1 <= 0:
        return 1e12 * np.ones(len(x))
    return offset + A1 * np.exp(-(x - x01) ** 2 / (2 * sig1 ** 2))


def f_unc(x, A1, x01, sig1, offset):
    """
    similar to the raw function call, but uses unp instead of np for uncertainties calculations.
    :return:
    """
    return offset + A1 * unp.exp(-(x - x01) ** 2 / (2 * sig1 ** 2))


def guess(key, values):
    """
    Returns guess values for the parameters of this function class based on the input. Used for fitting using this
    class.
    :param key:
    :param values:
    :return: guess for gaussian parameters
    """
    return [max(values) - min(values), key[np.argmax(values)], (max(key)-min(key))/4, min(values)]


def area_under(A1, x01, sig1, offset):
    return A1 * sig1 * np.sqrt(2 * np.pi)
