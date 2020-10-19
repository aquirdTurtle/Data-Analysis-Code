import numpy as np
import uncertainties.unumpy as unp
# based on gaussian
from . import gaussian


def fitCharacter(args):
    return args[1]

def fitCharacterErr(args, errs):
    return errs[1]

def getFitCharacterString():
    return "Fit-Center"


def args():
    return 'Amp', 'Center', r'$\sigma$', 'offset'


def f(x, A1, x01, sig1, offset):
    """
    The normal function call for this function. Performs checks on valid arguments, then calls the "raw" function.
    :return:
    """
    #if offset < 0:
    #    return np.ones(len(x))*10**10
    if A1 < 0:
        return np.ones(len(x)) * 10 ** 10
    return gaussian.f(x,A1,x01,sig1,offset)

def f_unc(x, A1, x01, sig1, offset):
    """
    similar to the raw function call, but uses unp instead of np for uncertainties calculations.
    :return:
    """
    return gaussian.f_unc(x,A1,x01,sig1,offset)


def guess(key, values):
    """
    Returns guess values for the parameters of this function class based on the input. Used for fitting using this
    class.
    :param key:
    :param values:
    :return: guess for gaussian parameters
    """
    return [max(values) - min(values), key[np.argmax(values)], (max(key)-min(key))/8, min(values)]

def area_under(A1, sig1):
    # ignoring the offset.
    return A1 * sig1 * np.sqrt(2 * np.pi)
