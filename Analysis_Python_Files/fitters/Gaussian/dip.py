import numpy as np
import uncertainties.unumpy as unp
# based on gaussian
from . import gaussian


def fitCharacter(args):
    return args[1]
def fitCharacterErr(fitV, fitE):
    return fitE[1]
def getFitCharacterString():
    return r'$x_0$'

def args():
    return 'A', r'$x_0$', r'$\sigma$', 'offset'


def f(x, A1, x01, sig1, offset):
    """
    The normal function call for this function. Performs checks on valid arguments.
    :return:
    """
    if offset > 1:
        return np.ones(len(x))*10**10
    if A1 > 0:
        return np.ones(len(x)) * 10 ** 10
    return gaussian.f(x, A1, x01, sig1, offset)

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
    :return:
    """
    return [min(values) - max(values), key[np.argmin(values)], (max(key)-min(key))/16, max(values)-0.01]
    #return [max(values) - min(values), 0.8, (max(key)-min(key))/4, max(values)]
