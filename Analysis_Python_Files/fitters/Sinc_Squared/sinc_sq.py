import numpy as np
import uncertainties.unumpy as unp


def center():
    return 2 # or the arg-number of the center.


def f(x, A, c, scale, offset):
    """
    The normal function call for this function. Performs checks on valid arguments, then calls the "raw" function.
    :return:
    """
#    if offset < 0:
#        return x * 10**10
#    if A < 0:
#        return x * 10**10
    return f_raw(x, A, c, scale, offset)

def fitCharacter(fitV):
    return fitV[1]

def fitCharacterErr(fitV, fitE):
    return fitE[1]

def getFitCharacterString():
    return "Center"

def args():
    return "Amplitude", "Center", "Width-Scale", "Offset" 


def f_raw(x, A, c, scale, offset):
    """
    The raw function call, performs no checks on valid parameters..
    :return:
    """
    return A * np.sinc((x - c)/scale)**2 + offset


def f_unc(x, A, c, scale, offset):
    """
    similar to the raw function call, but uses unp instead of np for uncertainties calculations.
    :return:
    """
    arg = np.pi*(x - c)/scale
    return A * (unp.sin(arg)/arg)**2 + offset


def guess(key, values):
    """
    Returns guess values for the parameters of this function class based on the input. Used for fitting using this
    class.
    :param key:
    :param values:
    :return:
    """
    return [max(values) - min(values), key[np.argmax(values)], (max(key)-min(key))/4, min(values)+0.001]

