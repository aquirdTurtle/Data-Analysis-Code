import numpy as np
import uncertainties.unumpy as unp
# based on the more general exponential_decay module.
from . import exponential_decay

limit = 0

def f(t,A,tau):
    return exponential_decay.f(t,A,tau,limit)

def args():
    return ["Amplitude", "Decay-Constant"]

def center():
    return None

def fitCharacter(params):
    return params[1]

def fitCharacterErr(params, errs):
    return errs[1]

def getFitCharacterString():
    return "Decay Constant"

def f_unc(t, A, tau):
    """
    similar to the raw function call, but uses unp instead of np for uncertainties calculations.
    :return:
    """
    return exponential_decay.f_unc(t,A,tau,limit)

def fitCharacter(params):
    return params[1]
def fitCharacterErr(params, Errs):
    return Errs[1]

def getFitCharacterString():
    return 'tau'

def guess(key, values):
    """
    Returns guess values for the parameters of this function class based on the input. Used for fitting using this
    class.
    :param key:
    :param values:
    :return:
    """
    return exponential_decay.guess(key,values)[:-1]

def fitCharacter(params):
    return params[1]

def fitCharacterErr(params, errors):
    return errors[1]

def getFitCharacterString():
    return "Decay constant"