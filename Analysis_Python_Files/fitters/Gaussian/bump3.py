import numpy as np
import uncertainties.unumpy as unp
from . import arb_1d_sum

numGauss = 3

def fitCharacter( params ):
    #return (params[2] + params[5] + params[7])/3
    #for raman spectra, assuming fits are in order from left to right, i.e. first fit is lowest freq
    r = params[7]/params[1]
    return r/(1-r) if not (r>=1) else 5

def args():
    arglist = ['Offset']
    for i in range(numGauss):
        j = i+1
        arglist += ['Amp'+str(j), 'Center'+str(j),'Sigma'+str(j)]
    return arglist

def getFitCharacterString():
    return r'$\bar{n}$'


def f(x, *params):
    """
    The normal function call for this function. Performs checks on valid arguments, then calls the "raw" function.
    :return:
    """
    if len(params) != 3*numGauss+1:
        raise ValueError('the bump'+str(numGauss)+' fitting function expects '+str(3*numGauss+1) + ' parameters and got ' + str(len(params)))
    penalty = 10**10 * np.ones(len(x))
    for i in range(numGauss):
        if params[3*i+1] < 0:
            # Penalize negative amplitude fits.
            return penalty
        if not (min(x) < params[3*i+2] < max(x)):
            # penalize fit centers outside of the data range (assuming if you want to see these that you've
            # at least put the gaussian in the scan)
            return penalty
    #if params[0] < 0:
        # penalize negative offset
    #    return penalty
    return f_raw(x, *params)


def f_raw(x, *params):
    """
    The raw function call, performs no checks on valid parameters..
    :return:
    """
    return arb_1d_sum.f(x, *params)


def f_unc(x, *params):
    """
    similar to the raw function call, but uses unp instead of np for uncertainties calculations.
    :return:
    """
    return arb_1d_sum.f_unc(x, *params)

def guess(key, values):
    """
    Returns guess values for the parameters of this function class based on the input. Used for fitting using this class.
    """
    a = (max(values)-min(values))/10
    return [min(values),
            a, -180, 3,
            a, -20, 3,
            a, 140, 3]
    

def areas(A1, x01, sig1, A2, x02, sig2):
    return np.array([A1*sig1,A2*sig2])*np.sqrt(2*np.pi)