# Symmetric Version of Bump3, forcing the two side bumps to be equally spaced from the center bump.

import numpy as np
import uncertainties.unumpy as unp
from . import arb_1d_sum

numGauss = 3

def fitCharacter( params ):
    [Offset, Amp1, Sigma1, Amp2, Sigma2, Amp3, Sigma3, Center, Spread] = params
    params_ = [Offset, Amp1, Center - Spread/2, Sigma1, Amp2, Center, Sigma2, Amp3, Center + Spread/2, Sigma3]
    # for raman spectra, assuming fits are in order from left to right, i.e. first fit is lowest freq
    r = params_[7] / params_[1]
    return r / ( 1 - r ) if not ( r >= 1 ) else np.inf

def fitCharacterErr(params, errs):
    [Offset, Amp1, Sigma1, Amp2, Sigma2, Amp3, Sigma3, Center, Spread] = params
    [Offset_e, Amp1_e, Sigma1_e, Amp2_e, Sigma2_e, Amp3_e, Sigma3_e, Center_e, Spread_e] = errs
    r = Amp3 / Amp1
    errR = np.sqrt(Amp3_e**2/Amp1**2 + Amp1_e**2 * (r**2/Amp1**2) )
    return errR/(1-r)**2

def axial_GSBC_guess():
    return  (0, 0.5, 10, 0.8, 10, 0.1, 10, 120, 50)

def args():
    arglist = ['Offset', "Amp1", "Sigma1", "Amp2", "Sigma2", "Amp3", "Sigma3", "Center", "Spread"]
    return arglist


def getFitCharacterString():
    return r'$\bar{n}$'


def f(x, Offset, Amp1, Sigma1, Amp2, Sigma2, Amp3, Sigma3, Center, Spread):
    """
    The normal function call for this function. Performs checks on valid arguments, then calls the "raw" function.
    :return:
    """
    penalty = 10**10 * np.ones(len(x))
    
    params = [Offset, Amp1, Center - Spread/2, Sigma1, Amp2, Center, Sigma2, Amp3, Center + Spread/2, Sigma3]
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


def f_unc(x, Offset, Amp1, Sigma1, Amp2, Sigma2, Amp3, Sigma3, Center, Spread):
    """
    similar to the raw function call, but uses unp instead of np for uncertainties calculations.
    :return:
    """
    params = [Offset, Amp1, Center - Spread/2, Sigma1, Amp2, Center, Sigma2, Amp3, Center + Spread/2, Sigma3]
    return arb_1d_sum.f_unc(x, *params)

def guess(key, values):
    """
    Returns guess values for the parameters of this function class based on the input. Used for fitting using this class.
    """
    #a = (max(values)-min(values))/10
    return [0.1,
            0.4, 10,
            0.4, 10,
            0.2, 10, 
            105, 60]    

def areas(A1, x01, sig1, A2, x02, sig2):
    return np.array([A1*sig1,A2*sig2])*np.sqrt(2*np.pi)