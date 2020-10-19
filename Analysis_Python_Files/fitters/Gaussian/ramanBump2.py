import numpy as np
import uncertainties.unumpy as unp
from .fitters.Gaussian import arb_1d_sum

numGauss = 2
carrierLoc = 0

def getExp(val):
    if val == 0:
        return 0
    return np.floor(np.log10(np.abs(val)))

def round_sig_str(x, sig=3):
    """
    round a float to some number of significant digits
    :param x: the numebr to round
    :param sig: the number of significant digits to use in the rounding
    :return the rounded number, as a string.
    """
    if sig<=0:
        return "0"
    if np.isnan(x):
        x = 0
    try:
        num = round(x, sig-int(np.floor(np.log10(abs(x)+2*np.finfo(float).eps)))-1)
        decimals = sig-getExp(num)-1
        if decimals == float('inf'):
            decimals = 3
        if decimals <= 0:
            decimals = 0
        result = ("{0:."+str(int(decimals))+"f}").format(num)
        # make sure result has the correct number of significant digits given the precision.
        return result
    except ValueError:
        print(abs(x))


def fitCharacter(params):
    offset, amp1, sig1, amp2, sig2, trapFreq = params
    # for raman spectra, return the nbar, assuming correct orientation of the 2 gaussians
    r = amp2/amp1
    return r/(1-r) if not (r>=1) else 5

def getFitCharacterString(params):
    offset, amp1, sig1, amp2, sig2, trapFreq = params
    r = amp2/amp1
    return round_sig_str(r/(1-r) if not (r>=1) else 5)


def args():
    arglist = ['Offset', 'Amp1', 'Sig1', 'Amp2', 'Sig2', 'trapFreq']
    return arglist


def getF(carrierF):
    """
    For this module, it is expected that the user wants to manually set the carrier frequency. otherwise this is the same effectively as bump2.
    So, you need an individually customizable fitting function but all the normal module functionality. This is my current way of doing that, 
    with some extra args 
    """
    return lambda x, offset, amp1, sig1, amp2, sig2, trapFreq: f(x, carrierF, offset, amp1, sig1, amp2, sig2, trapFreq)

def getF_unc(carrierF):
    return lambda x, offset, amp1, sig1, amp2, sig2, trapFreq: f_unc(x, carrierF, offset, amp1, sig1, amp2, sig2, trapFreq)



def f(x, carrierFreq, offset, amp1, sig1, amp2, sig2, trapFreq):
    """
    It is expected that this function is used to create a lambda
    The normal function call for this function. Performs checks on valid arguments, then calls the "raw" function.
    :return:
    """
    penalty = 10**10 * np.ones(len(x))
    if sig1 < 2 or sig2 < 2:
        # Penalize super-narrow fits
        return penalty
    if amp1 < 0 or amp2 < 0:
        # Penalize negative amplitude fits.
        return penalty
    if offset < 0:
        # penalize negative offset
        return penalty
    return f_raw(x, *[offset, amp1, carrierFreq - trapFreq, sig1, amp2, carrierFreq+trapFreq, sig2])


def f_raw(x, *params):
    """
    The raw function call, performs no checks on valid parameters..
    :return:
    """
    return arb_1d_sum.f(x, *params)


def f_unc(x, carrierFreq, offset, amp1, sig1, amp2, sig2, trapFreq):
    """
    similar to the raw function call, but uses unp instead of np for uncertainties calculations.
    :return:
    """
    return arb_1d_sum.f_unc(x, *[offset, amp1, carrierFreq - trapFreq, sig1, amp2, carrierFreq+trapFreq, sig2])

def guess(key, values):
    """
    Returns guess values for the parameters of this function class based on the input. Used for fitting using this class.
    """
    return [min(values), 0.2, 3, 0.2, 3, 115]
    

def areas(A1, x01, sig1, A2, x02, sig2):
    return np.array([A1*sig1,A2*sig2])*np.sqrt(2*np.pi)