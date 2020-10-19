import numpy as np
import uncertainties.unumpy as unp
from .fitters.Gaussian import arb_1d_sum

numGauss = 4
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


def fitCharacter( params ):
    # for raman spectra, assuming fits are in order from left to right, i.e. first fit is lowest freq
    r1 = params[10]/params[1]
    r2 = params[7]/params[4]
    nbar1 = r1/(1-r1) if not (r1>=1) else 5
    nbar2 = r2/(1-r2) if not (r2>=1) else 5
    avgNbar = (nbar1+nbar2)/2
    return avgNbar

def getFitCharacterString(params):
    r1 = params[10]/params[1]
    r2 = params[7]/params[4]
    nbar1 = r1/(1-r1) if not (r1>=1) else 5
    nbar2 = r2/(1-r2) if not (r2>=1) else 5
    avgNbar = (nbar1+nbar2)/2
    return "(" + round_sig_str(nbar1) + ", " + round_sig_str(nbar2) + ") => " + round_sig_str(avgNbar)

def args():
    arglist = ['Offset']
    for i in range(numGauss):
        j = i+1
        arglist += ['Amp'+str(j), 'Center'+str(j),'Sigma'+str(j)]
    return arglist


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
        if (params[3*i+3] < 1.5):
            # penalize super-narrow fits. 
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
            a, -190, 3,
            a, -170, 3,
            a, 130, 3,
            a, 150, 3]
    

def areas(A1, x01, sig1, A2, x02, sig2):
    return np.array([A1*sig1,A2*sig2])*np.sqrt(2*np.pi)