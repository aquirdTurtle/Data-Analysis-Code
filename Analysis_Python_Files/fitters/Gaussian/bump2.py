import numpy as np
import uncertainties.unumpy as unp
from . import arb_1d_sum

numGauss = 2

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
    # for raman spectra, return the nbar, assuming correct orientation of the 2 gaussians
    r = params[4]/params[1]
    return r/(1-r) if not (r>=1) else np.inf
    # return the diff/2
    #return (params[5] + params[2])/2

def fitCharacterErr(params, errs):
    # sqrt(f_x'^2 sig_x^2 + f_y'^2 sig_y^2)
    # error in r:
    # sqrt(1/b^2 sig_r^2 + (-r/b^2)^2 sig_b^2)
    r = params[4]/params[1]
    errR = np.sqrt(errs[4]**2/params[1]**2 + errs[1]**2 * (r**2/params[1]**2) )
    # error in nbar: 
    # sigma_r*((1-r)+r)/(1-r)**2 = 1/(1-r)**2 sigma_r
    return errR/(1-r)**2
    
def getFitCharacterString():
    return r'$\bar{n}$'


def args():
    arglist = ['Offset']
    for i in range(numGauss):
        j = i+1
        arglist += [r'$A_'+str(j)+'$', r'$x_'+str(j)+'$',r'$\sigma_'+str(j)+'$']
    return arglist

def f(x, *params):
    """
    The normal function call for this function. Performs checks on valid arguments, then calls the "raw" function.
    :return:
    """
    if len(params) != 3*numGauss+1:
        raise ValueError('the bump2 fitting function expects '+str(3*numGauss+1) + ' parameters and got ' + str(len(params)))
    penalty = 10**10 * np.ones(len(x))
    for i in range(numGauss):
        if params[3*i+3] < 3:
            # Penalize super-narrow fits
            return penalty
        if params[3*i+1] < 0:
            # Penalize negative amplitude fits.
            return penalty
        if not (min(x) < params[3*i+2] < max(x)):
            # penalize fit centers outside of the data range (assuming if you want to see these that you've
            # at least put the gaussian in the scan)
            return penalty
    if params[0] < 0:
        # penalize negative offset
        return penalty
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
    return sbcGuess()[0]
    #return [min(values),
    #        0.2, -150, 5,
    #        0.1, 95, 5]

def sbcGuess():
    return [[0,0.3,-150,10, 0.3, 150, 10]]
    
    
def areas(A1, x01, sig1, A2, x02, sig2):
    return np.array([A1*sig1,A2*sig2])*np.sqrt(2*np.pi)