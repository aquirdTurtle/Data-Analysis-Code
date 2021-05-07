import numpy as np
import uncertainties.unumpy as unp

def center():
    return [2, 3] # or the arg-number of the center.

def args():
    return 'amp', 'x0', 'y0', 'sig_x', 'sig_y', 'theta', 'offset'

def f(coordinates, amplitude, xo, yo, sigma_x, sigma_y, offset):
    """
    The normal function call for this function. Performs checks on valid arguments, then calls the "raw" function.
    :return:
    """
    if sigma_x > 50 or sigma_y > 50 or xo < 0 or yo < 0 or amplitude < 0 or sigma_x < 0 or sigma_y < 0:
        return 1e10*np.ones(len(coordinates[0])*len(coordinates[0][0]))
    res = f_raw(coordinates, amplitude, xo, yo, sigma_x, sigma_y, offset)
    return res

def f_noravel(coordinates, amplitude, xo, yo, sigma_x, sigma_y, offset):
    x = coordinates[0]
    y = coordinates[1]
    # sigma wavefunction = sqrt{hbar/(m\omega)}
    # sigma PDF = sqrt{hbar/(2m\omega)}    
    sigma_psix = np.sqrt(2)*sigma_x
    sigma_psiy = np.sqrt(2)*sigma_y
    xo = float(xo)
    yo = float(yo)
    a = 1/(sigma_psix**2)
    c = 1/(sigma_psiy**2)
    NormalizationCoef = 4/(np.pi*sigma_psix**3*sigma_psiy*2)
    return offset + amplitude*NormalizationCoef*(x-xo)**2*np.exp(- (a*((x-xo)**2) + c*((y-yo)**2)))

def f_raw(coordinates, amplitude, xo, yo, sigma_x, sigma_y, offset):
    """
    The raw function call, performs no checks on valid parameters..
    :return:
    """
    return f_noravel(coordinates, amplitude, xo, yo, sigma_x, sigma_y, offset).ravel()

def guess(key, values):
    """
    Returns guess values for the parameters of this function class based on the input. Used for fitting using this
    class.
    :param key:
    :param values:
    :return:
    """

