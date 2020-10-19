
import numpy as np
import uncertainties.unumpy as unp

def args():
    return ['Omega', 'resonance']

def f(time, Omega, Delta, maxAmp):
    """
    The normal function call for this function. Performs checks on valid arguments,
    then calls the "raw" function.
    :return:
    """
    return f_raw(time, Omega, Delta, maxAmp)

def f_raw( time, Omega, Delta, maxAmp ):
    """
    The raw function call, performs no checks on valid parameters..
    Delta: expected in angular units
    :return:
    """
    sin_arg = np.sqrt(Omega**2+Delta**2) * time / 2
    amp = maxAmp * Omega**2/(Omega**2+Delta**2)
    return amp*np.sin(sin_arg)**2

def f_unc(time, Omega, Delta, maxAmp):
    """
    similar to the raw function call, but uses unp instead of np for uncertainties calculations.
    :return:
    """
    sin_arg = unp.sqrt(Omega**2+Delta**2) * time / 2
    amp = maxAmp * Omega**2/(Omega**2+Delta**2)
    return amp*unp.sin(sin_arg)**2
