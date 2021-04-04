import numpy as np
import uncertainties.unumpy as unp

def getFitCharacterString():
    return "Amplitude";

def fitCharacter(vals):
    return vals[0];
    
def fitCharacterErr(vals, errs):
    return errs[0];
    
def args():
    return 'Amp', 'Decay', 'Freq', 'Phase', 'Offset'

def definition():
    return 'offset + A * np.exp(-x / tau) * np.cos(2 * np.pi * freq * x + phi)'

def getExtremes(f,phi,tau):
    """
    gets the first min and max of the cosine. Calculated the locations using sympy.
    """
    c = 1/(np.pi*f)
    d = 2*np.pi*f*tau
    return (c * (-phi/2 + np.arctan(d+np.sqrt(4*np.pi**2*f**2*tau**2+1))),
            c * (-phi/2 + np.arctan(d+np.sqrt(4*np.pi**2*f**2*tau**2+1)) + np.pi/2))

def f(x, A, tau, f, phi, offset):
    # Just for sanity. Keep some numbers positive.
    if A < 0 or A > 1.2:
        return x * 10 ** 10
    #if phi < 0:
    #    return x * 10 ** 10
    if offset < 0:
        return x * 10 ** 10
    # no growing fits.
    if tau < 0:
        return x * 10 ** 10
    return f_raw(x, A, tau, f, phi, offset)


def f_raw(x, A, tau, freq, phi, offset):
    return offset + A / 2 * np.exp(-x / tau) * np.cos(2 * np.pi * freq * x + phi)


def f_unc(x, A, tau, freq, phi, offset):
    return offset + A / 2 * unp.exp(-x / tau) * unp.cos(2 * np.pi * freq * x + phi)


def guess(key, vals):
    A_g = 0.3
    # A_g = (max(vals) - min(vals)) / 2
    tau_g = 5
    # tau_g = (max(key) - min(key)) * 2
    f_g = 0.3
    phi_g = 1
    offset_g = 0.5
    return [0.8, 10, 0.45, 1, 0.5]
