

import numpy as np
import uncertainties.unumpy as unp
from fitters import linear


def f(dacVal):
    return f_RelativeToResonance(dacVal)


def f_RelativeToResonance(dacVal):
    return 240 - f_raw(dacVal)


def f_raw(dacVal):
    """
    dacVal in volts, returns frequency in MHz that the VCO outputs.
    """
    return f_Feb2019(dacVal)
    
def f_June2018(dacVal):
    """
    Roughly june, unfortunately didn't record exact date
    """
    if dacVal > 3:
        raise ValueError("ERROR: dac value out of the range of the D1 calibration! VCO behavior is "
                         "not well defined here, the output power is very weak.")
    if dacVal < -3:
        raise ValueError("ERROR: dac value out of the range of the D1 calibraiton! The VCO outputs "
                         "a constant frequency below a voltage of -3.")
    return linear.f(dacVal, *[ -21.98656016,  212.9459437 ])

def f_Feb2019(dacVal):
    # february 27th 2019
    if dacVal > 3:
        raise ValueError("ERROR: dac value out of the range of the D1 calibration! VCO behavior is "
                         "not well defined here, the output power is very weak.")
    if dacVal < -3:
        raise ValueError("ERROR: dac value out of the range of the D1 calibraiton! The VCO outputs "
                         "a constant frequency below a voltage of -3.")
        
    return linear.f(dacVal, *[ -21.81325838,  216.41629956])

def f_August2018Correction(dacVal):
    """
    Tobias accidentally hit the tune knob for the vco, and the frequency calibration is a bit off now. 
    In principle we should at some point recalibrate this.
    """
    return f_June2018(dacVal) - 2.3

def units():
    return "D1 Frequency (MHz) (Relative to Free-Space Resonance)"