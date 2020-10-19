import numpy as np
from ...fitters import linear
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate

def f(volts):
    """
    Return units is hertz
    """
    return f_Aug_17th_2018_With_Offset(volts)


def f_Aug_17th_2018(volts):
    """
    A calibration curve corresponding to a given date
    """
    return linear.f(volts, -22783561.6206, 138789680.435)
    
    
def f_interp_Aug_30th_2017(volts):
    beatnoteFreq = [303.40,278.40,254.02,231.90,213.4,193.02,191.02,188.27,185.90,183.77,181.15,180.27,180.71,179.83,
                    179.71,178.65,178.65,177.21,176.58,176.08,175.50,175.21,175.15,174.58,174.15,173.46,173.40,172.71,
                    172.08,171.33,171.33,171.08,170.21,169.96,169.21,169.21,168.21,168.08,167.71,167.52,167.08,166.65,
                    166.4,166.15,165.77,165.4,165.15,164.96,164.58,161.77]
    dac20 = [-5,-4,-3,-2,-1,0,0.1,0.2,0.3,0.4,0.487,0.5,0.502,0.515,0.527,0.553,0.564,0.608,0.62,0.63,0.64,0.65,0.66,
             0.67,0.68,0.69,0.7,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.8,0.81,0.82,0.83,0.84,0.85,0.86,0.87,
             0.88,0.89,0.9,0.91,0.92,1]
    freqInterp = interpolate(dac20, beatnoteFreq, k=1);
    return freqInterp(volts)

def f_interp_March_2nd_2020(volts):
    # based on beatnote data, not just vco output
    volt  = list(reversed([0.5, 0, -0.5, -1, -1.5, -2, -2.5, -3, -3.5, -4, -4.5, -5, -5.5, -6, -6.5]))
    freq  = list(reversed([181, 194, 203.7, 213.8, 223.4, 234.15, 243.7, 255.06, 267, 279.55, 291.5, 304.64, 315.39, 327.936, 333.9]))
    freqInterp = interpolate(volt, freq, k=1);
    return freqInterp(volts)


def f_Aug_17th_2018_With_Offset(volts):
    return -linear.f(volts, -22783561.6206, 138789680.435) - 50e6 + 180e6
