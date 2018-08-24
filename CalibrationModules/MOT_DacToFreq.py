import numpy as np
from fitters import linear


def f(volts):
    """
    Return units is hertz
    """
    return f_Aug_17th_2018_With_Offset(volts)


def f_Aug_17th_2018_With_Offset(volts):
    return -linear.f(volts, -22783561.6206, 138789680.435) - 50e6 + 180e6

    
def f_Aug_17th_2018(volts):
    """
    A calibration curve corresponding to a given date
    """
    return linear.f(volts, -22783561.6206, 138789680.435)
    
    
    