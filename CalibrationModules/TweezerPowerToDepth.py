__version__ = "1.0"

"""
TEMPLATE:
"""

import numpy as np
import uncertainties.unumpy as unp

def f(power_mw):
    """
    Should call one of the f_date() functions, the most recent or best calibration
    """
    return f_August_7th_2018(power_mw)
    
def f_August_7th_2018(power_mw):
    """
    working on grey molasses stuff
    """
    # number comes from my light shift notebook calculations
    return 0.33228517568020316 * power_mw / 100
    
def units():
    return "Trap Depth: (mK)"