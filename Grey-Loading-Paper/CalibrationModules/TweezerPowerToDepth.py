__version__ = "1.0"

"""
TEMPLATE:
"""

import numpy as np
import uncertainties.unumpy as unp

def f(power_mw, numTraps=1):
    """
    Should call one of the f_date() functions, the most recent or best calibration
    """
    return f_August_7th_2018(power_mw, numTraps=numTraps)
    
def f_August_7th_2018(power_mw, numTraps=1):
    """
    working on grey molasses stuff
    """
    # number comes from my light shift notebook calculations
    return 0.33228517568020316 * power_mw / numTraps
    
def units():
    return "Trap Depth: (mK)"


