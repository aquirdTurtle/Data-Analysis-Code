__version__ = "1.0"

"""
Each class in this module contains all the information required for fitting data. That is, each class follows the
following TEMPLATE:
"""

import numpy as np
import uncertainties.unumpy as unp


def center():
    return None  # or the arg-number of the center.


def args():
    return None


def f():
    """
    The normal function call for this function. Performs checks on valid arguments, then calls the "raw" function.
    :return:
    """
    return f_raw()


def f_raw():
    """
    The raw function call, performs no checks on valid parameters..
    :return:
    """
    return  # ...


def f_unc():
    """
    similar to the raw function call, but uses unp instead of np for uncertainties calculations.
    :return:
    """


def guess(key, values):
    """
    Returns guess values for the parameters of this function class based on the input. Used for fitting using this
    class.
    :param key:
    :param values:
    :return:
    """