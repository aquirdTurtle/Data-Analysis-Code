"""
This fitting function assumes a separable quadratic potential. As such it's not as general as possible.
"""


def center():
    return None  # or the arg-number of the center.


def f(coords, x_0, y_0, w_x, w_y, offset):
    """
    The normal function call for this function. Performs checks on valid arguments, then calls the "raw" function.
    :return:
    """
    return f_raw(coords, x_0, y_0, w_x, w_y, offset)


def f_raw(coordinates, x_0, y_0, w_x, w_y, offset):
    """
    The raw function call, performs no checks on valid parameters..
    :return:
    """
    x, y = coordinates
    V_x = 0.5*w_x*(x-x_0)**2
    V_y = 0.5*w_y*(y-y_0)**2
    return (offset + V_x + V_y).flatten()


def f_unc(coords, x_0, y_0, w_x, w_y, offset):
    """
    similar to the raw function call, but uses unp instead of np for uncertainties calculations.
    :return:
    """
    return f_raw(coords, x_0, y_0, w_x, w_y, offset)


def guess(key, values):
    """
    Returns guess values for the parameters of this function class based on the input. Used for fitting using this
    class.
    :param key:
    :param values:
    :return:
    """