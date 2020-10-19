def center():
    return None  # or the arg-number of the center.


def f(x, A, center, width, offset):
    """
    The normal function call for this function. Performs checks on valid arguments, then calls the "raw" function.
    :return:
    """
    if offset < 0:
        return x * 10**10
    if A < 0:
        return x * 10**10
    return f_raw(x, A, center, width, offset)


def f_raw(x, A, center, width, offset):
    """
    The raw function call, performs no checks on valid parameters..
    :return:
    """
    return A / ((x - center)**2 + (width/2)**2)


def f_unc(x, A, center, width, offset):
    """
    similar to the raw function call, but uses unp instead of np for uncertainties calculations.
    :return:
    """
    return A / ((x - center)**2 + (width/2)**2)


def guess(key, values):
    """
    Returns guess values for the parameters of this function class based on the input. Used for fitting using this
    class.
    :param key:
    :param values:
    :return:
    """
