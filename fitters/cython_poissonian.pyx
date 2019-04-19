from libc.math cimport exp
import cython 


@cython.cdivision(True)
cdef double f_raw(double x, double k, double weight):
    """
    double x, double k, double weight
    The raw function call, performs no checks on valid parameters..
    This function calculates p_k{x} = weight * e^(-k) * k^x / x!.
    :param x: argument of the Poisson distribution
    :param k: order or (approximate) mean of the Poisson distribution.
    :param weight: a weight factor, related to the maximum data this is supposed to be fitted to, but typically over-
    weighted for the purposes of this function.
    :return: the Poisson distribution evaluated at x given the parameters.
    """
    cdef double term = 1
    # calculate the term k^x / x!. Can't do this directly, x! is too large.
    cdef size_t n
    cdef int xi = <int>x
    for n in range(0, xi):
        term *= k / (x - n) * exp(-k/xi)
    return term * weight


def guess(key, values):
    """
    Returns guess values for the parameters of this function class based on the input. Used for fitting using this
    class.
    :param key:
    :param values:
    :return:
    """
    
def f(int x, int k, double weight):
    """
    The normal function call for this function. Performs checks on valid arguments, then calls the "raw" function.
    This function calculates p_k{x} = weight * e^(-k) * k^x / x!.
    :param x: argument of the Poisson distribution
    :param k: order or (approximate) mean of the Poisson distribution.
    :param weight: a weight factor, related to the maximum data this is supposed to be fitted to, but typically over-
    weighted for the purposes of this function.
    :return: the Poisson distribution evaluated at x given the parameters.
    """
    return f_raw(x, k, weight)