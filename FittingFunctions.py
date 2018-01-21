__version__ = "1.0"

# THIS MODULE HAS BEEN DEPRECATED.
# we no longer use this module, instead we use the fitters module and submodules within.

import numpy as np
import uncertainties.unumpy as unp


def exponentialGrowth(x, A, tau):
    """

    :param x:
    :param A:
    :param tau:
    :return: A * np.exp(x/tau)
    """
    if A < 0:
        return np.ones(len(x)) * 1e10
    return A * np.exp(x/tau)


def exponentialSaturation(x, a, tau, c):
    """

    :param x:
    :param a:
    :param tau:
    :param c:
    :return: a * np.exp(- x/tau) + c
    """
    if a > 0: # a penalty for wrong parameters
        return np.ones(len(x)) * 1e10
    return a * np.exp(- x/tau) + c


def uncExponentialSaturation(x, a, tau, c):
    """

    :param x:
    :param a:
    :param tau:
    :param c:
    :return: a * unp.exp(- x/tau) + c
    """
    return a * unp.exp(- x/tau) + c


def exponentialDecay(x, A, tau):
    """

    :param x:
    :param A:
    :param tau:
    :return: A * np.exp(-x/tau)
    """
    if A < 0:
        return np.ones(len(x)) * 1e10
    return A * np.exp(-x/tau)


def uncExponentialDecay(x, A, tau):
    """

    :param x:
    :param A:
    :param tau:
    :return: A * unp.exp(-x/tau)
    """
    if A < 0:
        return np.ones(len(x)) * 1e10
    return A * unp.exp(-x/tau)


def linear(x, a, b):
    """
    :return: a * x + b
    """
    return a * x + b


def quadraticDip(x, a, b, x0):
    """
    This assumes downward facing. Best to write another function for upward facing if need be, I think.
    :return a + b*(x-x0)**2 , b < 0
    """
    if a < 0:
        return 10**10 * np.ones(len(x))
    if b < 0:
        return 10**10 * np.ones(len(x))
    return a + b*(x-x0)**2


def quadraticBump(x, a, b, x0):
    """
    This assumes downward facing. Best to write another function for upward facing if need be, I think.
    :return a + b*(x-x0)**2 , b > 0
    """
    if a < 0:
        return 10**10 * np.ones(len(x))
    if b > 0:
        return 10**10 * np.ones(len(x))
    return a + b*(x-x0)**2


def gaussian(x, A1, x01, sig1, offset):
    """

    :param x:
    :param A1:
    :param x01:
    :param sig1:
    :param offset:
    :return: offset + A1 * np.exp(-(x-x01)**2/(2*sig1**2))
    """
    if offset < 0:
        return 10**10
    return offset + A1 * np.exp(-(x-x01)**2/(2*sig1**2))


def doubleGaussian(x, A1, x01, sig1, A2, x02, sig2, offset):
    """

    :param x:
    :param A1:
    :param x01:
    :param sig1:
    :param A2:
    :param x02:
    :param sig2:
    :param offset:
    :return: offset + A1 * np.exp(-(x-x01)**2/(2*sig1**2)) + A2 * np.exp(-(x-x02)**2/(2*sig2**2))
    """
    if A1 < 0 or A2 < 0:
        # Penalize negative fits.
        return 10**10
    if offset < 0:
        return 10**10
    return offset + A1 * np.exp(-(x-x01)**2/(2*sig1**2)) + A2 * np.exp(-(x-x02)**2/(2*sig2**2))


def tripleGaussian(x, A1, x01, sig1, A2, x02, sig2, A3, x03, sig3, offset):
    """

    :param x:
    :param A1:
    :param x01:
    :param sig1:
    :param A2:
    :param x02:
    :param sig2:
    :param A3:
    :param x03:
    :param sig3:
    :param offset:
    :return: (offset + A1 * np.exp(-(x-x01)**2/(2*sig1**2)) + A2 * np.exp(-(x-x02)**2/(2*sig2**2))
            + A3 * np.exp(-(x-x03)**2/(2*sig3**2)))
    """
    if A1 < 0 or A2 < 0 or A3 < 0:
        # Penalize negative fits.
        return 10**10
    if offset < 0:
        return 10**10
    return (offset + A1 * np.exp(-(x-x01)**2/(2*sig1**2)) + A2 * np.exp(-(x-x02)**2/(2*sig2**2))
            + A3 * np.exp(-(x-x03)**2/(2*sig3**2)))


def gaussian_2D(coordinates, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    Stolen from "http://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian
    -function-using-scipy-optimize-curve-fit-valueerror-and-m"
    :param coordinates:
    :param amplitude:
    :param xo:
    :param yo:
    :param sigma_x:
    :param sigma_y:
    :param theta:
    :param offset:
    :return:
    """
    # limit the angle to a small range to prevent unncecessary flips of the axes. The 2D gaussian has two axes of
    # symmetry, so only a quarter of the 2pi is needed.
    if theta > np.pi/4 or theta < -np.pi/4:
        return 1e10
    x = coordinates[0]
    y = coordinates[1]
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()


def decayingCos(x, A, tau, f, phi, offset):
    """

    :param x:
    :param A:
    :param tau:
    :param f:
    :param phi:
    :param offset:
    :return:
    """
    # Just for sanity. Keep some numbers positive.
    if A < 0:
        return x * 10**10
    if phi < 0:
        return x * 10**10
    if offset < 0:
        return x * 10**10
    # no growing fits.
    if tau < 0:
        return x * 10**10
    return offset + (1 - A/2 * np.exp(-x/tau) * np.cos(2 * np.pi * f * x + phi))


def sincSquared(x, A, center, scale, offset):
    """

    :param x:
    :param A:
    :param center:
    :param scale:
    :param offset:
    :return:
    """
    if offset < 0:
        return x * 10**10
    if A < 0:
        return x * 10**10
    return A * np.sinc((x - center)/scale)**2 + offset


def uncGaussian(x, A1, x01, sig1, offset):
    """

    :param x:
    :param A1:
    :param x01:
    :param sig1:
    :param offset:
    :return:
    """
    if offset < 0:
        return 1e10
    return offset + A1 * unp.exp(-(x-x01)**2/(2*sig1**2))


def lorentzian(x, A, center, width, offset):
    """

    :param x:
    :param A:
    :param center:
    :param width:
    :param offset:
    :return:
    """
    if offset < 0:
        return x * 10**10
    if A < 0:
        return x * 10**10
    return A / ((x - center)**2 + (width/2)**2)


def RabiFlop(x, amp, Omega, phi):
    """

    :param x:
    :param amp:
    :param Omega:
    :param phi:
    :return:
    """
    if amp < 0:
        return -1e10
    else:
        return amp*np.sin(2*np.pi*Omega*x/2+phi)**2


def uncRabiFlop(x, amp, Omega, phi):
    """

    :param x:
    :param amp:
    :param Omega:
    :param phi:
    :return:
    """
    if amp < 0:
        return -10**10
    else:
        return amp * unp.sin(2*np.pi*Omega*x/2+phi)**2


def poissonian(x, k, weight):
    """
    This function calculates p_k{x} = weight * e^(-k) * k^x / x!.
    :param x: argument of the Poisson distribution
    :param k: order or (approximate) mean of the Poisson distribution.
    :param weight: a weight factor, related to the maximum data this is supposed to be fitted to, but typically over-
    weighted for the purposes of this function.
    :return: the Poisson distribution evaluated at x given the parameters.
    """

    term = 1
    # calculate the term k^x / x!. Can't do this directly, x! is too large.
    for n in range(0, int(x)):
        term *= k / (x - n) * np.exp(-k/int(x))
    return term * weight

