import numpy as np
import uncertainties.unumpy as unp


def center():
    return [2, 3] # or the arg-number of the center.


def args():
    return 'amp', 'x0', 'y0', 'sig_x', 'sig_y', 'theta', 'offset'


def f(coordinates, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    The normal function call for this function. Performs checks on valid arguments, then calls the "raw" function.
    :return:
    """
    if sigma_x > 50 or sigma_y > 50 or sigma_x < 0 or sigma_y < 0:
        #print('!',end='')
        return 1e10*np.ones(len(coordinates[0])*len(coordinates[0][0]))
    # limit the angle to a small range to prevent unncecessary flips of the axes. The 2D gaussian has two axes of
    # symmetry, so only a quarter of the 2pi is needed.
    if theta > np.pi/4 or theta < -np.pi/4:
        return 1e10*np.ones(len(coordinates[0])*len(coordinates[0][0]))
    res = f_raw(coordinates, amplitude, xo, yo, sigma_x, sigma_y, theta, offset)
    return res

def f_notheta(coordinates, amplitude, xo, yo, sigma_x, sigma_y, offset):
    return f(coordinates, amplitude, xo, yo, sigma_x, sigma_y, 0, offset)


def f_noravel(coordinates, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    #if sigma_x > 1 or sigma_y > 1:
    #    return 1e10

    x = coordinates[0]
    y = coordinates[1]
    #print('xy',x,y)
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp(- (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g


def f_raw(coordinates, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    The raw function call, performs no checks on valid parameters..
    :return:
    """
    return f_noravel(coordinates, amplitude, xo, yo, sigma_x, sigma_y, theta, offset).ravel()


def f_unc(coordinates, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    similar to the raw function call, but uses unp instead of np for uncertainties calculations.
    :return:
    """
    x = coordinates[0]
    y = coordinates[1]
    xo = float(xo)
    yo = float(yo)
    a = (unp.cos(theta)**2)/(2*sigma_x**2) + (unp.sin(theta)**2)/(2*sigma_y**2)
    b = -(unp.sin(2*theta))/(4*sigma_x**2) + (unp.sin(2*theta))/(4*sigma_y**2)
    c = (unp.sin(theta)**2)/(2*sigma_x**2) + (unp.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*unp.exp(- (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()


def areaUnder(amplitude, sigma_x, sigma_y):
    return amplitude * sigma_x * np.sqrt(2 * np.pi) * sigma_y * np.sqrt(2 * np.pi)


def guess(key, values):
    """
    Returns guess values for the parameters of this function class based on the input. Used for fitting using this
    class.
    :param key:
    :param values:
    :return:
    """

