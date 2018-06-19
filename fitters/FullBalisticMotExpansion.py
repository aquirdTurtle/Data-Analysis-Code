import numpy as np

def center():
    return None

def f(time, t_0, sigma_x, T, sigma_I):
    return  f_raw(time, t_0, sigma_x, T, sigma_I)

def f_raw(time, t_0, sigma_x, T, sigma_I):
    """
    Assumes Rubidium 87 to calculate sigma_v from T.
    
    :param time: the (known) time of the measured waists.
    :param t_0: the (effective) offset of the times of the measured waists. This can compensate for
        the wait that often exists between release and measurement, as well as non-perfect release of the
        MOT.
    :param sigma_x: The initial waist of the MOT.
    :param T: The temperature of the MOT.
    :param sigma_I: the waist of the imaging laser.
    """
    sigma_v = np.sqrt(mc.k_B * T / mc.Rb87_M)
    sigma_xt = np.sqrt(sigma_v**2*(time+t_0)**2 + sigma_x**2)
    sigma_m = sigma_I*sigma_xt/np.sqrt(sigma_I**2 + sigma_xt**2)
    return sigma_m


def guess(key, values):
    """
    Returns guess values for the parameters of this function class based on the input. Used for fitting using this
    class.
    """
    return 0, 10, 100e-6, 2e-3
