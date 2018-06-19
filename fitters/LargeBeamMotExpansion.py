import numpy as np
import MarksConstants as mc

def center():
    return None

def f(time, t_0, sigma_x, T):
    return  f_raw(time, t_0, sigma_x, T)

def f_raw(time, t_0, sigma_x, T):
    """
    Assumes Rubidium 87 to calculate sigma_v from T.
    Assumes that the imaging beam is large compared to the MOT.
    
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
    return sigma_xt


def guess():
    """
    Returns guess values for the parameters of this function class based on the input. Used for fitting using this
    class.
    """
    return 0, 0.0005, 100e-6

def args():
    return ['t_0', 'sigma_x', 'T']
