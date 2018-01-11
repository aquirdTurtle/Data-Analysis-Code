import numpy as np
import uncertainties.unumpy as unp


def center():
    return None


def args():
    return 'Amp', 'Decay', 'Freq', 'Phase', 'Offset'


def definition():
    return 'offset + A * np.exp(-x / tau) * np.cos(2 * np.pi * freq * x + phi)'


def f(x, A, tau, f, phi, offset):
    # Just for sanity. Keep some numbers positive.
    if A < 0:
        return x * 10 ** 10
    if phi < 0:
        return x * 10 ** 10
    if offset < 0:
        return x * 10 ** 10
    # no growing fits.
    if tau < 0:
        return x * 10 ** 10
    return f_raw(x, A, tau, f, phi, offset)


def f_raw(x, A, tau, freq, phi, offset):
    return offset + A * np.exp(-x / tau) * np.cos(2 * np.pi * freq * x + phi)


def f_unc(x, A, tau, freq, phi, offset):
    return offset + A * unp.exp(-x / tau) * unp.cos(2 * np.pi * freq * x + phi)


def guess(key, vals):
    A_g = (max(vals) - min(vals)) / 2
    tau_g = (max(key) - min(key)) * 2
    # assumes starts at zero then goes to max value or so. May need to modify.
    f_g = 1 / (2 * key[np.argmax(vals)])
    phi_g = np.pi
    offset_g = 0.5
    return [A_g, tau_g, f_g, phi_g, offset_g]
