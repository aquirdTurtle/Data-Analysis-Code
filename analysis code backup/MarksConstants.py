__version__ = "1.0"

""" all in mks """
import numpy as np
import sympy as sp
# ###################
# ## Common Constants

# planck's constant ?
h = 6.6260700e-34
# reduced planck's constant ?
hbar = 1.0545718e-34
# Boltzmann's constant ?
k_B = 1.3806488e-23
# speed of light (exact)
c = 299792458
# Stephan-Boltzmann constant, ?
sigma = 5.6704e-8
# atomic mass unit, ?
amu = 1.6605390e-27
# use numpy
pi = np.pi
# gravity constant, inexact
g = 9.80665
# fundamental charge (charge of electron & proton), in coulombs, inexact
qe = 1.6021766208e-19
# Bohr Radius, in m
a0 = 0.52917721067e-10
# Electric constant, vacuum permittivity, in Farads per meter, exact
epsilon0 = 8.854187817e-12
# Magnetic Constant, vacuum permeability, in Henrys / meter or newtons / Amp^2, exact
mu0 = 4e-7 * pi

# ######################
# ### Rubidium Constants

# rubidium 87 mass (inexact)
Rb87_M = 86.909180527 * amu
# linewidths, in s^-1
Rb87_D1Gamma = 36.10e6
Rb87_D1GammaUncertainty = 0.05e6
Rb87_D2Gamma = 38.11e6
Rb87_D2GammaUncertainty = 0.06e6
# for far-detuned approximations only.
# strictly, I should probably weight by Clebsch-Gordon coefficients or something to get
# a better far-detuned approximation.
Rb87_AvgGamma = (Rb87_D1Gamma + Rb87_D2Gamma)/2

# in mW/cm^2, 2-3', resonant & isotropic light.
Rb87_I_ResonantIsotropicSaturationIntensity = 3.576

# wavelengths are in vacuum.
# in m
Rb87_D2LineWavelength = 780.241209686e-9
Rb87_D2LineWavelengthUncertainty = 1.3e-17
# in Hz (1/s)
Rb87_D2LineFrequency = 384.2304844685e12
Rb87_D2LineFrequencyUncertainty = 6.2e3
# etc.
Rb87_D1LineWavelength = 794.9788509e-9
Rb87_D1LineWavelengthUncertainty = 8e-16
Rb87_D1LineFrequency = 377.1074635e12
Rb87_D1LineFrequencyUncertainty = 0.4e6

# #################
# ### Lab Constants

opBeamDacToVoltageConversionConstants = [8.5, -22.532, -1.9323, -0.35142]
# pixel sizes for various cameras we have
baslerScoutPixelSize = 7.4e-6
baslerAcePixelSize = 4.8e-6
andorPixelSize = 16e-6
dataRayPixelSize = 4.4e-6

# basler conversion... joules incident per grey-scale count.
# number from theory of camera operation
cameraConversion = 117e-18
# number from measurement. I suspect this is a little high because I think I underestimated the attenuation
# of the signal by the 780nm filter.
# C = 161*10**-18

# note: I don't know what the limiting aperture is, but I believe that it's a bit larger than either of these.
# This parameter could probably be extracted from Zeemax calculations.
# (in meters)

# ... what? these are both too big... I don't know why these are set to these numbers.
sillLensInputApertureDiameter = 40e-3
sillLensExitApertureDiameter = 40e-3

# need to look these up.
# tweezerBeamGaussianWaist = 4
# probeBeamGaussianWaist = ???

# ########################
# ### Trap Characteristics

# in nm
trapWavelength = 852e-9
# ( 0 for Pi-polarized trapping light (E along quantization axis) *)
EpsilonTrap = 0
# this is the complex spherical tensor representation of the trap polarization at the atom, the first
# entry is the sigma plus polarization component, the second is the pi-polarized component, and
# the third is the sigma minus polarization component.
uTrap = [0, 1, 0]

# Pauli Matrices
X = sigma_x = sp.Matrix([[0, 1], [1, 0]])
Y = sigma_y = sp.Matrix([[0, -1j], [1j, 0]])
Z = sigma_z = sp.Matrix([[1, 0], [0, -1]])
# Hadamard
H = hadamard = sp.Matrix([[1, 1], [1, -1]])
# Phase Gate
S = phaseGate = sp.Matrix([[1, 0], [0, 1j]])


def phaseShiftGate(phi):
    return sp.Matrix([[1, 0], [[0, sp.exp(1j * phi)]]])
