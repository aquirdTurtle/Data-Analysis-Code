from scipy.optimize import curve_fit as fit
import scipy.special as special
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from scipy.optimize import curve_fit as fit
from sympy.physics.wigner import wigner_6j
import uncertainties as unc
from uncertainties import unumpy as unp

from .CalibrationModules.Mot import MOT_DacToFreq
dacToFreq = MOT_DacToFreq.f_interp_Aug_30th_2017
dacToFreq2 = MOT_DacToFreq.f_interp_March_2nd_2020
from .fitters import linear
from .Miscellaneous import round_sig
from .Miscellaneous import transpose
from . import MarksConstants as mc

def SJS(a,b,c,d,e,f):
    return complex(wigner_6j(a,b,c,d,e,f))
from scipy.interpolate import interp1d 
def Interpolation(data):
    return interp1d(data[:,0],data[:,1],kind='cubic')
import numpy as np

def ReducedDipoleMatrixElement(Kappa, OmegaTrap, manifold):
    """
    I don't think "ReducedMatrixElement" is really the right name here. this appears as $\alpha^(K)_(nJ)$ In the Fam Le Kien paper.
    """
    J_i = float(manifold[-2]) / float(manifold[-1])
    k = Kappa
    OmTp = OmegaTrap
    #const = (-1)**(j_i + k + 1) * np.sqrt(2 * k + 1)
    const = (-1)**(J_i + k + 1) * np.sqrt(2 * k + 1)
    """
    IMPORTANT ACCURACY NOTE:
    Brian was using this one gamma for all the different contributions. I don't think that this is very accurate at all. 
    The gamma here should be the sum of the decay rates for the initial and final states, which in general will be much larger 
    than this value used here, I believe. This makes this calculation an overestimate of the reduced matrix element.
    """
    # should be \Gamma_{nJ} + \Gamma_{n'J'}
    G = 6.1e6
    sumTerm = 0
    # add the contribution from all transitions.
    for i, (key, val) in enumerate(mc.Rb87_Transition_rdme[manifold].items()):
        j_f = float(key[-2]) / float(key[-1])
        # Transition Frequency, or \omega_{n'J'} - \omega_{nJ}
        OmTs = (mc.Rb87_Energies[key] - mc.Rb87_Energies[manifold])/mc.hbar
        sumTerm += ((-1)**j_f * SJS(1, k, 1, J_i, j_f, J_i) * val**2 / mc.hbar 
                    * np.real( 1 / ( OmTs - OmTp - 1j * G / 2) 
                              + (-1)**k / (OmTs + OmTp + 1j * G / 2)))
    return  const * sumTerm

def ScalarPolarizability(wavelength, manifold):
    j_init = float(manifold[-2]) / float(manifold[-1])
    scalarShift = 1.0 / np.sqrt(3 * (2 * j_init + 1)) * ReducedDipoleMatrixElement(0, (2 * mc.pi * mc.c) / wavelength, manifold)
    return scalarShift

def VectorPolarizability(Fg, wavelength, manifold):
    j_init = float(manifold[-2]) / float(manifold[-1])
    vectorShift = ((-1)**(j_init + 3/2 + Fg) * np.sqrt((2 * Fg * (2 * Fg + 1))/(Fg + 1)) 
            * SJS(Fg, 1, Fg, j_init, 3/2, j_init)
            * ReducedDipoleMatrixElement(1, (2 * mc.pi * mc.c) / wavelength, manifold))
    return vectorShift

def TensorPolarizability(Fg, wavelength, manifold):
    j_init = float(manifold[-2]) / float(manifold[-1])
    tensorShift = ((-1)**(j_init + 3/2 + Fg + 1) * np.sqrt((2 * Fg * (2 * Fg - 1) * (2 * Fg + 1))/(3 * (Fg + 1) * (2 * Fg + 3))) 
        * SJS(Fg, 2, Fg, j_init, 3/2, j_init) * 
                   ReducedDipoleMatrixElement(2, (2*mc.pi*mc.c) / wavelength, manifold))
    return tensorShift

def ManifoldShift(Fg, mFg, I0, manifold, wavelength=852e-9):
    """
    Ground state shift, i.e. Trap Depth
    It's a long equation. Breaking it apart here.
    mc.Rb87_Transition_rdme.keys() = dict_keys(['5S12', '5P12', '5P32'])
    """
    first = (-0.25 * ((2 * I0)/(mc.c * mc.epsilon0)))
    scalarTerm = (ScalarPolarizability(wavelength, manifold))
    Ccoeff = np.conj(mc.uTrap[2]) * mc.uTrap[2] - np.conj(mc.uTrap[0]) * mc.uTrap[0];  
    vectorTerm =  (Ccoeff * VectorPolarizability(Fg, wavelength, manifold) * mFg / (2 * Fg))
    Dcoeff = 1 - 3 * np.conj(mc.uTrap[1]) * mc.uTrap[1];  
    tensorTerm = (- Dcoeff * TensorPolarizability(Fg, wavelength, manifold) 
               * (3 *mFg**2 - Fg * (Fg + 1))/( 2 * Fg * (2 * Fg - 1)))
    return first * (scalarTerm + vectorTerm + tensorTerm)

def getTrapDepth(power, wavelength=mc.trapWavelength):
    """
    :param power: refers to the power measured in the rail, in W.
    :returns: trap depth in mK
    """
    intensity = powerToIntensity*power
    return np.real(trapDepthFromIntensity(intensity,wavelength=wavelength))

def trapDepthFromIntensity(intensity, wavelength=mc.trapWavelength):
    return 1e3 * ManifoldShift(2, 2, intensity, '5S12', wavelength)/mc.k_B


def getGroundStateShift(power, wavelength=mc.trapWavelength):
    return getTrapDepth(power, wavelength=mc.trapWavelength)

def getStateShiftIn_mk(intensity, manifold, F=2, mF=2, wavelength=mc.trapWavelength):
    return np.real(1e3 * ManifoldShift(F, mF, intensity, manifold, wavelength=wavelength) / mc.k_B)

def getStateShiftIn_MHz(intensity, manifold, F=2, mF=2, wavelength=mc.trapWavelength):
    return np.real(1e-6 * ManifoldShift(F, mF, intensity, manifold, wavelength=wavelength) / mc.h)

def getDepthAndErr(power):
    global powerToIntensity
    powerToIntensity = resonanceShiftPerWattInRail / np.real(CyclingShift(1))
    m = np.real(getTrapDepth(power))
    powerToIntensity = resonanceShiftPerWattInRail_l / np.real(CyclingShift(1))
    l = np.real(getTrapDepth(power))
    powerToIntensity = resonanceShiftPerWattInRail_h / np.real(CyclingShift(1))
    h = np.real(getTrapDepth(power))
    if abs(h-m - (m-l)) < 1e-6:
        print(misc.round_sig(m,6),r'+-',misc.round_sig(h-m,6))
    else:
        print(m,'+',h-m,'-',m-l)
        
def getDepthAndErr_MHz(power):
    global powerToIntensity
    powerToIntensity = resonanceShiftPerWattInRail / np.real(CyclingShift(1))
    intensity = powerToIntensity*power
    m = np.real(getStateShiftIn_MHz(intensity, '5S12'))
    powerToIntensity = resonanceShiftPerWattInRail_l / np.real(CyclingShift(1))
    intensity = powerToIntensity*power
    l = np.real(getStateShiftIn_MHz(intensity, '5S12'))
    powerToIntensity = resonanceShiftPerWattInRail_h / np.real(CyclingShift(1))
    intensity = powerToIntensity*power
    h = np.real(getStateShiftIn_MHz(intensity, '5S12'))
    if abs(h-m - (m-l)) < 1e-6:
        print(misc.round_sig(m,6),r'+-',misc.round_sig(h-m,6))
    else:
        print(m,'+',h-m,'-',m-l)
        
def freqShift_june_3_2020(dacValue):
    freq = dacToFreq(dacValue)*1e6
    # fitVal[1] = 194.68607990470966
    shift = 242.36110231*1e6-freq
    return shift

def intensityFromDac_june_3_2020(dacValue):
    # shift / shift per unit intensity = intensity
    # np.real(CyclingShift(1)) = 0.010770874964772921
    return freqShift_june_3_2020(dacValue) / 0.010770874964772921

def trapDepthFromDac(dacValue, wavelength=mc.trapWavelength, intensityFromDac=intensityFromDac_june_3_2020):
    return np.real(1e3 * ManifoldShift(2, 2, intensityFromDac(dacValue), '5S12', wavelength)/mc.k_B)
