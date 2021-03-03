# right now mostly just a copy paste of misc functions and definitions being used for calibration analysis
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from . import ExpFile as exp
from . import AnalysisHelpers as ah
from . import Miscellaneous as misc
from . import MarksConstants as mc
from importlib import reload

@dataclass
class calPoint:
    value: float 
    error: np.ndarray
    timestamp: datetime

def loadAllTemperatureData():
    times, temps = [], [[],[],[],[]]
    for year_ in ['2020', '2021']:
        print('\n',year_)
        for month_ in ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']:
            print(month_,end=', ')
            for d in range(1,31):
                day_ = str(d)
                exp.setPath(day_,month_,year_)
                reload(ah)
                try:
                    xpts, data = ah.Temperature(show=False)
                    for x in xpts:
                        times.append(datetime.strptime(year_+':'+month_+':'+day_ + ':' + x, '%Y:%B:%d:%H:%M'))
                    for i in range(4):
                        temps[i] += list(data[3*(i+1)])
                except FileNotFoundError:
                    pass
    cTemps, cTimes = [[],[]]
    for i, (time, ts) in enumerate(zip(times, misc.transpose(temps))):
        bad_ = False
        for t in ts:
            try:
                x = float(t)
                if x <= 0:
                    raise ValueError()
            except ValueError:
                bad_ = True
                break
        if not bad_:
            cTemps.append([float(temp) for temp in ts])
            cTimes.append(time)
    return cTemps, cTimes        
        
def getWaist_fromRadialFreq(freq_r, depth_mk):
    """
    :@param freq: the radial trap frequency in non-angular Hz.
    :@param depth: the trap depth in mK
    """
    V = mc.k_B * depth_mk * 1e-3
    omega_r = 2*np.pi*freq_r
    return np.sqrt(4*V/(mc.Rb87_M * omega_r**2))


def getWaist_fromRadialFreq_err(freq_r, depth_mk, freq_r_err, depth_mk_err):
    m = mc.Rb87_M
    omega_r = 2*np.pi*freq_r
    V = depth_mk*mc.k_B*1e-3
    t1 = np.sqrt(4/(m * omega_r**2))*depth_mk_err*mc.k_B*1e-3
    t2 = np.sqrt(8*V/(m*omega_r**3))*freq_r_err*2*np.pi
    return np.sqrt(t1**2+t2**2)

def getWaistFromBothFreqs(nu_r, nu_z):
    return 850e-9/(np.sqrt(2)*np.pi)*(nu_r/nu_z)

def getWaist_fromAxialFreq(freq_z, depth_mk):
    """
    :@param freq: the radial trap frequency in non-angular Hz.
    :@param depth: the trap depth in mK
    """
    V = mc.k_B * depth_mk * 1e-3
    omega_z = 2*np.pi*freq_z
    wavelength=850e-9
    return (2*V/mc.Rb87_M)**(1/4)*np.sqrt(wavelength /(np.pi*omega_z))



