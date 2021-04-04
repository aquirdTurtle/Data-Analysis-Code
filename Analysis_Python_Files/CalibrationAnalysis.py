# right now mostly just a copy paste of misc functions and definitions being used for calibration analysis
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from . import ExpFile as exp
from . import AnalysisHelpers as ah
from . import Miscellaneous as misc
from . import MarksConstants as mc
from importlib import reload
from . import MatplotlibPlotters as mp
from . import PictureWindow as pw
from . import CalibrationAnalysis as ca
# It's important to explicitly import calPoint here or else pickling doesn't work.
from .fitters.Gaussian import dip, double_dip, bump, bump2, bump3, bump2r, gaussian, bump3_Sym
from .fitters.Sinc_Squared import sinc_sq3_Sym, sinc_sq
from .fitters import decaying_cos, exponential_decay_fixed_limit as decay, linear, LargeBeamMotExpansion
from . import LightShiftCalculations as lsc
import matplotlib.pyplot as plt
import IPython.display
import matplotlib.dates as mdates
import pickle

@dataclass
class calPoint:
    value: float 
    error: np.ndarray
    timestamp: datetime
        
# ValueError: time data '2021:February:12:3.7' does not match format '%Y:%B:%d:%H:%M'
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
                except ValueError:
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

def std_MOT_NUMBER(calData):
    with exp.ExpFile() as file:
        file.open_hdf5('MOT_NUMBER')
        exposureTime = file.f['Basler']['Exposure-Time'][0]*20*1e-6
    res = mp.plotMotNumberAnalysis( 'MOT_NUMBER', motKey=np.arange(0,10,0.1), exposureTime=exposureTime,
                                    window=pw.PictureWindow(30,80,30,80) );
    dt = exp.getStartDatetime("MOT_NUMBER")
    if (not (res[-2][-1] == np.zeros(res[-2][-1].shape)).all()) and not (res[1] < 0.1):
        calData['MOT_Size'] = ca.calPoint(res[0], 0, dt)
        calData['MOT_FillTime'] = ca.calPoint(res[1], res[3], dt)
    else:
        raise ValueError('BAD DATA!!!!')
    return calData, [res[-1]]

def std_BASIC_SINGLE_ATOMS(calData, atomLocations=[2,2,3,7,1]):
    res = mp.Survival("BASIC_SINGLE_ATOMS", atomLocations, forceNoAnnotation=True);
    dt = exp.getStartDatetime("BASIC_SINGLE_ATOMS")
    avgVals = [np.mean(vals) for vals in misc.transpose(res['Initial_Populations'])]
    calData['Loading'] = ca.calPoint(np.max(avgVals), np.std(misc.transpose(res['Initial_Populations'])[np.argmax(avgVals)]),dt)
    calData['ImagingSurvival'] = ca.calPoint(res['Average_Transfer'][0],res['Average_Transfer_Err'][0],dt)
    return calData, res['Figures']

def std_MOT_TEMPERATURE(calData, **plotMotTempArgs):
    res = mp.plotMotTemperature('MOT_TEMPERATURE', reps=15, fitWidthGuess=15, **plotMotTempArgs);
    dt = exp.getStartDatetime("MOT_TEMPERATURE")
    calData['MOT_Temperature'] = ca.calPoint(res[2], res[3], dt)
    return calData, res[-1]

def std_RED_PGC_TEMPERATURE(calData, **plotMotTempArgs):
    res = mp.plotMotTemperature('RED_PGC_TEMPERATURE', reps=15, fitWidthGuess=3, temperatureGuess=20e-6,
                                **plotMotTempArgs);
    dt = exp.getStartDatetime("RED_PGC_TEMPERATURE")
    calData['RPGC_Temperature'] = ca.calPoint(res[2], res[3], dt)
    return calData, res[-1]

def std_GREY_MOLASSES_TEMPERATURE(calData, **plotMotTempArgs):
    res = mp.plotMotTemperature('GREY_MOLASSES_TEMPERATURE', reps=15, fitWidthGuess=20, lastDataIsBackground=True, temperatureGuess=20e-6,
                                **plotMotTempArgs);
    dt = exp.getStartDatetime("GREY_MOLASSES_TEMPERATURE")
    calData['LGM_Temperature'] = ca.calPoint(res[2], res[3], dt)
    return calData, res[-1]

def std_3DSBC_TOP_CARRIER_RAMAN_SPECTROSCOPY(calData, atomLocations=[2,2,3,7,1]):
    res = mp.Survival( "3DSBC_TOP_CARRIER_RAMAN_SPECTROSCOPY", atomLocations, fitModules=[bump], forceNoAnnotation=True);
    dt = exp.getStartDatetime("3DSBC_TOP_CARRIER_RAMAN_SPECTROSCOPY")
    fit = res['Average_Transfer_Fit']
    if np.isinf(fit['errs'][1]) or np.isnan(fit['errs'][1]):
        print('BAD CARRIER SCAN!')
    else:
        calData['RadialCarrierLocation'] = ca.calPoint(fit['vals'][1], fit['errs'][1], dt) 
    return calData, res['Figures']

def std_THERMAL_TOP_SIDEBAND_RAMAN_SPECTROSCOPY(calData, atomLocations=[2,2,3,7,1]):
    res = mp.Survival( "THERMAL_TOP_SIDEBAND_RAMAN_SPECTROSCOPY", atomLocations, 
                       forceNoAnnotation=True, fitModules=[bump2], 
                       fitguess=[[0,0.3,-150,10, 0.3, 150, 10]]);
    dt = exp.getStartDatetime("THERMAL_TOP_SIDEBAND_RAMAN_SPECTROSCOPY")
    fit = res['Average_Transfer_Fit']
    calData['ThermalTrapFreq'] = ca.calPoint((fit['vals'][-2] - fit['vals'][2]) / 2, np.sqrt(fit['errs'][-2]**2/4+fit['errs'][2]**2/4), dt) 
    calData['ThermalNbar'] = ca.calPoint(bump2.fitCharacter(fit['vals']), bump2.fitCharacterErr(fit['vals'], fit['errs']), dt)
    return calData, res['Figures']

def std_3DSBC_AXIAL_RAMAN_SPECTROSCOPY(calData, atomLocations=[2,2,3,7,1], centerGuess=None, **SurvivalArgs):
    guess=bump3_Sym.guess([],[])
    if centerGuess is not None:
        guess[-2] = centerGuess
    res = mp.Survival( "3DSBC_AXIAL_RAMAN_SPECTROSCOPY", atomLocations, forceNoAnnotation=True, 
                       fitModules=bump3_Sym, showFitDetails=False, fitguess=[guess], **SurvivalArgs);
    dt = exp.getStartDatetime("3DSBC_AXIAL_RAMAN_SPECTROSCOPY")
    fitV = res['Average_Transfer_Fit']['vals']
    fitE = res['Average_Transfer_Fit']['errs']
    calData['AxialTrapFreq'] = ca.calPoint(fitV[-1]/2, fitE[-1]/2, dt) 
    calData['AxialCarrierLocation'] = ca.calPoint(fitV[-2], fitE[-2], dt) 
    calData['AxialNbar'] = ca.calPoint(bump3_Sym.fitCharacter(fitV), bump3_Sym.fitCharacterErr(fitV, fitE), dt)
    return calData, res['Figures']

def std_3DSBC_TOP_SIDEBAND_RAMAN_SPECTROSCOPY(calData, atomLocations=[2,2,3,7,1]):
    res = mp.Survival("3DSBC_TOP_SIDEBAND_RAMAN_SPECTROSCOPY", atomLocations, 
                      fitModules=[bump2], fitguess=[[0,0.3,-150,10, 0.3, 150, 10]], forceNoAnnotation=True);
    dt = exp.getStartDatetime("3DSBC_TOP_SIDEBAND_RAMAN_SPECTROSCOPY")
    fvals = res['Average_Transfer_Fit']['vals']
    ferrs = res['Average_Transfer_Fit']['errs']
    calData['RadialTrapFreq'] = ca.calPoint((fvals[-2]-fvals[2])/2, np.sqrt(ferrs[-2]**2/4+ferrs[2]**2/4), dt) 
    calData['RadialNbar'] = ca.calPoint(bump2.fitCharacter(fvals), bump2.fitCharacterErr(fvals, ferrs), dt) 
    if calData["AxialTrapFreq"] is not None:
        nur = calData["RadialTrapFreq"].value
        nuax = calData["AxialTrapFreq"].value
        calData["SpotSize2Freqs"] = ca.calPoint(ca.getWaistFromBothFreqs(nur,nuax), 0, dt)
    return calData, res['Figures']

def std_DEPTH_MEASUREMENT_DEEP(calData, atomLocations=[2,2,3,7,1]):
    res = mp.Survival("DEPTH_MEASUREMENT_DEEP", atomLocations, fitModules=dip, forceNoAnnotation=True, showFitDetails=True);
    dt = exp.getStartDatetime("DEPTH_MEASUREMENT_DEEP")
    fvals = res['Average_Transfer_Fit']['vals']
    ferrs = res['Average_Transfer_Fit']['errs']
    calData['DeepScatteringResonance'] = ca.calPoint(fvals[1], ferrs[1], dt) 
    rdepth = lsc.trapDepthFromDac(calData['DeepScatteringResonance'].value)
    calData['DeepDepth'] = ca.calPoint(rdepth, abs(rdepth-lsc.trapDepthFromDac(calData['DeepScatteringResonance'].value+calData['DeepScatteringResonance'].error)), dt) 
    calData['RamanDepth'] = calData['DeepDepth']
    if calData["RadialTrapFreq"] is not None:
        calData['SpotSizeRadialDepth'] = ca.calPoint(ca.getWaist_fromRadialFreq(calData["RadialTrapFreq"].value*1e3, -rdepth ),
                                                   ca.getWaist_fromRadialFreq_err(calData["RadialTrapFreq"].value*1e3, 
                                                                               -rdepth, calData["RadialTrapFreq"].error*1e3, calData['RamanDepth'].error) ,dt)
    if calData["AxialTrapFreq"] is not None:
        calData['SpotSizeAxDepth'] = ca.calPoint(ca.getWaist_fromAxialFreq(calData["AxialTrapFreq"].value*1e3, -calData['RamanDepth'].value ),0,dt)
    calData['IndvDeepScatteringResonances'] = [None for _ in res['Transfer_Fits']]
    calData['IndvRamanDepths'] = [None for _ in res['Transfer_Fits']]
    for fitn, fit in enumerate(res['Transfer_Fits']):
        fvals = fit['vals']
        ferrs = fit['errs']
        pt = ca.calPoint(fvals[1], ferrs[1], dt) 
        calData['IndvDeepScatteringResonances'][fitn] = pt
        rdepth = lsc.trapDepthFromDac(pt.value)
        calData['IndvRamanDepths'][fitn] = ca.calPoint(rdepth, abs(rdepth-lsc.trapDepthFromDac(pt.value+pt.error)), dt)
    return calData, res['Figures']

def std_DEPTH_MEASUREMENT_SHALLOW(calData, atomLocations=[2,2,3,7,1]):
    res = mp.Survival("DEPTH_MEASUREMENT_SHALLOW", atomLocations, fitModules=dip, forceNoAnnotation=True, showFitDetails=True);
    dt = exp.getStartDatetime("DEPTH_MEASUREMENT_SHALLOW")
    fvals = res['Average_Transfer_Fit']['vals']
    ferrs = res['Average_Transfer_Fit']['errs']
    calData['ShallowScatteringResonance'] = ca.calPoint(fvals[1], ferrs[1], dt)
    # bit of a cheap error calculation here
    calData['ShallowDepth'] = ca.calPoint(lsc.trapDepthFromDac(calData['ShallowScatteringResonance'].value),
                                        abs(lsc.trapDepthFromDac(calData['ShallowScatteringResonance'].value)
                                            -lsc.trapDepthFromDac(calData['ShallowScatteringResonance'].value+calData['ShallowScatteringResonance'].error)), 
                                        dt) 
    if calData['DeepScatteringResonance'] is not None and calData['DeepDepth'] is not None and calData['ShallowDepth'] is not None:
        calData['ResonanceDelta'] = ca.calPoint(calData['DeepScatteringResonance'].value - calData['ShallowScatteringResonance'].value, 
                                              np.sqrt(calData['DeepScatteringResonance'].error**2+calData['ShallowScatteringResonance'].error**2), dt)
        calData['ResonanceDepthDelta'] = ca.calPoint( calData['DeepDepth'].value-calData['ShallowDepth'].value, 
                                                    np.sqrt(calData['DeepDepth'].error**2+calData['ShallowDepth'].error**2), dt)
    return calData, res['Figures']

def std_LIFETIME_MEASUREMENT(calData, atomLocations=[2,2,3,7,1]):
    decay.limit = 0
    res = mp.Survival("LIFETIME_MEASUREMENT", atomLocations, fitModules=decay, forceNoAnnotation=True);
    dt = exp.getStartDatetime("LIFETIME_MEASUREMENT")
    fit = res['Average_Transfer_Fit']
    calData['LifeTime'] = ca.calPoint(fit['vals'][1], fit['errs'][1], dt)
    return calData, res['Figures']

def getInitCalData():
    ea = None
    return {"Loading":ea,"ImagingSurvival":ea,"MOT_Size":ea,"MOT_FillTime":ea, "MOT_Temperature":ea,
           "RPGC_Temperature":ea,"LGM_Temperature":ea,"ThermalTrapFreq":ea, "ThermalNbar":ea, 
           "AxialTrapFreq":ea, "AxialCarrierLocation":ea, "AxialNbar":ea, "RadialTrapFreq":ea,
           "RadialNbar":ea, "RadialCarrierLocation":ea, "DeepScatteringResonance":ea, "DeepDepth":ea, 
           "ShallowScatteringResonance":ea, "ShallowDepth":ea, "ResonanceDelta":ea, "SpotSize2Freqs":ea, 
           "SpotSizeRadialDepth":ea, "SpotSizeAxDepth":ea, "RamanDepth":ea, "ResonanceDepthDelta":ea, "Lifetime":ea }    

def std_analyzeAll(sCalData = getInitCalData(), displayResults=True, atomLocations=[2,2,3,7,1]):
    allErrs, allFigs = [],[]
    analysis_names = ["MOT_NUMBER", "MOT_TEMPERATURE", "RED_PGC_TEMPERATURE", "GREY_MOLASSES_TEMPERATURE",
                    "BASIC_SINGLE_ATOMS","3DSBC_TOP_CARRIER_RAMAN_SPECTROSCOPY",
                    "THERMAL_TOP_SIDEBAND_RAMAN_SPECTROSCOPY","3DSBC_AXIAL_RAMAN_SPECTROSCOPY",
                     "3DSBC_TOP_SIDEBAND_RAMAN_SPECTROSCOPY","DEPTH_MEASUREMENT_DEEP","DEPTH_MEASUREMENT_SHALLOW",
                    "LIFETIME_MEASUREMENT"]
    for std_func in [std_MOT_NUMBER, std_MOT_TEMPERATURE, std_RED_PGC_TEMPERATURE, std_GREY_MOLASSES_TEMPERATURE,
                    std_BASIC_SINGLE_ATOMS,std_3DSBC_TOP_CARRIER_RAMAN_SPECTROSCOPY,
                    std_THERMAL_TOP_SIDEBAND_RAMAN_SPECTROSCOPY,std_3DSBC_AXIAL_RAMAN_SPECTROSCOPY,
                     std_3DSBC_TOP_SIDEBAND_RAMAN_SPECTROSCOPY,std_DEPTH_MEASUREMENT_DEEP,std_DEPTH_MEASUREMENT_SHALLOW,
                    std_LIFETIME_MEASUREMENT]:
        #with open('temp.txt','a') as file:
        #    file.write(str(std_func) + ', ' + str(sCalData['AxialCarrierLocation']) + '\n')
        try:
            if std_func in [std_MOT_NUMBER, std_MOT_TEMPERATURE, std_RED_PGC_TEMPERATURE, std_GREY_MOLASSES_TEMPERATURE]:
                sCalData, figures = std_func(sCalData)
            else:
                sCalData, figures = std_func(sCalData, atomLocations)
            for fig in figures:
                plt.close(fig)
            allFigs.append(figures)
            allErrs.append(None)
        except Exception as error:
            print("Failed to do calibration: ", std_func, error)
            allFigs.append([])
            allErrs.append(error)
    IPython.display.clear_output()
    if displayResults:
        assert(len(analysis_names) == len(allFigs))
        for name, figs, err in zip(analysis_names, allFigs, allErrs):
            IPython.display.display(IPython.display.Markdown('### ' + name))
            for fig in figs:
                IPython.display.display(fig)
            if err is not None:
                print(err)
    return sCalData


def plotCalData(ax, dataV, pltargs={}, sf=1):
    err = np.array(np.array([data.error for data in dataV]).tolist())
    if len(np.array(err).shape) == 2:
        err = misc.transpose(err)
    ax.errorbar([data.timestamp for data in dataV], [data.value*sf for data in dataV], 
                yerr=err, **pltargs)
    
def addAnnotations(ax):
    ax.axvline(datetime(2020,6,20), color='k')
    ax.text(datetime(2020,6,20),0.5, 'Chimera 2.0', transform=ax.get_xaxis_transform(), color='k', rotation=-90)
    ax.axvline(datetime(2020,7,11), color='k')
    ax.text(datetime(2020,7,11), 0.5, "Mark's Vacation", transform=ax.get_xaxis_transform(), color='k', rotation=-90)
    ax.axvline(datetime(2020,7,21), color='k')
    ax.text(datetime(2020,7,21), 0.5, "Changed to 7 Atoms", transform=ax.get_xaxis_transform(), color='k', rotation=-90)
    ax.axvline(datetime(2020,9,21), color='k')
    ax.text(datetime(2020,9,21), 0.5, "Sprout Issues;\nSwitch To TA", transform=ax.get_xaxis_transform(), color='k', rotation=-90)
    ax.axvline(datetime(2020,12,1), color='k')
    ax.text(datetime(2020,12,1), 0.5, "Switch to GSBC", transform=ax.get_xaxis_transform(), color='k', rotation=-90)

def setAxis(ax, dataV=None, color='r', pad=0, annotate=True):
    ax.spines['right'].set_color(color)
    ax.yaxis.label.set_color(color)
    ax.tick_params(axis='y', colors=color)
    ax.tick_params(axis='y', which='major', pad=pad)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=0.15)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    if annotate:
        addAnnotations(ax)

def makeLegend(ax, loc = 'upper left', bbox=(0,1.05)):
    leg = ax.legend(loc=loc, bbox_to_anchor=bbox, ncol=3, framealpha=1)
    for text in leg.get_texts():
        text.set_color("k")

def simpleCheck(data, bounds, msg, errBound=np.inf):
    susData = []
    for dp in data:
        err = dp.error[0] if type(dp.error) == list or type(dp.error) == type(np.array([])) else dp.error
        if (bounds[0] is not None and bounds[0] > dp.value) or (bounds[1] is not None and dp.value > bounds[1]):
            susData.append(dp)        
        elif errBound < err:
            susData.append(dp)
    if susData:
        print(msg, ';\tReasonable data bounds:', bounds, '\tReasonable Error:', errBound)
        for sd in susData:
            print('\t',sd)
        
def checkData(calData):
    # check for zero errors, usually something fit but curve_fit couldn't figure out error
    susData = []
    for dataname in calData.keys():
        if dataname in ['MOT_Size', 'SpotSize2Freqs', 'SpotSizeAxDepth']:
            continue
        for val in calData[dataname]:
            err = val.error[0] if type(val.error) == list or type(val.error) == type(np.array([])) else val.error
            if  err == 0 or np.isinf(err) or np.isnan(err):
                susData.append([dataname, val])
    if susData:
        print('Data with no error:')
        for data in susData:
            print('\t',data)
    
    simpleCheck(calData['MOT_Size'], [1000,500000],'Suspicious MOT size data')
    simpleCheck(calData['MOT_FillTime'], [0.5,10],'Suspicious MOT Fill time data')
    simpleCheck(calData['MOT_Temperature'], [10,350],'Suspicious MOT Temperature data', 110)
    simpleCheck(calData['RPGC_Temperature'], [3,50],'Suspicious RPGC Temperature data', 50)
    simpleCheck(calData['LGM_Temperature'], [3,50],'Suspicious LGM Temperature data', 50)
    
    simpleCheck(calData['Loading'], [0.3,0.95],'Suspicious Atom Loading Rate', 0.1)
    simpleCheck(calData['ImagingSurvival'], [0, 1],'Suspicious Imaging Survival Data', 0.1)
    simpleCheck(calData['LifeTime'], [1000,10000],'Suspicious Atom Lifetime')
    
    simpleCheck(calData['ThermalTrapFreq'], [100, 200], 'Suspicious Thermal Radial Trap Frequency')
    simpleCheck(calData['RadialTrapFreq'], [100, 200], 'Suspicious Radial Trap Frequency')
    simpleCheck(calData['AxialTrapFreq'], [20, 45], 'Suspicious Axial Trap Frequency', 5)
    
    simpleCheck(calData['ShallowDepth'], [-0.5, 0.2], 'Suspicious Shallow Depth')
    simpleCheck(calData['DeepDepth'], [-1.5, -0.5], 'Suspicious Deep Depth')
    simpleCheck(calData['RamanDepth'], [-1.5, -0.5], 'Suspicious Raman Depth')
    
    simpleCheck(calData['SpotSize2Freqs'], [800e-9, 1100e-9], 'Suspicious Radial+Axial Spot Size Value')
    simpleCheck(calData['SpotSizeRadialDepth'], [600e-9, 900e-9], 'Suspicious Radial+Depth Spot Size Value')
    simpleCheck(calData['SpotSizeAxDepth'], [700e-9, 1000e-9], 'Suspicious Axial+Depth Spot Size Value')
    
    simpleCheck(calData['ThermalNbar'], [1, None], 'Suspicious Radial+Axial Spot Size Value')
    simpleCheck(calData['AxialNbar'], [0, 1], 'Suspicious Axial Nbar')
    simpleCheck(calData['RadialNbar'], [0, 1], 'Suspicious Radial Nbar')
    
    simpleCheck(calData['RadialCarrierLocation'], [6.8309, 6.831], 'Suspicious Carrier Location', 0.01)
    
def loadAllCalData(checkDates=False):
    ea = np.array([]) # empty array
    calData = {"Loading":ea,"ImagingSurvival":ea,"MOT_Size":ea,"MOT_FillTime":ea, "MOT_Temperature":ea,
               "RPGC_Temperature":ea,"LGM_Temperature":ea,"ThermalTrapFreq":ea, "ThermalNbar":ea, 
               "AxialTrapFreq":ea, "AxialCarrierLocation":ea, "AxialNbar":ea, "RadialTrapFreq":ea,
               "RadialNbar":ea, "RadialCarrierLocation":ea, "DeepScatteringResonance":ea, "DeepDepth":ea, 
               "ShallowScatteringResonance":ea, "ShallowDepth":ea, "ResonanceDelta":ea, "SpotSize2Freqs":ea, 
               "SpotSizeRadialDepth":ea, "SpotSizeAxDepth":ea, "RamanDepth":ea, "ResonanceDepthDelta":ea, "IndvRamanDepths":ea, "LifeTime": ea }
    for year_ in ['2020','2021']:
        print(year_)
        for month_ in ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november','december']:
            print(month_,end=',')
            for d_ in range(1,32):
                exp.setPath(str(d_),month_, year_)
                try:
                    with open(exp.dataAddress + 'CalibrationData.p', 'rb') as handle:
                        readCalData = pickle.loads(handle.read())
                    for key in calData.keys():
                        if key in readCalData:
                            if readCalData[key] is not None and readCalData[key] != []: 
                                if checkDates:
                                    if type(readCalData[key]) == list:
                                        for data in readCalData[key]:
                                            if data.timestamp.day != d_:
                                                print(data.timestamp, 'data for date',year_,month_,d_, "dates don't match!")
                                                continue
                                    else:
                                        if readCalData[key].timestamp.day != d_:
                                            print(readCalData[key].timestamp, 'data for date',year_,month_,d_, "dates don't match!")
                                            continue
                                # the data gets set to none if, e.g. there's a bad calibation set which was never fixed that day. 
                                calData[key] = np.append(calData[key], readCalData[key])
                except OSError:
                    pass
                except EOFError:
                    pass
    return calData
        
    
def standardPlotting(pltData, which="all", useSmartYLims=True):
    fs = 20
    fig, axs = plt.subplots(8 if which=="all" else 1,1, figsize=(20,35) if which=="all" else (20,5))
    plt.subplots_adjust(hspace=0.3)
    if which == "all" or which == 0:
        motAx = axs[0] if which == "all" else axs
        motAx_2 = motAx.twinx()
        motAx_3 = motAx.twinx()
        ca.plotCalData(motAx, pltData['MOT_Temperature'], {'marker':'o','label':'MOT_Temperature','capsize':5, 'color':'r'})
        ca.plotCalData(motAx_2, pltData['MOT_FillTime'], {'marker':'o','label':'MOT_FillTime','capsize':5, 'color':'b'})
        ca.plotCalData(motAx_3, pltData['MOT_Size'], {'marker':'o','label':'MOT_Size','capsize':5, 'color':'g'})
        motAx_2.set_title('MOT Characteristics', color='k', fontsize=fs)
        motAx.set_ylabel(r'MOT Temperature ($\mu K$)', fontsize=fs)
        motAx_2.set_ylabel(r'MOT Fill-Time', fontsize=fs)
        motAx_3.set_ylabel(r'MOT Size', fontsize=fs)
        ca.makeLegend(motAx)
        ca.makeLegend(motAx_2, 'upper right', (1,1.08))
        ca.makeLegend(motAx_3, 'upper right', (1,1.15))
        motAx.set_ylabel(r'MOT Temperature ($\mu K$)', fontsize=fs)
        motAx_2.set_ylabel(r'MOT Fill Time (Seconds)', fontsize=fs)
        motAx_3.set_ylabel(r'MOT Size (# Atoms)', fontsize=fs)
        ca.setAxis(motAx, pltData['MOT_Temperature'], color='r', pad=0)
        ca.setAxis(motAx_2, color='b', pad=0)
        ca.setAxis(motAx_3, color='g', pad=50)
    if which == "all" or which == 1:
        ax2 = axs[1] if which == "all" else axs
        ax2_MOT = ax2.twinx()
        ca.plotCalData(ax2_MOT, pltData['MOT_Temperature'], {'marker':'o','label':'MOT_Temperature','capsize':5, 'color':'r'})
        ca.plotCalData(ax2, pltData['RPGC_Temperature'], {'marker':'o','label':'RPGC_Temperature','capsize':5, 'color':'b'})
        ca.plotCalData(ax2, pltData['LGM_Temperature'], {'marker':'o','label':'LGM_Temperature','capsize':5, 'color':'g'})
        ax2.set_ylim(max(0,ax2.get_ylim()[0]),min(1e2,ax2.get_ylim()[1]))
        ax2_MOT.set_ylim(max(0,ax2_MOT.get_ylim()[0]),min(1e3,ax2_MOT.get_ylim()[1]))
        ca.makeLegend(ax2)
        ca.makeLegend(ax2_MOT, 'upper right', (1,1.05))
        ca.setAxis(ax2, color='k')
        ca.setAxis(ax2_MOT)
        ax2.set_title("Free Spacing Cooling Techniques", fontsize=fs)
        ax2_MOT.set_ylabel(r'MOT Temperature ($\mu K$)', fontsize=fs)
        ax2.set_ylabel(r'PGC Temperature ($\mu K$)', fontsize=fs)
    if which == "all" or which == 2:
        probAx = axs[2] if which == "all" else axs
        probAx.set_title('Basic Atom', color='k', fontsize=fs)
        lifeAx = probAx.twinx()
        ca.plotCalData(probAx, pltData['Loading'], {'marker':'o','label':'Loading','capsize':5})
        ca.plotCalData(probAx, pltData['ImagingSurvival'], {'marker':'o','label':'ImagingSurvival','capsize':5})
        ca.plotCalData(lifeAx, pltData['LifeTime'], {'marker':'o','label':'Lifetime','capsize':5, 'color':'r'})
        probAx.set_ylim(0,1)
        probAx.set_ylabel('Probability (/1)', fontsize=fs)
        lifeAx.set_ylabel('Lifetime (ms)', fontsize=fs)
        lifeAx.set_ylim(0,min(10000,lifeAx.get_ylim()[1]))
                
        ca.setAxis(probAx, color='k')
        ca.makeLegend(probAx)
        ca.setAxis(lifeAx, color='r')
        ca.makeLegend(lifeAx)
    if which == "all" or which == 3:
        trapAx_rfreqs = axs[3] if which == "all" else axs
        trapAx_axfreqs = trapAx_rfreqs.twinx()
        ca.plotCalData(trapAx_rfreqs, pltData['ThermalTrapFreq'], 
                       {'linestyle':':','marker':'o','label':'ThermalTrapFreq','capsize':5, 'color':'r'})
        ca.plotCalData(trapAx_rfreqs, pltData['RadialTrapFreq'], 
                       {'marker':'o','label':'RadialTrapFreq','capsize':5, 'color':'b'})
        ca.plotCalData(trapAx_axfreqs, pltData['AxialTrapFreq'], 
                       {'marker':'o','label':'AxialTrapFreq','capsize':5, 'color':'c'})
        trapAx_rfreqs.set_ylabel('Radial Trap Frequencies', fontsize=fs)
        trapAx_axfreqs.set_ylabel('Axial Trap Frequencies', fontsize=fs)
        trapAx_rfreqs.set_ylim(120,170)
        trapAx_axfreqs.set_ylim(20,45)
        ca.makeLegend(trapAx_rfreqs, 'upper right', (1, 1.1))
        ca.makeLegend(trapAx_axfreqs, 'upper right', (1,1.2))
        ca.setAxis(trapAx_rfreqs, color='b')
        ca.setAxis(trapAx_axfreqs, color='c', pad=100)
    if which == "all" or which == 4:
        trapAx_depths = axs[4] if which == "all" else axs
        trapAx_resonances = trapAx_depths.twinx()
        ca.plotCalData(trapAx_depths, pltData['ShallowDepth'], 
                       {'linestyle':':','marker':'o','label':'ShallowDepth','capsize':5, 'color':'g'})
        ca.plotCalData(trapAx_depths, pltData['DeepDepth'], 
                       {'linestyle':'--','marker':'o','label':'DeepDepth','capsize':5, 'color':'g'})
        ca.plotCalData(trapAx_depths, pltData['RamanDepth'], 
                       {'linestyle':'-','marker':'o','label':'RamanDepth','capsize':5, 'color':'g'})
        trapAx_depths.set_ylabel('Depth (V)', fontsize=fs)
        trapAx_depths.set_title('Trap Characterization', color='k', fontsize=fs)
        setAxis(trapAx_depths, color='g')
        makeLegend(trapAx_depths, 'upper left', (0,1.15))
        ca.plotCalData(trapAx_resonances, pltData['ResonanceDepthDelta'], {'marker':'o','label':'ResonanceDelta','capsize':5, 'color':'k'})
        trapAx_resonances.set_ylabel('Depth Delta (V)', fontsize=fs)
        trapAx_resonances.yaxis.tick_left()
        trapAx_resonances.yaxis.set_label_position("left")
        trapAx_resonances.set_ylim(-1.5,0)
        makeLegend(trapAx_resonances, 'upper left', (0,1.1))
        setAxis(trapAx_resonances, color='k', pad=50)
    if which == "all" or which == 5:
        trapAx_sizes = axs[5] if which == "all" else axs
        ca.plotCalData(trapAx_sizes, pltData['SpotSize2Freqs'],
                       {'linestyle':':','marker':'o','label':'SpotSize2Freqs','capsize':5, 'color':'r'}, sf=1e9)
        ca.plotCalData(trapAx_sizes, pltData['SpotSizeRadialDepth'],
                       {'linestyle':'-','marker':'o','label':'SpotSizeRadialDepth','capsize':5, 'color':'r'}, sf=1e9)
        ca.plotCalData(trapAx_sizes, pltData['SpotSizeAxDepth'],
                       {'linestyle':'--','marker':'o','label':'SpotSizeAxDepth','capsize':5, 'color':'r'}, sf=1e9)
        trapAx_sizes.set_ylabel('Spot Sizes (nm)', fontsize=fs)
        setAxis(trapAx_sizes, color='r', pad=50)
        makeLegend(trapAx_sizes, 'upper right', (1,1.15))
    if which == "all" or which == 6:
        nbar_ax = axs[6] if which == "all" else axs
        nbar_ax.set_title('NBar Values', color='k', fontsize=fs)
        plotCalData(nbar_ax, pltData['ThermalNbar'], {'marker':'o','label':'ThermalNbar','capsize':5})
        plotCalData(nbar_ax, pltData['AxialNbar'], {'marker':'o','label':'AxialNbar','capsize':5})
        plotCalData(nbar_ax, pltData['RadialNbar'], {'marker':'o','label':'RadialNbar','capsize':5})
        nbar_ax.set_ylim(0,2)
        makeLegend(nbar_ax)
        setAxis(nbar_ax)
    if which == "all" or which == 7:
        carrierAx = axs[7] if which == "all" else axs
        carrierAx.set_title('Carrier Drift', color='k', fontsize=fs)
        plotCalData(carrierAx, pltData['RadialCarrierLocation'], {'marker':'o','label':'Radial Carrier Location','capsize':5})
        setAxis(carrierAx, color='k')
        makeLegend(carrierAx, loc="upper right", bbox=(1,1.05))
        carrierAx.axvline(datetime(2020,6,16), color='k')
        carrierAx.text(datetime(2020,6,16),0.5, 'Started Recording\nAbosolute Freq', transform=carrierAx.get_xaxis_transform(), color='k');
        if useSmartYLims:
            carrierAx.set_ylim(max(6.8308,carrierAx.get_ylim()[0]),min(6.8311,carrierAx.get_ylim()[1]))
        