__version__ = "1.4"

import csv
import os # for linesep
import pandas as pd
from numpy import array as arr
import h5py as h5
from inspect import signature
import uncertainties as unc
import uncertainties.unumpy as unp
from warnings import warn
import matplotlib.pyplot as plt

import scipy.optimize as opt
import scipy.special as special
import scipy.interpolate as interp
import warnings

#from . import MatplotlibPlotters as mp
from . import MarksConstants as mc
from . import Miscellaneous as misc
from .Miscellaneous import what
from copy import copy, deepcopy
from .fitters import ( #cython_poissonian as poissonian, 
                      poissonian as poissonian,
                      FullBalisticMotExpansion, LargeBeamMotExpansion, exponential_saturation )
from .fitters.Gaussian import double as double_gaussian, gaussian_2d, arb_2d_sum, bump

from . import MainAnalysis as ma
from . import AtomThreshold
from . import ThresholdOptions
from . import ExpFile as exp
from .ExpFile import ExpFile, dataAddress
from .TimeTracker import TimeTracker
from . import PictureWindow as pw
from . import TransferAnalysisOptions as tao


import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

from statsmodels.stats.proportion import proportion_confint as confidenceInterval

import imageio
import numpy as np
from numpy import array as arr
import matplotlib as mpl
import matplotlib.cm
from IPython.display import Image, HTML, display

def softwareBinning(binningParams, rawData):
    if binningParams is not None:
        sb = binningParams
        if len(np.array(rawData).shape) == 3: 
            if not ((rawData.shape[1]/sb[0]).is_integer()): 
                raise ValueError('Vertical size ' + str(rawData.shape[1]) +  ' not divisible by binning parameter ' + str(sb[0]))
            if not ((rawData.shape[2]/sb[1]).is_integer()):
                raise ValueError('Horizontal size ' + str(rawData.shape[2]) +  ' not divisible by binning parameter ' + str(sb[1]))
            rawData = rawData.reshape(rawData.shape[0], rawData.shape[1]//sb[0], sb[0], rawData.shape[2]//sb[1], sb[1]).sum(4).sum(2)
        elif len(np.array(rawData).shape) == 2:
            if not ((rawData.shape[0]/sb[0]).is_integer()): 
                raise ValueError('Vertical size ' + str(rawData.shape[0]) +  ' not divisible by binning parameter ' + str(sb[0]))
            if not ((rawData.shape[1]/sb[1]).is_integer()):
                raise ValueError('Horizontal size ' + str(rawData.shape[1]) +  ' not divisible by binning parameter ' + str(sb[1]))
            rawData = rawData.reshape(rawData.shape[0]//sb[0], sb[0], rawData.shape[1]//sb[1], sb[1]).sum(3).sum(1)
        else:
            raise ValueError('Raw data must either 2 or 3 dimensions')            
    return rawData
    
def windowImage(image, window):
    if len(np.array(image).shape) == 2:
        return image[window[0]:window[1], window[2]:window[3]]
    else: 
        return image[:,window[0]:window[1], window[2]:window[3]]

def makeVid(pics, gifAddress, videoType, fileAddress=None, dur=1, lim=None, includeCount=True, lowLim=None, 
            finLabels=[], finTxt="Atom Reservoir Depleted", vidMap='inferno', maxMult=1, offset=0,
            resolutionMult=1):
    infernoMap = [mpl.cm.inferno(i)[:-1] for i in range(256)]
    viridisMap = [mpl.cm.viridis(i)[:-1] for i in range(256)]
    magmaMap = [mpl.cm.magma(i)[:-1] for i in range(256)]
    hotMap = [mpl.cm.hot(i)[:-1] for i in range(256)]
    cividisMap = [mpl.cm.cividis(i)[:-1] for i in range(256)]
    if vidMap == 'inferno':
        vidMap = infernoMap
    if vidMap == 'viridis':
        vidMap = viridisMap
    if vidMap == 'cividisMap':
        vidMap = cividisMap
    
    # global count
    # select subsection
    if lim is None:
        lim = len(pics)
    if lowLim is None:
        lowLim = 0
    pics = pics[lowLim:lim]
    # normalize to rgb scale
    pics = pics - min(pics.flatten())
    pics = np.uint16(pics / max(pics.flatten()) * 256 * maxMult)
    pics = arr([[[int(elem) for elem in row] for row in pic] for pic in pics])
    pics = arr(pics-min(pics.flatten()) - offset)
    pics = [[[vidMap[elem] if elem < 256 and elem >= 0 else vidMap[255] if elem >= 256 else vidMap[0] 
              for elem in row] for row in pic] for pic in pics]
    images = []
    sequenceCount = 1
    offset = 0
    for picCount, pic in enumerate(pics):
        fig = plt.figure()
        fig.set_size_inches([9,9])
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.grid(False)
        ax.imshow(pic, aspect='equal')
        if includeCount:
            ax.text(-0.1, 0.1, str(picCount+1-offset), color='white', fontsize=40)
        if picCount+1 in finLabels:
            ax.text(1.5, 14, finTxt, color='r', fontsize=40)
        name = "temp"+str(picCount+1)+".png"
        plt.savefig(name)
        images.append(imageio.imread(name))
        if picCount+1 in finLabels:
            sequenceCount += 1
            offset = picCount+1
            for _ in range(4):
                images.append(imageio.imread(name))
        plt.close('all')
    # make bigger
    pics = [np.repeat(np.repeat(pic, resolutionMult, axis=0), resolutionMult, axis=1) for pic in pics]
    imageio.mimsave(gifAddress, images, format=videoType, duration=dur)

def collapseImage(im, avg=True):
    vAvg = np.zeros(len(im[0]))
    for r in im:
        vAvg += r
    if avg:
        vAvg /= len(im)
    
    hAvg = np.zeros(len(im))
    for c in misc.transpose(im):
        hAvg += c
    if avg:
        hAvg /= len(im[0])
    return hAvg, vAvg

def jeffreyInterval(m,num):
    # sigma = 1-0.6827 gives the standard "1 sigma" intervals.
    i1, i2 = confidenceInterval(round(m*num), num, method='jeffreys', alpha=1-0.6827)
    return (m - i1, i2 - m)

def findImageMaxima(im, neighborhood_size=20, threshold=1):
    data_max = filters.maximum_filter(im, neighborhood_size)
    maxima = (im == data_max)
    data_min = filters.minimum_filter(im, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2    
        y.append(y_center)
    print('Found ' + str(len(x)) + ' Maxima.')
    return [p for p in zip([int(x_) for x_ in x],[int(y_) for y_ in y])]

def fitManyGaussianImage(im, numGauss, neighborhood_size=20, threshold=1, direct=True, widthGuess=1):
    """
    Maxima finding is based on the answer to this question:
    https://stackoverflow.com/questions/9111711/get-coordinates-of-local-maxima-in-2d-array-above-certain-value
    """
    maximaLocs = findImageMaxima(im, neighborhood_size=neighborhood_size, threshold=threshold)
    if len(maximaLocs) != numGauss:
        raise ValueError("ERROR: didn't find the right number of maxima!")
    guess = [min(im.flatten())]
    for loc in maximaLocs:
        guess += [im[loc[1],loc[0]], loc[0], loc[1], widthGuess, widthGuess]
    xpts = np.arange(len(im[0]))
    ypts = np.arange(len(im))
    X,Y = np.meshgrid(xpts,ypts)
    zpts = arb_2d_sum.f((X,Y), *guess).reshape(X.shape)
    f, ax = subplots(1,5,figsize=(20,10))
    ax[0].imshow(im)
    ax[0].set_title('Orig')
    ax[1].imshow(zpts)
    ax[1].set_title('Guess')
    ax[2].imshow(im-zpts)
    ax[2].set_title('Guess-Diff')
    optParam, optCov = opt.curve_fit(arb_2d_sum.f, (X,Y), im.flatten(), p0=guess)
    zpts_fit = arb_2d_sum.f((X,Y), *optParam).reshape(X.shape)
    ax[3].imshow(zpts_fit)
    ax[3].set_title('Fit')
    ax[4].imshow(im-zpts_fit)
    ax[4].set_title('Fit-Diff')
    return optParam

def temperatureAnalysis( data, magnification, temperatureGuess=100e-6, **standardImagesArgs ):
    res = ma.standardImages(data, scanType="Time(ms)", majorData='fits', fitPics=True, manualAccumulation=True, quiet=True, **standardImagesArgs)
    (key, rawData, dataMinusBg, dataMinusAvg, avgPic, pictureFitParams, fitCov, plottedData, v_params, v_errs, h_params, h_errs, intRawData) = res
    # convert to meters, convert from sigma to waist
    waists = 2 * mc.baslerScoutPixelSize * np.sqrt((pictureFitParams[:, 3]**2+pictureFitParams[:, 4]**2)/2) * magnification
    # waists_1D = 2 * mc.baslerScoutPixelSize * np.sqrt((v_params[:, 2]**2+h_params[:, 2]**2)/2) * magnification
    waists_1D = 2 * mc.baslerScoutPixelSize * v_params[:, 2] * magnification
    # convert to s
    times = key / 1000
    temp, fitVals, fitCov = calcBallisticTemperature(times, waists / 2, guess = [*LargeBeamMotExpansion.guess()[:-1], temperatureGuess])
    temp_1D, fitVals_1D, fitCov_1D = calcBallisticTemperature(times, waists_1D / 2, guess = [*LargeBeamMotExpansion.guess()[:-1], temperatureGuess])
    return ( temp, fitVals, fitCov, times, waists, rawData, pictureFitParams, key, plottedData, dataMinusBg, v_params, v_errs, h_params, h_errs,
             waists_1D, temp_1D, fitVals_1D, fitCov_1D )

def motFillAnalysis( dataSetNumber, motKey, exposureTime, window=pw.PictureWindow(), sidemotPower=2.05, diagonalPower=8, motRadius=8 * 8e-6,
                     imagingLoss=0.8, detuning=10e6, **standardImagesArgs ):
    res = ma.standardImages(dataSetNumber, key=motKey, scanType="time (s)", window=window, quiet=True, **standardImagesArgs)
    motKey, rawData = res[0], res[1]
    intRawData = integrateData(rawData)
    try:
        fitParams, pcov = opt.curve_fit( exponential_saturation.f, motKey, intRawData,
                                         p0=[np.min(intRawData) - np.max(intRawData), 1 / 2, np.max(intRawData)] )
    except RuntimeError:
        print('MOT # Fit failed!')
        # probably failed because of a bad guess. Show the user the guess fit to help them debug.
        popt = [np.min(intRawData) - np.max(intRawData), 1 / 2, np.max(intRawData)]
    fitErr = np.sqrt(np.diag(pcov))
    motNum, fluorescence = computeMotNumber(sidemotPower, diagonalPower, motRadius, exposureTime, imagingLoss, -fitParams[0],
                                            detuning=detuning)
    return rawData, intRawData, motNum, fitParams, fluorescence, motKey, fitErr

def getTodaysTemperatureData():
    path = dataAddress + 'Temperature_Data.csv'
    df = pd.read_csv(path, header=None, sep=',| ', engine='python')
    return df

def Temperature(show=True):
    df = getTodaysTemperatureData()

    legends = ['1: Master Computer', '2: B236', '3: Auxiliary Table', '4: Main Exp. (Near Ion Pump)']
    xpts = [x[:5] for x in df[1]]
    if not show:
        return xpts, df
    fig = plt.figure(figsize=(30,15))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)
    ax1.clear()
    for ind, leg in zip(np.arange(3,13,3), legends):
        pltx, data = [], []
        for num, dp in enumerate(df[ind]):
            try:
                data.append(float(dp))
                pltx.append(xpts[num])
            except ValueError:
                print('Bad Value!', dp, xpts[num])
                pass
        ax1.plot(pltx, data, label=leg)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5,1.2),ncol=4, fontsize=10)
    ax1.set_ylabel('Temperature (C)')
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.clear()
    for ind, leg in zip(np.arange(4,14,3), legends):
        pltx, data = [], []
        for num, dp in enumerate(df[ind]):
            try:
                data.append(float(dp))
                pltx.append(xpts[num])
            except ValueError:
                print('Bad Value!', dp, xpts[num])
        ax2.plot(pltx, data, label=leg)
    ax2.set_ylabel('Humidity (%)')
    incr = int(len(xpts)/20)+1
    ax2.set_xticks(xpts[::incr])
    plt.xlabel('Time (hour:minute)')
    plt.xticks(rotation=75);
    return xpts, df, ax1, ax2

def splitData(data, picsPerSplit, picsPerRep, runningOverlap=0):
    data = np.reshape(data, (picsPerSplit, int(data.shape[1]/picsPerSplit), data.shape[2], data.shape[3]))
    return data, int(picsPerSplit/picsPerRep), np.arange(0,int(data.shape[1]))

def parseRearrangeInfo(addr, limitedMoves=-1):
    moveList = []
    readyForAtomList = False
    with open(addr) as centerLog:
        for i, line in enumerate(centerLog):
            # this number depends on the size of the target matrix. it is height + 2.
            if i < 12:
                continue
            txt = line.split(' ')
            if txt[0] == 'Rep':
                moveList.append({'Rep': txt[2]})
                continue
            if txt[0] == 'Moves:\n':
                continue
            if txt[0] == 'Source:':
                moveList[-1]['Source'] = []
                for loc in txt[1:]:
                    if loc != ';' and loc != '\n':
                        moveList[-1]['Source'].append(int(loc[:-1]))
                continue
            if txt[0] == 'Target' and txt[1] == 'Location:':
                moveList[-1]['Target-Location'] = txt[2] + ',' + txt[3]
                moveList[-1]['Moves'] = []
                continue
            if not readyForAtomList:
                if len(moveList[-1]['Moves']) >= limitedMoves and limitedMoves != -1:
                    continue
                moveList[-1]['Moves'].append({'Flashed': bool(int(txt[1])), 'Direction': txt[2]})
                moveList[-1]['Moves'][-1]['Atoms'] = []
                readyForAtomList = True
                continue
            if len(txt) != 1:
                if len(moveList[-1]['Moves']) >= limitedMoves+1 and limitedMoves != -1:
                    continue
                moveList[-1]['Moves'][-1]['Atoms'].append((txt[0], txt[1]))
            else:
                # this blank line happens between moves.
                readyForAtomList = False
    return moveList

def handleKeyModifications(hdf5Key, numVariations, keyInput=None, keyOffset=0, groupData=False, keyConversion=None, keySlice=None ):
    """
    keySlice: mostly for handling the case of two concurrent variables that are varying the same, so it's not quite a multidimensional
    slice but I need to specify which value to use for the x-axis etc.
    """
    key = None
    key = hdf5Key if keyInput is None else keyInput
    if key is None: 
        key = arr([0]) if numVariations == 1 else arr([])
    if groupData:
        key = [0]
    if len(key.shape) == 1:
        key -= keyOffset
    if keyConversion is not None:
        key = [keyConversion.f(k) for k in key]
        #keyName += "; " + keyConversion.units()
    if len(key) != numVariations:
        raise ValueError("ERROR: The Length of the key doesn't match the data found. "
                         "Did you want to use a transfer-based function instead of a population-based function? Key:", 
                         len(key), "vars:", numVariations)
    if keySlice is not None:
        key = key[:,keySlice]
    return key

def modFitFunc(sign, hBiasIn, vBiasIn, depthIn, *testBiases):
    newDepths = extrapolateModDepth(sign, hBiasIn, vBiasIn, depthIn, testBiases)
    if newDepths is None:
        return 1e9
    return np.std(newDepths)

def genAvgDiscrepancyImage(data, shape, locs):
    """
    generates an image and determines color mins and maxes to 
    make the mean white on a normal diverging colormap.
    """
    me = np.mean(data)
    pic = np.ones(shape) * me
    for i, loc in enumerate(locs):
        pic[loc[0], loc[1]] = data[i]
    mi = min(pic.flatten())
    ma = max(pic.flatten())
    if me - mi > ma - me:
        vmin = mi
        vmax = 2*me - mi
    else:
        vmin = 2*me-ma
        vmax = ma
    return pic, vmin, vmax

def getBetterBiases(prevDepth, prev_V_Bias, prev_H_Bias, sign=1, hFreqs=None, vFreqs=None, hPhases=None, vPhases=None):
    for d in prevDepth.flatten():
        if d < 0:
            print('ERROR: This function cannot currently deal with negative arguments.')
    print('Assuming that (', prev_V_Bias[0],',',prev_V_Bias[-1], ') is the bias of the (highest, lowest)-frequency row')
    print('Assuming that (', prev_H_Bias[0],',',prev_H_Bias[-1], ') is the bias of the (lowest, highest)-frequency column')
    print('Please note that if using the outputted centers from Survival(), then you need to reshape the data'
          ' into a 2D numpy array correctly to match the ordering of the V and H biases. This is normally done'
          ' via a call to np.reshape() and a transpose to match the comments above.')
    print('Sign Argument should be -1 if numbers are pushout resonance locations.')
    if type(prevDepth) is not type(arr([])) or type(prevDepth[0]) is not type(arr([])):
        raise TypeError('ERROR: previous depth array must be 2D numpy array')
    result, modDepth = extrapolateEveningBiases(prev_H_Bias, prev_V_Bias, prevDepth, sign=sign);
    new_H_Bias = result['x'][:len(prev_H_Bias)]
    new_V_Bias = result['x'][len(prev_H_Bias):]
    print('Horizontal Changes')
    for prev, new in zip(prev_H_Bias, new_H_Bias):
        print(misc.round_sig(prev,4), '->', misc.round_sig(new,4), str(misc.round_sig(100 * (new - prev) / prev, 2)) + '%')
    print('Vertical Changes')
    for prev, new in zip(prev_V_Bias, new_V_Bias):
        print(misc.round_sig(prev,4), '->', misc.round_sig(new,4), str(misc.round_sig(100 * (new - prev) / prev, 2)) + '%')
    print('Previous Depth Relative Variation:', misc.round_sig(np.std(prevDepth),4), '/', 
          misc.round_sig(np.mean(prevDepth),4), '=', misc.round_sig(100 * np.std(prevDepth)/np.mean(prevDepth)), '%')
    print('Expected new Depth Relative Variation:', misc.round_sig(100*np.std(modDepth)/np.mean(prevDepth),4),'%')
    print('New Vertical Biases \n[',end='')
    for v in new_V_Bias:
        print(v, ',', end=' ')
    #print(']\nNew Horizontal Biases \n[', end='')
    #for h in new_H_Bias:
    #    print(h, ',', end=' ')
    #print(']\n')
    print(']\nNew Horizontal Biases \n[ ', end='')
    for h in new_H_Bias:
        print(h, ' ', end=' ')
    print(']\n')

    if hFreqs is None:
        return
    if not (len(new_H_Bias) == len(hFreqs) == len(hPhases)):
        raise ValueError('Lengths of horizontal data dont match')
    if not (len(new_V_Bias) == len(vFreqs) == len(vPhases)):
        raise ValueError('Lengths of vertical data dont match')
    with open('\\\\jilafile.colorado.edu\\scratch\\regal\\common\\LabData\\Quantum Gas Assembly\\Code_Files\\New-Depth-Evening-Config.txt','w') as file:
        file.write('HORIZONTAL:\n')
        for f, b, p in zip(hFreqs, new_H_Bias, hPhases):
            file.write(str(f) + '\t' + str(b) + '\t' + str(p) + '\n')
        file.write('VERTICAL:\n')
        for f, b, p in zip(vFreqs, reversed(new_V_Bias), vPhases):
            file.write(str(f) + '\t' + str(b) + '\t' + str(p) + '\n')

def extrapolateEveningBiases(hBiasIn, vBiasIn, depthIn, sign=1):
    """
    depth in is some measure of the trap depth which is assumed to be roughly linear with the trap depth. It need not be in the right units.
    """
    # normalize biases
    hBiasIn /= np.sum(hBiasIn)
    vBiasIn /= np.sum(vBiasIn)
    guess = np.concatenate((hBiasIn, vBiasIn))
    f = lambda g: modFitFunc(sign, hBiasIn, vBiasIn, depthIn, *g, )
    result = opt.minimize(f, guess)
    return result, extrapolateModDepth(sign, hBiasIn, vBiasIn, depthIn, result['x'])

def extrapolateModDepth(sign, hBiasIn, vBiasIn, depthIn, testBiases):
    """
    assumes that hBiasIn and vBiasIn are normalized.
    This function extrapolates what the depth of each tweezer should be based on the
    current depths and current biases. Basically, it assumes that if you change the bias by x%,
    then the depth for every atom in that row/column will change by x%.
    """
    hBiasTest = testBiases[:len(hBiasIn)]
    if len(hBiasTest) > 1:
        for b in hBiasTest:
            if b <= 0 or b > 1:
                return None
    vBiasTest = testBiases[len(hBiasIn):len(hBiasIn) + len(vBiasIn)]
    if len(vBiasTest) > 1:
        for b in vBiasTest:
            if b <= 0 or b > 1:
                return None
    # normalize tests
    hBiasTest /= np.sum(hBiasTest)
    vBiasTest /= np.sum(vBiasTest)
    modDepth = deepcopy(depthIn)
    for rowInc, _ in enumerate(depthIn):
        dif = (vBiasTest[rowInc] - vBiasIn[rowInc])/vBiasIn[rowInc]
        modDepth[rowInc] = modDepth[rowInc] * (1- sign * dif)
    for colInc, _ in enumerate(misc.transpose(depthIn)):
        dif = (hBiasTest[colInc] - hBiasIn[colInc])/hBiasIn[colInc]
        modDepth[:, colInc] = modDepth[:, colInc] * (1-sign * dif)
    return modDepth

def fitWithModule(module, key, vals, errs=None, guess=None, getF_args=[None], maxfev=2000):
    # this also works with class objects which have all the required member functions. 
    key = arr(key)
    xFit = (np.linspace(min(key), max(key), 1000) if len(key.shape) == 1 else np.linspace(min(misc.transpose(key)[0]),
                                                                                          max(misc.transpose(key)[0]), 1000))
    fitNom = fitStd = fitValues = fitErrs = fitCovs = fitGuess = rSq = None
    from numpy.linalg import LinAlgError
    try:
        fitF = module.getF(*getF_args) if hasattr(module, 'getF') else module.f
        fitF_unc = module.getF_unc(*getF_args) if hasattr(module, 'getF_unc') else module.f_unc
        if len(key) < len(signature(fitF).parameters) - 1:
            raise RuntimeError('Not enough data points to constrain a fit!')
        guessUsed = guess if guess is not None else module.guess(key,vals)
        fitValues, fitCovs = opt.curve_fit(fitF, key, vals, p0=guessUsed, maxfev=maxfev)
        fitErrs = np.sqrt(np.diag(fitCovs))
        corr_vals = unc.correlated_values(fitValues, fitCovs)
        fitUncObject = fitF_unc(xFit, *corr_vals)
        fitNom = unp.nominal_values(fitUncObject)
        fitStd = unp.std_devs(fitUncObject)
        fitFinished = True
        fitGuess = fitF(xFit, *guessUsed)
        residuals = vals - fitF(key, *fitValues)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((vals-np.mean(vals))**2)
        rSq = 1 - (ss_res / ss_tot)
    except (RuntimeError, LinAlgError, ValueError) as e:
        fitF = module.getF(*getF_args) if hasattr(module, 'getF') else module.f
        fitF_unc = module.getF_unc(*getF_args) if hasattr(module, 'getF_unc') else module.f_unc
        warn('Data Fit Failed! ' + str(e))
        print('stuff',key, vals, guessUsed)
        fitValues = module.guess(key, vals)
        fitNom = fitF(xFit, *fitValues)
        fitFinished = False        
        guessUsed = guess if guess is not None else module.guess(key,vals)
        fitGuess = fitF(xFit, *guessUsed)

    fitInfo = {'x': xFit, 'nom': fitNom, 'std': fitStd, 'vals': fitValues, 'errs': fitErrs, 'cov': fitCovs, 'guess': fitGuess, 'R-Squared': rSq}
    return fitInfo, fitFinished

def combineData(data, key):
    """
    combines similar key value data entries. data will be in order that unique key items appear in key.
    For example, if key = [1,3,5,3,7,1], returned key and corresponding data will be newKey = [1, 3, 5, 7]

    :param data:
    :param key:
    :return:
    """
    items = {}
    newKey = []
    newData = []
    for elem in key:
        if str(elem) not in items:
            indexes = [i for i, x in enumerate(key) if x == elem]
            # don't get it again
            items[str(elem)] = "!"
            newKey.append(elem)
            newItem = np.zeros((data.shape[1], data.shape[2]))
            # average together the corresponding data.
            for index in indexes:
                newItem += data[index]
            newItem /= len(indexes)
            newData.append(newItem)
    return arr(newData), arr(newKey)

def fitPic(picture, showFit=True, guessSigma_x=1, guessSigma_y=1, guess_x=None, guess_y=None, fitF=gaussian_2d.f_notheta, guessOffset=None, extraGuess=None):
    """
        Fit an individual picture with a 2d gaussian, and fit the horizontal and vertical averages with 1d gaussians
    """
    pos = arr(np.unravel_index(np.argmax(picture), picture.shape))
    pos[1] = guess_x if guess_x is not None else pos[1]
    pos[0] = guess_y if guess_x is not None else pos[0]

    pic = picture.flatten()
    x = np.linspace(0, picture.shape[1], picture.shape[1])
    y = np.linspace(0, picture.shape[0], picture.shape[0])
    X, Y = np.meshgrid(x, y)
    ### 2D Fit
    initial_guess = [(np.max(pic) - np.min(pic)), pos[1], pos[0], guessSigma_x, guessSigma_y, np.min(pic) if guessOffset is None else guessOffset]
    # for fitting functions with more arguments
    if extraGuess is not None:
        initial_guess += extraGuess
    try:
        print('fitting...')
        popt, pcov = opt.curve_fit(fitF, (X, Y), pic, p0=initial_guess)#, epsfcn=0.01, ftol=0)
    except RuntimeError:
        popt = np.zeros(len(initial_guess))
        pcov = np.zeros((len(initial_guess), len(initial_guess)))
        warn('2D Gaussian Picture Fitting Failed!')
    ### Vertical (i.e. collapse in the vertical direction) Average Fit
    vAvg = np.zeros(len(picture[0]))
    for r in picture:
        vAvg += r
    vAvg /= len(picture)
    vGuess = [np.max(vAvg) - np.min(vAvg), x[np.argmax(vAvg)], guessSigma_x, np.min(vAvg)]
    try:
        popt_v, pcov_v = opt.curve_fit(bump.f, x, vAvg, vGuess)
    except RuntimeError:
        popt_v = np.zeros(len(vGuess))
        pcov_v = np.zeros((len(vGuess), len(vGuess)))
        warn('Vertical Average Picture Fitting Failed!')
    
    ### Horizontal Average Fit
    hAvg = np.zeros(len(picture))
    for c in misc.transpose(picture):
        hAvg += c
    hAvg /= len(picture[0])
    hGuess = [np.max(hAvg) - np.min(hAvg), y[np.argmax(hAvg)], guessSigma_y, np.min(hAvg)]
    try:
        popt_h, pcov_h = opt.curve_fit(bump.f, y, hAvg, hGuess)
    except RuntimeError:
        popt_h = np.zeros(len(hGuess))
        pcov_h = np.zeros((len(hGuess), len(hGuess)))
        warn('Horizontal Average Picture Fitting Failed!')
        
    if showFit:
        print(fitF)
        data_fitted = fitF((X,Y), *popt)
        fig, axs = plt.subplots(1, 3)
        plt.grid(False)
        im = axs[0].imshow(picture, origin='lower')#, extent=(x.min(), x.max(), y.min(), y.max()))
        data_fitted = data_fitted.reshape(picture.shape[0],picture.shape[1])
        axs[0].contour(x, y, data_fitted, 4, colors='w', alpha=0.2)
        mp.addAxColorbar(fig, axs[0], im)
        axs[0].set_title('Raw Data')
        
        im = axs[1].imshow( data_fitted, origin='lower')
        mp.addAxColorbar(fig, axs[1], im)
        axs[1].set_title('Fit')
        
        im = axs[2].imshow( picture - data_fitted, origin='lower' )
        mp.addAxColorbar(fig, axs[2], im)
        axs[2].contour(x, y, data_fitted, 4, colors='w', alpha=0.2)
        axs[2].set_title('Residuals')
        
    return initial_guess, popt, np.sqrt(np.diag(pcov)), popt_v, np.sqrt(np.diag(pcov_v)), popt_h, np.sqrt(np.diag(pcov_h))

def fitPictures(pictures, dataRange, guessSigma_x=1, guessSigma_y=1, quiet=False, firstIsGuide=True):
    """
    fit an array of pictures with gaussians
    
    if firstIsGuide is true then use the fit from the first pic as the guide for the next pictures.
    :param pictures:
    :param dataRange:
    :param guessSigma_x:
    :param guessSigma_y:
    :return:
    """
    fitParameters, fitErrors, vParams, vErrs, hParams, hErrs = [[] for _ in range(6)]
    count = 0
    warningHasBeenThrown = False
    if not quiet:
        print('fitting picture Number...')
    for picInc, picture in enumerate(pictures):
        if not quiet:
            print(picInc, ',', end='')
        if count not in dataRange:
            parameters, errors = [np.zeros(7) for _ in range(2)]
            v_param, v_err, h_param, h_err = [np.zeros(4) for _ in range(4)]
        else:
            try:
                if firstIsGuide and picInc != 0:
                    # amplitude, xo, yo, sigma_x, sigma_y, theta, offset
                    _, parameters, errors, v_param, v_err, h_param, h_err = fitPic(picture, showFit=False, 
                                                                                   guess_x = fitParameters[0][1], guess_y = fitParameters[0][2], 
                                                                                   guessSigma_x=fitParameters[0][3], guessSigma_y=fitParameters[0][4])
                else:
                    _, parameters, errors, v_param, v_err, h_param, h_err = fitPic(picture, showFit=False, 
                                                                                   guessSigma_x=guessSigma_x, guessSigma_y=guessSigma_y)
            except RuntimeError:
                if not warningHasBeenThrown:
                    print("Warning! Not all picture fits were able to fit the picture signal to a 2D Gaussian.\n"
                          "When the fit fails, the fit parameters are all set to zero.")
                    warningHasBeenThrown = True
                parameters, errors = [np.zeros(7) for _ in range(2)]
                v_param, v_err, h_param, h_err = [np.zeros(4) for _ in range(2)]
        # append things regardless of whether the fit succeeds or not in order to keep things the right length.
        fitParameters.append(parameters)
        fitErrors.append(errors)
        vParams.append(v_param)
        vErrs.append(v_err)
        hParams.append(h_param)
        hErrs.append(h_err)
        count += 1
    return np.array(fitParameters), np.array(fitErrors), np.array(vParams), np.array(vErrs), np.array(hParams), np.array(hErrs)

def fitDoubleGaussian(binCenters, binnedData, fitGuess, quiet=False):
    try:
        fitVals, fitCovNotUsed = opt.curve_fit( lambda x, a1, a2, a3, a4, a5, a6:
                                            double_gaussian.f(x, a1, a2, a3, a4, a5, a6, 0),
                                            binCenters, binnedData, fitGuess )
    except opt.OptimizeWarning as err:
        if not quiet:
            print('Double-Gaussian Fit Failed! (Optimization Warning)', err)
        fitVals = (0, 0, 0, 0, 0, 0)
    except RuntimeError as err:
        if not quiet:
            print('Double-Gaussian Fit Failed! (Runtime error)', err)
        fitVals = (0, 0, 0, 0, 0, 0)        
    return [*fitVals,0]

def fitGaussianBeamWaist(data, key, wavelength):
    # expects waists as inputs
    initial_guess = [min(data.flatten()), key[int(3*len(key)/4)]]
    try:
        # fix the wavelength
        # beamWaistExpansion(z, w0, wavelength)
        popt, pcov = fit(lambda x, a, b: beamWaistExpansion(x, a, b, wavelength), key, data, p0=initial_guess)
    except RuntimeError:
        popt, pcov = [0, 0]
        warn('Fit Failed!')
    return popt, pcov

# #############################
# ### Analyzing machine outputs

def load_SRS_SR780(fileAddress):
    """
    from a TXT file from the SRS, returns the frequencies (the [0] element) and the powers (the [1] element)
    """
    data = pd.read_csv(fileAddress, delimiter=',', header=None)
    return data[0], data[1]

def load_HP_4395A(fileAddress):
    """
    Analyzing HP 4395A Spectrum & Network Analyzer Data
    """
    data = pd.read_csv(fileAddress, delimiter='\t', header=11)
    return data["Frequency"], data["Data Trace"]

def load_RSA_6114A(fileLocation):
    """
    return xData, yData, yUnits, xUnits
    """
    lines = []
    count = 0
    yUnits = ""
    xUnits = ""
    xPointNum, xStart, xEnd = [0, 0, 0]
    with open(fileLocation) as file:
        for line in iter(file.readline, ''):
            count += 1
            # 18 lines to skip.
            if count == 11:
                yUnits = str(line[line[:].index('>')+1:line[1:].index('<')+1])
                continue
            elif count == 12:
                xUnits = str(line[line[:].index('>')+1:line[1:].index('<')+1])
                continue
            elif count == 16:
                xPointNum = float(line[line[:].index('>')+1:line[1:].index('<')+1])
                continue
            elif count == 17:
                xStart = float(line[line[:].index('>')+1:line[1:].index('<')+1])
                continue
            elif count == 18:
                xEnd = float(line[line[:].index('>')+1:line[1:].index('<')+1])
                continue
            elif count <= 18:
                continue
            try:
                lines.append(line[line[:].index('>')+1:line[1:].index('<')+1])
            except ValueError:
                pass
    yData = np.float64(arr(lines))
    xData = np.linspace(xStart, xEnd, xPointNum)
    return xData, yData, yUnits, xUnits

# ##########################
# ### Some AOM Optimizations

def getOptimalAomBiases(minX, minY, spacing, widthX, widthY):
    # This function has been deprecated. It was only valid for the old Gooch and Housego aoms.
    return "FUNCTION-DEPRECATED"
    """

    :param minX:
    :param minY:
    :param spacing:
    :param widthX:
    :param widthY:
    :return:
    """
    # these calibrations were taken on 9/11/2017\n",
    # At Vertical Frequency = 80 MHz. \n",
    horFreq =     [70,   75,  65, 67.5, 72.5,  80,  85,  90,  95,   60,  50,  55,   45, 62.5, 57.5, 52.5]
    powerInRail = [209, 197, 180,  198,  205, 186, 156, 130, 72.5, 181, 109, 179, 43.5,  174,  182, 165]
    #def orderData(data, key, keyDim=None, otherDimValues=None):
    powerInRail, horFreq, _ = orderData(powerInRail, horFreq)
    relativeHorPowerInRail = arr(powerInRail) / max(powerInRail) * 100
    horAomCurve = interp.InterpolatedUnivariateSpline(horFreq, relativeHorPowerInRail)
    # at horizontal freq of 70MHz\n",
    vertFreq = [80, 82.5, 77.5, 75, 85, 90, 95, 100, 105, 70, 65, 60, 55, 50, 52.5, 57.5, 62.5]
    vertPowerInRail = [206, 204, 202, 201, 197, 184, 145, 126, 64, 193, 185, 140, 154, 103, 141, 140, 161]
    vertPowerInRail, vertFreq, _ = orderData(vertPowerInRail, vertFreq)
    relativeVertPowerInRail = arr(vertPowerInRail) / max(vertPowerInRail) * 100
    #vertAomCurve = interp.interp1d(vertFreq, relativeVertPowerInRail)
    vertAomCurve = interp.InterpolatedUnivariateSpline(vertFreq, relativeVertPowerInRail)
    xFreqs = [minX + num * spacing for num in range(widthX)]
    #xAmps = arr([100 / base_interp(xFreq, horFreq, relativeHorPowerInRail) for xFreq in xFreqs])
    xAmps = arr([100 / horAomCurve(xFreq) for xFreq in xFreqs])
    xAmps /= np.sum(xAmps) / len(xAmps)
    yFreqs = [minY + num * spacing for num in range(widthY)]
    #yAmps = arr([100 / base_interp(yFreq, vertFreq, relativeVertPowerInRail) for yFreq in yFreqs])
    yAmps = arr([100 / vertAomCurve(yFreq) for yFreq in yFreqs])
    yAmps /= np.sum(yAmps) / len(yAmps)
    #yAmps = [100 / vertAomCurve(yFreq) for yFreq in yFreqs]
    return xFreqs, xAmps, yFreqs, yAmps

def maximizeAomPerformance(horCenterFreq, vertCenterFreq, spacing, numTweezersHor, numTweezersVert, iterations=10, paperGuess=True, metric='max',
                          vertAmps=None, horAmps=None, carrierFre=255):
    """
    computes the amplitudes and phases to maximize the AOM performance.
    :param horCenterFreq:
    :param vertCenterFreq:
    :param spacing:
    :param numTweezersHor:
    :param numTweezersVert:
    :param iterations:
    :return:
    """
    horFreqs  = [horCenterFreq - spacing * (numTweezersHor  - 1) / 2.0 + i * spacing for i in range(numTweezersHor )]
    vertFreqs = [horCenterFreq - spacing * (numTweezersVert - 1) / 2.0 + i * spacing for i in range(numTweezersVert)]
    actualHFreqs = 255 - arr(horFreqs)
    actualVFreqs = 255 - arr(vertFreqs)
    
    if vertAmps is None:
        vertAmps = np.ones(numTweezersVert)
    if horAmps is None:
        horAmps = np.ones(numTweezersHor)
    def calcWaveCos(xPts, phases, freqs, amps):
        volts = np.zeros(len(xPts))
        phases += [0]
        for phase, freq, amp in zip(phases, freqs, amps):
            volts += amp * np.cos(2*np.pi*freq * 1e6 * xPts + phase)
        return volts
    def calcWave(xPts, phases, freqs, amps):
        volts = np.zeros(len(xPts))
        phases += [0]
        for phase, freq, amp in zip(phases, freqs, amps):
            volts += amp * np.sin(2*np.pi*freq * 1e6 * xPts + phase)
        return volts
    
    def getXMetric(phases):
        x = np.linspace(0, 3e-6, 20000)
        if metric=='max':
            return max(abs(calcWave(x, phases, actualHFreqs, horAmps)))
        elif metric == 'std':
            return np.std(calcWave(x, phases, actualHFreqs, horAmps))

    def getYMetric(phases):
        x = np.linspace(0, 3e-6, 20000)
        if metric=='max':
            return max(abs(calcWave(x, phases, actualVFreqs, vertAmps)))
        elif metric == 'std':
            return np.std(calcWave(x, phases, actualVFreqs, vertAmps))

    xBounds = [(0, 2 * mc.pi) for _ in range(numTweezersHor-1)]
    #
    if paperGuess:
        xGuess = arr([np.pi * i**2/numTweezersHor for i in range(numTweezersHor-1)])
    else:
        xGuess = arr([0 for _ in range(numTweezersHor-1)])
    minimizer_kwargs = dict(method="L-BFGS-B", bounds=xBounds)
    xPhases = opt.basinhopping(getXMetric, xGuess, minimizer_kwargs=minimizer_kwargs, niter=iterations, stepsize=0.2)
    xPhases = list(xPhases.x) + [0]
    print('horFreqs', horFreqs)
    print('horAmps', horAmps)
    print('Hor-Phases:', [misc.round_sig_str(x,10) for x in xPhases])

    if paperGuess:
        yGuess = arr([np.pi * i**2/numTweezersVert for i in range(numTweezersVert-1)])
    else:
        yGuess = arr([0 for _ in range(numTweezersVert-1)])
    yBounds = [(0, 2 * mc.pi) for _ in range(numTweezersVert-1)]
    minimizer_kwargs = dict(method="L-BFGS-B", bounds=yBounds)
    yPhases = opt.basinhopping(getYMetric, yGuess, minimizer_kwargs=minimizer_kwargs, niter=iterations, stepsize=0.2)
    yPhases = list(yPhases.x) + [0]
    for i, xp in enumerate(yPhases):
        yPhases[i] = misc.round_sig(xp, 10)

    print('vertFreqs', vertFreqs)
    print('vertAmps', vertAmps)
    print('Vert-Phases:', [misc.round_sig_str(y,10) for y in yPhases])

    xpts = np.linspace(0, 1e-6, 10000)
    ypts_x = calcWave(xpts, xPhases, actualHFreqs, horAmps)
    yptsOrig = calcWaveCos(xpts, arr([0 for _ in range(numTweezersHor)]), actualHFreqs, horAmps)
    plt.title('X-Axis')
    plt.plot(xpts, ypts_x, ':', label='X-Optimization')
    plt.plot(xpts, yptsOrig, ':', label='X-Worst-Case')
    plt.legend()

    plt.figure()
    yptsOrig = calcWave(xpts, arr([0 for _ in range(numTweezersVert)]), actualVFreqs, vertAmps)
    ypts_y = calcWaveCos(xpts, yPhases, actualVFreqs, vertAmps)
    plt.title('Y-Axis')
    plt.plot(xpts, ypts_y, ':', label='Y-Optimization')
    plt.plot(xpts, yptsOrig, ':', label='Y-Worst-Case')
    plt.legend()
    return xpts, ypts_x, ypts_y, 

def integrateData(pictures):
    """

    :param pictures:
    :return:
    """
    if len(pictures.shape) == 3:
        integratedData = np.zeros(pictures.shape[0])
        picNum = 0
        for pic in pictures:
            for row in pic:
                for elem in row:
                    integratedData[picNum] += elem
            picNum += 1
    else:
        integratedData = 0
        for row in pictures:
            for elem in row:
                integratedData += elem
    return integratedData

def beamIntensity(power, waist, radiusOfInterest=0):
    """
    computes the average beam intensity, in mW/cm^2, of a beam over some radius of interest.

    :param power: power of the laser beam, in mW
    :param waist: waist of the laser beam, in cm.
    :param radiusOfInterest: the radius of interest. In the case that this is << waist, the equation below
        reduces to a simpler, more commonly referenced form. The literal math gives 0/0 though, so I
        include the reduced form.
    """
    if radiusOfInterest == 0:
        return 2 * power / (mc.pi * waist ** 2)
    else:
        return power * (1 - np.exp(-2 * radiusOfInterest ** 2 / waist ** 2)) / (mc.pi * radiusOfInterest ** 2)

def computeBaslerGainDB(rawGain):
    """
    Gain (NOT currently used in fluorescence calc...)
    """
    G_c = 20 * np.log10((658 + rawGain)/(658 - rawGain))
    if 110 <= rawGain <= 511:
        gainDB = 20 * np.log10((658 + rawGain)/(658 - rawGain)) - G_c
    elif 511 <= rawGain <= 1023:
        gainDB = 0.0354 * rawGain - G_c
    else:
        gainDB = None
        warn('raw gain out of range! gainDB set to None/')
    return gainDB

def computeScatterRate(totalIntensity, D2Line_Detuning):
    """
    Computes the rate of photons scattering off of a single atom. From steck, equation 48.

    Assumes 2-Level approximation, good for near resonant light since the near-resonant transition
    will be dominant.

    Assumes D2 2 to 3' transition.

    :param totalIntensity: the total intensity (from all beams) shining on the atoms.
    :param D2Line_Detuning: the detuning, in Hz, of the light shining on the atoms from the D2 transition.
    """
    isat = mc.Rb87_I_Sat_ResonantIsotropic_2_to_3
    rate = (mc.Rb87_D2Gamma / 2) * (totalIntensity / isat) / (1 + 4 * (D2Line_Detuning / mc.Rb87_D2Gamma) ** 2
                                                                  + totalIntensity / isat)
    return rate

def computeFlorescence(greyscaleReading, imagingLoss, imagingLensDiameter, imagingLensFocalLength, exposure ):
    """
    TODO: incorporate gain into the calculation, currently assumes gain = X1... need to check proper conversion
    from basler software. I'd expect a power conversion so a factor of 20,
    Fluorescence

    :param greyscaleReading:
    :param imagingLoss:
    :param imagingLensDiameter:
    :param imagingLensFocalLength:
    :param exposure:
    :return:
    """
    term1 = greyscaleReading * mc.cameraConversion / (mc.h * mc.c / mc.Rb87_D2LineWavelength)
    term2 = 1 * imagingLoss * (imagingLensDiameter**2 / (16 * imagingLensFocalLength**2)) * exposure
    fluorescence = term1 / term2
    return fluorescence

# mot radius is in cm
def computeMotNumber(sidemotPower, diagonalPower, motRadius, exposure, imagingLoss, greyscaleReading, detuning=10e6):
    """
    :param sidemotPower: power in the sidemot beam, in mW.  Code Assumes 3.3mm sidemot waist
    :param diagonalPower: power in an individual diagonal mot beam, in mW
    :param motRadius: the approximate radius of the MOT. Used as a higher order part of the calculation which takes into
        account the spread of the intensity of the beams over the finite size of the MOT. Less needed for
        big MOT beams.
    :param exposure: exposure time of the camera, in seconds.
    :param imagingLoss: Approximate amount of light lost in the imaging line due to mirrors efficiency, filter
        efficiency, etc.
    :param greyscaleReading: the integrated greyscale count reading from the camera.

    ===
    The mot number is determined via the following formula:

    MOT # = (Scattered Light Collected) / (Scattered light predicted per atom)

    Here with sideMOT power in mW assuming 3.3mm radius waist and a very rough estimation of main MOT diameter
    one inch, motRadius using the sigma of the MOT size but should not matter if it's small enough, and exposure
    in sec, typically 0.8 for the imaging loss accounting for the line filter, greyscaleReading is the integrated gray
    scale count with 4by4 binning on the Basler camera, and assuming gain set to 260 which is  unity gain for Basler
    """
    # in cm 
    sidemotWaist = .33 / (2 * np.sqrt(2))
    # in cm 
    diagonalWaist = 2.54 / 2
    # intensities 
    sidemotIntensity = beamIntensity(sidemotPower, sidemotWaist, motRadius)
    diagonalIntensity = beamIntensity(diagonalPower, diagonalWaist, motRadius)
    totalIntensity = sidemotIntensity + 2 * diagonalIntensity
    rate = computeScatterRate(totalIntensity, detuning)
    imagingLensDiameter = 2.54
    imagingLensFocalLength = 10
    fluorescence = computeFlorescence(greyscaleReading, imagingLoss, imagingLensDiameter, imagingLensFocalLength,
                                      exposure)
    motNumber = fluorescence / rate
    return motNumber, fluorescence


def calcBallisticTemperature(times, sizeSigmas, guess = LargeBeamMotExpansion.guess(), sizeErrors=None):
    """ Small wrapper around a fit 
    expects time in s, sigma in m
    return temp, vals, cov
    """
    warnings.simplefilter("error", opt.OptimizeWarning)
    try:
        fitVals, fitCovariances = opt.curve_fit(LargeBeamMotExpansion.f, times, sizeSigmas, p0=guess, sigma = sizeErrors)
        temperature = fitVals[2]
    except opt.OptimizeWarning as error:
        warn('Mot Temperature Expansion Fit Failed!' + str(error))
        try:
            fitValsTemp, fitCovTemp = opt.curve_fit(lambda t,x,y: LargeBeamMotExpansion.f(t, x, 0, y), times, sizeSigmas, p0=[guess[0], guess[2]], sigma = sizeErrors)
            temperature = fitValsTemp[1]
            fitVals = [fitValsTemp[0], 0, fitValsTemp[1]]
            fitCovariances = np.zeros((len(guess),len(guess)))
            fitCovariances[0,0] = fitCovTemp[0,0]
            fitCovariances[2,0] = fitCovTemp[1,0]
            fitCovariances[0,2] = fitCovTemp[0,1]
            fitCovariances[2,2] = fitCovTemp[1,1]
        except opt.OptimizeWarning:
            fitVals = np.zeros(len(guess))
            fitCovariances = np.zeros((len(guess), len(guess)))
            temperature = 0
            warn('Restricted Mot Temperature Expansion Fit Failed Too with optimize error!')
        except RuntimeError:
            fitVals = np.zeros(len(guess))
            fitCovariances = np.zeros((len(guess), len(guess)))
            temperature = 0
            warn('Mot Temperature Expansion Fit Failed with Runtime error!')
    except RuntimeError:
        fitVals = np.zeros(len(guess))
        fitCovariances = np.zeros((len(guess), len(guess)))
        temperature = 0
        warn('Mot Temperature Expansion Fit Failed!')
    warnings.simplefilter("default", opt.OptimizeWarning)
    return temperature, fitVals, fitCovariances


def orderData(data, key, keyDim=None, otherDimValues=None):
    """
        return arr(data), arr(key), arr(otherDimValues)
    """
    zipObj = (zip(key, data, otherDimValues) if otherDimValues is not None else zip(key, data))
    if keyDim is not None:
        key, data, otherDimValues = list(zip(*sorted(zipObj, key=lambda x: x[0][keyDim])))
        # assuming 2D
        count = 0
        for val in key:
            if val[keyDim] == key[0][keyDim]:
                count += 1
        majorKeySize = int(len(key) / count)
        tmpKey = arr(key[:])
        tmpVals = arr(data[:])
        tmpKey.resize([majorKeySize, count, 2])
        tmpVals.resize([majorKeySize, count, arr(data).shape[1], arr(data).shape[2], arr(data).shape[3]])
        finKey = []
        finData = []
        for k, d in zip(tmpKey, tmpVals):
            k1, d1 = list(zip(*sorted(zip(k, d), key=lambda x: x[0][int(not keyDim)])))
            for k2, d2 in zip(k1, d1):
                finKey.append(arr(k2))
                finData.append(arr(d2))
        return arr(finData), arr(finKey), arr(otherDimValues)
    else:
        if otherDimValues is None:
            key, data = list(zip(*sorted(zipObj, key=lambda x: x[0])))
        else:
            key, data, otherDimValues = list(zip(*sorted(zipObj, key=lambda x: x[0])))
    return arr(data), arr(key), arr(otherDimValues)

def groupMultidimensionalData(key, varyingDim, atomLocations, survivalData, survivalErrs, loadingRate):
    """
    Normally my code takes all the variations and looks at different locations for all those variations.
    In the multi-dim case, this there are multiple variations for the same primary key value. I need to
    split up those multiple variations.
    """
    if len(key.shape) == 1:
        # no grouping needed
        return (key, atomLocations, survivalErrs, survivalData, loadingRate,
                [None for _ in range(len(key)*len(atomLocations))])
    # make list of unique indexes for each dimension
    uniqueSecondaryAxisValues = []
    newKey = []
    for keyValNum, secondaryValues in enumerate(misc.transpose(key)):
        if keyValNum == varyingDim:
            for val in secondaryValues:
                if val not in newKey:
                    newKey.append(val)
            continue
        uniqueSecondaryAxisValues.append([])
        for val in secondaryValues:
            if val not in uniqueSecondaryAxisValues[-1]:
                uniqueSecondaryAxisValues[-1].append(val)
    extraDimValues = 1
    for i, dim in enumerate(uniqueSecondaryAxisValues):
        extraDimValues *= len(dim)
    newLoadingRate, newTransferData, newErrorData, locationsList, otherDimsList = [[] for _ in range(5)]
    allSecondaryDimVals = arr(uniqueSecondaryAxisValues).flatten()
    # iterate through all locations
    for loc, locData, locErrs, locLoad in zip(atomLocations, survivalData, survivalErrs, loadingRate):
        newData = locData[:]
        newErr = locErrs[:]
        newLoad = locLoad[:]
        newData.resize(int(len(locData)/extraDimValues), extraDimValues)
        newData = misc.transpose(newData)
        newErr.resize(int(len(locData)/extraDimValues), extraDimValues)
        newErr = misc.transpose(newErr)
        newLoad.resize(int(len(locData)/extraDimValues), extraDimValues)
        newLoad = misc.transpose(newLoad)
        # iterate through all extra dimensions in the locations
        secondIndex = 0
        for val, err, load in zip(newData, newErr, newLoad):
            newTransferData.append(val)
            newErrorData.append(err)
            newLoadingRate.append(load)
            locationsList.append(loc)
            otherDimsList.append(allSecondaryDimVals[secondIndex])
            secondIndex += 1
    return (arr(newKey), arr(locationsList), arr(newErrorData), arr(newTransferData), arr(newLoadingRate),
            arr(otherDimsList))

def getFitsDataFrame(fits, fitModules, avgFit):
    uniqueModules = set(fitModules)
    fitDataFrames = [pd.DataFrame() for _ in uniqueModules]
    for moduleNum, fitModule in enumerate(uniqueModules):
        # for each column in dataframe.
        for argnum, arg in enumerate(fitModule.args()):
            vals = []
            for fitData, mod in zip(fits, fitModules):
                if mod is fitModule:
                    vals.append(fitData['vals'][argnum])
            errs = []
            for fitData, mod in zip(fits, fitModules):
                if mod is fitModule:
                    if fitData['errs'] is not None:
                        errs.append(fitData['errs'][argnum])
                    else:
                        errs.append(0)
            vals.append(np.mean(vals))
            vals.append(np.median(vals))
            vals.append(np.std(vals))            
        
            if fitModule == fitModules[-1]:
                vals.append(avgFit['vals'][argnum])
            errs.append(np.mean(errs))
            errs.append(np.median(errs))
            errs.append(np.std(errs))

            if fitModule == fitModules[-1]:
                if avgFit['errs'] is not None:
                    errs.append(avgFit['errs'][argnum])
                else:
                    errs.append(0)
            fitDataFrames[moduleNum][arg] = vals
            fitDataFrames[moduleNum][arg + '-Err'] = errs
            
            
            
        
        characters = []
        for fitData, mod in zip(fits, fitModules):
            if mod is fitModule:
                characters.append(fitModule.fitCharacter(fitData['vals']))
        characters.append(np.mean(characters))
        characters.append(np.median(characters))
        characters.append(np.std(characters))
        if fitModule == fitModules[-1]:
            characters.append(fitModule.fitCharacter(avgFit['vals']))
        fitDataFrames[moduleNum][fitModule.getFitCharacterString()] = characters
        
        
        indexStrs = []
        for i in range(len(fits)):
            if fitModules[i] is fitModule:
                indexStrs.append('fit ' +str(i))
        indexStrs.append('Mean Val')
        indexStrs.append('Median Val')
        indexStrs.append('Std Val')
        if fitModule == fitModules[-1]:
            indexStrs.append('Fit of Avg')
        fitDataFrames[moduleNum].index = indexStrs
        #disp.display(disp.Markdown(fitModules[-1].getFitCharacterString() + ': ' + misc.round_sig_str(fitModules[-1].fitCharacter(avgFit['vals']))))
    return fitDataFrames

    
def getThresholds( picData, tOptions=ThresholdOptions.ThresholdOptions(), quiet=True):
    th = AtomThreshold.AtomThreshold()
    th.rawData = picData
    th.binCenters, th.binHeights = getBinData( tOptions.histBinSize, th.rawData )
    # inner outwards. e.g. [0,1,2,3,4] -> [2,1,3,0,4]
    binGuessIteration = [th.binCenters[(len(th.binCenters) + (~i, i)[i%2]) // 2] for i in range(len(th.binCenters))]
    # binGuessIteration = list(reversed(bins[:len(bins)//2]))
    # binGuessIteration = list(bins[len(bins)//2:])
    gWidth = 25
    ampFac = 0.35
    if not tOptions.manualThreshold:
        guessPos1, guessPos2 = guessGaussianPeaks( th.binCenters, th.binHeights )
        if guessPos1 == guessPos2:
            guessPos2 += 1
        guess = arr([max(th.binHeights), guessPos1, gWidth, max(th.binHeights)*ampFac, guessPos2, gWidth])
        th.fitVals = fitDoubleGaussian(th.binCenters, th.binHeights, guess, quiet=quiet)
        th.t, th.fidelity = calculateAtomThreshold(th.fitVals)
        th.rmsResidual = getNormalizedRmsDeviationOfResiduals(th.binCenters, th.binHeights, double_gaussian.f, th.fitVals)
        if tOptions.rigorousThresholdFinding:
            for guessNum in range(len(binGuessIteration)):
                if th.fidelity - th.rmsResidual*0.05 > 0.97:
                    break # good enough. When fitting is good, Fidelity should be high and residual should be low. 
                newGuessPosition = binGuessIteration[guessNum]
                guess = arr([max(th.binHeights), guessPos1, gWidth, max(th.binHeights)*ampFac, newGuessPosition, gWidth])
                t2 = copy(th)
                t2.fitVals = fitDoubleGaussian(t2.binCenters, t2.binHeights, guess, quiet=quiet)
                t2.t, t2.fidelity = calculateAtomThreshold(t2.fitVals)
                t2.rmsResidual = getNormalizedRmsDeviationOfResiduals(t2.binCenters, t2.binHeights, double_gaussian.f, t2.fitVals)
                if t2.fidelity - t2.rmsResidual > th.fidelity - th.rmsResidual: # keep track of the best result in the t variable.
                    th.t, th.fitVals, th.fidelity, th.rmsResidual = t2.t, t2.fitVals, t2.fidelity, t2.rmsResidual
                else:
                    pass
    elif tOptions.autoHardThreshold:
        th.fitVals = None
        th.t, th.fidelity = ((max(th.rawData) + min(th.rawData))/2.0, 0) 
        th.rmsResidual=0
    elif tOptions.autoThresholdFittingGuess:
        guess = arr([max(th.binHeights), (max(th.rawData) + min(th.rawData))/4.0, gWidth,
                     max(th.binHeights)*0.5, 3*(max(th.rawData) + min(th.rawData))/4.0, gWidth])
        th.fitVals = fitDoubleGaussian(th.binCenters, th.binHeights, guess, quiet=quiet)
        th.t, th.fidelity = calculateAtomThreshold(th.fitVals)
        th.rmsResidual = getNormalizedRmsDeviationOfResiduals(th.binCenters, th.binHeights, double_gaussian.f, th.fitVals)
    else:
        print('no option???')
        th.fitVals = None
        th.t, th.fidelity = (tOptions.manualThresholdValue, 0)
        th.rmsResidual=0
    return th

def getNormalizedRmsDeviationOfResiduals(xdata, ydata, function, fitVals):
    """
    calculates residuals, calculates the rms average of them, and then normalizes by the average of the actual data.
    """
    residuals = ydata - function(xdata, *fitVals)
    return np.sqrt(sum(residuals**2) / len(residuals)) / np.mean(ydata)

def getSurvivalBoolData(ICounts, TCounts, IThreshold, TThreshold):
    """    
    I stands for init, as in initialCounts, T stands for Transfered, as in transfered Counts.
    """
    IAtoms, TAtoms = [[] for _ in range(2)]
    for IPoint, TPoint in zip(ICounts, TCounts):
        IAtoms.append(IPoint > IThreshold)
        TAtoms.append(TPoint > TThreshold)
    return IAtoms, TAtoms
    
def getAtomBoolData(pic1Data, threshold):
    atomCount = 0
    pic1Atom = []
    for point in pic1Data:
        if point > threshold:
            atomCount += 1
            pic1Atom.append(1)
        else:
            pic1Atom.append(0)
    return pic1Atom, atomCount

def getAtomCountsData( pics, picsPerRep, whichPic, loc, subtractEdges=True ):
    borders = getAvgBorderCount(pics, whichPic, picsPerRep) if subtractEdges else np.zeros(int(len(pics)/picsPerRep))
    pic1Data = normalizeData(pics, loc, whichPic, picsPerRep, borders)
    return list(pic1Data)

def calculateAtomThreshold(fitVals):
    """
    :param fitVals = [Amplitude1, center1, sigma1, amp2, center2, sigma2]
    """
    if fitVals[5] + fitVals[2] == 0:
        return 200, 0
    else:
        TCalc = (fitVals[4] - fitVals[1])/(np.abs(fitVals[5]) + np.abs(fitVals[2]))
        threshold = abs(fitVals[1] + TCalc * fitVals[2])
    if np.isnan(threshold):
        threshold = 200
    fidelity = getFidelity(threshold, fitVals)
    return threshold, fidelity

def getFidelity(threshold, fitVals):
    a1, x1, s1, a2, x2, s2, o = fitVals
    return 0.5 * (1 + 0.5 *
                  (  special.erf(np.abs(threshold-x1)/(np.sqrt(2)*s1)) 
                   + special.erf(np.abs(threshold-x2)/(np.sqrt(2)*s2))))

def getMaxFidelityThreshold(fitVals):
    # difference between centers divided by sum of sigma?
    from scipy import optimize as opt
    a1, x1, s1, a2, x2, s2 = fitVals
    def minFunc(t):
        return 1 - getFidelity(t, fitVals)
    res = opt.minimize(minFunc, (x2+x1)/2, bounds=[(x1,x2)])
    threshold = res.x[0]
    fidelity = getFidelity(threshold, fitVals)
    return threshold, fidelity


def postSelectOnAssembly( pic1AtomData, pic2AtomData, analysisOpts, justReformat=False, extraDataToPostSelect=None ): 
    totalRepNum = len(pic1AtomData[0])
    dataSetTotalNum = len(analysisOpts.postSelectionConditions)
    if extraDataToPostSelect is not None:
        if len(extraDataToPostSelect) != totalRepNum:
            raise ValueError('extraDataToPostSelect must have a value for every repetition. Length of extraDataToPostSelect = ' 
                             + str(len(extraDataToPostSelect)) + ' while totalRepNum = ' + str(totalRepNum))
    # the "justReformat" arg is a bit hackish here. 
    if analysisOpts.postSelectionConditions is None:
        analysisOpts.postSelectionConditions = [[] for _ in analysisOpts.positiveResultConditions]
    # 2d conditions... ps for post-selected
    psPic1AtomData, psPic2AtomData, postSelectedExtraData = [[[] for _ in analysisOpts.postSelectionConditions] for _ in range(3)]    
    for dataSetInc in range(dataSetTotalNum):
        conditionHits = [None for condition in analysisOpts.postSelectionConditions[dataSetInc]]
        for conditionInc, condition in enumerate(analysisOpts.postSelectionConditions[dataSetInc]):
            # see getEnsembleHits for more discussion on how the post-selection is working here. 
            conditionHits[conditionInc] = getConditionHits([pic1AtomData, pic2AtomData], condition)        
        #print('hello, world!')
        #print(conditionHits, pic1AtomData, analysisOpts.postSelectionConditions[dataSetInc])        
        for repNum in range(totalRepNum):
            allCondMatch = True
            for condition in conditionHits:
                if condition[repNum] == False:
                    allCondMatch = False
            #print('!' if allCondMatch else '.', end='')
            if allCondMatch or justReformat:
                indvAtoms1, indvAtoms2 = [], []
                for atomInc in range(len(pic1AtomData)):
                    indvAtoms1.append(pic1AtomData[atomInc][repNum])
                    indvAtoms2.append(pic2AtomData[atomInc][repNum])
                psPic1AtomData[dataSetInc].append(indvAtoms1)
                psPic2AtomData[dataSetInc].append(indvAtoms2)
                if extraDataToPostSelect is not None:
                    postSelectedExtraData[dataSetInc].append(extraDataToPostSelect[repNum])  
                    
            # else nothing! discard the data.
        if len(psPic1AtomData[dataSetInc]) == 0:
            print("No data left after post-selection! Data Set #" + str(dataSetInc))
            indvAtoms1, indvAtoms2 = [], []
            for atomInc in range(len(pic1AtomData)):
                indvAtoms1.append(pic1AtomData[atomInc][repNum])
                indvAtoms2.append(pic2AtomData[atomInc][repNum])
            psPic1AtomData[dataSetInc].append(indvAtoms1)
            psPic2AtomData[dataSetInc].append(indvAtoms2)
            if extraDataToPostSelect is not None:
                postSelectedExtraData[dataSetInc].append(extraDataToPostSelect[repNum])       
    return psPic1AtomData, psPic2AtomData, postSelectedExtraData

def getConditionHits(atomPresenceData, hitCondition, verbose=False):
    """
    Returns:
        ensembleHits (1D array of bool or int): ensembleHits[whichPicture] one to one with each picture in the 
        atomList. This list is the answer to whether a given picture matched the hit condition or not.
        If partial credit, it is instead of a bool an int which records the number of aotms in the picture. 
    """
    #assert(type(hitCondition) == type(tao.condition()))
    ensembleHits = []
    for picInc, _ in enumerate(atomPresenceData[0][0]):
        numMatch = 0
        for atomInc, whichAtom in enumerate(hitCondition.whichAtoms):
            #atoms = misc.transpose(atomPresenceData[hitCondition.whichPic[atomInc]])[picInc]
            needAtom = hitCondition.conditions[atomInc]
            if needAtom == None:
                continue # no requirement
            if ((atomPresenceData[hitCondition.whichPic[atomInc]][whichAtom][picInc] and needAtom)
                or (not atomPresenceData[hitCondition.whichPic[atomInc]][whichAtom][picInc] and not needAtom)):
                numMatch += 1
            #if (atoms[whichAtom] and needAtom) or (not atoms[whichAtom] and not needAtom):
            #    numMatch += 1
        hit = False
        if type(hitCondition.numRequired) == list:
            # interpret a list of numbers as an inclusive "or" condition.
            for num in hitCondition.numRequired:
                if (num == -1 and numMatch == len(hitCondition.whichAtoms)) or numMatch == num:
                    hit = True
        elif (hitCondition.numRequired == -1 and numMatch == len(hitCondition.whichAtoms)) or numMatch == hitCondition.numRequired:
            hit = True
        ensembleHits.append(hit)
    return ensembleHits


def getEnsembleHits(atomPresenceData, hitCondition=None, requireConsecutive=False, partialCredit=False):
    """
    This function determines whether an ensemble of atoms was hit in a given picture. Give it whichever
    picture data you need.

    NEW: this function is now involved in post-selection work

    Args:
        atomPresenceData (2D Array of bool): atomPresenceData[whichAtom][whichPicture]
            First dimension is atom index, second dimension is the picture, and the value is whether the
            given atom was present in the given picture. 
        hitCondition (1D array of bool or int):
            if 1D array of bool:
                The picture of the expected configuration which counts as a hit.
            if int:
                In this case, the number of atoms in the atomPresenceData which must be present to count as a hit.
        requireConsecutive (bool): 
            (only relevant for int hitCondition.) An option which specifies if the number of atoms 
            specified by the hit condition must be consequtive in the list in order to count as a hit. 
        partialCredit (bool):
            (only relevant for array hitCondition). An option which specifies if the user wants a relative
            measurement of how many atoms made it to the given configuration. Note: currently doesn't 
            actually use the hit condition for some reason?
    
    Returns:
        ensembleHits (1D array of bool or int): ensembleHits[whichPicture] one to one with each picture in the 
        atomList. This list is the answer to whether a given picture matched the hit condition or not.
        If partial credit, it is instead of a bool an int which records the number of aotms in the picture. 
        
    """
    if hitCondition is None:
        hitCondition = np.ones(atomPresenceData.shape[0])
    ensembleHits = []
    if type(hitCondition) is int:
        # condition is, e.g, 5 out of 6 of the ref pic, and if consecutive, all atoms should be connected somehow.
        for atoms in misc.transpose(atomPresenceData):
            matches = 0
            consecutive = True
            for atom in atoms:
                if atom:
                    matches += 1
                # else there's no atom. 3 possibilities: before string of atoms, after string, or in middle.
                # if in middle, consecutive requirement is not met.
                elif 0 < matches < hitCondition:
                    consecutive = False
            if requireConsecutive:
                ensembleHits.append((matches == hitCondition) and consecutive)
            else:
                ensembleHits.append(matches == hitCondition)
    else:
        if partialCredit:
            for inc, atoms in enumerate(misc.transpose(atomPresenceData)):
                ensembleHits.append(sum(atoms)) # / len(atoms))
        else:
            for inc, atoms in enumerate(misc.transpose(atomPresenceData)):
                ensembleHits.append(True)
                for atom, needAtom in zip(atoms, hitCondition):
                    if needAtom == None:
                        continue # no condition
                    if not atom and needAtom:
                        ensembleHits[inc] = False
                    if atom and not needAtom:
                        ensembleHits[inc] = False
                        
    return ensembleHits

def getAvgBorderCount(data, p, ppe):
    """
    data: the array of pictures
    p: which picture to start on
    ppe: pictures Per Experiment
    """
    if len(data.shape) == 4:
        rawData = data.reshape((data.shape[0] * data.shape[1], data.shape[2], data.shape[3]))
    else:
        rawData = data
    normFactor = (2*len(rawData[0][0][:])+2*len(rawData[0][:][0]))
    avgBorderCount = (np.sum(rawData[p::ppe,0,:], axis=1) + np.sum(rawData[p::ppe,-1,:], axis=1) 
                      + np.sum(rawData[p::ppe,:,0], axis=1) + np.sum(rawData[p::ppe,:,-1], axis=1)).astype(float)
    corners = rawData[p::ppe,0,0] + rawData[p::ppe,0,-1] + rawData[p::ppe,-1,0] + rawData[p::ppe,-1,-1]
    avgBorderCount -= corners
    avgBorderCount /= normFactor - 4
    return avgBorderCount

def normalizeData(data, atomLocation, picture, picturesPerExperiment, borders):
    """
    :param picturesPerExperiment:
    :param picture:
    :param subtractBorders:
    :param data: the array of pictures
    :param atomLocation: The location to analyze
    :return: The data at atomLocation with the background subtracted away (commented out at the moment).
    """
    allData = arr([])
    # if given data separated into different variations, flatten the variation separation.
    dimensions = data.shape
    if len(dimensions) == 4:
        rawData = data.reshape((data.shape[0] * data.shape[1], data.shape[2], data.shape[3]))
    else:
        rawData = data
    dimensions = rawData.shape
    count=0
    for imageInc in range(0, dimensions[0]):
        if (imageInc + picturesPerExperiment - picture) % picturesPerExperiment == 0:
            if len(atomLocation) != 2:
                raise TypeError('AtomLocation, which has value ' + str(atomLocation) + ', should be 2 elements.')
            if len(borders) <= count:
                raise IndexError('borders, of len ' + str(len(borders)), 'is not long enough!')
            try:
                allData = np.append(allData, rawData[imageInc][atomLocation[0]][atomLocation[1]] - borders[count])
            except IndexError:
                raise
            count += 1
    return allData

def guessGaussianPeaks(binCenters, binnedData):
    """
    This function guesses where the gaussian peaks of the data are. It assumes one is near the maximum of the binned
    data. Then, from the binned data it subtracts an over-weighted (i.e. extra tall) poissonion distribution e^-k k^n/n!
    From the binned data. This should squelch the peak that it found. It then assumes that the second peak is near the
    maximum of the (data-poissonian) array.
    :param binCenters: The pixel-numbers corresponding to the binned data data points.
    :param binnedData: the binned data data points.
    :return: the two guesses.
    """
    if len(binCenters) == 0 or len(binnedData) == 0:
        raise ValueError("inputted data was empty?!?" + str(binCenters) + str(binnedData))
    # This offset is to prevent negative x values while working with the poissonian. If set wrong guesses can start to work funny.
    # The offset is only use to get an appropriate width for the no-atoms peak. Arguably since I use this to manually shift the width, I should
    # just use a gaussian instead of a poissonian.
    randomOffset = 800
    poisonAmplitude = 2
    binCenters += randomOffset
    # get index corresponding to global max
    guess1Index = np.argmax(binnedData)
    # get location of global max
    guess1Location = binCenters[guess1Index]
    binnedDataNoPoissonian = []
    for binInc in range(0, len(binCenters)):
        binnedDataNoPoissonian.append(binnedData[binInc] 
                                      - poissonian.f(binCenters[binInc], guess1Location, poisonAmplitude * max(binnedData) /
                                                     poissonian.f(guess1Location, guess1Location, 1)))
    guess2Index = np.argmax(binnedDataNoPoissonian)
    guess2Location = binCenters[guess2Index]
    binCenters -= randomOffset
    return guess1Location - randomOffset, guess2Location - randomOffset

def getBinData(binWidth, data):
    # I feel like there's probably a better built in way to do this...
    if min(data) == max(data):
        raise ValueError("Data for binning was all same value of " + str(min(data)))
    binBorderLocation = min(data)
    binsBorders = arr([])
    while binBorderLocation < max(data):
        binsBorders = np.append(binsBorders, binBorderLocation)
        binBorderLocation = binBorderLocation + binWidth
    binnedData, trash = np.histogram(data, binsBorders)
    binCenters = binsBorders[0:binsBorders.size-1]
    return binCenters, binnedData

def getGenStatistics(genData, repetitionsPerVariation):
    # Take the previous data, which includes entries when there was no atom in the first picture, and convert it to
    # an array of just loaded and survived or loaded and died.
    genAverages = np.array([])
    genErrors = np.array([])
    if genData.size < repetitionsPerVariation:
        repetitionsPerVariation = genData.size
    for variationInc in range(0, int(genData.size / repetitionsPerVariation)):
        genList = np.array([])
        for repetitionInc in range(0, repetitionsPerVariation):
            if genData[variationInc * repetitionsPerVariation + repetitionInc] != -1:
                genList = np.append(genList, genData[variationInc * repetitionsPerVariation + repetitionInc])
        if genList.size == 0:
            # catch the case where there's no relevant data, typically if laser becomes unlocked.
            genErrors = np.append(genErrors, [0])
            genAverages = np.append(genAverages, [0])
        else:
            # normal case
            genErrors = np.append(genErrors, np.std(genList) / np.sqrt(genList.size))
            genAverages = np.append(genAverages, np.average(genList))
    return genAverages, genErrors

def getGenerationEvents(loadAtoms, finAtomsAtoms):
    """
    This is more or less the opposite of "GetSurvivalEvents". It counts events as +1 when you start with no atom and end
    with an atom. This could be used to characterize all sorts of things, e.g. hopping, background catches, rearranging, etc.
    :param loadAtoms:
    :param finAtomsAtoms:

    :return:
    """
    # this will include entries for when there is no atom in the first picture.
    genData = np.array([])
    genData.astype(int)
    # this doesn't take into account loss, since these experiments are feeding-back on loss.
    # there shoukld be a smarter / faster way to do this like the survival method.
    for atom1, atom2 in zip(loadAtoms, finAtomsAtoms):
        if atom1:
            # not interesting for generation
            genData = np.append(genData, [-1])
        elif atom2:
            # atom was generated.
            genData = np.append(genData, [1])
        else:
            # never an atom.
            genData = np.append(genData, [0])
    return genData

def groupEventsIntoVariations(bareList, repsPerVar):
    varList = [None for _ in range(int(bareList.size / repsPerVar))]
    for varInc in range(0, int(bareList.size / repsPerVar)):
        varList[varInc] = arr([x for x in bareList[varInc * repsPerVar:(varInc+1) * repsPerVar] if x != -1])
    return varList

def getAvgPic(picSeries):
    if len(picSeries.shape) == 3:
        avgPic = np.zeros(picSeries[0].shape)
        for pic in picSeries:
            avgPic += pic
        avgPic = avgPic / len(picSeries)
        return avgPic
    elif len(picSeries.shape) == 4:
        avgPic = np.zeros(picSeries[0][0].shape)
        for variation in picSeries:
            for pic in variation:
                avgPic += pic
        avgPic = avgPic / (len(picSeries) * len(picSeries[0]))
        return avgPic

def getAvgPics(pics, picsPerRep=2):
    if len(pics.shape) == 3:
        avgPics = []
        for picNum in range(picsPerRep):
            avgPic = np.zeros(pics[0].shape)
            for pic_inc in range(int(pics.shape[0]/picsPerRep)):
                avgPic += pics[int(pic_inc * picsPerRep) + picNum]
            avgPics.append(avgPic / (len(pics)/picsPerRep))
        return avgPics
    elif len(pics.shape) == 4:
        avgPics = []
        for picNum in range(picsPerRep):
            avgPic = np.zeros(pics[0][0].shape)
            for var in pics:
                for pic_inc in range(int(var.shape[0]/picsPerRep)):
                    avgPic += var[int(pic_inc * picsPerRep) + picNum]
            avgPics.append(avgPic / (len(pics)/picsPerRep))
        return avgPics
    else:
        raise ValueError(pics.shape)

def processSingleImage(rawData, bg, window, xMin, xMax, yMin, yMax, accumulations, zeroCorners, smartWindow,
                       manuallyAccumulate=True):
    """
    Process the original data, giving back data that has been ordered and windowed as well as two other versions that
    have either the background or the average of the pictures subtracted out.

    This is a helper function that is expected to be embedded in a package. As such, many parameters are simply
    passed through some other function in order to reach this function, and all parameters are required.
    """
    # handle manual accumulations, where the code just sums pictures together.
    if manuallyAccumulate and not len(rawData.shape) == 3:
        print('ERROR: Requested manual accumulation but raw data doesn"t have the correct shape for that.')
    if manuallyAccumulate:
        avgPics = np.zeros((rawData.shape[1], rawData.shape[2]))
        count = 0
        for pic in rawData:
            avgPics += pic
            count += 1
        rawData = avgPics
    # handle windowing defaults
    allXPts = np.arange(1, rawData.shape[1])
    allYPts = np.arange(1, rawData.shape[0])

    if smartWindow:
        maxLocs = coordMax(rawData)
        xMin = maxLocs[1] - rawData.shape[1] / 5
        xMax = maxLocs[1] + rawData.shape[1] / 5
        yMin = maxLocs[0] - rawData.shape[0] / 5
        yMax = maxLocs[0] + rawData.shape[0] / 5
    elif window != (0, 0, 0, 0):
        xMin = window[0]
        xMax = window[1]
        yMin = window[2]
        yMax = window[3]
    else:
        if xMax == 0:
            xMax = len(rawData[0])
        if yMax == 0:
            yMax = len(rawData)
        if xMax < 0:
            xMax = 0
        if yMax < 0:
            yMax = 0

    xPts = allXPts[xMin:xMax]
    yPts = allYPts[yMin:yMax]

    # window images.
    rawData = np.copy(arr(rawData[yMin:yMax, xMin:xMax]))

    # final normalized data
    normData = rawData / accumulations

    # ### -Background Analysis
    # if user just entered a number, assume that it's a file number.
    if type(bg) == int and not bg == 0:
        print('loading background file ', bg)
        bg, _, _, _ = loadHDF5(bg)
        if manuallyAccumulate:
            avgPics = np.zeros((bg.shape[1], bg.shape[2]))
            count = 0
            for pic in bg:
                avgPics += pic
                count += 1
            bg = avgPics
        else:
            bg = bg[0]
        bg /= accumulations
    # window the background
    if not bg.size == 1:
        bg = np.copy(arr(bg[yMin:yMax, xMin:xMax]))
    dataMinusBg = np.copy(normData)
    dataMinusBg -= bg

    # it's important and consequential that the zeroing here is done after the background / corner is subtracted.
    if zeroCorners:
        cornerAvg = (pic[0, 0] + pic[0, -1] + pic[-1, 0] + pic[-1, -1]) / 4
        dataMinusBg -= cornerAvg
        cornerAvg = (pic[0, 0] + pic[0, -1] + pic[-1, 0] + pic[-1, -1]) / 4
        normData -= cornerAvg
    return normData, dataMinusBg, xPts, yPts

def processImageData(key, rawData, bg, window, accumulations, dataRange, zeroCorners,
                     smartWindow, manuallyAccumulate=False):
    """
    Process the orignal data, giving back data that has been ordered and windowed as well as two other versions that
    have either the background or the average of the pictures subtracted out.

    This is a helper function that is expected to be embedded in a package. As such, many parameters are simply
    passed through some other function in order to reach this function, and all parameters are required.
    """
    # handle windowing defaults
    if smartWindow:
        maxLocs = []
        for dat in rawData:
            maxLocs.append(coordMax(dat))
        maxLocs = arr(maxLocs)
        xMin = min(maxLocs[:, 0])
        xMax = max(maxLocs[:, 0])
        yMin = min(maxLocs[:, 1])
        yMax = max(maxLocs[:, 1])
        xRange = rawData.shape[2] / 2
        yRange = rawData.shape[1] / 2
        if xRange < xMax - xMin:
            xRange = xMax - xMin
        if yRange < yMax - yMin:
            yRange = yMax - yMin
        xMin -= 0.2 * xRange
        xMax += 0.2 * xRange
        yMin -= 0.2 * yRange
        yMax += 0.2 * yRange
        window = pw.PictureWindow( xMin, xMax, yMin, yMax )
    if manuallyAccumulate:
        # ignore shape[1], which is the number of pics in each variation. These are what are getting averaged.
        avgPics = np.zeros((int(rawData.shape[0] / accumulations), rawData.shape[1], rawData.shape[2]))
        varCount = 0
        for var in avgPics:
            for picNum in range(accumulations):
                var += rawData[varCount * accumulations + picNum]
            varCount += 1
        rawData = avgPics    
    
    if rawData.shape[0] != len(key):
        raise ValueError("ERROR: number of pictures (after manual accumulations) " + str(rawData.shape[0]) + 
                         " data doesn't match length of key " + str(len(key)) + "!")
    # combine and order data.
    rawData, key = combineData(rawData, key)
    rawData, key, _ = orderData(rawData, key)

    # window images.
    rawData = np.array([window.window(pic) for pic in rawData])
    # pull out the images to be used for analysis.
    if dataRange is not None:
        rawData = rawData[dataRange[0]:dataRange[-1]]
        key = key[dataRange[0]:dataRange[-1]]
        # final normalized data
    normData = rawData / accumulations

    # ### -Background Analysis
    # if user just entered a number, assume that it's a file number.
    if type(bg) == int and not bg == 0:
        with exp.ExpFile() as fid:
            fid.open_hdf5(bg)
            bg = np.mean(fid.get_basler_pics(),0)
    # window the background
    if not bg.size == 1:
        bg = np.array(window.window(bg))
    dataMinusBg = np.copy(normData)
    for pic in dataMinusBg:
        pic -= bg
    # ### -Average Analysis
    # make a picture which is an average of all pictures over the run.
    avgPic = 0
    for pic in normData:
        avgPic += pic
    avgPic /= len(normData)
    dataMinusAvg = np.copy(normData)
    for pic in dataMinusAvg:
        pic -= avgPic

    # it's important and consequential that the zeroing here is done after the background / corner is subtracted.
    if zeroCorners:
        for pic in dataMinusBg:
            cornerAvg = (pic[0, 0] + pic[0, -1] + pic[-1, 0] + pic[-1, -1]) / 4
            pic -= cornerAvg
        for pic in dataMinusAvg:
            cornerAvg = (pic[0, 0] + pic[0, -1] + pic[-1, 0] + pic[-1, -1]) / 4
            pic -= cornerAvg
        for pic in normData:
            cornerAvg = (pic[0, 0] + pic[0, -1] + pic[-1, 0] + pic[-1, -1]) / 4
            pic -= cornerAvg
    return key, normData, dataMinusBg, dataMinusAvg, avgPic

def unpackAtomLocations(locs, avgPic=None):
    """
    :param locs: expects locs to be format [bottomLeftRow, bottomLeftColumn, spacing, width, height]
    :return: a list of the coordinates of each tweezer in the image.
    """
    if type(locs) == type(9): # location is an integer??? I forget how this was used...
        return locs 
    if not (type(locs[0]) == int): # already unpacked
        return locs
    # assume atom grid format.
    bottomLeftRow, bottomLeftColumn, spacing, width, height = locs
    locArray = []
    for widthInc in range(width):
        for heightInc in range(height):
            locArray.append([bottomLeftRow + spacing * heightInc, bottomLeftColumn + spacing * widthInc])
    # this option looks for the X brightest spots in the average picture and assumes that this is where the 
    # atoms are. Note that it can mess up the ordering of different locations.
    if type(locArray) == type(9) and avgPic is not None:
        res = np.unravel_index(avgPic.flatten().argsort()[-locArray:][::-1],avgPic.shape)
        locArray = [x for x in zip(res[0],res[1])]
    return locArray

def getGridDims(locs):
    bottomLeftRow, bottomLeftColumn, spacing, width, height = locs
    return width, height

def sliceMultidimensionalData(dimSlice, origKey, rawData, varyingDim=None):
    """
    :param dimSlice: e.g. [80, None]
    :param origKey:
    :param rawData:
    :param varyingDim:
    :return:
    """
    key = origKey[:]
    if dimSlice is not None:
        runningKey = key[:]
        runningData = rawData[:]
        for dimnum, dimSpec in enumerate(dimSlice):
            if dimSpec is None:
                varyingDim = dimnum
                continue
            tempKey = []
            tempData = []
            for elemnum, elem in enumerate(misc.transpose(runningKey)[dimnum]):
                if abs(elem - dimSpec) < 1e-6:
                    tempKey.append(runningKey[elemnum])
                    tempData.append(runningData[elemnum])
            runningKey = tempKey[:]
            runningData = tempData[:]
        key = runningKey[:]
        rawData = runningData[:]
    otherDimValues = None
    if varyingDim is not None:
        otherDimValues = []
        print('key',key)
        for keyVal in key:
            otherDimValues.append('')
            for valNum, dimVal in enumerate(keyVal):
                if not valNum == varyingDim:
                    otherDimValues[-1] += str(dimVal) + ","
    if dimSlice is not None:
        key = arr(misc.transpose(key)[varyingDim])
    if varyingDim is None and len(arr(key).shape) > 1:
        key = arr(misc.transpose(key)[0])
    return arr(key), arr(rawData), otherDimValues, varyingDim

def applyDataRange(dataRange, groupedDataRaw, key):
    
    if dataRange is not None:
        if type(groupedDataRaw) == type({}):
            
            groupedData, newKey = {}, []
            for variNum, keyVal in enumerate(key):
                keyValStr = misc.round_sig_str(keyVal)
                if variNum in dataRange:
                    groupedData[keyValStr] = groupedDataRaw[keyValStr]
                    newKey.append(key[variNum])
            key = arr(newKey)
        else:
            #Expects a multidimensional array
            groupedData, newKey = [[] for _ in range(2)]
            for variNum, varPics in enumerate(groupedDataRaw):
                if variNum in dataRange:
                    groupedData.append(varPics)
                    newKey.append(key[variNum])
            groupedData = arr(groupedData)
            key = arr(newKey)
    else:
        groupedData = groupedDataRaw
    return key, groupedData

def getNetLossStats(netLoss, reps):
    lossAverages = np.array([])
    lossErrors = np.array([])
    for variationInc in range(0, int(len(netLoss) / reps)):
        lossList = np.array([])
        # pull together the data for only this variation
        for repetitionInc in range(0, reps):
            if netLoss[variationInc * reps + repetitionInc] != -1:
                lossList = np.append(lossList, netLoss[variationInc * reps + repetitionInc])
        if lossList.size == 0:
            # catch the case where there's no relevant data, typically if laser becomes unlocked.
            lossErrors = np.append(lossErrors, [0])
            lossAverages = np.append(lossAverages, [0])
        else:
            # normal case, compute statistics
            lossErrors = np.append(lossErrors, np.std(lossList)/np.sqrt(lossList.size))
            lossAverages = np.append(lossAverages, np.average(lossList))
    return lossAverages, lossErrors

def getNetLoss(pic1Atoms, pic2Atoms):
    """
    Calculates the net loss fraction for every experiment. Assumes 2 pics per experiment.
    Useful for experiments where atoms move around, e.g. rearranging.
    """
    netLoss = []
    for inc, (atoms1, atoms2) in enumerate(zip(misc.transpose(pic1Atoms), misc.transpose(pic2Atoms))):
        loadNum, finNum = [0.0 for _ in range(2)]

        for atom1, atom2 in zip(atoms1, atoms2):
            if atom1:
                loadNum += 1.0
            if atom2:
                finNum += 1.0
        if loadNum == 0:
            netLoss.append(0)
        else:
            netLoss.append(1 - float(finNum) / loadNum)
    return netLoss

def getAtomInPictureStatistics(atomsInPicData, reps):
    """
    assumes atomsInPicData is a 2D array. atomsInPicData[0,:] refers to all of the atom events for a single location,
    atomsInPicData[1,:] refers to all the events for the second, etc.
    """
    stats = []
    for singleLocData in atomsInPicData:
        singleLocData = arr(singleLocData)
        variationData = singleLocData.reshape([int(len(singleLocData)/reps), reps])
        avgs = [np.average(singleVarData) for singleVarData in variationData]
        errs = [np.std(singleVarData)/np.sqrt(len(singleVarData)) for singleVarData in variationData]
        stats.append({'avg': avgs, 'err': errs})
    return stats

def getEnhancement(loadAtoms, assemblyAtoms, normalized=False):
    """
    determines how many atoms were added to the assembly, another measure of how well the rearranging is working.
    """
    enhancement = []
    for inc, (loaded, assembled) in enumerate(zip(misc.transpose(loadAtoms), misc.transpose(assemblyAtoms))):
        enhancement.append(sum(assembled) - sum(loaded))
        if normalized:
            enhancement[-1] /= len(assembled)
    return enhancement

def getEnsembleStatistics(ensembleData, reps):
    """
    EnsembleData is a list of "hits" of the deisgnated ensemble of atoms in a given picture, for different variations.
    This function calculates some statistics on that list.
    """
    ensembleAverages = np.array([])
    ensembleErrors = np.array([])
    for variationInc in range(0, int(len(ensembleData) / reps)):
        ensembleList = np.array([])
        # pull together the data for only this variation
        for repetitionInc in range(0, reps):
            if ensembleData[variationInc * reps + repetitionInc] != -1:
                ensembleList = np.append(ensembleList, ensembleData[variationInc * reps + repetitionInc])
        if ensembleList.size == 0:
            # catch the case where there's no relevant data, typically if laser becomes unlocked.
            ensembleErrors = np.append(ensembleErrors, [0])
            ensembleAverages = np.append(ensembleAverages, [0])
        else:
            # normal case, compute statistics
            ensembleErrors = np.append(ensembleErrors, np.std(ensembleList)/np.sqrt(ensembleList.size))
            ensembleAverages = np.append(ensembleAverages, np.average(ensembleList))
    ensembleStats = {'avg': ensembleAverages, 'err': ensembleErrors}
    return ensembleStats

