__version__ = "1.4"
"""
Recent Changes:
- General Cleaning up code
"""

from numpy import array as arr
from pandas import DataFrame

from Miscellaneous import getStats, round_sig
from MarksFourierAnalysis import fft
from matplotlib.pyplot import *
from scipy.optimize import curve_fit as fit

import FittingFunctions as fitFunc

from AnalysisHelpers import (loadHDF5, getAvgPic, getBinData, getEnsembleHits, getEnsembleStatistics,
                             handleFitting, groupMultidimensionalData, getNetLoss,
                             getAtomInPictureStatistics, normalizeData, fitDoubleGaussian,
                             getSurvivalData, getSurvivalEvents, unpackAtomLocations, guessGaussianPeaks,
                             calculateAtomThreshold, outputDataToMmaNotebook, getLoadingData, orderData,
                             postSelectOnAssembly, fitWithClass, getNetLossStats, organizeTransferData,
                             getGenerationEvents, getGenStatistics)
import fitters.linear

def analyzeCodeTimingData(num, talk=True, numTimes=3):
    """
    Analyzing code timing data. Data is outputted in the following format:
    numTimes total times taken for a given experiment repetition.
    Each time for a given experiment is outputted on a line with a space between the different times measured that rep.
    Each experiment repetition outputted on a new line.
    """
    filename = ("J:\\Data Repository\\New Data Repository\\2017\\September\\September 8"
                "\\Raw Data\\rearrangementlog" + str(num) + ".txt")
    with open(filename) as f:
        num_lines = sum(1 for _ in open(filename)) - 1
        allTimes = [[0] * num_lines for _ in range(numTimes)]
        totalTime = [0] * num_lines
        names = ["" for _ in range(numTimes)]
        for count, line in enumerate(f):
            if count == 0:
                for i, name in enumerate(line.strip('\n').split(' ')):
                    names[i] = name
                continue
            eventTimes = line.strip('\n').split(' ')
            totalTime[count-1] = np.sum(arr(eventTimes).astype(float))
            for inc, time in enumerate(eventTimes):
                allTimes[inc][count-1] = time
        if talk:
            for inc, timeInterval in enumerate(allTimes):
                print(names[inc])
                getStats(arr(timeInterval).astype(float))
                print('\n')
            print('Total Time:')
            getStats(totalTime)
            print('\n')
        return allTimes


def analyzeNiawgWave(fileIndicator, ftPts=None):
    """
    fileIndicator: can be a number (in which case assumes Debug-Output folder), or a full file address

    Analysis is based on a simple format where each (interweaved) value is outputted to a file one after another.
    :param fileIndicator:
    :return tPts, chan1, chan2, fftInfoC1, fftInfoC2
    """
    if isinstance(fileIndicator, int):
        address = ('C:/Users/Mark-Brown/Chimera-Control/Debug-Output/Wave_' + str(fileIndicator) + '.txt')
    else:
        address = fileIndicator
    # current as of october 15th, 2017
    sampleRate = 320000000
    with open(address) as f:
        data = []
        for line in f:
            for elem in line.split(' ')[:-1]:
                data.append(float(elem))
        chan1 = data[::2]
        chan2 = data[1::2]
        tPts = [t / sampleRate for t in range(len(chan1))]
        if ftPts is None:
            fftInfoC1 = fft(chan1, tPts, normalize=True)
            fftInfoC2 = fft(chan2, tPts, normalize=True)
        else:
            fftInfoC1 = fft(chan1[:ftPts], tPts[:ftPts], normalize=True)
            fftInfoC2 = fft(chan2[:ftPts], tPts[:ftPts], normalize=True)
        # returns {'Freq': freqs, 'Amp': fieldFFT}
        return tPts, chan1, chan2, fftInfoC1, fftInfoC2


def analyzeScatterData(fileNumber, atomLocs1, connected=False, loadPic=1, transferPic=2, picsPerRep=3,
                       subtractEdgeCounts=True, histSecondPeakGuess=False, manualThreshold=None,
                       normalizeForLoadingRate=False, **transferOrganizeArgs):
    """
        does all the post-selection conditions and only one import of the data. previously I did this by importing the data
        for each condition.

    :param fileNumber:
    :param atomLocs1:
    :param connected:
    :param loadPic:
    :param transferPic:
    :param picsPerRep:
    :param subtractEdgeCounts:
    :param histSecondPeakGuess:
    :param manualThreshold:
    :param normalizeForLoadingRate:
    :param transferOrganizeArgs:
    :return:
    """

    (groupedData, atomLocs1, atomLocs2, keyName, repetitions,
     key) = organizeTransferData(fileNumber, atomLocs1, atomLocs1, picsPerRep=picsPerRep,
                                 **transferOrganizeArgs)
    # initialize arrays
    (pic1Data, pic2Data, atomCounts, bins, binnedData, thresholds,
     loadingRate, pic1Atoms, pic2Atoms, survivalFits) = arr([[None] * len(atomLocs1)] * 10)
    survivalData, survivalErrs = [[[] for _ in range(len(atomLocs1))] for _ in range(2)]
    for i, (loc1, loc2) in enumerate(zip(atomLocs1, atomLocs2)):
        pic1Data[i] = normalizeData(groupedData, loc1, loadPic, picsPerRep, subtractBorders=subtractEdgeCounts)
        pic2Data[i] = normalizeData(groupedData, loc2, transferPic, picsPerRep, subtractBorders=subtractEdgeCounts)
        atomCounts[i] = arr([a for a in arr(list(zip(pic1Data[i], pic2Data[i]))).flatten()])
        bins[i], binnedData[i] = getBinData(10, pic1Data[i])
        guess1, guess2 = guessGaussianPeaks(bins[i], binnedData[i])
        guess = arr([max(binnedData[i]), guess1, 30, max(binnedData[i]) * 0.75,
                     200 if histSecondPeakGuess is not None else histSecondPeakGuess, 10])
        gaussianFitVals = fitDoubleGaussian(bins[i], binnedData[i], guess)
        thresholds[i], thresholdFid = (((manualThreshold, 0) if manualThreshold is not None
                                        else calculateAtomThreshold(gaussianFitVals)))
        pic1Atoms[i], pic2Atoms[i] = [[] for _ in range(2)]
        for point1, point2 in zip(pic1Data[i], pic2Data[i]):
            pic1Atoms[i].append(point1 > thresholds[i])
            pic2Atoms[i].append(point2 > thresholds[i])

    key, psSurvivals, psErrors = [], [], []
    for condition in range(len(pic2Atoms)):
        tempData, tempErr, tempRate = arr([[None for _ in range(len(atomLocs1))] for _ in range(3)])
        temp_pic1Atoms, temp_pic2Atoms = postSelectOnAssembly(pic1Atoms, pic2Atoms, condition + 1,
                                                              connected=connected)
        tempData = arr(tempData.tolist())
        if len(temp_pic1Atoms[0]) != 0:
            for i in range(len(atomLocs1)):
                survivalList = getSurvivalEvents(temp_pic1Atoms[i], temp_pic2Atoms[i])
                tempData[i], tempErr[i], loadingRate[i] = getSurvivalData(survivalList, repetitions)
            # weight the sum with loading percentage
            if normalizeForLoadingRate:
                psSurvivals.append(sum(tempData * loadingRate) / sum(loadingRate))
                # the errors here are not normalized for the loading rate!
                psErrors.append(np.sqrt(np.sum(tempErr ** 2)) / len(atomLocs1))
            else:
                # print('condition', condition, tempData, np.mean(tempData))
                psSurvivals.append(np.mean(tempData))
                psErrors.append(np.sqrt(np.sum(tempErr ** 2)) / len(atomLocs1))
            key.append(condition + 1)
        for i in range(len(atomLocs1)):
            survivalData[i] = np.append(survivalData[i], tempData[i])
            survivalErrs[i] = np.append(survivalErrs[i], tempErr[i])
    key = arr(key)
    psErrors = arr(psErrors)
    psSurvivals = arr(psSurvivals)
    fitInfo, fitFinished = fitWithClass(fitters.linear, key, psSurvivals.flatten(), errs=psErrors.flatten())
    for i, (data, err) in enumerate(zip(survivalData, survivalErrs)):
        survivalFits[i], _ = fitWithClass(fitters.linear, key, data.flatten(), errs=err.flatten())
    return key, psSurvivals, psErrors, fitInfo, fitFinished, survivalData, survivalErrs, survivalFits, atomLocs1


def standardTransferAnalysis(fileNumber, atomLocs1, atomLocs2, key=None, picsPerRep=2, manualThreshold=None,
                             fitModule=None, window=None, xMin=None, xMax=None, yMin=None, yMax=None, dataRange=None,
                             histSecondPeakGuess=None, keyOffset=0, outputMma=False, dimSlice=None,
                             varyingDim=None, subtractEdgeCounts=True, loadPic=0, transferPic=1,
                             postSelectionCondition=None, groupData=False, quiet=False, postSelectionConnected=False,
                             getGenerationStats=True, normalizeForLoadingRate=False, rerng=False):
    """
    Standard data analysis package for looking at survival rates throughout an experiment.
    Returns key, survivalData, survivalErrors

    :param fileNumber: for the HDF5 File. Path is automatic.
    :param atomLocs1: usually the loading picture arrangement.
    :param atomLocs2:
    :param key: manually entered key. Overrides key from HDF5 file.
    :param manualThreshold:
    :param fitModule: a submodule from the fitters module for fitting.
    :param window: (left, top, right bottom)? quick way of setting xmin/max and ymin/max.
    :param xMin:
    :param xMax:
    :param yMin:
    :param yMax:
    :param dataRange: which data points to use. 0-indexed. For example, use this to remove bad points to do a proper
        fit with the remaining points.
    :param histSecondPeakGuess:
    :param keyOffset: added to all key values. e.g. for microwave, this can be set to ~6.34e9 to get the actual freqs
        on the x-axis.
    :param outputMma: an option used by tobias and Yiheng because they wanted to work in mathematica (for some reason).
    :param dimSlice: for multidimensional scans.
    :param varyingDim: for multidimensional scans.
    :param subtractEdgeCounts: subtracts background from each picture individually. Background calculated as the average
        from the edge of the picture.
    :param loadPic: 0-indexed.
    :param transferPic: 0-indexed.
    :param picsPerRep: e.g. 1 for simple load, 2 for simple survival, 3 for rearrange-based things or probe images, etc.
    :param postSelectionCondition: can be "which atoms of the atomLocs1 to require" or if an int, this is the # of
        atomLocs required during the post-selection. e.g. "5" when atomLocs is 6 locs means 5/6 locations must have been
        loaded.
    :param rerng: if true, set loadPic=1, transferPic=2, picsPerRep=3. This is a simple shortcut.
    :param groupData: collapses many variations into a single data point.
    :param quiet: doesn't output text during analysis if true.
    :param postSelectionConnected: requrie that a # of atoms required by the postSelectionCondition arg are
        consecutive.
    :param getGenerationStats: get "generation" events, where no atom was loaded but an atom appears in the second pic.
    :param normalizeForLoadingRate:

    :return: a lot.
    """
    if rerng:
        loadPic, transferPic, picsPerRep = 1, 2, 3
    (groupedData, atomLocs1, atomLocs2, keyName,
     repetitions, key) = organizeTransferData(fileNumber, atomLocs1, atomLocs2, key=key, window=window, xMin=xMin, xMax=xMax,
                                         yMin=yMin, yMax=yMax, dataRange=dataRange, keyOffset=keyOffset,
                                         picsPerRep=picsPerRep, dimSlice=dimSlice, varyingDim=varyingDim,
                                         groupData=groupData, quiet=quiet)
    avgPic = getAvgPic(groupedData)
    # initialize arrays
    (pic1Data, pic2Data, atomCounts, bins, binnedData, thresholds, survivalData, survivalErrs,
     loadingRate, pic1Atoms, pic2Atoms, genAvgs, genErrs) = arr([[None] * len(atomLocs1)] * 13)
    for i, (loc1, loc2) in enumerate(zip(atomLocs1, atomLocs2)):
        pic1Data[i] = normalizeData(groupedData, loc1, loadPic, picsPerRep, subtractBorders=subtractEdgeCounts)
        pic2Data[i] = normalizeData(groupedData, loc2, transferPic, picsPerRep, subtractBorders=subtractEdgeCounts)
        atomCounts[i] = arr([a for a in arr(list(zip(pic1Data[i], pic2Data[i]))).flatten()])
        bins[i], binnedData[i] = getBinData(10, pic1Data[i])
        guess1, guess2 = guessGaussianPeaks(bins[i], binnedData[i])
        guess = arr([max(binnedData[i]), guess1, 30, max(binnedData[i])*0.75,
                     200 if histSecondPeakGuess is not None else histSecondPeakGuess, 10])
        gaussianFitVals = fitDoubleGaussian(bins[i], binnedData[i], guess)
        thresholds[i], thresholdFid = (((manualThreshold, 0) if manualThreshold is not None
                                       else calculateAtomThreshold(gaussianFitVals)))
        pic1Atoms[i], pic2Atoms[i] = [[] for _ in range(2)]
        for point1, point2 in zip(pic1Data[i], pic2Data[i]):
            pic1Atoms[i].append(point1 > thresholds[i])
            pic2Atoms[i].append(point2 > thresholds[i])

    if postSelectionCondition is not None:
        pic1Atoms, pic2Atoms = postSelectOnAssembly(pic1Atoms, pic2Atoms, postSelectionCondition,
                                                    connected=postSelectionConnected)

    for i in range(len(atomLocs1)):
        survivalList = getSurvivalEvents(pic1Atoms[i], pic2Atoms[i])
        survivalData[i], survivalErrs[i], loadingRate[i] = getSurvivalData(survivalList, repetitions)
        if getGenerationStats:
            genList = getGenerationEvents(pic1Atoms[i], pic2Atoms[i])
            genAvgs[i], genErrs[i] = getGenStatistics(genList, repetitions)
        else:
            genAvgs[i], genErrs[i] = [None, None]
    (key, locationsList, survivalErrs, survivalData, loadingRate,
     otherDims) = groupMultidimensionalData(key, varyingDim, atomLocs1, survivalData, survivalErrs, loadingRate)
    # need to change for loop!
    fits = [None] * len(locationsList)
    if fitModule is not None:
        for i, _ in enumerate(locationsList):
            fits[i], _ = fitWithClass(fitModule, key, survivalData[i])
    pic1Data = arr(pic1Data.tolist())
    atomCounts = arr(atomCounts.tolist())
    # calculate average values
    avgSurvivalData, avgSurvivalErr, avgFit = [None]*3
    # weight the sum with loading percentage
    if normalizeForLoadingRate:
        avgSurvivalData = sum(survivalData*loadingRate)/sum(loadingRate)
    else:
        avgSurvivalData = np.mean(survivalData)
    avgSurvivalErr = np.sqrt(np.sum(survivalErrs**2))/len(atomLocs1)

    if fitModule is not None:
        avgFit, _ = fitWithClass(fitModule, key, avgSurvivalData)
    if outputMma:
        outputDataToMmaNotebook(fileNumber, survivalData, survivalErrs, loadingRate, key)
    return (atomLocs1, atomLocs2, atomCounts, survivalData, survivalErrs, loadingRate, pic1Data, keyName, key,
            repetitions, thresholds, fits, avgSurvivalData, avgSurvivalErr, avgFit, avgPic, otherDims, locationsList,
            genAvgs, genErrs)


def standardLoadingAnalysis(fileNum, atomLocations, picsPerRep=1, analyzeTogether=False, loadingPicture=0,
                            manualThreshold=None, loadingFitModule=None, showTotalHist=True, keyInput=None):
    """
    
    :param fileNum:
    :param atomLocations:
    :param picsPerRep:
    :param analyzeTogether:
    :param loadingPicture:
    :param manualThreshold:
    :param loadingFitModule:
    :param showTotalHist: 
    :param keyInput: if not none, this will be used instead of the key in the HDF5 file. 
    :return: pic1Data, thresholds, avgPic, key, loadingRateErr, loadingRateList, allLoadingRate, allLoadingErr, loadFits,
            loadingFitType, keyName, totalPic1AtomData, rawData, showTotalHist, atomLocations, avgFits
    """
    atomLocations = unpackAtomLocations(atomLocations)
    rawData, keyName, key, repetitions = loadHDF5(fileNum)
    numOfPictures = rawData.shape[0]
    numOfVariations = int(numOfPictures / (repetitions * picsPerRep))
    # handle defaults.
    if numOfVariations == 1:
        if showTotalHist is None:
            showTotalHist = False
        if key is None:
            key = arr([0])
    else:
        if showTotalHist is None:
            showTotalHist = True
        if key is None:
            key = arr([])
    if keyInput is not None:
        key = keyInput
    # make it the right size
    if len(arr(atomLocations).shape) == 1:
        atomLocations = [atomLocations]
    print("Key Values, in experiment's order: ", key)
    print('Total # of Pictures:', numOfPictures)
    print('Number of Variations:', numOfVariations)
    if not len(key) == numOfVariations:
        raise ValueError("ERROR: The Length of the key doesn't match the data found.")
    # ## Initial Data Analysis
    s = rawData.shape
    if analyzeTogether:
        newShape = (1, s[0], s[1], s[2])
    else:
        newShape = (numOfVariations, repetitions * picsPerRep, s[1], s[2])
    groupedData = rawData.reshape(newShape)
    groupedData, key, _ = orderData(groupedData, key)
    print('Data Shape:', groupedData.shape)
    loadingRateList, loadingRateErr, loadFits = [[[] for _ in range(len(atomLocations))] for _ in range(3)]
    print('Analyzing Variation... ', end='')
    allLoadingRate, allLoadingErr = [[[]] * len(groupedData) for _ in range(2)]
    totalPic1AtomData = []
    (pic1Data, pic1Atom, thresholds, thresholdFid, fitVals, bins, binData,
     atomCount) = arr([[[None for _ in atomLocations] for _ in groupedData] for _ in range(8)])
    for dataInc, data in enumerate(groupedData):
        print(str(dataInc) + ', ', end='')
        # fill lists with data
        allAtomPicData = []
        for i, atomLoc in enumerate(atomLocations):
            (pic1Data[dataInc][i], pic1Atom[dataInc][i], thresholds[dataInc][i], thresholdFid[dataInc][i],
             fitVals[dataInc][i], bins[dataInc][i], binData[dataInc][i],
             atomCount[dataInc][i]) = getLoadingData(data, atomLoc, loadingPicture, picsPerRep, manualThreshold,
                                                     10)
            totalPic1AtomData.append(pic1Atom[dataInc][i])
            allAtomPicData.append(np.mean(pic1Atom[dataInc][i]))
            loadingRateList[i].append(np.mean(pic1Atom[dataInc][i]))
            loadingRateErr[i].append(np.std(pic1Atom[dataInc][i]) / np.sqrt(len(pic1Atom[dataInc][i])))
        allLoadingRate[dataInc] = np.mean(allAtomPicData)
        allLoadingErr[dataInc] = np.std(allAtomPicData) / np.sqrt(len(allAtomPicData))
    for i, load in enumerate(loadingRateList):
        loadFits[i] = handleFitting(loadingFitModule, key, load)
    avgFits = handleFitting(loadingFitModule, key, allLoadingRate)
    avgPic = getAvgPic(rawData)
    # get averages across all variations
    (pic1Data, pic1Atom, thresholds, thresholdFid, fitVals, bins, binData,
     atomCount) = arr([[None] * len(atomLocations)] * 8)
    for i, atomLoc in enumerate(atomLocations):
        (pic1Data[i], pic1Atom[i], thresholds[i], thresholdFid[i],
         fitVals[i], bins[i], binData[i], atomCount[i]) = getLoadingData(rawData, atomLoc, loadingPicture,
                                                                         picsPerRep, manualThreshold, 5)
    pic1Data = arr(pic1Data.tolist())
    return (pic1Data, thresholds, avgPic, key, loadingRateErr, loadingRateList, allLoadingRate, allLoadingErr, loadFits,
            loadingFitModule, keyName, totalPic1AtomData, rawData, showTotalHist, atomLocations, avgFits)


def standardAssemblyAnalysis(fileNumber, atomLocs1, pic1Num, atomLocs2=None, keyOffset=0,
                             window=None, picsPerRep=2, dataRange=None, histSecondPeakGuess=None,
                             manualThreshold=None, fitModule=None, allAtomLocs1=None, allAtomLocs2=None, keyInput=None):
    """
    
    :param fileNumber:
    :param atomLocs1: 
    :param pic1Num: 
    :param atomLocs2: 
    :param keyOffset: 
    :param window: 
    :param picsPerRep: 
    :param dataRange: 
    :param histSecondPeakGuess: 
    :param manualThreshold: 
    :param fitModule: 
    :param allAtomLocs1: 
    :param allAtomLocs2: 
    :return: 
    """
    atomLocs1 = unpackAtomLocations(atomLocs1)
    atomLocs2 = (atomLocs1[:] if atomLocs2 is None else unpackAtomLocations(atomLocs2))
    allAtomLocs1 = (atomLocs1[:] if allAtomLocs1 is None else unpackAtomLocations(allAtomLocs1))
    allAtomLocs2 = (allAtomLocs1[:] if allAtomLocs2 is None else unpackAtomLocations(allAtomLocs2))
    # Get the array from the fits file.
    rawData, keyName, key, repetitions = loadHDF5(fileNumber)
    if keyInput is not None:
        key = keyInput
    key -= keyOffset
    print("Key Values, in Time Order: ", key)
    # window the images images.
    xMin, yMin, xMax, yMax = window if window is not None else [0, 0] + list(reversed(list(arr(rawData[0]).shape)))
    rawData = np.copy(arr(rawData[:, yMin:yMax, xMin:xMax]))
    # gather some info about the run
    numberOfPictures = int(rawData.shape[0])
    # numberOfRuns = int(numberOfPictures / picsPerRep)
    numberOfVariations = int(numberOfPictures / (repetitions * picsPerRep))
    print('Total # of Pictures:', numberOfPictures)
    print('Number of Variations:', numberOfVariations)
    if not len(key) == numberOfVariations:
        raise RuntimeError("The Length of the key doesn't match the data found.")
    # ## Initial Data Analysis
    # Group data into variations.
    groupedDataRaw = rawData.reshape((numberOfVariations, repetitions * picsPerRep, rawData.shape[1], rawData.shape[2]))
    groupedDataRaw, key, _ = orderData(groupedDataRaw, key)
    if dataRange is not None:
        groupedData, newKey = [[] for _ in range(2)]
        for count, variation in enumerate(groupedDataRaw):
            if count in dataRange:
                groupedData.append(variation)
                newKey.append(key[count])
        groupedData = arr(groupedData)
        key = arr(newKey)
        # numberOfPictures = groupedData.shape[0] * groupedData.shape[1]
        # numberOfRuns = int(numberOfPictures / picsPerRep)
        # numberOfVariations = len(groupedData)
    else:
        groupedData = groupedDataRaw
    print('Data Shape:', groupedData.shape)
    # avgPic = getAvgPic(groupedData)

    (pic1Data, pic2Data, atomCounts, bins, binnedData, thresholds, pic1Atoms,
     pic2Atoms) = arr([[None] * len(atomLocs1)] * 8)
    for i, (loc1, loc2) in enumerate(zip(atomLocs1, atomLocs2)):
        pic1Data[i] = normalizeData(groupedData, loc1, 0, picsPerRep)
        pic2Data[i] = normalizeData(groupedData, loc2, 1, picsPerRep)
        atomCounts[i] = arr([])
        for pic1, pic2 in zip(pic1Data[i], pic2Data[i]):
            atomCounts[i] = np.append(atomCounts[i], [pic1, pic2])
        bins[i], binnedData[i] = getBinData(10, pic1Data[i])
        guess1, guess2 = guessGaussianPeaks(bins[i], binnedData[i])
        guess = arr([max(binnedData[i]), guess1, 30, max(binnedData[i]) * 0.75,
                     200 if histSecondPeakGuess is None else histSecondPeakGuess, 10])
        gaussianFitVals = fitDoubleGaussian(bins[i], binnedData[i], guess)
        thresholds[i], thresholdFid = ((manualThreshold, 0) if manualThreshold is not None
                                       else calculateAtomThreshold(gaussianFitVals))
        pic1Atoms[i], pic2Atoms[i] = [[] for _ in range(2)]
        for point1, point2 in zip(pic1Data[i], pic2Data[i]):
            pic1Atoms[i].append(point1 > thresholds[i])
            pic2Atoms[i].append(point2 > thresholds[i])
    (allPic1Data, allPic2Data, allPic1Atoms, allPic2Atoms, bins, binnedData,
     thresholds) = arr([[None] * len(allAtomLocs1)] * 7)
    for i, (locs1, locs2) in enumerate(zip(allAtomLocs1, allAtomLocs2)):
        allPic1Data[i] = normalizeData(groupedData, locs1, 0, picsPerRep)
        allPic2Data[i] = normalizeData(groupedData, locs2, 1, picsPerRep)
        bins[i], binnedData[i] = getBinData(10, allPic1Data[i])
        guess1, guess2 = guessGaussianPeaks(bins[i], binnedData[i])
        guess = arr([max(binnedData[i]), guess1, 30, max(binnedData[i]) * 0.75,
                     200 if histSecondPeakGuess is None else histSecondPeakGuess, 10])
        gaussianFitVals = fitDoubleGaussian(bins[i], binnedData[i], guess)
        thresholds[i], thresholdFid = ((manualThreshold, 0) if manualThreshold is not None
                                       else calculateAtomThreshold(gaussianFitVals))
        allPic1Atoms[i], allPic2Atoms[i] = [[] for _ in range(2)]
        for point1, point2 in zip(allPic1Data[i], allPic2Data[i]):
            allPic1Atoms[i].append(point1 > thresholds[i])
            allPic2Atoms[i].append(point2 > thresholds[i])
    netLossList = getNetLoss(allPic1Atoms, allPic2Atoms)
    lossAvg, lossErr = getNetLossStats(netLossList, repetitions)
    ensembleHits = (getEnsembleHits(pic1Atoms) if pic1Num == 1 else getEnsembleHits(pic2Atoms))
    ensembleStats = getEnsembleStatistics(ensembleHits, repetitions)
    indvStatistics = getAtomInPictureStatistics(pic1Atoms if pic1Num == 1 else pic2Atoms, repetitions)
    fitData = handleFitting(fitModule, key, ensembleStats['avg'])
    avgPic = getAvgPic(rawData)
    pic1Data = arr(pic1Data.tolist())
    pic2Data = arr(pic2Data.tolist())
    return (atomLocs1, atomLocs2, key, thresholds, pic1Data, pic2Data, fitData, ensembleStats, avgPic, atomCounts,
            keyName, indvStatistics, lossAvg, lossErr)


def AnalyzeRearrangeMoves(rerngInfoAddress, fileNumber, locations, picNumber=2, threshold=300,
                          splitByNumberOfMoves=False, allLocsList=None, splitByTargetLocation=False,
                          fitData=True):
    """
    Analyzes the rearrangement move log file and displays statistics for different types of moves.
    Updated to handle new info in the file that tells where the final location of the rearrangement was.

    :param rerngInfoAddress:
    :param fileNumber:
    :param locations:
    :param picNumber:
    :param threshold:
    :param splitByNumberOfMoves:
    :param splitByTargetLocation:
    :param fitData:
    :param allLocsList:
    :return:
    """
    locations = unpackAtomLocations(locations)
    if allLocsList is not None:
        allLocsList = unpackAtomLocations(allLocsList)
    # Open file and create list of moves.
    moveList = []
    with open(rerngInfoAddress) as centerLog:
        for i, line in enumerate(centerLog):
            if i < 5:
                continue
            txt = line.split(' ')
            if txt[0] == 'Rep' or txt[0] == 'Source:':
                continue
            if txt[0] == 'Moves:\n':
                continue
            if txt[0] == 'Target' and txt[1] == 'Location:':
                moveList.append({'Target-Location': txt[2] + ',' + txt[3], 'Moves': []})
                continue
            txt[4] = txt[4][:-1]
            move = '(' + str(txt[1]) + ',' + str(txt[3]) + ')->(' + str(txt[2]) + ',' + str(txt[4] + ')')
            moveList[-1]['Moves'].append(move)
    pics, _, _, repetitions = loadHDF5(fileNumber)
    # a dictionary of dictionaries. Higher level splits by target location, lower level contains move list and
    # picture list for that location.
    dataByLocation = {}
    if splitByTargetLocation:
        for i, move in enumerate(moveList):
            name = move['Target-Location']
            if name not in dataByLocation:
                dataByLocation[name] = {'Move-List': [move], 'Picture-List': [pics[2 * i], pics[2 * i + 1]]}
            else:
                dataByLocation[name]['Move-List'].append(move)
                dataByLocation[name]['Picture-List'].append(pics[2 * i])
                dataByLocation[name]['Picture-List'].append(pics[2 * i + 1])
    else:
        for i, move in enumerate(moveList):
            name = 'No-Split'
            if name not in dataByLocation:
                dataByLocation[name] = {'Move-List': [move], 'Picture-List': [pics[2 * i], pics[2 * i + 1]]}
            else:
                dataByLocation[name]['Move-List'].append(move)
                dataByLocation[name]['Picture-List'].append(pics[2 * i])
                dataByLocation[name]['Picture-List'].append(pics[2 * i + 1])

    # Get and print average statsistics over the whole set.
    (allPic1Data, allPic2Data, allPic1Atoms, allPic2Atoms,
     allLocsPic1Data, allLocsPic2Data, allLocsPic1Atoms, allLocsPic2Atoms) = [[] for _ in range(8)]
    for loc in locations:
        allPic1Data.append(normalizeData(pics, loc, 0, 2))
        allPic2Data.append(normalizeData(pics, loc, 1, 2))
    for point1, point2 in zip(allPic1Data, allPic2Data):
        allPic1Atoms.append(point1 > threshold)
        allPic2Atoms.append(point2 > threshold)
    if allLocsList is not None:
        for loc in allLocsList:
            allLocsPic1Data.append(normalizeData(pics, loc, 0, 2))
            allLocsPic2Data.append(normalizeData(pics, loc, 1, 2))
        for point1, point2 in zip(allLocsPic1Data, allLocsPic2Data):
            allLocsPic1Atoms.append(point1 > threshold)
            allLocsPic2Atoms.append(point2 > threshold)
    else:
        (allLocsPic1Data, allLocsPic2Data, allLocsPic1Atoms,
         allLocsPic2Atoms) = allPic1Data, allPic2Data, allPic1Atoms, allPic2Atoms
    allEvents = (getEnsembleHits(allPic2Atoms) if picNumber == 2 else getEnsembleHits(allPic1Atoms))
    allLossList = getNetLoss(allLocsPic1Atoms, allLocsPic2Atoms)
    allLossAvg, allLossErr = getNetLossStats(allLossList, len(allLossList))
    print('Average Loss:', allLossAvg, '\nLoss Error:', allLossErr)
    print('Total Average Assembly:', round_sig(np.mean(allEvents)),
          round_sig(np.std(allEvents) / np.sqrt(len(allEvents))))
    allData = {}
    fits = {}
    for targetLoc, data in dataByLocation.items():
        moveData = {}
        if splitByNumberOfMoves:
            numberMovesList = []
            # nomoves handled separately because can refer to either loaded a 1x6 or loaded <6.
            noMoves = 0
            print('\nSplitting location:', targetLoc)
            print('Number of Repetitions Rearranging to this location:', len(data['Move-List']))
            for i, move in enumerate(data['Move-List']):
                moveName = len(move['Moves'])
                if len(move['Moves']) != 0:
                    numberMovesList.append(len(move['Moves']))
                else:
                    noMoves += 1
                if moveName not in moveData:
                    moveData[moveName] = [data['Picture-List'][2 * i], data['Picture-List'][2 * i + 1]]
                else:
                    moveData[moveName].append(data['Picture-List'][2 * i])
                    moveData[moveName].append(data['Picture-List'][2 * i + 1])
            print('Average Number of Moves, excluding zeros:', np.mean(numberMovesList))
            print('Number of repetitions with no moves:', noMoves)
        else:
            for i, move in enumerate(data['Move-List']):
                if len(move['Moves']) == 0:
                    moveName = 'No-Move'
                else:
                    moveName = '{'
                    for m in move['Moves']:
                        moveName += m + ','
                    moveName = moveName[:-2] + ')}'
                if moveName not in moveData:
                    moveData[moveName] = [data['Picture-List'][2 * i], data['Picture-List'][2 * i + 1]]
                else:
                    moveData[moveName].append(data['Picture-List'][2 * i])
                    moveData[moveName].append(data['Picture-List'][2 * i + 1])
        """
        netLossList = getNetLoss(pic1Atoms, pic2Atoms)
        lossAvg, lossErr = getNetLossStats(netLossList, repetitions)
        """
        d = DataFrame()
        lossAvgList, allLossErr = [[], []]
        for keyName, pics in moveData.items():
            pics = arr(pics)
            (pic1Data, pic1Atoms, pic2Data, pic2Atoms, pic1AllLocsData, pic1AllLocsAtoms, pic2AllLocsData,
             pic2AllLocsAtoms) = [[] for _ in range(8)]
            for loc in locations:
                pic1Data.append(normalizeData(pics, loc, 0, 2).tolist())
                pic2Data.append(normalizeData(pics, loc, 1, 2).tolist())
                pic1Atoms.append([])
                pic2Atoms.append([])
                for (point1, point2) in zip(pic1Data[-1], pic2Data[-1]):
                    pic1Atoms[-1].append(point1 > threshold)
                    pic2Atoms[-1].append(point2 > threshold)
            if allLocsList is not None:
                for loc in allLocsList:
                    pic1AllLocsData.append(normalizeData(pics, loc, 0, 2).tolist())
                    pic2AllLocsData.append(normalizeData(pics, loc, 1, 2).tolist())
                    pic1AllLocsAtoms.append([])
                    pic2AllLocsAtoms.append([])
                    for (point1, point2) in zip(pic1AllLocsData[-1], pic2AllLocsData[-1]):
                        pic1AllLocsAtoms[-1].append(point1 > threshold)
                        pic2AllLocsAtoms[-1].append(point2 > threshold)
                lossList = getNetLoss(pic1AllLocsAtoms, pic2AllLocsAtoms)
                a, e = getNetLossStats(lossList, len(lossList))
                allLossAvg.append(a[0])
                allLossErr.append(e[0])
            atomEvents = (getEnsembleHits(pic2Atoms) if picNumber == 2 else getEnsembleHits(pic1Data))
            d[keyName] = [int(len(data) / 2), np.mean(atomEvents), np.std(atomEvents) / np.sqrt(len(atomEvents))]
        allLossAvg = arr(allLossAvg)
        d = d.transpose()
        d.columns = ['occurances', 'success', 'error']
        d = d.sort_values('occurances', ascending=False)
        allData[targetLoc] = d
        if fitData:
            # avoid the longest moves, as these usually only happen a very few number of times and
            # I don't have good statstics for them.
            nums = []
            for val in d.transpose().columns:
                nums.append(val)
            orderedData, nums, _ = orderData(list(d['success']), nums)
            fitValues, fitCov = fit(fitFunc.exponentialDecay, nums[1:-3], orderedData[1:-3], p0=[1, 3])
            fits[targetLoc] = fitValues
        else:
            fits[targetLoc] = None
    return allData, fits
