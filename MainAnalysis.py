__version__ = "1.4"
"""
"""

from numpy import array as arr
from pandas import DataFrame

from Miscellaneous import getStats, round_sig, errString
from MarksFourierAnalysis import fft
from matplotlib.pyplot import *
from scipy.optimize import curve_fit as fit
import FittingFunctions as fitFunc
from AnalysisHelpers import *
import fitters.linear
from ExpFile import ExpFile
from TimeTracker import TimeTracker


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

    (rawData, groupedData, atomLocs1, atomLocs2, keyName, repetitions,
     key) = organizeTransferData(fileNumber, atomLocs1, atomLocs1, picsPerRep=picsPerRep,
                                 **transferOrganizeArgs)
    # initialize arrays
    (pic1Data, pic2Data, atomCounts, bins, binnedData, thresholds,
     loadingRate, pic1Atoms, pic2Atoms, survivalFits) = arr([[None] * len(atomLocs1)] * 10)
    survivalData, survivalErrs = [[[] for _ in range(len(atomLocs1))] for _ in range(2)]
    if subtractEdgeCounts:
        borders_load = getAvgBorderCount(groupedData, loadPic, picsPerRep)
        borders_trans = getAvgBorderCount(groupedData, transerPic, picsPerRep)
    else:
        borders_load = borders_trans = np.zeros(len(groupedData.shape[0]*groupedData.shape[1]))
    for i, (loc1, loc2) in enumerate(zip(atomLocs1, atomLocs2)):
        pic1Data[i] = normalizeData(groupedData, loc1, loadPic, picsPerRep, borders_load)
        pic2Data[i] = normalizeData(groupedData, loc2, transferPic, picsPerRep, borders_trans)
        atomCounts[i] = arr([a for a in arr(list(zip(pic1Data[i], pic2Data[i]))).flatten()])
        bins[i], binnedData[i] = getBinData(10, pic1Data[i])
        guess1, guess2 = guessGaussianPeaks(bins[i], binnedData[i])
        guess = arr([max(binnedData[i]), guess1, 30, max(binnedData[i]) * 0.75,
                     200 if histSecondPeakGuess is not None else histSecondPeakGuess, 10])
        gaussianFitVals = fitDoubleGaussian(bins[i], binnedData[i], guess)
        thresholds[i], thresholdFid = ((manualThreshold, 0) if manualThreshold is not None
                                        else getMaxFidelityThreshold(gaussianFitVals))
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


def standardTransferAnalysis( fileNumber, atomLocs1, atomLocs2, picsPerRep=2, manualThreshold=None,
                              fitModule=None, histSecondPeakGuess=None, outputMma=False, varyingDim=None,
                              subtractEdgeCounts=True, loadPic=0, transferPic=1, postSelectionCondition=None,
                              postSelectionConnected=False, getGenerationStats=False, normalizeForLoadingRate=False, 
                              rerng=False, tt=None, **organizerArgs ):
    """
    Standard data analysis package for looking at survival rates throughout an experiment.
    Returns key, survivalData, survivalErrors

    :param fileNumber: for the HDF5 File. Path is automatic.
    :param atomLocs1: the loading atom arrangement.
    :param atomLocs2: the transfer atom arrangement
    :param key: manually entered key. Overrides key from HDF5 file.
    :param manualThreshold: 
    :param fitModule: a submodule from the fitters module for fitting.
    :param window: (left, top, right bottom)? quick way of setting xmin/max and ymin/max.
    :param xMin:   :param xMax:     :param yMin:     :param yMax:  see window
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
    :param loadPic: 0-indexed!
    :param transferPic: 0-indexed!
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
    :param repRange: similar to data range but for the pictures before any grouping happens. Convenient e.g. if you are
        only taking 1 data point and something happens midway through and you want to only use all the pics before the
        event.
    :return: a lot. (atomLocs1, atomLocs2, atomCounts, survivalData, survivalErrs, loadingRate, loadPicData, keyName, key,
            repetitions, thresholds, fits, avgSurvivalData, avgSurvivalErr, avgFit, avgPics, otherDims, locationsList,
            genAvgs, genErrs, threshFitVals, tt)
    """
    if tt is None:
        tt = TimeTracker()
    if rerng:
        loadPic, transferPic, picsPerRep = 1, 2, 3
    (rawData, groupedData, atomLocs1, atomLocs2, keyName, repetitions, 
     key) = organizeTransferData(fileNumber, atomLocs1, atomLocs2,  picsPerRep=picsPerRep, varyingDim=varyingDim, 
                                 **organizerArgs)
    allPics = getAvgPics(groupedData)
    avgPics = [allPics[loadPic], allPics[transferPic]]
    
    (fullPixelCounts, thresholds, threshFids, threshFitVals, threshBins, threshBinData) = arr([[None] * len(atomLocs1)] * 6)
    for i, atomLoc in enumerate(atomLocs1):
        fullPixelCounts[i] = getAtomCountsData( rawData, picsPerRep, loadPic, atomLoc, subtractEdges=subtractEdgeCounts )
        res = getThresholds( fullPixelCounts[i], 5, manualThreshold )
        thresholds[i], threshFids[i], threshFitVals[i], threshBins[i], threshBinData[i] = res
    
    (loadPicData, transPicData, atomCounts, bins, binnedData, survivalData, survivalErrs,
     loadingRate, loadAtoms, transAtoms, genAvgs, genErrs) = arr([[None] * len(atomLocs1)] * 12)
    if subtractEdgeCounts:
        borders_load = getAvgBorderCount(groupedData, loadPic, picsPerRep)
        borders_trans = getAvgBorderCount(groupedData, transferPic, picsPerRep)
    else:
        borders_load = borders_trans = np.zeros(len(groupedData.shape[0]*groupedData.shape[1]))
    for i, (loc1, loc2) in enumerate(zip(atomLocs1, atomLocs2)):
        loadPicData[i] = normalizeData(groupedData, loc1, loadPic, picsPerRep, borders_load)
        transPicData[i] = normalizeData(groupedData, loc2, transferPic, picsPerRep, borders_trans)
        atomCounts[i] = arr([a for a in arr(list(zip(loadPicData[i], transPicData[i]))).flatten()])
        loadAtoms[i], transAtoms[i] = [[] for _ in range(2)]
        for point1, point2 in zip(loadPicData[i], transPicData[i]):
            loadAtoms[i].append(point1 > thresholds[i])
            transAtoms[i].append(point2 > thresholds[i])
    if postSelectionCondition is not None:
        loadAtoms, transAtoms = postSelectOnAssembly(loadAtoms, transAtoms, postSelectionCondition,
                                                    connected=postSelectionConnected)
    for i in range(len(atomLocs1)):
        survivalList = getSurvivalEvents(loadAtoms[i], transAtoms[i])
        survivalData[i], survivalErrs[i], loadingRate[i] = getSurvivalData(survivalList, repetitions)
        if getGenerationStats:
            genList = getGenerationEvents(loadAtoms[i], transAtoms[i])
            genAvgs[i], genErrs[i] = getGenStatistics(genList, repetitions)
        else:
            genAvgs[i], genErrs[i] = [None, None]
    # Positioning of this is very important.
    res = groupMultidimensionalData(key, varyingDim, atomLocs1, survivalData, survivalErrs, loadingRate)
    (key, locationsList, survivalErrs, survivalData, loadingRate, otherDims) = res

    loadPicData = arr(loadPicData.tolist())
    atomCounts = arr(atomCounts.tolist())
    # calculate average values
    avgSurvivalData, avgSurvivalErr, avgFit = [None]*3
    # weight the sum with loading percentage
    if normalizeForLoadingRate:
        avgSurvivalData = sum(survivalData*loadingRate)/sum(loadingRate)
    else:
        avgSurvivalData = np.mean(survivalData)
    avgSurvivalErr = np.sqrt(np.sum(survivalErrs**2))/len(atomLocs1)
    
    fits = [None] * len(locationsList)
    if fitModule is not None:
        for i, _ in enumerate(locationsList):
            fits[i], _ = fitWithClass(fitModule, key, survivalData[i])
        avgFit, _ = fitWithClass(fitModule, key, avgSurvivalData)
    if outputMma:
        outputDataToMmaNotebook(fileNumber, survivalData, survivalErrs, loadingRate, key)
    return (atomLocs1, atomLocs2, atomCounts, survivalData, survivalErrs, loadingRate, loadPicData, keyName, key,
            repetitions, thresholds, fits, avgSurvivalData, avgSurvivalErr, avgFit, avgPics, otherDims, locationsList,
            genAvgs, genErrs, threshFitVals, tt)


def standardPopulationAnalysis( fileNum, atomLocations, whichPic, picsPerRep, analyzeTogether=False, 
                                manualThreshold=None, fitModule=None, keyInput=None, fitIndv=False, subtractEdges=True,
                                keyConversion=None, quiet=False):
    """
    keyConversion should be a calibration which takes in a single value as an argument and converts it.
        It needs a calibration function f() and a units function units()
    return: ( fullPixelCounts, thresholds, avgPic, key, avgLoadingErr, avgLoading, allLoadingRate, allLoadingErr, loadFits,
             fitModule, keyName, totalAtomData, rawData, atomLocations, avgFits, atomImages, threshFitVals )
    """
    atomLocations = unpackAtomLocations(atomLocations)
    with ExpFile(fileNum) as f:
        rawData, keyName, key, repetitions = f.pics, f.key_name, f.key, f.reps 
        if not quiet:
            f.get_basic_info()
    numOfPictures = rawData.shape[0]
    numOfVariations = int(numOfPictures / (repetitions * picsPerRep))
    # handle defaults.
    if numOfVariations == 1:
        if key is None:
            key = arr([0])
    else:
        if key is None:
            key = arr([])
    if keyInput is not None:
        key = keyInput
    if len(arr(atomLocations).shape) == 1:
        atomLocations = [atomLocations]
    if not len(key) == numOfVariations:
        raise ValueError("ERROR: The Length of the key doesn't match the data found. "
                         "Did you want to use a transfer-based function instead of a population-based function? Key:", 
                         len(key), "vars:", numOfVariations)
    if keyConversion is not None:
        key = [keyConversion.f(k) for k in key]
        keyName += "; " + keyConversion.units()
    # ## Initial Data Analysis
    s = rawData.shape
    if analyzeTogether:
        newShape = (1, s[0], s[1], s[2])
    else:
        newShape = (numOfVariations, repetitions * picsPerRep, s[1], s[2])    
    # Split the rawData by variations
    groupedData = rawData.reshape(newShape)
    groupedData, key, _ = orderData(groupedData, key)
    avgLoading, avgLoadingErr, loadFits = [[[] for _ in range(len(atomLocations))] for _ in range(3)]

    allLoadingRate, allLoadingErr = [[[]] * len(groupedData) for _ in range(2)]
    totalAtomData = []
    
    (fullPixelCounts, fullAtomData, thresholds, threshFids, threshFitVals, threshBins, threshBinData,
     fullAtomCount) = arr([[None] * len(atomLocations)] * 8)
    for i, atomLoc in enumerate(atomLocations):
        fullPixelCounts[i] = getAtomCountsData( rawData, picsPerRep, whichPic, atomLoc, subtractEdges=subtractEdges )
        res = getThresholds( fullPixelCounts[i], 5, manualThreshold )
        thresholds[i], threshFids[i], threshFitVals[i], threshBins[i], threshBinData[i] = res
        fullAtomData[i], fullAtomCount[i] = getAtomBoolData(fullPixelCounts[i], thresholds[i])
    fullAtomData = arr(fullAtomData.tolist())
    fullPixelCounts = arr(fullPixelCounts.tolist())
    if not quiet:
        print('Analyzing Variation... ', end='')    
    (variationPixelData, variationAtomData, atomCount) = arr([[[None for _ in atomLocations] for _ in groupedData] for _ in range(3)])
    for dataInc, data in enumerate(groupedData):
        if not quiet:
            print(str(dataInc) + ', ', end='')
        allAtomPicData = []
        for i, atomLoc in enumerate(atomLocations):
            variationPixelData[dataInc][i] = getAtomCountsData( data, picsPerRep, whichPic, atomLoc, subtractEdges=subtractEdges )
            variationAtomData[dataInc][i], atomCount[dataInc][i] = getAtomBoolData(variationPixelData[dataInc][i], thresholds[i])            
            totalAtomData.append(variationAtomData[dataInc][i])
            allAtomPicData.append(np.mean(variationAtomData[dataInc][i]))
            avgLoading[i].append(np.mean(variationAtomData[dataInc][i]))
            avgLoadingErr[i].append(np.std(variationAtomData[dataInc][i]) / np.sqrt(len(variationAtomData[dataInc][i])))
        allLoadingRate[dataInc] = np.mean(allAtomPicData)
        allLoadingErr[dataInc] = np.std(allAtomPicData) / np.sqrt(len(allAtomPicData))
    # 
    avgFits = None
    if fitModule is not None:
        if fitIndv:
            for i, load in enumerate(avgLoading):
                loadFits[i], _ = fitWithClass(fitModule, key, load)
        avgFits, _ = fitWithClass(fitModule, key, allLoadingRate)
    avgPic = getAvgPic(rawData)
    # get averages across all variations
    atomImages = [np.zeros(rawData[0].shape) for _ in range(int(numOfPictures/picsPerRep))]
    atomImagesInc = 0
    for picInc in range(int(numOfPictures)):
        if picInc % picsPerRep != whichPic:
            continue
        for locInc, loc in enumerate(atomLocations):
            atomImages[atomImagesInc][loc[0]][loc[1]] = fullAtomData[locInc][atomImagesInc]
        atomImagesInc += 1

    return ( fullPixelCounts, thresholds, avgPic, key, avgLoadingErr, avgLoading, allLoadingRate, allLoadingErr, loadFits,
             fitModule, keyName, totalAtomData, rawData, atomLocations, avgFits, atomImages, threshFitVals )


def standardAssemblyAnalysis(fileNumber, atomLocs1, assemblyPic, atomLocs2=None, keyOffset=0, dataRange=None,
                             window=None, picsPerRep=2, histSecondPeakGuess=None, partialCredit=False,
                             manualThreshold=None, fitModule=None, allAtomLocs1=None, allAtomLocs2=None, keyInput=None,
                             loadPic=0):
    """
    :param fileNumber:
    :param atomLocs1: 
    :param assemblyPic: 
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
    if assemblyPic == 1:
        print('Assesing Loading-Assembly???')
    atomLocs1 = unpackAtomLocations(atomLocs1)
    atomLocs2 = (atomLocs1[:] if atomLocs2 is None else unpackAtomLocations(atomLocs2))
    allAtomLocs1 = (atomLocs1[:] if allAtomLocs1 is None else unpackAtomLocations(allAtomLocs1))
    allAtomLocs2 = (allAtomLocs1[:] if allAtomLocs2 is None else unpackAtomLocations(allAtomLocs2))
    with ExpFile(fileNumber) as f:
        rawData, keyName, key, repetitions = f.pics, f.key_name, f.key, f.reps         
    if keyInput is not None:
        key = keyInput
    key -= keyOffset
    print("Key Values, in Time Order: ", key)
    # window the images images.
    xMin, yMin, xMax, yMax = window if window is not None else [0, 0] + list(reversed(list(arr(rawData[0]).shape)))
    rawData = np.copy(arr(rawData[:, yMin:yMax, xMin:xMax]))
    # gather some info about the run
    numberOfPictures = int(rawData.shape[0])
    numberOfVariations = int(numberOfPictures / (repetitions * picsPerRep))
    print('Total # of Pictures:', numberOfPictures)
    print('Number of Variations:', numberOfVariations)
    if not len(key) == numberOfVariations:
        raise RuntimeError("The Length of the key doesn't match the shape of the data???")
    
    groupedDataRaw = rawData.reshape((numberOfVariations, repetitions * picsPerRep, rawData.shape[1], rawData.shape[2]))
    groupedDataRaw, key, _ = orderData(groupedDataRaw, key)
    key, groupedData = applyDataRange(dataRange, groupedDataRaw, key)
    print('Data Shape:', groupedData.shape)
    
    borders_load = getAvgBorderCount(groupedData, loadPic, picsPerRep)
    borders_assembly = getAvgBorderCount(groupedData, assemblyPic, picsPerRep)
    (loadPicData, assemblyPicData, atomCounts, bins, binnedData, thresholds, loadAtoms,
     assemblyAtoms) = arr([[None] * len(atomLocs1)] * 8)
    for i, (loc1, loc2) in enumerate(zip(atomLocs1, atomLocs2)):
        loadPicData[i]     = normalizeData(groupedData, loc1, loadPic,     picsPerRep, borders_load)
        assemblyPicData[i] = normalizeData(groupedData, loc2, assemblyPic, picsPerRep, borders_assembly)
        atomCounts[i] = arr([])
        for pic1, pic2 in zip(loadPicData[i], assemblyPicData[i]):
            atomCounts[i] = np.append(atomCounts[i], [pic1, pic2])
        bins[i], binnedData[i] = getBinData(10, loadPicData[i])
        guess1, guess2 = guessGaussianPeaks(bins[i], binnedData[i])
        guess = arr([max(binnedData[i]), guess1, 30, max(binnedData[i]) * 0.75,
                     200 if histSecondPeakGuess is None else histSecondPeakGuess, 10])
        if manualThreshold is None:
            gaussianFitVals = fitDoubleGaussian(bins[i], binnedData[i], guess)
        thresholds[i], thresholdFid = ((manualThreshold, 0) if manualThreshold is not None
                                       else calculateAtomThreshold(gaussianFitVals))
        loadAtoms[i], assemblyAtoms[i] = [[] for _ in range(2)]
        for point1, point2 in zip(loadPicData[i], assemblyPicData[i]):
            loadAtoms[i].append(point1 > thresholds[i])
            assemblyAtoms[i].append(point2 > thresholds[i])
    # now analyze the atom data
    enhancement = getEnhancement(loadAtoms, assemblyAtoms)
    enhancementStats = getEnsembleStatistics(enhancement, repetitions)
    ensembleHits = getEnsembleHits(assemblyAtoms, partialCredit=partialCredit)
    ensembleStats = getEnsembleStatistics(ensembleHits, repetitions)
    indvStatistics = getAtomInPictureStatistics(assemblyAtoms, repetitions)
    fitData = handleFitting(fitModule, key, ensembleStats['avg'])

    # similar for other set of locations.
    (allPic1Data, allPic2Data, allPic1Atoms, allPic2Atoms, bins, binnedData,
     thresholds) = arr([[None] * len(allAtomLocs1)] * 7)
    for i, (locs1, locs2) in enumerate(zip(allAtomLocs1, allAtomLocs2)):
        allPic1Data[i] = normalizeData(groupedData, locs1, loadPic, picsPerRep, borders_load)
        allPic2Data[i] = normalizeData(groupedData, locs2, assemblyPic, picsPerRep, borders_assembly)
        bins[i], binnedData[i] = getBinData(10, allPic1Data[i])
        guess1, guess2 = guessGaussianPeaks(bins[i], binnedData[i])
        guess = arr([max(binnedData[i]), guess1, 30, max(binnedData[i]) * 0.75,
                     200 if histSecondPeakGuess is None else histSecondPeakGuess, 10])
        if manualThreshold is None:
            gaussianFitVals = fitDoubleGaussian(bins[i], binnedData[i], guess)
        thresholds[i], thresholdFid = ((manualThreshold, 0) if manualThreshold is not None
                                       else calculateAtomThreshold(gaussianFitVals))
        allPic1Atoms[i], allPic2Atoms[i] = [[] for _ in range(2)]
        for point1, point2 in zip(allPic1Data[i], allPic2Data[i]):
            allPic1Atoms[i].append(point1 > thresholds[i])
            allPic2Atoms[i].append(point2 > thresholds[i])
    netLossList = getNetLoss(allPic1Atoms, allPic2Atoms)
    lossAvg, lossErr = getNetLossStats(netLossList, repetitions)
    
    avgPic = getAvgPic(rawData)
    loadPicData = arr(loadPicData.tolist())
    assemblyPicData = arr(assemblyPicData.tolist())
    return (atomLocs1, atomLocs2, key, thresholds, loadPicData, assemblyPicData, fitData, ensembleStats, avgPic, atomCounts,
            keyName, indvStatistics, lossAvg, lossErr, fitModule, enhancementStats)



def AnalyzeRearrangeMoves(rerngInfoAddress, fileNumber, locations, loadPic=0, rerngedPic=1, picsPerRep=2,
                          splitByNumberOfMoves=False, allLocsList=None, splitByTargetLocation=False,
                          fitData=False, sufficientLoadingPostSelect=True, includesNoFlashPostSelect=False,
                          includesParallelMovePostSelect=False, isOnlyParallelMovesPostSelect=False,
                          noParallelMovesPostSelect=False, parallelMovePostSelectSize=None,
                          postSelectOnNumberOfMoves=False, limitedMoves=-1, SeeIfMovesMakeSense=True, 
                          postSelectOnLoading=False, **popArgs):
    """
    Analyzes the rearrangement move log file and displays statistics for different types of moves.
    Updated to handle new info in the file that tells where the final location of the rearrangement was.
    """
    def append_all(moveList, picNums, picList, move, pics, i):
        moveList.append(move)
        picList.append(pics[2 * i])
        picList.append(pics[2 * i + 1])
        picNums.append(2*i)

    locations = unpackAtomLocations(locations)
    if allLocsList is not None:
        allLocsList = unpackAtomLocations(allLocsList)
    # Open file and create list of moves.
    moveList = parseRearrangeInfo(rerngInfoAddress, limitedMoves=limitedMoves)
    with ExpFile(fileNumber) as f:
        rawPics, repetitions = f.pics, f.reps 
        #f.get_basic_info()
    print(len(rawPics),'...')
    picNums = list(np.arange(1,len(rawPics),1))
    if sufficientLoadingPostSelect:
        tmpPicNums, tmpPicList, tmpMoveList = [[], [], []]
        for i, move in enumerate(moveList):
            if not np.sum(move['Source']) < len(locations):
                append_all(tmpMoveList, tmpPicNums, tmpPicList, move, rawPics, i)
        picNums, moveList, rawPics = arr(tmpPicNums), tmpMoveList, arr(tmpPicList)
        
    if includesNoFlashPostSelect:
        tmpPicNums, tmpPicList, tmpMoveList = [[], [], []]
        for i, move in enumerate(moveList):
            includesNoFlash = False
            for indvMove in move['Moves']:
                if not indvMove['Flashed']:
                    includesNoFlash = True
            if includesNoFlash:
                append_all(tmpMoveList, tmpPicNums, tmpPicList, move, rawPics, i)
        picNums, moveList, rawPics = arr(tmpPicNums), tmpMoveList, arr(tmpPicList)
        
    if includesParallelMovePostSelect:
        tmpPicNums, tmpPicList, tmpMoveList = [[], [], []]
        for i, move in enumerate(moveList):
            includesParallelMove = False
            for indvMove in move['Moves']:
                if parallelMovePostSelectSize is None:
                    if len(indvMove['Atoms']) > 1:
                        includesParallelMove = True
                elif len(indvMove['Atoms']) == parallelMovePostSelectSize:
                    includesParallelMove = True
            if includesParallelMove:
                append_all(tmpMoveList, tmpPicNums, tmpPicList, move, rawPics, i)
        picNums, moveList, rawPics = arr(tmpPicNums), tmpMoveList, arr(tmpPicList)
        
    if isOnlyParallelMovesPostSelect:
        tmpPicNums, tmpPicList, tmpMoveList = [[], [], []]
        for i, move in enumerate(moveList):
            isParallel = True
            for indvMove in move['Moves']:
                if parallelMovePostSelectSize is None:
                    if len(indvMove['Atoms']) == 1:
                        isParallel = False
                elif len(indvMove['Atoms']) != parallelMovePostSelectSize:
                    isParallel = False
            if isParallel:
                append_all(tmpMoveList, tmpPicNums, tmpPicList, move, rawPics, i)
        picNums, moveList, rawPics = arr(tmpPicNums), tmpMoveList, arr(tmpPicList)
        
    if noParallelMovesPostSelect:
        tmpPicNums, tmpPicList, tmpMoveList = [[], [], []]
        for i, move in enumerate(moveList):
            containsParallel = False
            for indvMove in move['Moves']:
                if len(indvMove['Atoms']) > 1:
                    containsParallel = True
            if not containsParallel:
                append_all(tmpMoveList, tmpPicNums, tmpPicList, move, rawPics, i)
        picNums, moveList, rawPics = arr(tmpPicNums), tmpMoveList, arr(tmpPicList)
        
    if postSelectOnNumberOfMoves:
        tmpPicNums, tmpPicList, tmpMoveList = [[], [], []]
        for i, move in enumerate(moveList):
            if len(move['Moves']) == postSelectOnNumberOfMoves:
                append_all(tmpMoveList, tmpPicNums, tmpPicList, move, rawPics, i)
        picNums, moveList, rawPics = arr(tmpPicNums), tmpMoveList, arr(tmpPicList)
    dataByLocation = {}
    for i, move in enumerate(moveList):
        name = (move['Target-Location'] if splitByTargetLocation else 'No-Target-Split')
        if name not in dataByLocation:
            dataByLocation[name] = {'Move-List': [move], 'Picture-List': [rawPics[2 * i], rawPics[2 * i + 1]],
                                   'Picture-Nums': [2 * i, 2 * i + 1]}
        else:
            append_all( dataByLocation[name]['Move-List'], dataByLocation[name]['Picture-Nums'], 
                       dataByLocation[name]['Picture-List'], move, rawPics, i)
    # Get and print average statsistics over the whole set.
    borders_load = getAvgBorderCount(rawPics, loadPic, picsPerRep)
    borders_trans = getAvgBorderCount(rawPics, rerngedPic, picsPerRep)
    allData, fits = {}, {}
    # this is usually just a 1x loop.
    for targetLoc, data in dataByLocation.items():
        moveData = {}
        # final assembly of move-data
        if splitByNumberOfMoves:
            numberMovesList = []
            # nomoves handled separately because can refer to either loaded a 1x6 or loaded <6.
            noMoves = 0
            if len(dataByLocation.keys()) != 1:
                print('\nSplitting location:', targetLoc, '\nNumber of Repetitions Rearranging to this location:',
                      len(data['Move-List']))
            for i, move in enumerate(data['Move-List']):
                moveName = len(move['Moves'])
                if len(move['Moves']) != 0:
                    numberMovesList.append(len(move['Moves']))
                else:
                    noMoves += 1
                if moveName not in moveData:
                    moveData[moveName] = [2*i]
                else:
                    moveData[moveName].append(2*i)
                    
            print('Average Number of Moves, excluding zeros:', np.mean(numberMovesList))
            print('Number of repetitions with no moves:', noMoves)
        else:
            for i, move in enumerate(data['Move-List']):
                if len(move['Moves']) == 0:
                    moveName = 'No-Move'
                else:
                    moveName = ''
                    for m in move['Moves']:
                        for a in m['Atoms']:
                            moveName += '(' + a[0] + ',' + a[1].rstrip() + ')'
                        directions = ['U','D','L','R']
                        moveName += directions[int(m['Direction'])] + ', '
                if moveName not in moveData:
                    moveData[moveName] = [2*i]
                else:
                    moveData[moveName].append(2*i)
        
        res = standardPopulationAnalysis( fileNumber, locations, rerngedPic, picsPerRep, **popArgs)
        allRerngedAtoms = res[11]
        res = standardPopulationAnalysis( fileNumber, locations, loadPic, picsPerRep, **popArgs)
        allLoadedAtoms = res[11]        
        (loadData, loadAtoms, rerngedData, rerngedAtoms, loadAllLocsData, loadAllLocsAtoms, rerngedAllLocsData,
         rerngedAllLocsAtoms) = [[] for _ in range(8)]
        d = DataFrame()
        # looping through diff target locations...
        print(arr(allRerngedAtoms).shape,'hi')
        for keyName, categoryPicNums in moveData.items():
            if postSelectOnLoading:
                rerngedAtoms = arr([[locAtoms[int(i/2)] for i in categoryPicNums if not bool(allLoadedAtoms[j,int(i/2)])] 
                                    for j, locAtoms in enumerate(allRerngedAtoms)])
            else:
                rerngedAtoms = arr([[locAtoms[int(i/2)] for i in categoryPicNums] for j, locAtoms in enumerate(allRerngedAtoms)])
            atomEvents = getEnsembleHits(rerngedAtoms)            
            # set the occurances, mean, error
            if len(atomEvents) == 0:
                d[keyName] = [int(len(rerngedAtoms[0])), 0, 0]
            else:
                d[keyName] = [int(len(rerngedAtoms[0])), np.mean(atomEvents), np.std(atomEvents) / np.sqrt(len(atomEvents))]
                
        d = d.transpose()
        d.columns = ['occurances', 'success', 'error']
        d = d.sort_values('occurances', ascending=False)
        allData[targetLoc] = d
        if fitData:
            nums = []
            for val in d.transpose().columns:
                nums.append(val)
            orderedData, nums, _ = orderData(list(d['success']), nums)
            fitValues, fitCov = fit(fitFunc.exponentialDecay, nums[1:-3], orderedData[1:-3], p0=[1, 3])
            fits[targetLoc] = fitValues
        else:
            fits[targetLoc] = None
    return allData, fits, rawPics, moveList

