
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
import ExpFile as exp
from TimeTracker import TimeTracker
import AnalysisHelpers as ah


def standardImages(data, 
                   # Cosmetic Parameters
                   scanType="", xLabel="", plotTitle="", convertKey=False, 
                   colorMax=-1, individualColorBars=False, majorData='counts',
                   # Global Data Manipulation Options
                   loadType='andor', window=(0, 0, 0, 0), smartWindow=False, xMin=0, xMax=0, yMin=0, yMax=0,
                   accumulations=1, key=arr([]), zeroCorners=False, dataRange=(0, 0), manualAccumulation=False,
                   # Local Data Manipulation Options
                   plottedData=None, bg=arr([0]), location=(-1, -1), fitBeamWaist=False, fitPics=False,
                   cameraType='dataray', fitWidthGuess=80, quiet=False):
    """
    """
    if plottedData is None:
        plottedData = ["raw"]
    # Check for incompatible parameters.
    if fitBeamWaist and not fitPics:
        raise ValueError(
            "ERROR: Can't use fitBeamWaist and not fitPics! The fitBeamWaist attempts to use the fit values "
            "found by the gaussian fits.")

    # the key
    """
    if key.size == 0:
        origKey, keyName = loadDetailedKey(data)
    else:
        origKey = key
    # this section could be expanded to handle different types of conversions.
    if convertKey:
        a = consts.opBeamDacToVoltageConversionConstants
        key = [a[0] + x * a[1] + x ** 2 * a[2] + x ** 3 * a[3] for x in origKey]
    else:
        # both keys the same.
        key = origKey
    if len(key) == 0:
        raise RuntimeError('key was empty!')
    """
    """ ### Handle data ### 
    If the corresponding inputs are given, all data gets...
    - normalized for accumulations
    - normalized using the normData array
    - like values in the key & data are averaged
    - key and data is ordered in ascending order.
    - windowed.
    """
    if type(data) == int or (type(data) == np.array and type(data[0]) == int):
        if loadType == 'andor':
            rawData, _, _, _ = loadHDF5(data)
        elif loadType == 'scout' or  loadType == 'ace':
            rawData = loadCompoundBasler(data, loadType)
        elif loadType == 'basler':
            with exp.ExpFile() as f:
                f.open_hdf5(data,True)
                rawData = f.get_basler_pics()
        elif loadType == 'dataray':
            raise ValueError('Loadtype of "dataray" has become deprecated and needs to be reimplemented.')
        else:
            raise ValueError('Bad value for LoadType.')
    elif type(data) == type('a string'):
        # assume a file address for an HDF5 file.
        with exp.ExpFile() as f:
            f.open_hdf5(data,True)
            if loadType == 'andor':
                rawData = f.get_pics()
            elif loadType == 'basler':
                rawData = f.get_basler_pics()
            if key is None:
                kn, key = f.get_key()
    else:
        # assume the user inputted a picture or array of pictures.
        if not quiet:
            print('Assuming input is a picture or array of pictures.')
        rawData = data
    if not quiet:
        print('Data Loaded.')
    res = processImageData( key, rawData, bg, window, xMin, xMax, yMin, yMax, accumulations, dataRange, zeroCorners,
                            smartWindow, manuallyAccumulate=manualAccumulation )
    key, rawData, dataMinusBg, dataMinusAvg, avgPic = res
    if fitPics:
        # should improve this to handle multiple sets.
        if '-bg' in plottedData:
            if not quiet:
                print('fitting background-subtracted data.')
            pictureFitParams, pictureFitErrors = fitPictures(dataMinusBg, range(len(key)), guessSigma_x=fitWidthGuess,
                                                             guessSigma_y=fitWidthGuess, quiet=quiet)
        elif '-avg' in plottedData:
            if not quiet:
                print('fitting average-subtracted data.')
            pictureFitParams, pictureFitErrors = fitPictures(dataMinusAvg, range(len(key)), guessSigma_x=fitWidthGuess,
                                                             guessSigma_y=fitWidthGuess, quiet=quiet)
        else:
            if not quiet:
                print('fitting raw data.')
            pictureFitParams, pictureFitErrors = fitPictures(rawData, range(len(key)), guessSigma_x=fitWidthGuess,
                                                             guessSigma_y=fitWidthGuess*0.6, quiet=quiet)
    else:
        pictureFitParams, pictureFitErrors = np.zeros((len(key), 7)), np.zeros((len(key), 7))

    # convert to normal optics convention. the equation uses gaussian as exp(x^2/2sigma^2), I want the waist,
    # which is defined as exp(2x^2/waist^2):
    waists = 2 * abs(arr([pictureFitParams[:, 3], pictureFitParams[:, 4]]))
    positions = arr([pictureFitParams[:, 1], pictureFitParams[:, 2]])
    if cameraType == 'dataray':
        pixelSize = consts.dataRayPixelSize
    elif cameraType == 'andor':
        pixelSize = consts.andorPixelSize
    elif cameraType == 'ace':
        pixelSize = consts.baslerAcePixelSize
    elif cameraType == 'scout':
        pixelSize = consts.baslerScoutPixelSize
    else:
        raise ValueError("Error: Bad Value for 'cameraType'.")
    waists *= pixelSize
    positions *= pixelSize
    # average of the two dimensions
    avgWaists = []
    for pair in np.transpose(arr(waists)):
        avgWaists.append((pair[0] + pair[1]) / 2)
    if fitBeamWaist:
        try:
            waistFitParamsX, waistFitErrsX = fitGaussianBeamWaist(waists[0], key, 850e-9)
            waistFitParamsY, waistFitErrsY = fitGaussianBeamWaist(waists[1], key, 850e-9)
            waistFitParams = [waistFitParamsX, waistFitParamsY]
            # assemble the data structures for plotting.
            countData, fitData = assemblePlotData(rawData, dataMinusBg, dataMinusAvg, positions, waists,
                                                  plottedData, scanType, xLabel, plotTitle, location,
                                                  waistFits=waistFitParams, key=key)
        except RuntimeError:
            print('gaussian waist fit failed!')
            # assemble the data structures for plotting.
            countData, fitData = assemblePlotData(rawData, dataMinusBg, dataMinusAvg, positions, waists,
                                                  plottedData, scanType, xLabel, plotTitle, location)
    else:
        # assemble the data structures for plotting.
        countData, fitData = assemblePlotData(rawData, dataMinusBg, dataMinusAvg, positions, waists,
                                              plottedData, scanType, xLabel, plotTitle, location)
    if majorData == 'counts':
        majorPlotData, minorPlotData = countData, fitData
    elif majorData == 'fits':
        minorPlotData, majorPlotData = countData, fitData
    else:
        raise ValueError("incorect 'majorData' argument")
    return key, rawData, dataMinusBg, dataMinusAvg, avgPic, pictureFitParams, pictureFitErrors



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


def analyzeScatterData( fileNumber, atomLocs1, connected=False, loadPic=1, transferPic=2, picsPerRep=3,
                        subtractEdgeCounts=True, histSecondPeakGuess=False, manualThreshold=None,
                        normalizeForLoadingRate=False, **transferOrganizeArgs ):
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
    fitInfo, fitFinished = fitWithModule(fitters.linear, key, psSurvivals.flatten(), errs=psErrors.flatten())
    for i, (data, err) in enumerate(zip(survivalData, survivalErrs)):
        survivalFits[i], _ = fitWithModule(fitters.linear, key, data.flatten(), errs=err.flatten())
    return key, psSurvivals, psErrors, fitInfo, fitFinished, survivalData, survivalErrs, survivalFits, atomLocs1

def standardTransferAnalysis( fileNumber, atomLocs1, atomLocs2, picsPerRep=2, manualThreshold=None,
                              fitModules=None, histSecondPeakGuess=None, outputMma=False, varyingDim=None,
                              subtractEdgeCounts=True, initPic=0, transPic=1, postSelectionCondition=None,
                              postSelectionConnected=False, getGenerationStats=False, rerng=False, tt=None,
                              rigorousThresholdFinding=True, transThresholdSame=True, fitguess=[None], forceAnnotation=True,
                              **organizerArgs ):
    """
    "Survival" is a special case of transfer where the initial location and the transfer location are the same location.
    """
    if tt is None:
        tt = TimeTracker()
    if rerng:
        initPic, transPic, picsPerRep = 1, 2, 3
    ( rawData, groupedData, atomLocs1, atomLocs2, keyName, repetitions, 
      key, numOfPictures, avgPics, basicInfoStr ) = organizeTransferData( fileNumber, atomLocs1, atomLocs2,  picsPerRep=picsPerRep, varyingDim=varyingDim,
                                                              initPic=initPic, transPic=transPic, **organizerArgs )
    (initPixelCounts, initThresholds, transPixelCounts, transThresholds) =  arr([[None] * len(atomLocs1)] * 4)
    for i, (loc1, loc2) in enumerate(zip(atomLocs1, atomLocs2)):
        initPixelCounts[i] = getAtomCountsData( rawData, picsPerRep, initPic, loc1, subtractEdges=subtractEdgeCounts )
        initThresholds[i] = getThresholds( initPixelCounts[i], 5, manualThreshold, rigorous=rigorousThresholdFinding )        
        if transThresholdSame:
            transThresholds[i] = copy(initThresholds[i])
        else: 
            transPixelCounts[i] = getAtomCountsData( rawData, picsPerRep, transPic, loc2, subtractEdges=subtractEdgeCounts )
            transThresholds[i] = getThresholds( transPixelCounts[i], 5, manualThreshold, rigorous=rigorousThresholdFinding )
     
    if subtractEdgeCounts:
        borders_init = getAvgBorderCount(groupedData, initPic, picsPerRep)
        borders_trans = getAvgBorderCount(groupedData, transPic, picsPerRep)
    else:
        borders_init = borders_trans = np.zeros(groupedData.shape[0]*groupedData.shape[1])
    (allInitPicCounts, allTransPicCounts, allInitAtoms, allTransAtoms) = arr([[None] * len(atomLocs1)] * 4)
    for i, (loc1, loc2) in enumerate(zip(atomLocs1, atomLocs2)):
        allInitPicCounts[i]  = normalizeData(groupedData, loc1, initPic, picsPerRep, borders_init)
        allTransPicCounts[i] = normalizeData(groupedData, loc2, transPic, picsPerRep, borders_trans)
        allInitAtoms[i], allTransAtoms[i] = getSurvivalBoolData(allInitPicCounts[i], allTransPicCounts[i], initThresholds[i].t, transThresholds[i].t)
    # transAtomsVarAvg, transAtomsVarErrs: the given an atom and a variation, the mean of the 
    # transfer events (mapped to a bernoili distribution), and the error on that mean. 
    # initAtoms (transAtoms): a list of the atom events in the initial (trans) picture, mapped to a bernoilli distribution 
    (initPicCounts, transPicCounts, bins, binnedData, transAtomsVarAvg, transAtomsVarErrs,
     initPopulation, initAtoms, transAtoms, genAvgs, genErrs, transList) = arr([[None] * len(atomLocs1)] * 12)
    for i, (loc1, loc2) in enumerate(zip(atomLocs1, atomLocs2)):
        initPicCounts[i]  = normalizeData(groupedData, loc1, initPic, picsPerRep, borders_init)
        transPicCounts[i] = normalizeData(groupedData, loc2, transPic, picsPerRep, borders_trans)
        initAtoms[i], transAtoms[i] = getSurvivalBoolData(initPicCounts[i], transPicCounts[i], initThresholds[i].t, transThresholds[i].t)
    if postSelectionCondition is not None:
        initAtoms, transAtoms = postSelectOnAssembly( initAtoms, transAtoms, postSelectionCondition,
                                                      connected=postSelectionConnected )
    for i in range(len(atomLocs1)):
        transList[i] = getTransferEvents(initAtoms[i], transAtoms[i])
        transAtomsVarAvg[i], transAtomsVarErrs[i], initPopulation[i] = getTransferStats(transList[i], repetitions)
        if getGenerationStats:
            genList = getGenerationEvents(initAtoms[i], transAtoms[i])
            genAvgs[i], genErrs[i] = getGenStatistics(genList, repetitions)
        else:
            genAvgs[i], genErrs[i] = [None, None]
    # Positioning of this is very important.
    #res = groupMultidimensionalData(key, varyingDim, atomLocs1, transferData, transAtomsVarErrs, initPopulation)
    res = (key, atomLocs1, transAtomsVarErrs, transAtomsVarAvg, initPopulation, [None for _ in range(len(key)*len(atomLocs1))])
    (key, locationsList, transAtomsVarErrs, transAtomsVarAvg, initPopulation, otherDims) = res

    initPicCounts = arr(initPicCounts.tolist())
    transPicCounts = arr(transPicCounts.tolist())
    
    # an atom average. The answer to the question: if you picked a random atom for a given variation, 
    # what's the mean [mean value] that you would find, and what is the error on that mean?
    # transAtomVarAvg, transAtomVarErr
    avgTransData, avgTransErr, avgFit = [None]*3
    # weight the sum with initial percentage
    avgTransData = np.mean(transAtomsVarAvg)
    avgTransErr = np.sqrt(np.sum(transAtomsVarErrs**2)/len(atomLocs1))
    ### ###
    # averaged over all events, summed over all atoms. this data has very small error bars
    # becaues of the factor of 100 in the number of atoms.
    transVarAvg, transVarErr = [[],[]]
    transVarList = [ah.groupEventsIntoVariations(atomsList, repetitions) for atomsList in transList]
    #print(transVarList.shape)
    allAtomsListByVar = [[z for atomList in transVarList for z in atomList[i]] for i in range(len(transVarList[0]))]
    #allAtomsListByVar = [sum(transVarList[:,i]) for i in range(len(transVarList[0]))]
    for varData in allAtomsListByVar:
        p = np.mean(varData)
        transVarAvg.append(p)
        transVarErr.append(ah.jeffreyInterval(p, len(arr(varData).flatten())))
        #transVarErr.append(np.sqrt(p*(1-p)/len(varData)))
       
    fits = [None] * len(locationsList)
    if len(fitModules) == 1: 
        fitModules = [fitModules[0] for _ in range(len(locationsList)+1)]
    if fitModules[0] is not None:
        if type(fitModules) != list:
            raise TypeError("ERROR: fitModules must be a list of fit modules. If you want to use only one module for everything,"
                            " then set this to a single element list with the desired module.")
        if len(fitguess) == 1:
            fitguess = [fitguess[0] for _ in range(len(locationsList)+1) ]
        if len(fitModules) != len(locationsList)+1:
            raise ValueError("ERROR: length of fitmodules should be" + str(len(locationsList)+1) + "(Includes avg fit)")
        for i, (loc, module) in enumerate(zip(locationsList, fitModules)):
            fits[i], _ = fitWithModule(module, key, transAtomsVarAvg[i], guess=fitguess[i])
        avgFit, _ = fitWithModule(fitModules[-1], key, avgTransData, guess=fitguess[-1])
    
    initAtomImages = [np.zeros(rawData[0].shape) for _ in range(int(numOfPictures/picsPerRep))]
    transAtomImages = [np.zeros(rawData[0].shape) for _ in range(int(numOfPictures/picsPerRep))]
    transImagesInc, initImagesInc = [0,0]
    for picInc in range(int(numOfPictures)):
        if picInc % picsPerRep == initPic:
            for locInc, loc in enumerate(atomLocs1):
                initAtomImages[initImagesInc][loc[0]][loc[1]] = allInitAtoms[locInc][initImagesInc]            
            initImagesInc += 1
        elif picInc % picsPerRep == transPic:
            for locInc, loc in enumerate(atomLocs2):
                transAtomImages[transImagesInc][loc[0]][loc[1]] = allTransAtoms[locInc][transImagesInc]
            transImagesInc += 1
    return (atomLocs1, atomLocs2, transAtomsVarAvg, transAtomsVarErrs, initPopulation, initPicCounts, keyName, key,
            repetitions, initThresholds, fits, avgTransData, avgTransErr, avgFit, avgPics, otherDims, locationsList,
            genAvgs, genErrs, tt, transVarAvg, transVarErr, initAtomImages, transAtomImages, transPicCounts, 
            transPixelCounts, transThresholds, fitModules, transThresholdSame, basicInfoStr)


def standardPopulationAnalysis( fileNum, atomLocations, whichPic, picsPerRep, analyzeTogether=False, 
                                manualThreshold=None, fitModules=[None], keyInput=None, fitIndv=False, subtractEdges=True,
                                keyConversion=None, quiet=False, dataRange=None, picSlice=None, keyOffset=0):
    """
    keyConversion should be a calibration which takes in a single value as an argument and converts it.
        It needs a calibration function f() and a units function units()
    return: ( fullPixelCounts, thresholds, avgPic, key, avgLoadingErr, avgLoading, allLoadingRate, allLoadingErr, loadFits,
             fitModule, keyName, totalAtomData, rawData, atomLocations, avgFits, atomImages, threshFitVals )
    """
    atomLocations = unpackAtomLocations(atomLocations)
    with ExpFile(fileNum) as f:
        rawData, keyName, hdf5Key, repetitions = f.pics, f.key_name, f.key, f.reps 
        if not quiet:
            f.get_basic_info()
    numOfPictures = rawData.shape[0]
    numOfVariations = int(numOfPictures / (repetitions * picsPerRep))
    key = ah.handleKeyModifications(hdf5Key, numOfVariations, keyInput=keyInput, keyOffset=keyOffset, groupData=False, keyConversion=keyConversion )
    # ## Initial Data Analysis
    s = rawData.shape
    groupedData = rawData.reshape((1, s[0], s[1], s[2]) if analyzeTogether else (numOfVariations, repetitions * picsPerRep, s[1], s[2]))
    key, groupedData = applyDataRange(dataRange, groupedData, key)
    if picSlice is not None:
        rawData = rawData[picSlice[0]:picSlice[1]]
        numOfPictures = rawData.shape[0]
        numOfVariations = int(numOfPictures / (repetitions * picsPerRep))
    print(rawData.shape[0], numOfPictures, numOfVariations,'hi')
    #groupedData, key, _ = orderData(groupedData, key)
    avgPopulation, avgPopulationErr, popFits = [[[] for _ in range(len(atomLocations))] for _ in range(3)]
    allPopulation, allPopulationErr = [[[]] * len(groupedData) for _ in range(2)]
    totalAtomData = []
    # get full data... there's probably a better way of doing this...
    (fullPixelCounts, fullAtomData, thresholds, fullAtomCount) = arr([[None] * len(atomLocations)] * 4)
    for i, atomLoc in enumerate(atomLocations):
        fullPixelCounts[i] = getAtomCountsData( rawData, picsPerRep, whichPic, atomLoc, subtractEdges=subtractEdges )
        thresholds[i] = getThresholds( fullPixelCounts[i], 5, manualThreshold )
        fullAtomData[i], fullAtomCount[i] = getAtomBoolData(fullPixelCounts[i], thresholds[i].t)
    flatTotal = arr(arr(fullAtomData).tolist()).flatten()
    totalAvg = np.mean(flatTotal)
    totalErr = np.std(flatTotal) / np.sqrt(len(flatTotal))
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
            variationAtomData[dataInc][i], atomCount[dataInc][i] = getAtomBoolData(variationPixelData[dataInc][i], thresholds[i].t)            
            totalAtomData.append(variationAtomData[dataInc][i])
            mVal = np.mean(variationAtomData[dataInc][i])
            allAtomPicData.append(mVal)
            avgPopulation[i].append(mVal)
            # avgPopulationErr[i].append(np.std(variationAtomData[dataInc][i]) / np.sqrt(len(variationAtomData[dataInc][i])))
            avgPopulationErr[i].append(ah.jeffreyInterval(mVal, len(variationAtomData[dataInc][i])))
            # np.std(variationAtomData[dataInc][i]) / np.sqrt(len(variationAtomData[dataInc][i])))
        meanVal = np.mean(allAtomPicData)
        allPopulation[dataInc] = meanVal
        allPopulationErr[dataInc] = ah.jeffreyInterval(meanVal, len(variationAtomData[dataInc][i])*len(arr(allAtomPicData).flatten()))
        # Old error: np.std(allAtomPicData) / np.sqrt(len(allAtomPicData))
    # 
    avgFits = None
    if len(fitModules) == 1: 
        fitModules = [fitModules[0] for _ in range(len(avgPopulation)+1)]
    if fitModules[0] is not None:
        if type(fitModules) != list:
            raise TypeError("ERROR: fitModules must be a list of fit modules. If you want to use only one module for everything,"
                            " then set this to a single element list with the desired module.")
        if len(fitModules) == 1: 
            fitModules = [fitModules[0] for _ in range(len(avgPopulation)+1)]
        if fitIndv:
            for i, (pop, module) in enumerate(zip(avgPopulation, module)):
                popFits[i], _ = fitWithModule(module, key, pop)
        avgFits, _ = fitWithModule(fitModules[-1], key, allPopulation)
    avgPics = getAvgPics(rawData, picsPerRep=picsPerRep)
    avgPic = avgPics[whichPic]
    # get averages across all variations
    atomImages = [np.zeros(rawData[0].shape) for _ in range(int(numOfPictures/picsPerRep))]
    atomImagesInc = 0
    for picInc in range(int(numOfPictures)):
        if picInc % picsPerRep != whichPic:
            continue
        for locInc, loc in enumerate(atomLocations):
            atomImages[atomImagesInc][loc[0]][loc[1]] = fullAtomData[locInc][atomImagesInc]
        atomImagesInc += 1

    return ( fullPixelCounts, thresholds, avgPic, key, avgPopulationErr, avgPopulation, allPopulation, allPopulationErr, popFits,
             fitModules, keyName, totalAtomData, rawData, atomLocations, avgFits, atomImages, totalAvg, totalErr )


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
        atomCounts[i] = arr([])
        for pic1, pic2 in zip(loadPicData[i], assemblyPicData[i]):
            atomCounts[i] = np.append(atomCounts[i], [pic1, pic2])
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

