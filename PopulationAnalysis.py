from numpy import array as arr
from pandas import DataFrame

from Miscellaneous import getStats, round_sig, errString
from MarksFourierAnalysis import fft
from matplotlib.pyplot import *
from scipy.optimize import curve_fit as fit
import fitters.linear
from ExpFile import ExpFile
import ExpFile as exp
from TimeTracker import TimeTracker
import AnalysisHelpers as ah
import MarksConstants as mc
import copy
import PictureWindow as pw
import ThresholdOptions as to
import sys, os

# Disable printing 
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
    return close(os.devnull)


def standardPopulationAnalysis( fileNum, atomLocations, whichPic, picsPerRep, analyzeTogether=False, 
                                thresholdOptions=to.ThresholdOptions(), fitModules=[None], keyInput=None, fitIndv=False, subtractEdges=True,
                                keyConversion=None, quiet=False, dataRange=None, picSlice=None, keyOffset=0, softwareBinning=None,
                                window=None, yMin=None, yMax=None, xMin=None, xMax=None, expFile_version=4, useBaseA=True, **StandardArgs ):
    """
    keyConversion should be a calibration which takes in a single value as an argument and converts it.
        It needs a calibration function f() and a units function units()
    return: ( fullPixelCounts, thresholds, avgPic, key, avgLoadingErr, avgLoading, allLoadingRate, allLoadingErr, loadFits,
             fitModule, keyName, totalAtomData, rawData, atomLocations, avgFits, atomImages, threshFitVals )
    """
    atomLocations = ah.unpackAtomLocations(atomLocations)
    with ExpFile(fileNum, expFile_version=expFile_version, useBaseA=useBaseA) as f:
        rawData, keyName, hdf5Key, repetitions = f.pics, f.key_name, f.key, f.reps 
        if not quiet:
            f.get_basic_info()
    numOfPictures = rawData.shape[0]
    numOfVariations = int(numOfPictures / (repetitions * picsPerRep))
    key = ah.handleKeyModifications(hdf5Key, numOfVariations, keyInput=keyInput, keyOffset=keyOffset, groupData=False, keyConversion=keyConversion )
    # ## Initial Data Analysis
    # window the images images.
    if window is not None:
        xMin, yMin, xMax, yMax = window
    rawData = np.copy(arr(rawData[:, yMin:yMax, xMin:xMax]))

    if softwareBinning is not None:
        sb = softwareBinning
        print(rawData.shape)
        rawData = rawData.reshape(rawData.shape[0], rawData.shape[1]//sb[0], sb[0], rawData.shape[2]//sb[1], sb[1]).sum(4).sum(2)
    s = rawData.shape
    groupedData = rawData.reshape((1, s[0], s[1], s[2]) if analyzeTogether else (numOfVariations, repetitions * picsPerRep, s[1], s[2]))
    key, groupedData = ah.applyDataRange(dataRange, groupedData, key)
    
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
        fullPixelCounts[i] = ah.getAtomCountsData( rawData, picsPerRep, whichPic, atomLoc, subtractEdges=subtractEdges )
        thresholds[i] = ah.getThresholds( fullPixelCounts[i], 5, thresholdOptions )
        fullAtomData[i], fullAtomCount[i] = ah.getAtomBoolData(fullPixelCounts[i], thresholds[i].t)
    flatTotal = arr(arr(fullAtomData).tolist()).flatten()
    totalAvg = np.mean(flatTotal)
    totalErr = np.std(flatTotal) / np.sqrt(len(flatTotal))
    fullAtomData = arr(fullAtomData.tolist())
    fullPixelCounts = arr(fullPixelCounts.tolist())
    if not quiet:
        #print('Analyzing Variation... ', end='')
        (variationPixelData, variationAtomData, atomCount) = arr([[[None for _ in atomLocations] for _ in groupedData] for _ in range(3)])
    for dataInc, data in enumerate(groupedData):
        if not quiet:
            #print(str(dataInc) + ', ', end='')
            blockPrint()
            allAtomPicData = []
        for i, atomLoc in enumerate(atomLocations):
            variationPixelData[dataInc][i] = ah.getAtomCountsData( data, picsPerRep, whichPic, atomLoc, subtractEdges=subtractEdges )
            variationAtomData[dataInc][i], atomCount[dataInc][i] = ah.getAtomBoolData(variationPixelData[dataInc][i], thresholds[i].t)            
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
                popFits[i], _ = ah.fitWithModule(module, key, pop)
        avgFits, _ = ah.fitWithModule(fitModules[-1], key, allPopulation)
    avgPics = ah.getAvgPics(rawData, picsPerRep=picsPerRep)
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


