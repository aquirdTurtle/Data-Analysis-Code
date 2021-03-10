import numpy as np
import copy
from . import AnalysisHelpers as ah
from . import ExpFile as exp
from . import PictureWindow as pw
from . import ThresholdOptions as to
from . import TransferAnalysisOptions as ao
from . import Miscellaneous as misc
from .Miscellaneous import what

def organizeTransferData( fileNumber, analysisOpts, key=None, win=pw.PictureWindow(), dataRange=None, keyOffset=0, 
                          dimSlice=None, varyingDim=None, groupData=False, quiet=False, picsPerRep=2, repRange=None, 
<<<<<<< HEAD
                          keyConversion=None, binningParams=None, removePics=None, expFile_version=4, useBaseA=True):
=======
                          keyConversion=None, softwareBinning=None, removePics=None, expFile_version=4, useBaseA=True, 
                          keyParameter=None):
>>>>>>> e0d9c50a3fbfd2164e5e4358aaeec307f4096e9e
    """
    Unpack inputs, properly shape the key, picture array, and run some initial checks on the consistency of the settings.
    """
    with exp.ExpFile(fileNumber, expFile_version=expFile_version, useBaseA=useBaseA, keyParameter=keyParameter) as f:
        rawData, keyName, hdf5Key, repetitions = f.pics, f.key_name, f.key, f.reps
        if not quiet:
            basicInfoStr = f.get_basic_info()
    if removePics is not None:
        for index in reversed(sorted(removePics)):
            rawData = np.delete(rawData, index, 0)
            # add zero pics to the end to keep the total number consistent.
            rawData = np.concatenate((rawData, [np.zeros(rawData[0].shape)]), 0 )
    if repRange is not None:
        repetitions = repRange[1] - repRange[0]
        rawData = rawData[repRange[0]*picsPerRep:repRange[1]*picsPerRep]

    rawData = np.array([win.window(pic) for pic in rawData])
    rawData = ah.softwareBinning(binningParams, rawData)
    # Group data into variations.
    numberOfPictures = int(rawData.shape[0])
    if groupData:
        repetitions = int(numberOfPictures / picsPerRep)
    numberOfVariations = int(numberOfPictures / (repetitions * picsPerRep))
    key = ah.handleKeyModifications(hdf5Key, numberOfVariations, keyInput=key, keyOffset=keyOffset, groupData=groupData, keyConversion=keyConversion )
    groupedDataRaw = rawData.reshape((numberOfVariations, repetitions * picsPerRep, rawData.shape[1], rawData.shape[2]))
    res = ah.sliceMultidimensionalData(dimSlice, key, groupedDataRaw, varyingDim=varyingDim)
    (_, slicedData, otherDimValues, varyingDim) = res
    slicedOrderedData = slicedData
    key, groupedData = ah.applyDataRange(dataRange, slicedOrderedData, key)
    # check consistency
    numberOfPictures = int(groupedData.shape[0] * groupedData.shape[1])
    numberOfVariations = int(numberOfPictures / (repetitions * picsPerRep))
    numOfPictures = groupedData.shape[0] * groupedData.shape[1]
    allAvgPics = ah.getAvgPics(groupedData, picsPerRep=picsPerRep)
    avgPics = [allAvgPics[analysisOpts.initPic], allAvgPics[analysisOpts.tferPic]]
    return rawData, groupedData, keyName, repetitions, key, numOfPictures, avgPics, basicInfoStr, analysisOpts

def getGeneralEvents(pic1Atoms, pic2Atoms, positiveResultCondition):
    eventList = ah.getConditionHits( [pic1Atoms, pic2Atoms], positiveResultCondition);
    return np.array(eventList)

def getTransferStats(tferList):
    # Take the previous data, which includes entries when there was no atom in the first picture, and convert it to
    # an array of just loaded and survived or loaded and died.
    transferErrors = np.zeros([2])
    tferVarList = np.array([x for x in tferList if x != -1])
    if tferVarList.size == 0:
        # catch the case where there's no relevant data, typically if laser becomes unlocked.
        transferErrors = [0,0]
        transferAverages = 0
    else:
        # normal case
        transferAverages = np.average(tferVarList)
        transferErrors = ah.jeffreyInterval(transferAverages, len(tferVarList))
    return transferAverages, transferErrors


def getTransferThresholds(analysisOpts, rawData, groupedData, picsPerRep, tOptions=[to.ThresholdOptions()]):
    # some initialization...
    (initThresholds, tferThresholds) =  np.array([[None] * len(analysisOpts.initLocs())] * 2)
    for atomThresholdInc, _ in enumerate(initThresholds):
        initThresholds[atomThresholdInc] = np.array([None for _ in range(groupedData.shape[0])])
        tferThresholds[atomThresholdInc] = np.array([None for _ in range(groupedData.shape[0])])
    if len(tOptions) == 1:
        tOptions = [tOptions[0] for _ in range(len(analysisOpts.initLocs()))]
    # getting thresholds
    for i, (loc1, loc2) in enumerate(zip(analysisOpts.initLocs(), analysisOpts.tferLocs())):
        opt = tOptions[i]
        if opt.indvVariationThresholds:
            for j, variationData in enumerate(groupedData):
                initPixelCounts = ah.getAtomCountsData( variationData, picsPerRep, analysisOpts.initPic, loc1, subtractEdges=opt.subtractEdgeCounts )
                initThresholds[i][j] = ah.getThresholds( initPixelCounts, 5, opt )        
        else:
            # calculate once with full raw data and then copy to all slots. 
            initPixelCounts = ah.getAtomCountsData( rawData, picsPerRep, analysisOpts.initPic, loc1, subtractEdges=opt.subtractEdgeCounts )
            initThresholds[i][0] = ah.getThresholds( initPixelCounts, 5, opt )        
            for j, _ in enumerate(groupedData):
                initThresholds[i][j] = initThresholds[i][0]
        if opt.tferThresholdSame:
            tferThresholds[i] = copy.copy(initThresholds[i])
        else: 
            if opt.indvVariationThresholds:
                for j, variationData in enumerate(groupedData):
                    tferPixelCounts = ah.getAtomCountsData( variationData, picsPerRep, analysisOpts.tferPic, loc2, 
                                                           subtractEdges=opt.subtractEdgeCounts )
                    tferThresholds[i][j] = ah.getThresholds( tferPixelCounts, 5, opt )
            else:
                tferPixelCounts = ah.getAtomCountsData( rawData, picsPerRep, analysisOpts.tferPic, loc1, 
                                                       subtractEdges=opt.subtractEdgeCounts )
                tferThresholds[i][0] = ah.getThresholds( tferPixelCounts, 5, opt )        
                for j, _ in enumerate(groupedData):
                    tferThresholds[i][j] = tferThresholds[i][0]
    if tOptions[0].subtractEdgeCounts:
        borders_init = ah.getAvgBorderCount(groupedData, analysisOpts.initPic, picsPerRep)
        borders_tfer = ah.getAvgBorderCount(groupedData, analysisOpts.tferPic, picsPerRep)
    else:
        borders_init = borders_tfer = np.zeros(groupedData.shape[0]*groupedData.shape[1])
    return borders_init, borders_tfer, initThresholds, tferThresholds

def getTransferAtomImages( analysisOpts, groupedData, numOfPictures, picsPerRep, initAtoms, tferAtoms ):
    initAtomImages, tferAtomImages = [np.zeros(groupedData.shape) for _ in range(2)]
    for varNum, varPics in enumerate(groupedData):
        tferImagesInc, initImagesInc = [0,0]
        for picInc in range(len(varPics)):
            if picInc % picsPerRep == analysisOpts.initPic:
                for locInc, loc in enumerate(analysisOpts.initLocs()):
                    initAtomImages[varNum][initImagesInc][loc[0]][loc[1]] = initAtoms[varNum][locInc][initImagesInc]            
                initImagesInc += 1
            elif picInc % picsPerRep == analysisOpts.tferPic:
                for locInc, loc in enumerate(analysisOpts.tferLocs()):
                    tferAtomImages[varNum][tferImagesInc][loc[0]][loc[1]] = tferAtoms[varNum][locInc][tferImagesInc]
                tferImagesInc += 1
    return initAtomImages, tferAtomImages

def determineTransferAtomPrescence( analysisOpts, groupedData, picsPerRep, borders_init, borders_tfer, initThresholds, tferThresholds):
    # tferAtomsVarAvg, tferAtomsVarErrs: the given an atom and a variation, the mean of the 
    # tferfer events (mapped to a bernoili distribution), and the error on that mean. 
    # initAtoms (tferAtoms): a list of the atom events in the initial (tfer) picture, mapped to a bernoilli distribution
    numAtomLocations = len(analysisOpts.initLocs())
    numDataSets = len(analysisOpts.positiveResultConditions)
    (initAtoms, tferAtoms) = [[[[] for _ in range(numAtomLocations)] for _ in groupedData] for _ in range(2)]
    initPicCounts, tferPicCounts = [[[] for _ in range(numAtomLocations)] for _ in range(2)]
    for i, (loc1, loc2) in enumerate(zip(analysisOpts.initLocs(), analysisOpts.tferLocs())):
        for variationNum, varData in enumerate(groupedData):
            initCounts = ah.normalizeData(varData, loc1, analysisOpts.initPic, picsPerRep, borders_init )
            tferCounts = ah.normalizeData(varData, loc2, analysisOpts.tferPic, picsPerRep, borders_tfer )
            Iatoms, Tatoms = ah.getSurvivalBoolData(initCounts, tferCounts, initThresholds[i][variationNum].t, tferThresholds[i][variationNum].t)
            initAtoms[variationNum][i] = Iatoms 
            tferAtoms[variationNum][i] = Tatoms
            tferPicCounts[i] += list(tferCounts)
            initPicCounts[i] += list(initCounts)
    return initAtoms, tferAtoms, np.array(initPicCounts), np.array(tferPicCounts)

def handleTransferFits(analysisOpts, fitModules, key, avgTferData, fitguess, getFitterArgs, tferAtomsVarAvg):
    numDataSets = len(analysisOpts.positiveResultConditions)
    numAtomLocations = len(analysisOpts.initLocs())
    fits = [None] * numDataSets
    avgFit = None
    if type(fitModules) is not list:
        fitModules = [fitModules]
    if len(fitModules) == 1: 
        fitModules = [fitModules[0] for _ in range(numDataSets+1)]
    if fitModules[0] is not None:
        if type(fitModules) != list:
            raise TypeError("ERROR: fitModules must be a list of fit modules. If you want to use only one module for everything,"
                            " then set this to a single element list with the desired module.")
        if len(fitguess) == 1:
            fitguess = [fitguess[0] for _ in range(numDataSets+1) ]
        if len(getFitterArgs) == 1:
            getFitterArgs = [getFitterArgs[0] for _ in range(numDataSets+1) ]
        if len(fitModules) != numDataSets+1:
            raise ValueError("ERROR: length of fitmodules should be" + str(numDataSets+1) + "(Includes avg fit)")
        for i, (loc, module) in enumerate(zip(range(analysisOpts.numDataSets()), fitModules)):
            fits[i], _ = ah.fitWithModule(module, key, tferAtomsVarAvg[i], guess=fitguess[i], getF_args=getFitterArgs[i])
        avgFit, _ = ah.fitWithModule(fitModules[-1], key, avgTferData, guess=fitguess[-1], getF_args=getFitterArgs[-1])
    return fits, avgFit, fitModules

def getTransferAvgs(analysisOpts, initAtomsPs, tferAtomsPs, prConditions=None):
    (tferList, tferAtomsVarAvg, tferAtomsVarErrs) = [[[None for _ in initAtomsPs] for _ in range(analysisOpts.numDataSets())] for _ in range(3)]
    if prConditions is None:
        prConditions = analysisOpts.positiveResultConditions
    for dsetInc in range(analysisOpts.numDataSets()):
        if prConditions[dsetInc] is None:
            print('using default positive result condition...')
        for varInc in range(len(initAtomsPs)):
            if prConditions[dsetInc] is None:
                prConditions[dsetInc] = ao.condition(name='Default Survival Condition', whichPic=[1],
                                                     whichAtoms=[dsetInc],conditions=[True],numRequired=-1);
            tferList[dsetInc][varInc] = getGeneralEvents(misc.transpose(initAtomsPs[varInc][dsetInc]), misc.transpose(tferAtomsPs[varInc][dsetInc]),
                                                         prConditions[dsetInc])
            tferAtomsVarAvg[dsetInc][varInc], tferAtomsVarErrs[dsetInc][varInc] = getTransferStats( tferList[dsetInc][varInc] )
    # an atom average. The answer to the question: if you picked a random atom for a given variation, 
    # what's the mean [mean value] that you would find (averaging over the atoms), and what is the error on that mean?
    # weight the sum with initial percentage
    avgTferData = np.mean(tferAtomsVarAvg, 0)
    avgTferErr = np.sqrt(np.sum(np.array(tferAtomsVarErrs)**2,0) / analysisOpts.numDataSets())
    # averaged over all events, summed over all atoms. this data has very small error bars
    # becaues of the factor of 100 in the number of atoms.
    tferVarAvg, tferVarErr = [[],[]]
    allAtomsListByVar = [[] for _ in tferList[0]]
    for atomInc, atomList in enumerate(tferList):
        for varInc, varList in enumerate(atomList):
            for dp in varList:
                if dp != -1:
                    allAtomsListByVar[varInc].append(dp)    
    for varData in allAtomsListByVar:
        mv = np.mean(varData)
        tferVarAvg.append(mv)
        tferVarErr.append(ah.jeffreyInterval(mv, len(np.array(varData).flatten())))
    return avgTferData, avgTferErr, tferVarAvg, tferVarErr, tferAtomsVarAvg, tferAtomsVarErrs, tferList


def standardTransferAnalysis( fileNumber, analysisOpts, picsPerRep=2, fitModules=[None], varyingDim=None, getGenerationStats=False, 
                              fitguess=[None], forceAnnotation=True, tOptions=[to.ThresholdOptions()], getFitterArgs=[None], **organizerArgs ):
    """
    "Survival" is a special case of transfer where the initial location and the transfer location are the same location.
    """
    assert(type(analysisOpts) == ao.TransferAnalysisOptions)
    print("sta: Organizing Transfer Data...")
    ( rawData, groupedData, keyName, repetitions, key, numOfPictures, avgPics, 
      basicInfoStr, analysisOpts ) = organizeTransferData( fileNumber, analysisOpts, picsPerRep=picsPerRep, varyingDim=varyingDim, **organizerArgs )
    print("sta: Getting Transfer Thresholds...")
    res = getTransferThresholds( analysisOpts, rawData, groupedData, picsPerRep, tOptions )
    borders_init, borders_tfer, initThresholds, tferThresholds = res
    print("sta: Determining Atom Prescence...")
    res = determineTransferAtomPrescence( analysisOpts, groupedData, picsPerRep, borders_init, borders_tfer, initThresholds, tferThresholds)
    initAtoms, tferAtoms, initPicCounts, tferPicCounts = res
    print("sta: Getting Transfer Atom Images...")
    initAtomImages, tferAtomImages = getTransferAtomImages( analysisOpts, groupedData, numOfPictures, picsPerRep, initAtoms, tferAtoms )    
    initAtomsPs, tferAtomsPs, ensembleHits = [[None for _ in initAtoms] for _ in range(3)]
    print("sta: Post-Selecting...",end='')
    for varInc in range(len(initAtoms)):
        print('.',end='')
        initAtomsPs[varInc], tferAtomsPs[varInc], ensembleHits[varInc] = ah.postSelectOnAssembly(initAtoms[varInc], tferAtoms[varInc], analysisOpts )
        initAtoms[varInc], tferAtoms[varInc], _ = ah.postSelectOnAssembly(initAtoms[varInc], tferAtoms[varInc], analysisOpts, justReformat=True)
    print("sta: Getting Transfer Averages...")
    res = getTransferAvgs(analysisOpts, initAtomsPs, tferAtomsPs)
    avgTferData, avgTferErr, tferVarAvg, tferVarErr, tferAtomsVarAvg, tferAtomsVarErrs, tferList = res
    print("sta: Getting Load Averages...")
    loadConditions = []
    if len(analysisOpts.postSelectionConditions[0]) == 0:
        # no post-selection, probably loading, in which case this data isn't useful, but okay. 
        (avgloadData, avgloadErr, loadVarAvg, loadVarErr, loadAtomsVarAvg, loadAtomsVarErrs, 
         loadList) = getTransferAvgs(analysisOpts, initAtoms, tferAtoms, analysisOpts.positiveResultConditions)
    else:
        # assumes that the first post-selection condition is loading somewhere. 
        (avgloadData, avgloadErr, loadVarAvg, loadVarErr, loadAtomsVarAvg, loadAtomsVarErrs, 
         loadList) = getTransferAvgs(analysisOpts, initAtoms, tferAtoms, [conditions[0] for conditions in analysisOpts.postSelectionConditions])
    print("sta: Handling Fitting...")
    fits, avgFit, fitModules = handleTransferFits( analysisOpts, fitModules, key, avgTferData, fitguess, getFitterArgs, tferAtomsVarAvg )
    genAvgs, genErrs = [None, None]
    return (tferAtomsVarAvg, tferAtomsVarErrs, loadAtomsVarAvg, initPicCounts, keyName, key, repetitions, initThresholds, 
            fits, avgTferData, avgTferErr, avgFit, avgPics, genAvgs, genErrs, tferVarAvg, tferVarErr, initAtomImages, 
            tferAtomImages, tferPicCounts, tferThresholds, fitModules, basicInfoStr, ensembleHits, tOptions, analysisOpts,
            tferAtomsPs, tferAtomsPs, tferList)
