import AnalysisHelpers as ah
import ExpFile as exp
import numpy as np
import copy

def organizeTransferData( fileNumber, initLocs, tferLocs, key=None, window=None, xMin=None, xMax=None, yMin=None,
                          yMax=None, dataRange=None, keyOffset=0, dimSlice=None, varyingDim=None, groupData=False,
                          quiet=False, picsPerRep=2, repRange=None, initPic=0, tferPic=1, keyConversion=None, softwareBinning=None,
                          removePics=None, expFile_version=3):
    """
    Unpack inputs, properly shape the key, picture array, and run some initial checks on the consistency of the settings.
    """
    with exp.ExpFile(fileNumber, expFile_version=expFile_version) as f:
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
    if softwareBinning is not None:
        sb = softwareBinning
        rawData = rawData.reshape(rawData.shape[0], rawData.shape[1]//sb[0], sb[0], rawData.shape[2]//sb[1], sb[1]).sum(4).sum(2)
    # window the images images.
    if window is not None:
        xMin, yMin, xMax, yMax = window
    rawData = np.copy(np.array(rawData[:, yMin:yMax, xMin:xMax]))
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
    avgPics = [allAvgPics[initPic], allAvgPics[tferPic]]
    atomLocs1 = ah.unpackAtomLocations(initLocs, avgPic=avgPics[0])
    atomLocs2 = ah.unpackAtomLocations(tferLocs, avgPic=avgPics[1])
    return rawData, groupedData, atomLocs1, atomLocs2, keyName, repetitions, key, numOfPictures, avgPics, basicInfoStr

def getTransferEvents(pic1Atoms, pic2Atoms):
    """
    It returns a raw array that includes every survival data point, including points where the the atom doesn't get
    loaded at all.
    """
    # this will include entries for when there is no atom in the first picture.
    pic1Atoms = np.array(np.array(pic1Atoms).tolist())
    pic2Atoms = np.array(np.array(pic2Atoms).tolist())
    # -1 = no atom in the first place.
    tferferData = np.zeros(pic1Atoms.shape) - 1
    # convert to 1 if atom and atom survived
    tferferData += 2 * pic1Atoms * pic2Atoms
    # convert to 0 if atom and atom didn't survive. This and the above can't both evaluate to non-zero.
    tferferData += pic1Atoms * (~pic2Atoms)
    return tferferData.flatten()

def getTransferStats(tferList, repsPerVar):
    # Take the previous data, which includes entries when there was no atom in the first picture, and convert it to
    # an array of just loaded and survived or loaded and died.
    numVars = int(tferList.size / repsPerVar)
    transferAverages = np.array([])
    loadingProbability = np.array([])
    transferErrors = np.zeros([numVars,2])
    if tferList.size < repsPerVar:
        # probably a single variation that has been sliced to cut out bad data.
        repsPerVar = tferList.size
    for variationInc in range(0, numVars):
        tferVarList = np.array([x for x in tferList[variationInc * repsPerVar:(variationInc+1) * repsPerVar] if x != -1])
        if tferVarList.size == 0:
            # catch the case where there's no relevant data, typically if laser becomes unlocked.
            transferErrors[variationInc] = [0,0]
            loadingProbability = np.append(loadingProbability, [0])
            transferAverages = np.append(transferAverages, [0])
        else:
            # normal case
            meanVal = np.average(tferVarList)
            transferErrors[variationInc] = ah.jeffreyInterval(meanVal, len(tferVarList))
            # old method
            loadingProbability = np.append(loadingProbability, tferVarList.size / repsPerVar)
            transferAverages = np.append(transferAverages, meanVal)    
    return transferAverages, transferErrors, loadingProbability


def getTransferThresholds(atomLocs1, atomLocs2, rawData, groupedData, thresholdOptions, indvVariationThresholds, picsPerRep, initPic, tferPic, 
                          subtractEdgeCounts, rigorousThresholdFinding, tferThresholdSame):
    # some initialization...
    (initThresholds, tferThresholds) =  np.array([[None] * len(atomLocs1)] * 2)
    for atomThresholdInc, _ in enumerate(initThresholds):
        initThresholds[atomThresholdInc] = np.array([None for _ in range(groupedData.shape[0])])
        tferThresholds[atomThresholdInc] = np.array([None for _ in range(groupedData.shape[0])])
    if thresholdOptions == None:
        thresholdOptions = [None for _ in range(len(atomLocs1))]
    if len(thresholdOptions) == 1:
        thresholdOptions = [thresholdOptions[0] for _ in range(len(atomLocs1))]
    # gettubg thresholds
    for i, (loc1, loc2) in enumerate(zip(atomLocs1, atomLocs2)):
        if indvVariationThresholds:
            for j, variationData in enumerate(groupedData):
                initPixelCounts = ah.getAtomCountsData( variationData, picsPerRep, initPic, loc1, subtractEdges=subtractEdgeCounts )
                initThresholds[i][j] = ah.getThresholds( initPixelCounts, 5, thresholdOptions[i], rigorous=rigorousThresholdFinding )        
        else:
            # calculate once with full raw data and then copy to all slots. 
            initPixelCounts = ah.getAtomCountsData( rawData, picsPerRep, initPic, loc1, subtractEdges=subtractEdgeCounts )
            initThresholds[i][0] = ah.getThresholds( initPixelCounts, 5, thresholdOptions[i], rigorous=rigorousThresholdFinding )        
            for j, _ in enumerate(groupedData):
                initThresholds[i][j] = initThresholds[i][0]
        if tferThresholdSame:
            tferThresholds[i] = copy.copy(initThresholds[i])
        else: 
            if indvVariationThresholds:
                for j, variationData in enumerate(groupedData):
                    tferPixelCounts = ah.getAtomCountsData( variationData, picsPerRep, tferPic, loc2, subtractEdges=subtractEdgeCounts )
                    tferThresholds[i][j] = ah.getThresholds( tferPixelCounts, 5, thresholdOptions[i], rigorous=rigorousThresholdFinding )
            else:
                tferPixelCounts = ah.getAtomCountsData( rawData, picsPerRep, initPic, loc1, subtractEdges=subtractEdgeCounts )
                tferThresholds[i][0] = ah.getThresholds( tferPixelCounts, 5, thresholdOptions[i], rigorous=rigorousThresholdFinding )        
                for j, _ in enumerate(groupedData):
                    tferThresholds[i][j] = tferThresholds[i][0]
    if subtractEdgeCounts:
        borders_init = ah.getAvgBorderCount(groupedData, initPic, picsPerRep)
        borders_tfer = ah.getAvgBorderCount(groupedData, tferPic, picsPerRep)
    else:
        borders_init = borders_tfer = np.zeros(groupedData.shape[0]*groupedData.shape[1])
    return borders_init, borders_tfer, initThresholds, tferThresholds

def getTransferAtomImages(atomLocs1, atomLocs2, rawData, numOfPictures, picsPerRep, initPic, tferPic, initAtoms, tferAtoms):
    initAtomImages = [np.zeros(rawData[0].shape) for _ in range(int(numOfPictures/picsPerRep))]
    tferAtomImages = [np.zeros(rawData[0].shape) for _ in range(int(numOfPictures/picsPerRep))]
    tferImagesInc, initImagesInc = [0,0]
    for picInc in range(int(numOfPictures)):
        if picInc % picsPerRep == initPic:
            for locInc, loc in enumerate(atomLocs1):
                initAtomImages[initImagesInc][loc[0]][loc[1]] = initAtoms[locInc][initImagesInc]            
            initImagesInc += 1
        elif picInc % picsPerRep == tferPic:
            for locInc, loc in enumerate(atomLocs2):
                tferAtomImages[tferImagesInc][loc[0]][loc[1]] = tferAtoms[locInc][tferImagesInc]
            tferImagesInc += 1
    return initAtomImages, tferAtomImages

def determineTransferAtomPrescence( atomLocs1, atomLocs2, groupedData, initPic, tferPic, picsPerRep, borders_init, borders_tfer,
                                    initThresholds, tferThresholds):
    # tferAtomsVarAvg, tferAtomsVarErrs: the given an atom and a variation, the mean of the 
    # tferfer events (mapped to a bernoili distribution), and the error on that mean. 
    # initAtoms (tferAtoms): a list of the atom events in the initial (tfer) picture, mapped to a bernoilli distribution 
    (initPicCounts, tferPicCounts, bins, binnedData, tferAtomsVarAvg, tferAtomsVarErrs,
     initPopulation, initAtoms, tferAtoms, genAvgs, genErrs, tferList) = [[[] for _ in range(len(atomLocs1))] for _ in range(12)]
    for i, (loc1, loc2) in enumerate(zip(atomLocs1, atomLocs2)):
        for j, varData in enumerate(groupedData):
            initCounts = ah.normalizeData(varData, loc1, initPic, picsPerRep, borders_init )
            tferCounts = ah.normalizeData(varData, loc2, tferPic, picsPerRep, borders_tfer )
            Iatoms, Tatoms = ah.getSurvivalBoolData(initCounts, tferCounts, initThresholds[i][j].t, tferThresholds[i][j].t)
            initAtoms[i] += Iatoms 
            tferAtoms[i] += Tatoms
            tferPicCounts[i] += list(tferCounts)
            initPicCounts[i] += list(initCounts)
    return initAtoms, tferAtoms, tferList, tferAtomsVarAvg, tferAtomsVarErrs, initPopulation, genAvgs, genErrs, np.array(initPicCounts), np.array(tferPicCounts)

def handleTransferFits(locationsList, fitModules, key, avgTferData, fitguess, getFitterArgs):
    fits = [None] * len(locationsList)
    avgFit = None
    if len(fitModules) == 1: 
        fitModules = [fitModules[0] for _ in range(len(locationsList)+1)]
    if fitModules[0] is not None:
        if type(fitModules) != list:
            raise TypeError("ERROR: fitModules must be a list of fit modules. If you want to use only one module for everything,"
                            " then set this to a single element list with the desired module.")
        if len(fitguess) == 1:
            fitguess = [fitguess[0] for _ in range(len(locationsList)+1) ]
        if len(getFitterArgs) == 1:
            getFitterArgs = [getFitterArgs[0] for _ in range(len(locationsList)+1) ]
        if len(fitModules) != len(locationsList)+1:
            raise ValueError("ERROR: length of fitmodules should be" + str(len(locationsList)+1) + "(Includes avg fit)")
        for i, (loc, module) in enumerate(zip(locationsList, fitModules)):
            fits[i], _ = ah.fitWithModule(module, key, tferAtomsVarAvg[i], guess=fitguess[i], getF_args=getFitterArgs[i])
        avgFit, _ = ah.fitWithModule(fitModules[-1], key, avgtferData, guess=fitguess[-1], getF_args=getFitterArgs[-1])
    return fits, avgFit

def getTransferAvgs(tferAtomsVarAvg, tferAtomsVarErrs, atomLocs1, repetitions, tferList, initAtoms, tferAtoms, initPopulation, getGenerationStats, genAvgs, genErrs):
    for i in range(len(atomLocs1)):
        tferList[i] = getTransferEvents(initAtoms[i], tferAtoms[i])
        tferAtomsVarAvg[i], tferAtomsVarErrs[i], initPopulation[i] = getTransferStats(tferList[i], repetitions)
        if getGenerationStats:
            genList = getGenerationEvents(initAtoms[i], tferAtoms[i])
            genAvgs[i], genErrs[i] = getGenStatistics(genList, repetitions)
        else:
            genAvgs[i], genErrs[i] = [None, None]

    # an atom average. The answer to the question: if you picked a random atom for a given variation, 
    # what's the mean [mean value] that you would find (averaging over the atoms), and what is the error on that mean?
    # weight the sum with initial percentage
    avgTferData = np.mean(tferAtomsVarAvg, 0)
    avgTferErr = np.sqrt(np.sum(np.array(tferAtomsVarErrs)**2,0)/len(atomLocs1))
    ### ###
    # averaged over all events, summed over all atoms. this data has very small error bars
    # becaues of the factor of 100 in the number of atoms.
    tferVarAvg, tferVarErr = [[],[]]
    tferVarList = [ah.groupEventsIntoVariations(atomsList, repetitions) for atomsList in tferList]
    allAtomsListByVar = [[z for atomList in tferVarList for z in atomList[i]] for i in range(len(tferVarList[0]))]
    for varData in allAtomsListByVar:
        p = np.mean(varData)
        tferVarAvg.append(p)
        tferVarErr.append(ah.jeffreyInterval(p, len(np.array(varData).flatten())))
    return avgTferData, avgTferErr, tferVarAvg, tferVarErr, genAvgs, genErrs

def standardTransferAnalysis( fileNumber, atomLocs1, atomLocs2, picsPerRep=2, thresholdOptions=None,
                              fitModules=[None], histSecondPeakGuess=None, varyingDim=None,
                              subtractEdgeCounts=True, initPic=0, tferPic=1, postSelectionCondition=None,
                              postSelectionConnected=False, getGenerationStats=False, rerng=False, tt=None,
                              rigorousThresholdFinding=True, tferThresholdSame=True, fitguess=[None], 
                              forceAnnotation=True, indvVariationThresholds=False, getFitterArgs=[None], 
                              **organizerArgs ):
    """
    "Survival" is a special case of transfer where the initial location and the transfer location are the same location.
    """
    if rerng:
        initPic, tferPic, picsPerRep = 1, 2, 3
    ( rawData, groupedData, atomLocs1, atomLocs2, keyName, repetitions, key, numOfPictures, avgPics, 
     basicInfoStr ) = organizeTransferData( fileNumber, atomLocs1, atomLocs2,  picsPerRep=picsPerRep, varyingDim=varyingDim,
                                            initPic=initPic, tferPic=tferPic, **organizerArgs )
    
    res = getTransferThresholds( atomLocs1, atomLocs2, rawData, groupedData, thresholdOptions, indvVariationThresholds, picsPerRep, initPic, tferPic,
                                 subtractEdgeCounts, rigorousThresholdFinding, tferThresholdSame )
    borders_init, borders_tfer, initThresholds, tferThresholds = res
    res = determineTransferAtomPrescence( atomLocs1, atomLocs2, groupedData, initPic, tferPic, picsPerRep, borders_init, borders_tfer, initThresholds, tferThresholds)
    initAtoms, tferAtoms, tferList, tferAtomsVarAvg, tferAtomsVarErrs, initPopulation, genAvgs, genErrs, initPicCounts, tferPicCounts = res
    initAtomImages, tferAtomImages = getTransferAtomImages( atomLocs1, atomLocs2, rawData, numOfPictures, picsPerRep, initPic,
                                                            tferPic, initAtoms, tferAtoms )
    initAtoms, tferAtoms, ensembleHits = ah.postSelectOnAssembly( initAtoms, tferAtoms, postSelectionCondition, connected=postSelectionConnected )
    res = getTransferAvgs(tferAtomsVarAvg, tferAtomsVarErrs, atomLocs1, repetitions, tferList, 
                          initAtoms, tferAtoms, initPopulation, getGenerationStats, genAvgs, genErrs)
    avgTferData, avgTferErr, tferVarAvg, tferVarErr, genAvgs, genErrs = res
    fits, avgFit = handleTransferFits(atomLocs1, fitModules, key, avgTferData, fitguess, getFitterArgs)    
    return (atomLocs1, atomLocs2, tferAtomsVarAvg, tferAtomsVarErrs, initPopulation, initPicCounts, keyName, key, repetitions, initThresholds, 
            fits, avgTferData, avgTferErr, avgFit, avgPics, None, atomLocs1, genAvgs, genErrs, tt, tferVarAvg, tferVarErr, initAtomImages, 
            tferAtomImages, tferPicCounts, tferThresholds, fitModules, tferThresholdSame, basicInfoStr, ensembleHits)

