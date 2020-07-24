import AnalysisHelpers as ah
import ExpFile as exp
import numpy as np
import copy
import PictureWindow as pw
import ThresholdOptions as to
import AnalysisOptions as ao

def organizeTransferData( fileNumber, analysisOpts, key=None, win=pw.PictureWindow(), dataRange=None, keyOffset=0, 
                          dimSlice=None, varyingDim=None, groupData=False, quiet=False, picsPerRep=2, repRange=None, 
                          keyConversion=None, softwareBinning=None, removePics=None, expFile_version=4, useBaseA=True):
    """
    Unpack inputs, properly shape the key, picture array, and run some initial checks on the consistency of the settings.
    """
    with exp.ExpFile(fileNumber, expFile_version=expFile_version, useBaseA=useBaseA) as f:
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
    rawData = np.array([win.window(pic) for pic in rawData])
    # Group data into variations.
    numberOfPictures = int(rawData.shape[0])
    if groupData:
        repetitions = int(numberOfPictures / picsPerRep)
    print("repsOTD:", repetitions)
    print("picsOTD:", picsPerRep)
    numberOfVariations = int(numberOfPictures / (repetitions * picsPerRep))
    key = ah.handleKeyModifications(hdf5Key, numberOfVariations, keyInput=key, keyOffset=keyOffset, groupData=groupData, keyConversion=keyConversion )
    print("numvarOTD:", numberOfVariations)
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
    print("numvarOTD2:", numberOfVariations)
    return rawData, groupedData, keyName, repetitions, key, numOfPictures, avgPics, basicInfoStr, analysisOpts

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
    transferErrors = np.zeros([2])
    if tferList.size < repsPerVar:
        # probably a single variation that has been sliced to cut out bad data.
        repsPerVar = tferList.size
    tferVarList = np.array([x for x in tferList if x != -1])
    if tferVarList.size == 0:
        # catch the case where there's no relevant data, typically if laser becomes unlocked.
        transferErrors = [0,0]
        loadingProbability = 0
        transferAverages = 0
    else:
        # normal case
        meanVal = np.average(tferVarList)
        transferErrors = ah.jeffreyInterval(meanVal, len(tferVarList))
        # old method
        loadingProbability = tferVarList.size / repsPerVar
        transferAverages = meanVal
    return transferAverages, transferErrors, loadingProbability


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
                    tferPixelCounts = ah.getAtomCountsData( variationData, picsPerRep, analysisOpts.tferPic, loc2, subtractEdges=opt.subtractEdgeCounts )
                    tferThresholds[i][j] = ah.getThresholds( tferPixelCounts, 5, opt )
            else:
                tferPixelCounts = ah.getAtomCountsData( rawData, picsPerRep, analysisOpts.tferPic, loc1, subtractEdges=opt.subtractEdgeCounts )
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
    #(initAtoms, tferAtoms) = [[[[] for _ in groupedData] for _ in atomLocs1] for _ in range(2)]
    numAtoms = len(analysisOpts.initLocs())
    (initAtoms, tferAtoms) = [[[[] for _ in range(numAtoms)] for _ in groupedData] for _ in range(2)]
    initPicCounts, tferPicCounts = [[[] for _ in range(numAtoms)] for _ in range(2)]
    for i, (loc1, loc2) in enumerate(zip(analysisOpts.initLocs(), analysisOpts.tferLocs())):
        for j, varData in enumerate(groupedData):
            initCounts = ah.normalizeData(varData, loc1, analysisOpts.initPic, picsPerRep, borders_init )
            tferCounts = ah.normalizeData(varData, loc2, analysisOpts.tferPic, picsPerRep, borders_tfer )
            Iatoms, Tatoms = ah.getSurvivalBoolData(initCounts, tferCounts, initThresholds[i][j].t, tferThresholds[i][j].t)
            initAtoms[j][i] = Iatoms 
            tferAtoms[j][i] = Tatoms
            tferPicCounts[i] += list(tferCounts)
            initPicCounts[i] += list(initCounts)
    return initAtoms, tferAtoms, np.array(initPicCounts), np.array(tferPicCounts)

def handleTransferFits(analysisOpts, fitModules, key, avgTferData, fitguess, getFitterArgs, tferAtomsVarAvg):
    numAtoms = len(analysisOpts.initLocs())
    fits = [None] * numAtoms
    avgFit = None
    if len(fitModules) == 1: 
        fitModules = [fitModules[0] for _ in range(numAtoms+1)]
    if fitModules[0] is not None:
        if type(fitModules) != list:
            raise TypeError("ERROR: fitModules must be a list of fit modules. If you want to use only one module for everything,"
                            " then set this to a single element list with the desired module.")
        if len(fitguess) == 1:
            fitguess = [fitguess[0] for _ in range(numAtoms+1) ]
        if len(getFitterArgs) == 1:
            getFitterArgs = [getFitterArgs[0] for _ in range(numAtoms+1) ]
        if len(fitModules) != numAtoms+1:
            raise ValueError("ERROR: length of fitmodules should be" + str(numAtoms+1) + "(Includes avg fit)")
        for i, (loc, module) in enumerate(zip(analysisOpts.initLocs(), fitModules)):
            fits[i], _ = ah.fitWithModule(module, key, tferAtomsVarAvg[i], guess=fitguess[i], getF_args=getFitterArgs[i])
        avgFit, _ = ah.fitWithModule(fitModules[-1], key, avgTferData, guess=fitguess[-1], getF_args=getFitterArgs[-1])
    return fits, avgFit, fitModules

def getTransferAvgs(analysisOpts, repetitions, initAtoms, tferAtoms, getGenerationStats):
    genAvgs, genErrs, initPopulation, tferList, tferAtomsVarAvg, tferAtomsVarErrs = [[[None for _ in initAtoms] for _ in range(len(analysisOpts.initLocs()))] for _ in range(6)]
    for atomInc in range(len(analysisOpts.initLocs())):
        for varInc in range(len(initAtoms)):
            tferList[atomInc][varInc] = getTransferEvents(initAtoms[varInc][atomInc], tferAtoms[varInc][atomInc])
            tferAtomsVarAvg[atomInc][varInc], tferAtomsVarErrs[atomInc][varInc], initPopulation[atomInc][varInc] = getTransferStats(tferList[atomInc][varInc], repetitions)
            if getGenerationStats:
                genList = getGenerationEvents(initAtoms[atomInc][varInc], tferAtoms[atomInc][varInc])
                genAvgs[atomInc][varInc], genErrs[atomInc][varInc] = getGenStatistics(genList, repetitions)
            else:
                genAvgs, genErrs = [None, None]
    # an atom average. The answer to the question: if you picked a random atom for a given variation, 
    # what's the mean [mean value] that you would find (averaging over the atoms), and what is the error on that mean?
    # weight the sum with initial percentage
    avgTferData = np.mean(tferAtomsVarAvg, 0)
    avgTferErr = np.sqrt(np.sum(np.array(tferAtomsVarErrs)**2,0) / len(analysisOpts.initLocs()))
    ### ###
    # averaged over all events, summed over all atoms. this data has very small error bars
    # becaues of the factor of 100 in the number of atoms.
    tferVarAvg, tferVarErr = [[],[]]
    #tferVarList = [ah.groupEventsIntoVariations(atomsList, repetitions) for atomsList in tferList]
    #allAtomsListByVar = [[z for atomList in tferList for z in atomList[i]] for i in range(len(tferList[0]))]
    allAtomsListByVar = [[] for _ in tferList[0]]
    for atomInc, atomList in enumerate(tferList):
        for varInc, varList in enumerate(atomList):
            for dp in varList:
                if dp != -1:
                    allAtomsListByVar[varInc].append(dp)    
    for varData in allAtomsListByVar:
        p = np.mean(varData)
        tferVarAvg.append(p)
        tferVarErr.append(ah.jeffreyInterval(p, len(np.array(varData).flatten())))
    return avgTferData, avgTferErr, tferVarAvg, tferVarErr, genAvgs, genErrs, tferAtomsVarAvg, tferAtomsVarErrs, initPopulation


def standardTransferAnalysis( fileNumber, analysisOpts, picsPerRep=2, fitModules=[None], varyingDim=None, getGenerationStats=False, 
                              fitguess=[None], forceAnnotation=True, tOptions=[to.ThresholdOptions()], getFitterArgs=[None], **organizerArgs ):
    """
    "Survival" is a special case of transfer where the initial location and the transfer location are the same location.
    """
    assert(type(analysisOpts) == ao.AnalysisOptions)
    ( rawData, groupedData, keyName, repetitions, key, numOfPictures, avgPics, 
      basicInfoStr, analysisOpts ) = organizeTransferData( fileNumber, analysisOpts, picsPerRep=picsPerRep, varyingDim=varyingDim, **organizerArgs )
    res = getTransferThresholds( analysisOpts, rawData, groupedData, picsPerRep, tOptions )
    borders_init, borders_tfer, initThresholds, tferThresholds = res
    
    res = determineTransferAtomPrescence( analysisOpts, groupedData, picsPerRep, borders_init, borders_tfer, initThresholds, tferThresholds)
    initAtoms, tferAtoms, initPicCounts, tferPicCounts = res
    
    initAtomImages, tferAtomImages = getTransferAtomImages( analysisOpts, groupedData, numOfPictures, picsPerRep, initAtoms, tferAtoms )    
    ensembleHits = [None for _ in initAtoms]
    for varInc in range(len(initAtoms)):
        initAtoms[varInc], tferAtoms[varInc], ensembleHits[varInc] = ah.postSelectOnAssembly(initAtoms[varInc], tferAtoms[varInc], analysisOpts )
    
    res = getTransferAvgs(analysisOpts, repetitions, initAtoms, tferAtoms, getGenerationStats)
    avgTferData, avgTferErr, tferVarAvg, tferVarErr, genAvgs, genErrs, tferAtomsVarAvg, tferAtomsVarErrs, initPopulation = res
    fits, avgFit, fitModules = handleTransferFits( analysisOpts, fitModules, key, avgTferData, fitguess, getFitterArgs, tferAtomsVarAvg )
    
    return (tferAtomsVarAvg, tferAtomsVarErrs, initPopulation, initPicCounts, keyName, key, repetitions, initThresholds, 
            fits, avgTferData, avgTferErr, avgFit, avgPics, genAvgs, genErrs, tferVarAvg, tferVarErr, initAtomImages, 
            tferAtomImages, tferPicCounts, tferThresholds, fitModules, basicInfoStr, ensembleHits, tOptions, analysisOpts)

