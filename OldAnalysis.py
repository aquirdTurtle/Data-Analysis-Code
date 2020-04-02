"""
This is more or less a recycling bin for old code files which I think are no longer used but I want to be a bit slow about throwing them out. 
"""
def ballisticMotExpansion(t, sigma_y0, sigma_vy, sigma_I):
    """
    You can see a derivation of this in a different notebook for temperature calculations. I don't know why, but
    this function seems somewhat unstable as a fitting function. It doesn't always work very sensibly.

    :param t:
    :param sigma_y0:
    :param sigma_vy:
    :param sigma_I:
    :return:
    """
    return sigma_I*np.sqrt((sigma_y0**2 + sigma_vy**2 * t**2)/(sigma_y0**2+sigma_vy**2*t**2+sigma_I**2))

def simpleMotExpansion(t, sigma_y0, sigma_vy):
    """
    For some reason calculations involving this tend to be wildly off... I need to look into this more the next time
    I need to calculate a temperature.

    this simpler version ignores the size of the beam waist of the atoms.

    :param t:
    :param sigma_y0:
    :param sigma_vy:
    :return:

    """
    return sigma_y0 + sigma_vy * t

def beamWaistExpansion(z, w0, z0, wavelength):
    """
    assuming gaussian intensity profile of I~exp{-2z^2/w{z}^2}
    :param z:
    :param w0:
    :param z0:
    :param wavelength:
    :return:
    """
    return w0 * np.sqrt(1+(wavelength*(z-z0)/(np.pi*w0**2))**2)

def fillPlotDataDefaults(plotData):
    """

    :param plotData:
    :return:
    """
    if 'ax1' not in plotData:
        raise RuntimeError('plot data must have ax1 defined!')
    else:
        if 'data' not in plotData['ax1']:
            raise RuntimeError('ax1 of plot data must contain data!')
        if 'ylabel' not in plotData['ax1']:
            plotData['ax1']['ylabel'] = ''
        if 'legendLabels' not in plotData['ax1']:
            if plotData['ax1']['data'].ndim == 2:
                # create empty labels
                plotData['ax1']['legendLabels'] = ['' for _ in range(plotData['ax1']['data'].shape[0])]
            else:
                plotData['ax1']['legendLabels'] = ''
    if 'ax2' in plotData:
        if 'data' not in plotData['ax2']:
            raise RuntimeError('if ax2 is defined, it must contain data!')
        if 'ylabel' not in plotData['ax2']:
            plotData['ax2']['ylabel'] = ''
        if 'legendLabels' not in plotData['ax2']:
            if plotData['ax2']['data'].ndim == 2:
                # create empty labels
                plotData['ax2']['legendLabels'] = ['' for _ in range(plotData['ax2']['data'].shape[0])]
            else:
                plotData['ax2']['legendLabels'] = ''
    if 'xlabel' not in plotData:
        plotData['xlabel'] = ""
    if 'title' not in plotData:
        plotData['title'] = ""

        
def calcMotTemperature(times, sigmas):
    # print(sigmas[0])
    guess = [sigmas[0], 0.1]
    # guess = [0.001, 0.1]
    # in cm...?
    # sidemotWaist = .33 / (2*np.sqrt(2))
    sidemotWaist = 8 / (2*np.sqrt(2))
    # sidemotWaist^2/2 = 2 sigma_sidemot^2
    # different gaussian definitions
    sigma_I = sidemotWaist / 2
    # convert to m
    sigma_I /= 100
    # modify roughly for angle of sidemot
    # sigma_I /= np.cos(2*pi/3)
    sigma_I /= np.cos(mc.pi/4)
    sigma_I = 100
    fitVals, fitCovariances = opt.curve_fit(lambda x, a, b: ballisticMotExpansion(x, a, b, sigma_I), times, sigmas, p0=guess)
    simpleVals, simpleCovariances = opt.curve_fit(simpleMotExpansion, times, sigmas, p0=guess)
    temperature = mc.Rb87_M / mc.k_B * fitVals[1]**2
    tempFromSimple = mc.Rb87_M / mc.k_B * simpleVals[1]**2
    return temperature, tempFromSimple, fitVals, fitCovariances, simpleVals, simpleCovariances

def getLoadingData(picSeries, loc, whichPic, picsPerRep, manThreshold, binWidth, subtractEdges=True):
    borders = getAvgBorderCount(picSeries, whichPic, picsPerRep) if subtractEdges else np.zeros(len(picSeries))
    pic1Data = normalizeData(picSeries, loc, whichPic, picsPerRep, borders)
    bins, binnedData = getBinData(binWidth, pic1Data)
    guess1, guess2 = guessGaussianPeaks(bins, binnedData)
    guess = arr([max(binnedData), guess1, 30, max(binnedData)*0.75, guess2, 30])
    if manThreshold is None:
        gaussianFitVals = fitDoubleGaussian(bins, binnedData, guess)
        threshold, thresholdFid = calculateAtomThreshold(gaussianFitVals)
    elif manThreshold=='auto':
        gaussianFitVals = None
        threshold, thresholdFid = ((max(pic1Data) + min(pic1Data))/2.0, 0) 
    else:
        gaussianFitVals = None
        threshold, thresholdFid = (manThreshold, 0)
    atomCount = 0
    pic1Atom = []
    for point in pic1Data:
        if point > threshold:
            atomCount += 1
            pic1Atom.append(1)
        else:
            pic1Atom.append(0)
    return list(pic1Data), pic1Atom, threshold, thresholdFid, gaussianFitVals, bins, binnedData, atomCount


def normalizeData_2(data, atomLocation, picture, picturesPerExperiment, subtractBorders=True):
    """
    :param picturesPerExperiment:
    :param picture:
    :param subtractBorders:
    :param data: the array of pictures
    :param atomLocation: The location to analyze
    :return: The data at atomLocation with the background subtracted away (commented out at the moment).
    """
    allData = arr([])
    dimensions = data.shape
    if len(dimensions) == 4:
        rawData = data.reshape((data.shape[0] * data.shape[1], data.shape[2], data.shape[3]))
    else:
        rawData = data
    dimensions = rawData.shape
    for imageInc in range(0, dimensions[0]):
        if (imageInc + picturesPerExperiment - picture) % picturesPerExperiment == 0:
            pic = rawData[imageInc]
            border = 0
            if subtractBorders:
                normFactor = (2*len(pic[0][:])+2*len(pic[:][0]))
                border += (np.sum(pic[0][:]) + np.sum(pic[:][0]) + np.sum(pic[dimensions[1] - 1][:])
                           + np.sum([pic[i][dimensions[2] - 1] for i in range(len(pic))]))
                corners = (pic[0][0] + pic[dimensions[1]-1][dimensions[2] - 1] + pic[0][dimensions[2] - 1]
                           + pic[dimensions[1]-1][0])
                border -= corners
                # the number of pixels counted in the border
                border /= normFactor - 4
            allData = np.append(allData, pic[atomLocation[0]][atomLocation[1]] - border)
    return allData

def getTransferEvents_slow(pic1Atoms, pic2Atoms):
    """
    It returns a raw array that includes every survival data point, including points where the the atom doesn't get
    loaded at all.
    """
    # this will include entries for when there is no atom in the first picture.
    transferData = np.array([])
    transferData.astype(int)
    # flattens variations & locations?
    # if len(data.shape) == 4:
    #     data.reshape((data.shape[0] * data.shape[1], data.shape[2], data.shape[3]))

    # this doesn't take into account loss, since these experiments are feeding-back on loss.
    for atom1, atom2 in zip(pic1Atoms, pic2Atoms):
        if atom1 and atom2:
            # atom survived
            transferData = np.append(transferData, [1])
        elif atom1 and not atom2:
            # atom didn't survive
            transferData = np.append(transferData, [0])
        else:
            # no atom in the first place
            transferData = np.append(transferData, [-1])
    return transferData

def getTransferStats_fast(transferData, repetitionsPerVariation):
    # Take the previous data, which includes entries when there was no atom in the first picture, and convert it to
    # an array of just loaded and survived or loaded and died.
    transferAverages = np.array([])
    loadingProbability = np.array([])
    transferErrors = np.array([])
    if transferData.size < repetitionsPerVariation:
        repetitionsPerVariation = transferData.size
    for variationInc in range(0, int(transferData.size / repetitionsPerVariation)):
        transferList = np.array([])
        for repetitionInc in range(0, repetitionsPerVariation):
            if transferData[variationInc * repetitionsPerVariation + repetitionInc] != -1:
                transferList = np.append(transferList,
                                         transferData[variationInc * repetitionsPerVariation + repetitionInc])
        if transferList.size == 0:
            # catch the case where there's no relevant data, typically if laser becomes unlocked.
            transferErrors = np.append(transferErrors, [0])
            loadingProbability = np.append(loadingProbability, [0])
            transferAverages = np.append(transferAverages, [0])
        else:
            # normal case
            meanVal = np.average(transferList)
            transferErrors = np.append(transferErrors, jeffreyInterval(meanVal, len(transVarList)))
            #transferErrors = np.append(transferErrors, np.std(transferList)/np.sqrt(transferList.size))
            loadingProbability = np.append(loadingProbability, transferList.size / repetitionsPerVariation)
            transferAverages = np.append(transferAverages, meanVal)
    return transferAverages, transferErrors, loadingProbability

def handleFitting(fitType, key, data):
    """

    :param fitType:
    :param key:
    :param data:
    :return: returns fitInfo, fitType, where fitInfo = {'x': xFit, 'nom': fitNom, 'std': fitStd, 'center': centerIndex,
            'vals': fitValues, 'errs': fitErrs, 'cov': fitCovs}
    """
    xFit = (np.linspace(min(key), max(key), 1000) if len(key.shape) == 1 else np.linspace(min(transpose(key)[0]),
                                                                                          max(transpose(key)[0]), 1000))
    fitNom = fitStd = centerIndex = fitValues = fitErrs = fitCovs = None
    if fitType == 'Quadratic-Bump':
        try:
            # 3 parameters
            if len(key) < len(signature(fitFunc.quadraticBump).parameters) - 1:
                print('Not enough data points to constrain a fit!')
                raise RuntimeError()
            widthGuess = np.std(key) / 2
            centerGuess = key[list(data).index(max(data))]
            fitValues, fitCovs = opt.curve_fit(fitFunc.quadraticBump, key, data, p0=[max(data), -1/widthGuess, centerGuess])
            fitErrs = np.sqrt(np.diag(fitCovs))
            a, b, c = unc.correlated_values(fitValues, fitCovs)
            fitObject = fitFunc.quadraticBump(xFit, a, b, c)
            fitNom = unp.nominal_values(fitObject)
            fitStd = unp.std_devs(fitObject)
            centerIndex = 2
        except RuntimeError:
            warn('Data Fit Failed!')
            fitType = None
    elif fitType == 'Gaussian-Bump':
        try:
            if len(key) < len(signature(fitFunc.gaussian).parameters) - 1:
                print('Not enough data points to constrain a fit!')
                raise RuntimeError()
            widthGuess = np.std(key) / 2
            # Get all the atoms
            centerGuess = key[list(data).index(max(data))]
            fitValues, fitCovs = opt.curve_fit(fitFunc.gaussian, key, data, p0=[-0.95, centerGuess, widthGuess, 0.95])
            fitErrs = np.sqrt(np.diag(fitCovs))
            a, b, c, d = unc.correlated_values(fitValues, fitCovs)
            fitObject = fitFunc.uncGaussian(xFit, a, b, c, d)
            fitNom = unp.nominal_values(fitObject)
            fitStd = unp.std_devs(fitObject)
            centerIndex = 1
        except RuntimeError:
            warn('Data Fit Failed!')
            fitType=None
    elif fitType == 'Gaussian-Dip':
        try:
            if len(key) < len(signature(fitFunc.gaussian).parameters) - 1:
                print('Not enough data points to constrain a fit!')
                raise RuntimeError()

            widthGuess = np.std(key) / 2
            centerGuess = key[list(data).index(min(data))]
            fitValues, fitCovs = opt.curve_fit(fitFunc.gaussian, key, data, p0=[-0.95, centerGuess, widthGuess, 0.95])
            fitErrs = np.sqrt(np.diag(fitCovs))
            a, b, c, d = unc.correlated_values(fitValues, fitCovs)
            fitObject = fitFunc.uncGaussian(xFit, a, b, c, d)
            fitNom = unp.nominal_values(fitObject)
            fitStd = unp.std_devs(fitObject)
        except RuntimeError:
            warn('Data Fit Failed!')
            fitType = None
    elif fitType == 'Exponential-Decay':
        try:
            if len(key) < len(signature(fitFunc.exponentialDecay).parameters) - 1:
                print('Not enough data points to constrain a fit!')
                raise RuntimeError()

            decayConstantGuess = np.std(key)
            ampGuess = data[0]
            fitValues, fitCovs = opt.curve_fit(fitFunc.exponentialDecay, key, data, p0=[ampGuess, decayConstantGuess])
            fitErrs = np.sqrt(np.diag(fitCovs))
            a, b = unc.correlated_values(fitValues, fitCovs)
            fitObject = fitFunc.uncExponentialDecay(xFit, a,b)
            fitNom = unp.nominal_values(fitObject)
            fitStd = unp.std_devs(fitObject)
        except RuntimeError:
            warn('Data Fit Failed!')
            fitType = None
    elif fitType == 'Exponential-Saturation':
        try:
            if len(key) < len(signature(fitFunc.exponentialSaturation).parameters) - 1:
                print('Not enough data points to constrain a fit!')
                fitType=None
                raise RuntimeError()

            decayConstantGuess = (np.max(key)+np.min(key))/4
            ampGuess = data[-1]
            fitValues, fitCovs = opt.curve_fit(fitFunc.exponentialSaturation, key, data,
                                     p0=[-ampGuess, decayConstantGuess, ampGuess])
            fitErrs = np.sqrt(np.diag(fitCovs))
            a, b, c = unc.correlated_values(fitValues, fitCovs)
            fitObject = fitFunc.uncExponentialSaturation(xFit, a, b, c)
            fitNom = unp.nominal_values(fitObject)
            fitStd = unp.std_devs(fitObject)
        except RuntimeError:
            warn('Data Fit Failed!')
            fitType = None
    elif fitType == 'DecayingSine':
        try:
            if len(key) < len(signature(fitFunc.RabiFlop).parameters) - 1:
                print('Not enough data points to constrain a fit!')
                raise RuntimeError()
            ampGuess = 1
            phiGuess = 0
            OmegaGuess = np.pi / np.max(key)
            # Omega is the Rabi rate
            fitValues, fitCovs = opt.curve_fit(fitFunc.RabiFlop, key, data, p0=[ampGuess, OmegaGuess, phiGuess])
            fitErrs = np.sqrt(np.diag(fitCovs))
            a, b, c = unc.correlated_values(fitValues, fitCovs)
            fitObject = fitFunc.uncRabiFlop(xFit, a, b, c)
            fitNom = unp.nominal_values(fitObject)
            fitStd = unp.std_devs(fitObject)
        except RuntimeError:
            warn('Data Fit Failed!')
            fitType = None
    elif fitType is not None:
        warn("fitType Not Understood! Not Trying to Fit!")
        fitType = None
    # these are None if fit fail
    fitInfo = {'x': xFit, 'nom': fitNom, 'std': fitStd, 'center': centerIndex, 'vals': fitValues, 'errs': fitErrs,
               'cov': fitCovs}
    return fitInfo, fitType



def outputDataToMmaNotebook(fileNumber, survivalData, survivalErrs, captureArray, key):
    """

    :param fileNumber:
    :param survivalData:
    :param survivalErrs:
    :param captureArray:
    :param key:
    :return:
    """
    runNum = fileNumber
    try:
        f = open(dataAddress + '\\run' + str(runNum)+'_key.txt', "w")
        for item in key:
            f.write("%s\n" % item)
        f.close()
        f = open(dataAddress + '\\run' + str(runNum)+'_survival.txt', "w")
        for i in survivalData:
            tmp = ''
            for j in i:
                tmp += "%s," % round_sig(j, 9)
            tmp2 = (tmp.replace("[", "").replace("]", ""))[0:-1]+"\n"
            f.write(tmp2)
        f.close()
        f = open(dataAddress + '\\run' + str(runNum)+'_err.txt', "w")
        for i in survivalErrs:
            tmp = ''
            for j in i:
                tmp += "%s," % round_sig(j, 9)
            tmp2 = (tmp.replace("[", "").replace("]", ""))[0:-1]+"\n"
            f.write(tmp2)
        f.close()
        f = open(dataAddress + '\\run' + str(runNum) + '_loading.txt', "w")
        for i in captureArray:
            tmp = ''
            for j in i:
                tmp += "%s," % round_sig(j, 9)
            tmp2 = (tmp.replace("[", "").replace("]", ""))[0:-1]+"\n"
            f.write(tmp2)
        f.close()
    except:
        print("Error while outputting data to mathematica file.")

        
def getLabels(plotType):
    """

    :param plotType:
    :return:
    """
    if plotType == "Detuning":
        xlabelText = "Detuning (dac value)"
        titleText = "Detuning Scan"
    elif plotType == "Field":
        xlabelText = "Differential Field Change / 2 (dac value)"
        titleText = "Differential Magnetic Field Scan"
    elif plotType == "Time(ms)":
        xlabelText = "Time(ms)"
        titleText = "Time Scan"
    elif plotType == "Power":
        xlabelText = "Power (dac units)"
        titleText = "Power Scan"
    else:
        xlabelText = plotType
        titleText = ""
    return xlabelText, titleText


def assemblePlotData(rawData, dataMinusBg, dataMinusAverage, positions, waists, plottedData, scanType,
                     xLabel, plotTitle, location, waistFits=None, key=None):
    """
    take the data and organize it into the appropriate structures.
    """
    if key is None:
        key = []
    if waistFits is None:
        waistFits = []
    countData = {}
    countData['xlabel'], countData['title'] = getLabels(scanType)
    if not xLabel == "":
        countData['xlabel'] = xLabel
    if not plotTitle == "":
        countData['title'] = plotTitle
    ax1 = {}
    if location == (-1, -1):
        data = []
        legendLabels = []
        if "raw" in plottedData:
            data.append(integrateData(rawData))
            legendLabels.append(['Int(Raw Data)'])
        if "-bg" in plottedData:
            data.append(integrateData(dataMinusBg))
            legendLabels.append('Int(Data - Background)')
        if "-avg" in plottedData:
            data.append(integrateData(dataMinusAverage))
            legendLabels.append('Int(Data - Average)')
        ax1['data'] = arr(data)
        ax1['legendLabels'] = arr(legendLabels)
        ax1['ylabel'] = 'Camera Counts (Integrated over Picture)'

        ax2 = {}
        data = []
        legendLabels = []
        if "raw" in plottedData:
            data.append([np.max(pic, axis=(1, 0)) for pic in rawData])
            legendLabels.append(['Max Value in Raw Data'])
        if "-bg" in plottedData:
            data.append([np.max(pic, axis=(1, 0)) for pic in dataMinusBg])
            legendLabels.append('Max Value in Data Without Background')
        if "-avg" in plottedData:
            data.append([np.max(pic, axis=(1, 0)) for pic in dataMinusAverage])
            legendLabels.append('Max Value in Data Without Average')
        ax2['data'] = arr(data)
        ax2['legendLabels'] = arr(legendLabels)
        ax2['ylabel'] = 'Maximum count in picture'
        countData['ax2'] = ax2
    else:
        ax1['data'] = arr([rawData[:, location[0], location[1]]])
        ax1['ylabel'] = 'Camera Counts (single pixel)'
    countData['ax1'] = ax1
    ax1['data'] = {'data': waists}
    ax1['ylabel'] = r"Waist ($2\sigma$) (pixels)"
    if len(waistFits) == 0:
        ax1['legendLabels'] = [r"fit $w_x$", r"fit $w_y$"]
    else:
        print('...')
        ax1['legendLabels'] = [r"fit $w_x$", r"fit $w_y$", 'Fitted X: ' + str(waistFits[0]),
                               'Fitted Y: ' + str(waistFits[1])]
        fitYData = []
        xpts = np.linspace(min(key), max(key), 1000)
        for fitParams in waistFits:
            fitYData.append(beamWaistExpansion(xpts, fitParams[0], fitParams[1], 850e-9))
        ax1['fitYData'] = fitYData
        ax1['fitXData'] = [xpts, xpts]
    fitData = {'ax1': ax1}
    ax2 = {'data': positions, 'ylabel': "position (pixels)", 'legendLabels': ["Center x", "Center y"]}
    fitData['ax2'] = ax2
    fitData['title'] = "Fit Information"
    fitData['xlabel'] = countData['xlabel']

    return countData, fitData


        
def indvHists(dat, thresh, colors, extra=None, extraname=None, extra2=None, extra2Name=None, gaussianFitVals=None):
    f, axs = subplots(10,10, figsize=(25,18))
    for i, (d,t,c) in enumerate(zip(dat, thresh, colors[1:])):
        ax = axs[len(axs[0]) - i%len(axs[0]) - 1][int(i/len(axs))]
        heights, _, _ = ax.hist(d, 100, color=c, histtype='stepfilled')
        ax.axvline(t, color='w', ls=':')
        if gaussianFitVals is not None:
            g = gaussianFitVals[i]
            xpts = np.linspace(min(d), max(d), 1000)
            ax.plot(xpts, double_gaussian.f(xpts, *g), 'w')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlim(min(dat.flatten()), max(dat.flatten()))
        ax.set_ylim(0,max(heights))
        ax.grid(False)
        if extra is not None:
            txt = extraname + misc.round_sig_str( np.mean(extra[i])) if extraname is not None else misc.round_sig_str(np.mean(extra[i]))
            t = ax.text( 0.25, max(heights)-5, txt, fontsize=12 )
            t.set_bbox(dict(facecolor='k', alpha=0.3))
    f.subplots_adjust(wspace=0, hspace=0)
