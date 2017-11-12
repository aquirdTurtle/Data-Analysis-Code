__version__ = "1.0"

from numpy import array as arr

from Miscellaneous import getStats, transpose
from MarksFourierAnalysis import fft
import MarksConstants as consts

from matplotlib.pyplot import *
from scipy.optimize import curve_fit as fit

import FittingFunctions as fitFunc

from AnalysisHelpers import (loadDetailedKey, processImageData, loadHDF5, loadCompoundBasler, loadDataRay, fitPictures,
                             getAvgPic, getBinData, getEnsembleHits, getEnsembleStatistics, fitGaussianBeamWaist,
                             showPics, showBigPics, handleFitting, computeMotNumber, assemblePlotData, integrateData,
                             sliceMultidimensionalData, groupMultidimensionalData, getNetLoss, getNetLossStats,
                             getAtomInPictureStatistics, normalizeData, showPicComparisons, fitDoubleGaussian,
                             getSurvivalData, getSurvivalEvents, unpackAtomLocations, guessGaussianPeaks,
                             calculateAtomThreshold, outputDataToMmaNotebook, simpleMotExpansion, calcMotTemperature,
                             ballisticMotExpansion, getLoadingData, orderData, applyDataRange)


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
        num_lines = sum(1 for line in open(filename)) - 1
        allTimes = [[0] * num_lines for x in range(numTimes)]
        totalTime = [0] * num_lines
        names = ["" for x in range(numTimes)]
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


def analyzeNiawgWave(fileIndicator):
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
        fftInfoC1 = fft(chan1, tPts, normalize=True)
        fftInfoC2 = fft(chan2, tPts, normalize=True)
        # returns {'Freq': freqs, 'Amp': fieldFFT}
        return tPts, chan1, chan2, fftInfoC1, fftInfoC2


def plotNiawg(fileIndicator, points=300):
    """
    plots the first part of the niawg wave and the fourier transform of the total wave.
    """
    t, c1, c2, fftc1, fftc2 = analyzeNiawgWave(fileIndicator)
    plot(t[:points], c1[:points], 'o:', label='Vertical Channel', markersize=4, linewidth=1)
    plot(t[:points], c2[:points], 'o:', label='Horizontal Channel', markersize=4, linewidth=1)
    title('Niawg Output, first ' + str(points) + ' points.')
    ylabel('Relative Voltage (before NIAWG Gain)')
    xlabel('Time (s)')
    legend()
    figure()
    semilogy(fftc1['Freq'], abs(fftc1['Amp']) ** 2, 'o:', label='Vertical Channel', markersize=4, linewidth=1)
    semilogy(fftc2['Freq'], abs(fftc2['Amp']) ** 2, 'o:', label='Horizontal Channel', markersize=4, linewidth=1)
    title('Fourier Transform of NIAWG output')
    ylabel('Transform amplitude')
    xlabel('Frequency (Hz)')
    legend()
    # this is half the niawg sample rate. output is mirrored around x=0.
    xlim(0, 160e6)
    show()


def standardImages(data,
                   # Cosmetic Parameters
                   scanType="", xLabel="", yLabel="", title="", convertKey=False, showPictures=True, show=True,
                   allThePics=False, bigPics=False, colorMax=-1, individualColorBars=False, majorData='counts',
                   # Global Data Manipulation Options
                   loadType='andor', window=(0, 0, 0, 0), smartWindow=False, xMin=0, xMax=0, yMin=0, yMax=0,
                   accumulations=1, key=arr([]),
                   normData=np.array([]), zeroCorners=False, dataRange=(0, 0), initDataNum=0, manualAccumulation=False,
                   # Local Data Manipulation Options
                   plottedData=None, bg=arr([0]), location=(-1, -1), fitBeamWaist=False, fitPics=False,
                   cameraType='dataray', fitWidthGuess=80
                   ):
    """
    This function analyzes and plots fits pictures. It does not know anything about atoms,
    it just looks at pixels or integrates regions of the picture.

    ############
    ### Returns
    :return key, rawData, dataMinusBg, dataMinusAvg, avgPic, pictureFitParams
    ###############
    ### Parameters:
    ###
    Required parameters
    ###
    :param data: the number of the fits file and (by default) the number of the key file. The function knows where to look
            for the file with the corresponding name. Alternatively, an array of pictures to be used directly.

    ###
    optional parameters
    ### Cosmetic parameters
    * Change the way the data is diplayed, but not how it is analyzed.
    :param scanType: a string which characterizes what was scanned during the run. Depending on this string, axis names
        are assigned to the axis of the plots produced by this function. This parameter, if it is to do
        anything, must be one of a defined list of types, which can be looked up in the getLabels function.
    :param xLabel: Specify the xlabel that appears on the plots. Will override an xlabel specified by the scan type,
        but leave other scanType options un-marred.
    :param title: Specify the title that appears on the plots. Will override a title specified by the scan type,
        but leave other scanType options un-marred.
    :param convertKey: For frequency scans. If this is true, use the dacToFrequency conversion constants declared in the
        constants section of the base notebook to convert the key into frequency units instead of voltage
        units. This should probably eventually be expanded to handle other conversions as well.
    :param showPictures: if this is True, show all of the pictures taken during the experiment.
    :param show: if this is true, the function plots the integrated or point data. This would only be false if you just
            want the data arrays to do data visualization yourself.
    :param allThePics: If this is true, then when the pictures are shown, the raw, -bg, and -avg pictures will all be
                    plotted side-by-side for comparison. showPictures must also be true.
    :param bigPics: if this is true, then when the pictures are shown, instead of compressing the pictures to all fit in a
                reasonably sized figure, each picture gets its own figure. allThePics must be false, showPictures must
                be true.
    :param colorMax: by default the colorbars in the displayed pictures are chosen to range from the minimum value in the pic
                to the maximum value. If this is set, then instead you can specify an offset for the maximum (e.g. -5
                for 5th highest value in the picture). This is usefull, e.g. if cosmic rays are messing with your
                pictures.
    :param individualColorBars: by default, the pictures are all put on the same colorbar for comparison. If this is true,
                            each picture gets its own colorbar, which can make it easier to see features in each
                            individual picture, but generally makes comparison between pictures harder.
    :param majorData: (expects one of 'counts', 'fits') Identifies the data to appear in the big plot, the other gets shown
                    in the small plot.
    ### GLOBAL Data Manipulation Options
    * these manipulate all the data that is analyzed and plotted.
    :param loadType: (expects one of 'fits', 'dataray', 'basler') determines the loading function used to load the image
        data.
    :param window: (expects format (xMin, xMax, yMin, xMax)). Specifies a window of the picture to be analyzed in lieu of the
        entire picture. This command overrides the individual arguments (e.g. xMin) if both are present. The
        rest of the picture is completely discarded and not used for any data analysis, e.g. fitting. This
        defaults to cover the entire picture
    :param :xMin, xMax, yMin, and yMax: Specify a particular bound on the image to be analyzed; other parameters are left
        untouched. See above description of the window parameter. These default to cover
        the entire picture.
    :param :accumulations: If using accumulation mode on the Andor, set this parameter to the number of accumulations per
        image so that the data can be properly normalized for comparison to other pictures, backgrounds, etc.
        Defaults to 1.
    :param key: give the function a custom key. By default, the function looks for a key in the raw data file, but sometimes
        scans are done without the master code specifying the key.
    :param normData: This can be used to normalize each picture individually, e.g. normalizing for the size of the MOT during
        a picture. I don't think that this is actually working well yet, so check how this is beign used.
    :param zeroCorners: If this is true, average the four corners of the picture and subtract this average from every picture.
        This applies to the raw, -bg, and -avg data. for the latter two, the average is calculated and
        subtracted after the bg or avg is subtracted.
    :param dataRange: Specify which key values to analyze. By default, analyze all of them. (0,-1) will drop the last
        key value, etc.
    :param initDataNum: for basler scout analysis only right now. The offset of the data number so that the function can figure out
        all of the data numbers that correspond to this run.
    ### LOCAL Data Manipulation Parameters
    * These add extra analysis or change certain parts of the data analysis while leaving other parts intact.
    :param plottedData: (can include "raw", "-bg", and or "-avg") An array of strings which tells the function which data to
        plot. Can be used to plot multiple sets of data simultaneously, if needed. Defaults to raw. If only
        a single set is plotted, then that set is also shown in the pictures. In the case that multiple are
        shown, it's a bit funny right now.
    :param bg: A background picture, or a constant value, which is subtracted from each picture.
        defaults to subtracting nothing.
    :param location: Specify a specific pixel to be analyzed instead of the whole picture.
    :param fitPics: Attempt a 2D gaussian fit to each picture.
    :param fitBeamWaist: Don't think this works yet. The idea is that if gaussianFitPics is also true, then you can use this
        to fit a gaussian beam profile to the expanding gaussian fits. This could be useful, e.g. when
        calibrating the camera position.
    """
    if plottedData is None:
        plottedData = ["raw"]
    # Check for incompatible parameters.
    if bigPics and allThePics:
        raise ValueError("ERROR: can't use both bigPics and allThePics.")
    if fitBeamWaist and not fitPics:
        raise ValueError(
            "ERROR: Can't use fitBeamWaist and not fitPics! The fitBeamWaist attempts to use the fit values "
            "found by the gaussian fits.")
    if bigPics and not showPictures:
        raise ValueError("Can't show bigPics if not showPics!")
    if allThePics and not showPictures:
        raise ValueError("Can't show allThePics if not showPics!")

    # the key
    if key.size == 0:
        origKey, keyName = loadDetailedKey(data)
    else:
        origKey = key
    # this section could be expanded to handle different types of conversions.
    if convertKey:
        a = consts.opBeamDacToVoltageConversion
        key = [a[0] + x * a[1] + x ** 2 * a[2] + x ** 3 * a[3] for x in origKey]
    else:
        # both keys the same.
        key = origKey

    print("Key Values, in Time Order: ", key)
    if len(key) == 0:
        raise RuntimeError('key was empty!')

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
        elif loadType == 'scout':
            rawData = loadCompoundBasler(data, 'scout')
        elif loadType == 'ace':
            rawData = loadCompoundBasler(data, 'ace')
        elif loadType == 'dataray':
            rawData = [[] for x in range(data)]
            # assume user inputted an array of ints.
            for dataNum in data:
                rawData[keyInc][repInc] = loadDataRay(data)
        else:
            raise ValueError('Bad value for LoadType.')
    else:
        # assume the user inputted a picture or array of pictures.
        print('Assuming input is a picture or array of pictures.')
        rawData = data
    print('Data Loaded.')
    print('init shape: ' + str(rawData.shape))
    (key, rawData, dataMinusBg, dataMinusAvg,
     avgPic) = processImageData(key, rawData, bg, window, xMin, xMax, yMin, yMax, accumulations, dataRange, zeroCorners,
                                smartWindow, manuallyAccumulate=manualAccumulation)
    print('after process: ' + str(rawData.shape))
    if fitPics:
        # should improve this to handle multiple sets.
        if '-bg' in plottedData:
            print('fitting background-subtracted data.')
            pictureFitParams, pictureFitErrors = fitPictures(dataMinusBg, range(len(key)), guessSigma_x=fitWidthGuess,
                                                             guessSigma_y=fitWidthGuess)
        elif '-avg' in plottedData:
            print('fitting average-subtracted data.')
            pictureFitParams, pictureFitErrors = fitPictures(dataMinusAvg, range(len(key)), guessSigma_x=fitWidthGuess,
                                                             guessSigma_y=fitWidthGuess)
        else:
            print('fitting raw data.')
            pictureFitParams, pictureFitErrors = fitPictures(rawData, range(len(key)), guessSigma_x=fitWidthGuess,
                                                             guessSigma_y=fitWidthGuess)
    else:
        pictureFitParams, pictureFitErrors = np.zeros((len(key), 7)), np.zeros((len(key), 7))

    # convert to normal optics convention. the equation uses gaussian as exp(x^2/2sigma^2), I want the waist,
    # which is defined as exp(2x^2/waist^2):
    waists = 2 * arr([pictureFitParams[:, 3], pictureFitParams[:, 4]])
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
                                                  plottedData, scanType, xLabel, title, location,
                                                  waistFits=waistFitParams, key=key)
        except RuntimeError:
            print('gaussian waist fit failed!')
            # assemble the data structures for plotting.
            countData, fitData = assemblePlotData(rawData, dataMinusBg, dataMinusAvg, positions, waists,
                                                  plottedData, scanType, xLabel, title, location)
    else:
        # assemble the data structures for plotting.
        countData, fitData = assemblePlotData(rawData, dataMinusBg, dataMinusAvg, positions, waists,
                                              plottedData, scanType, xLabel, title, location)

    if majorData == 'counts':
        majorPlotData, minorPlotData = countData, fitData
    elif majorData == 'fits':
        minorPlotData, majorPlotData = countData, fitData
    else:
        raise ValueError("incorect 'majorData' argument")

    if show:
        plotPoints(key, majorPlotData, minorPlot=minorPlotData, picture=avgPic, picTitle="Average Picture")

    if show and showPictures:
        if allThePics:
            data = []
            for inc in range(len(rawData)):
                data.append([rawData[inc], dataWithoutBg[inc], dataWithoutAverage[inc]])
            showPicComparisons(arr(data), key, fitParameters=pictureFitParams, colorMax=colorMax)
        else:
            if "raw" in plottedData:
                if bigPics:
                    showBigPics(rawData, key, fitParameters=pictureFitParams, colorMax=colorMax,
                                individualColorBars=individualColorBars)
                else:
                    showPics(rawData, key, fitParameters=pictureFitParams, colorMax=colorMax,
                             individualColorBars=individualColorBars)
            if "-bg" in plottedData:
                if bigPics:
                    showBigPics(dataMinusBg, key, fitParameters=pictureFitParams, colorMax=colorMax,
                                individualColorBars=individualColorBars)
                else:
                    showPics(dataMinusBg, key, fitParameters=pictureFitParams, colorMax=colorMax,
                             individualColorBars=individualColorBars)
            if "-avg" in plottedData:
                if bigPics:
                    showBigPics(dataMinusAvg, key, fitParameters=pictureFitParams, colorMax=colorMax,
                                individualColorBars=individualColorBars)
                else:
                    showPics(dataMinusAvg, key, fitParameters=pictureFitParams, colorMax=colorMax,
                             individualColorBars=individualColorBars)
    return key, rawData, dataMinusBg, dataMinusAvg, avgPic, pictureFitParams


def plotMotTemperature(data, xLabel="", yLabel="", plotTitle="", window=(0, 0, 0, 0), xMin=0, xMax=0, yMin=0, yMax=0,
                       accumulations=1, key=arr([]), dataRange=(0, 0), fitWidthGuess=100):
    """
    Calculate the mot temperature, and plot the data that led to this.

    :param data:
    :param xLabel:
    :param yLabel:
    :param plotTitle:
    :param window:
    :param xMin:
    :param xMax:
    :param yMin:
    :param yMax:
    :param accumulations:
    :param key:
    :param dataRange:
    :param fitWidthGuess:
    :return:
    """
    (key, rawData, dataMinusBg, dataMinusAvg, avgPic,
     pictureFitParams) = standardImages(data, showPictures=False, show=False, scanType="Time(ms)", xLabel=xLabel,
                                        yLabel=yLabel, title=plotTitle, majorData='fits', loadType='scout',
                                        window=window, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax,
                                        accumulations=accumulations, key=key, dataRange=dataRange, fitPics=True,
                                        manualAccumulation=True, fitWidthGuess=fitWidthGuess)
    # convert to meters
    waists = 2 * consts.baslerScoutPixelSize * pictureFitParams[:, 3]
    # convert to s
    times = key / 1000
    temp, simpleTemp, fitVals, fitCov, simpleFitVals, simpleFitCov = calcMotTemperature(times, waists / 2)
    figure()
    plot(times, waists, 'o', label='Raw Data Waist')
    plot(times, 2 * ballisticMotExpansion(times, *fitVals, 100), label='balistic MOT expansion Fit Waist')
    plot(times, 2 * simpleMotExpansion(times, *simpleFitVals), label='simple MOT expansion Fit Waist ')
    title('Measured atom cloud size over time')
    xlabel('time (s)')
    ylabel('gaussian fit waist (m)')
    legend()

    showPics(rawData, key, fitParameters=pictureFitParams)

    print("PGC Temperture (full ballistic):", temp * 1e6, 'uK')
    # the simple balistic fit doesn't appear to work
    print("MOT Temperature (simple):", simpleTemp * 1e6, 'uK')  #


def plotMotNumber(dataSetNumber, motKey, exposureTime, window=(0, 0, 0, 0), cameraType='scout',
              showStandardImages=False, sidemotPower=2.05, diagonalPower=8, motRadius=8 * 8e-6,
              imagingLoss=0.8, detuning=10e6):
    """
    Calculate the MOT number and plot the data that resulted in the #.

    :param dataSetNumer: the number corresponding to the data set you want to analyze.
    :param motKey: the x-axis of the data. Should an array where each element corresponds to the time at which
        its corresponding picture was taken.
    :param exposureTime: the time the camera was exposed for to take the picture. Important in calculating the
        actual fluorescence rate, and this can change significantly from experiment to experiment.
    :param window: an optional specification of a subset region of the picture to analyze.
    :param cameraType: type of camera used to take the picture. Important for converting counts to photons.
    :param showStandardImages: show the images produced by the standardImages function, the actual images of the mot
        at different times.
    :param sidemotPower: measured sidemot power during the measurement.
    :param diagonalPower: measured diagonal power (of a single beam) during the experiment.
    :param motRadius: approximate radius of the MOT. Used to take into account the spread of intensity of the small
        side-mot beam across the finite area of the MOT. Default number comes from something like 8 pixels
        and 8um per pixel scaling, a calculation I don't have on me at the moment.
    :param imagingLoss: the loss in the imaging path due to filtering, imperfect reflections, etc.
    :param detuning: detuning of the mot beams during the imaging.
    """
    _, rawData, _, _, _, _ = standardImages(dataSetNumber, key=motKey, scanType="time (s)",
                                            window=window, loadType=cameraType, show=showStandardImages)
    intRawData = integrateData(rawData)
    try:
        # its always an exponential saturation fit for this data.
        popt, pcov = fit(fitFunc.exponentialSaturation, motKey, intRawData, p0=[np.min(intRawData) - np.max(intRawData),
                                                                                1 / 2, np.max(intRawData)])
    except RuntimeError:
        print('MOT # Fit failed!')
        # probably failed because of a bad guess. Show the user the guess fit to help them debug.
        popt = [np.min(intRawData) - np.max(intRawData), 1 / 2, np.max(intRawData)]
    figure()
    plot(motKey, intRawData, 'bo', label='data', color='b')
    xfitPts = np.linspace(min(motKey), max(motKey), 1000)
    plot(xfitPts, fitFunc.exponentialSaturation(xfitPts, *popt), 'b-', label='fit', color='r', linestyle=':')
    xlabel('loading time (s)')
    ylabel('integrated counts')
    title('Mot Fill Curve')
    print("integrated saturated counts subtracting background =", -popt[0])
    print("loading time 1/e =", popt[1], "s")
    motNum = computeMotNumber(sidemotPower, diagonalPower, motRadius, exposureTime, imagingLoss, -popt[0])
    print('MOT Number:', motNum)
    return motNum


def standardTransferAnalysis(fileNumber, atomLocs1, atomLocs2, accumulations=1, key=None, picsPerRep= 2,
                             manualThreshold=None, fitType=None, window=None, xMin=None, xMax=None, yMin=None,
                             yMax=None, dataRange=None, histSecondPeakGuess=None, keyOffset=0, sumAtoms=True,
                             outputMma=False, dimSlice=None, varyingDim=None, subtractEdgeCounts=True):
    """
    Standard data analysis package for looking at survival rates throughout an experiment.
    Returns key, survivalData, survivalErrors
    """
    atomLocs1 = unpackAtomLocations(atomLocs1)
    atomLocs2 = unpackAtomLocations(atomLocs2)
    # ### Load Fits File & Get Dimensions
    # Get the array from the fits file. That's all I care about.
    rawData, keyName, key, repetitions = loadHDF5(fileNumber)
    if len(key.shape) == 1:
        key -= keyOffset
    # window the images images.
    xMin, yMin, xMax, yMax = window if window is not None else [0, 0] + list(reversed(list(arr(rawData[0]).shape)))
    rawData = np.copy(arr(rawData[:,yMin:yMax, xMin:xMax]))
    # ## Initial Data Analysis
    # Group data into variations.
    numberOfPictures = int(rawData.shape[0])
    numberOfRuns = int(numberOfPictures / picsPerRep)
    numberOfVariations = int(numberOfPictures / (repetitions * picsPerRep))
    groupedDataRaw = rawData.reshape((numberOfVariations, repetitions * picsPerRep, rawData.shape[1],
                                      rawData.shape[2]))
    (key, slicedData, otherDimValues,
     varyingDim) = sliceMultidimensionalData(dimSlice, key, groupedDataRaw, varyingDim=varyingDim)
    slicedOrderedData, key, otherDimValues = orderData(slicedData, key, keyDim=varyingDim,
                                                       otherDimValues=otherDimValues)
    key, groupedData = applyDataRange(dataRange, slicedOrderedData, key)
    # gather some info about the run
    numberOfPictures = int(groupedData.shape[0] * groupedData.shape[1])
    numberOfRuns = int(numberOfPictures / picsPerRep)
    numberOfVariations = int(numberOfPictures / (repetitions * picsPerRep))
    print('Total # of Pictures:', numberOfPictures,'\n','Number of Variations:', numberOfVariations)
    print('Data Shape:', groupedData.shape)
    if not len(key) == numberOfVariations:
        raise RuntimeError("The Length of the key (" + str(len(key)) + ") doesn't match the data found ("
                           + str(numberOfVariations) + ").")
    avgPic = getAvgPic(groupedData)
    # initialize arrays
    (pic1Data, pic2Data, atomCounts, bins, binnedData, thresholds, survivalData, survivalErrs,
     loadingRate,) = arr([[None] * len(atomLocs1)] * 9)
    for i, (loc1, loc2) in enumerate(zip(atomLocs1, atomLocs2)):
        pic1Data[i] = normalizeData(groupedData, loc1, 0, picsPerRep, subtractBorders=subtractEdgeCounts)
        pic2Data[i] = normalizeData(groupedData, loc2, 1, picsPerRep, subtractBorders=subtractEdgeCounts)
        atomCounts[i] = arr([a for a in arr(list(zip(pic1Data[i], pic2Data[i]))).flatten()])
        bins[i], binnedData[i] = getBinData(10, pic1Data[i])
        guess1, guess2 = guessGaussianPeaks(bins[i], binnedData[i])
        guess = arr([max(binnedData[i]), guess1, 30, max(binnedData[i])*0.75,
                     200 if histSecondPeakGuess else histSecondPeakGuess, 10])
        gaussianFitVals = fitDoubleGaussian(bins[i], binnedData[i], guess)
        thresholds[i], thresholdFid = (((manualThreshold, 0) if manualThreshold is not None
                                       else calculateAtomThreshold(gaussianFitVals)))
        survivalList = getSurvivalEvents(atomCounts[i], thresholds[i], numberOfRuns)
        survivalData[i], survivalErrs[i], loadingRate[i] = getSurvivalData(survivalList, repetitions)
    (key, locationsList, survivalErrs, survivalData, loadingRate,
     otherDims) = groupMultidimensionalData(key, varyingDim, atomLocs1, survivalData, survivalErrs, loadingRate)
    # need to change for loop!
    fits = [None] * len(locationsList)
    for i, _ in enumerate(locationsList):
        xFit = (np.linspace(min(key), max(key), 1000) if len(key.shape) == 1
                else np.linspace(min(transpose(key)[0]), max(transpose(key)[0]), 1000))
        fits[i] = handleFitting(fitType,key,survivalData[i])
    pic1Data = arr(pic1Data.tolist())
    atomCounts = arr(atomCounts.tolist())
    # calculate average values
    avgSurvivalData, avgSurvivalErr, avgFit = [None]*3
    if sumAtoms:
        # weight the sum with loading percentage
        avgSurvivalData = sum(survivalData*loadingRate)/sum(loadingRate)
        avgSurvivalErr = np.sqrt(np.sum(survivalErrs**2))/len(atomLocs1)
        avgFit = handleFitting(fitType, key, avgSurvivalData)
    if outputMma:
        outputDataToMmaNotebook(fileNumber, runNum, survivalData, survivalErrs, loadingRate)
    return (atomLocs1, atomLocs2, atomCounts, survivalData, survivalErrs, loadingRate,
            pic1Data, keyName, key, repetitions, thresholds, fits,
            avgSurvivalData, avgSurvivalErr, avgFit, avgPic, otherDims, locationsList)


def standardLoadingAnalysis(fileNum, atomLocations, accumulations=1, picsPerExperiment = 1,
                            analyzeTogether=False, picture=0, manualThreshold=None,
                            loadingFitType=None, showTotalHist=True):
    atomLocations = unpackAtomLocations(atomLocations)
    rawData, keyName, key, repetitions = loadHDF5(fileNum)
    numOfPictures = rawData.shape[0]
    numOfVariations = int(numOfPictures / repetitions)
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
        newShape = (numOfVariations, repetitions,s[1],s[2])
    groupedData = rawData.reshape(newShape)
    groupedData, key, _ = orderData(groupedData, key)
    print('Data Shape:', groupedData.shape)
    loadingRateList, loadingRateErr, loadFits = [[[] for x in range(len(atomLocations))] for x in range(3)]
    print( 'Analyzing Variation... ', end='')
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
             atomCount[dataInc][i]) = getLoadingData( data, atomLoc, picture, picsPerExperiment, manualThreshold, 10)
            totalPic1AtomData.append(pic1Atom[dataInc][i])
            allAtomPicData.append(np.mean(pic1Atom[dataInc][i]))
            loadingRateList[i].append(np.mean(pic1Atom[dataInc][i]))
            loadingRateErr[i].append(np.std(pic1Atom[dataInc][i]) / np.sqrt(len(pic1Atom[dataInc][i])))
        allLoadingRate[dataInc] = np.mean(allAtomPicData)
        allLoadingErr[dataInc] = np.std(allAtomPicData) / np.sqrt(len(allAtomPicData))
    for i, load in enumerate(loadingRateList):
        loadFits[i] = handleFitting(loadingFitType,key,load)
    avgFits = handleFitting(loadingFitType, key, allLoadingRate)
    avgPic = getAvgPic(rawData)
    # get averages across all variations
    (pic1Data, pic1Atom, thresholds, thresholdFid, fitVals, bins, binData,
     atomCount) = arr([[None] * len(atomLocations)] * 8)
    for i, atomLoc in enumerate(atomLocations):
        (pic1Data[i], pic1Atom[i], thresholds[i], thresholdFid[i],
         fitVals[i], bins[i], binData[i], atomCount[i]) = getLoadingData(rawData, atomLoc, picture, picsPerExperiment,
                                                                         manualThreshold, 5)
        xFit = (np.linspace(min(key), max(key), 1000) if len(key.shape) == 1
                else np.linspace(min(transpose(key)[0]), max(transpose(key)[0]), 1000))
    pic1Data = arr(pic1Data.tolist())
    return (pic1Data, thresholds, avgPic, key, loadingRateErr, loadingRateList, allLoadingRate,
            allLoadingErr, loadFits, loadingFitType, keyName, totalPic1AtomData, rawData, showTotalHist, avgFits)


def standardAssemblyAnalysis(fileNumber, atomLocs1, pic1Num, atomLocs2=None, pic2Num=None, keyOffset=0,
                             window=None, picsPerRep=2, dataRange=None, histSecondPeakGuess=None,
                             manualThreshold=None, fitType=None, allAtomLocs1=None, allAtomLocs2=None):
    atomLocs1 = unpackAtomLocations(atomLocs1)
    atomLocs2 = (atomLocs1[:] if atomLocs2 is None else unpackAtomLocations(atomLocs2))
    allAtomLocs1 = (atomLocs1[:] if allAtomLocs1 is None else unpackAtomLocations(allAtomLocs1))
    allAtomLocs2 = (allAtomLocs1[:] if allAtomLocs2 is None else unpackAtomLocations(allAtomLocs2))
    # Get the array from the fits file.
    rawData, keyName, key, repetitions = loadHDF5(fileNumber)
    key -= keyOffset
    print("Key Values, in Time Order: ", key)
    # window the images images.
    xMin, yMin, xMax, yMax = window if window is not None else [0, 0] + list(reversed(list(arr(rawData[0]).shape)))
    rawData = np.copy(arr(rawData[:, yMin:yMax, xMin:xMax]))
    # gather some info about the run
    numberOfPictures = int(rawData.shape[0])
    numberOfRuns = int(numberOfPictures / picsPerRep)
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
        numberOfPictures = groupedData.shape[0] * groupedData.shape[1]
        numberOfRuns = int(numberOfPictures / picsPerRep)
        numberOfVariations = len(groupedData)
    else:
        groupedData = groupedDataRaw
    print('Data Shape:', groupedData.shape)
    avgPic = getAvgPic(groupedData)

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
        thresholds[i], thresholdFid = (
        (manualThreshold, 0) if manualThreshold is not None else calculateAtomThreshold(gaussianFitVals))
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
    fitData = handleFitting(fitType, key, ensembleStats['avg'])
    avgPic = getAvgPic(rawData)
    pic1Data = arr(pic1Data.tolist())
    pic2Data = arr(pic2Data.tolist())
    return (atomLocs1, atomLocs2, key, thresholds, pic1Data, pic2Data, fitData, ensembleStats, avgPic, atomCounts,
            keyName, indvStatistics, lossAvg, lossErr)
