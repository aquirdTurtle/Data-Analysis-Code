__version__ = "1.1"

from MainAnalysis import standardLoadingAnalysis, analyzeNiawgWave, standardTransferAnalysis
from numpy import array as arr
from random import randint
from Miscellaneous import getColors, round_sig
from matplotlib.pyplot import *
import matplotlib as mpl
from scipy.optimize import curve_fit as fit
from AnalysisHelpers import (loadHDF5, loadDataRay, loadCompoundBasler, processSingleImage, orderData,
                             normalizeData, getBinData, getSurvivalData, getSurvivalEvents, fitDoubleGaussian,
                             guessGaussianPeaks, calculateAtomThreshold, getAvgPic, getEnsembleHits,
                             getEnsembleStatistics, handleFitting, getLoadingData, loadDetailedKey, processImageData,
                             fitPictures, fitGaussianBeamWaist, assemblePlotData, showPics, showBigPics,
                             showPicComparisons, ballisticMotExpansion, simpleMotExpansion, calcMotTemperature,
                             integrateData, computeMotNumber, getFitsDataFrame)
import MarksConstants as consts
import FittingFunctions as fitFunc


def plotNiawg(fileIndicator, points=300, plotTogether=True, plotVolts=False):
    """
    plots the first part of the niawg wave and the fourier transform of the total wave.
    """
    t, c1, c2, fftc1, fftc2 = analyzeNiawgWave(fileIndicator, ftPts=points)
    if plotVolts:
        figure(figsize=(20,10))
        title('Niawg Output, first ' + str(points) + ' points.')
        ylabel('Relative Voltage (before NIAWG Gain)')
        xlabel('Time (s)')
        plot(t[:points], c1[:points], 'o:', label='Vertical Channel', markersize=4, linewidth=1)
        legend()    
        if not plotTogether:
            figure(figsize=(20,10))
            title('Niawg Output, first ' + str(points) + ' points.')
            ylabel('Relative Voltage (before NIAWG Gain)')
            xlabel('Time (s)')
        plot(t[:points], c2[:points], 'o:', label='Horizontal Channel', markersize=4, linewidth=1)
        if not plotTogether:
            legend()
    figure(figsize=(20,10))
    title('Fourier Transform of NIAWG output')
    ylabel('Transform amplitude')
    xlabel('Frequency (Hz)')
    semilogy(fftc1['Freq'], abs(fftc1['Amp']) ** 2, 'o:', label='Vertical Channel', markersize=4, linewidth=1)
    legend()
    if not plotTogether:
        figure(figsize=(20,10))
        title('Fourier Transform of NIAWG output')
        ylabel('Transform amplitude')
        xlabel('Frequency (Hz)')
    semilogy(fftc2['Freq'], abs(fftc2['Amp']) ** 2, 'o:', label='Horizontal Channel', markersize=4, linewidth=1)
    if not plotTogether:
        legend()
    
    # this is half the niawg sample rate. output is mirrored around x=0.
    xlim(0, 160e6)
    show()


def standardImages(data,
                   # Cosmetic Parameters
                   scanType="", xLabel="", plotTitle="", convertKey=False, showPictures=True, showPlot=True,
                   allThePics=False, bigPics=False, colorMax=-1, individualColorBars=False, majorData='counts',
                   # Global Data Manipulation Options
                   loadType='andor', window=(0, 0, 0, 0), smartWindow=False, xMin=0, xMax=0, yMin=0, yMax=0,
                   accumulations=1, key=arr([]), zeroCorners=False, dataRange=(0, 0), manualAccumulation=False,
                   # Local Data Manipulation Options
                   plottedData=None, bg=arr([0]), location=(-1, -1), fitBeamWaist=False, fitPics=False,
                   cameraType='dataray', fitWidthGuess=80):
    """
    This function analyzes and plots fits pictures. It does not know anything about atoms,
    it just looks at pixels or integrates regions of the picture. It is commonly used, to look at background noise
    or the MOT.
    :return key, rawData, dataMinusBg, dataMinusAvg, avgPic, pictureFitParams
    ### Required parameters
    :param data: the number of the fits file and (by default) the number of the key file. The function knows where to
        look for the file with the corresponding name. Alternatively, an array of pictures to be used directly.
    ### Cosmetic parameters
    * Change the way the data is diplayed, but not how it is analyzed.
    :param scanType: a string which characterizes what was scanned during the run. Depending on this string, axis names
        are assigned to the axis of the plots produced by this function. This parameter, if it is to do
        anything, must be one of a defined list of types, which can be looked up in the getLabels function.
    :param xLabel: Specify the xlabel that appears on the plots. Will override an xlabel specified by the scan type,
        but leave other scanType options un-marred.
    :param plotTitle: Specify the title that appears on the plots. Will override a title specified by the scan type,
        but leave other scanType options un-marred.
    :param convertKey: For frequency scans. If this is true, use the dacToFrequency conversion constants declared in the
        constants section of the base notebook to convert the key into frequency units instead of voltage
        units. This should probably eventually be expanded to handle other conversions as well.
    :param showPictures: if this is True, show all of the pictures taken during the experiment.
    :param showPlot: if this is true, the function plots the integrated or point data. This would only be false if you
        justwant the data arrays to do data visualization yourself.
    :param allThePics: If this is true, then when the pictures are shown, the raw, -bg, and -avg pictures will all be
        plotted side-by-side for comparison. showPictures must also be true.
    :param bigPics: if this is true, then when the pictures are shown, instead of compressing the pictures to all fit
        in a reasonably sized figure, each picture gets its own figure. allThePics must be false, showPictures must
        be true.
    :param colorMax: by default the colorbars in the displayed pictures are chosen to range from the minimum value in
        the pic to the maximum value. If this is set, then instead you can specify an offset for the maximum (e.g. -5
        for 5th highest value in the picture). This is usefull, e.g. if cosmic rays are messing with your pictures.
    :param individualColorBars: by default, the pictures are all put on the same colorbar for comparison. If this is
        true, each picture gets its own colorbar, which can make it easier to see features in each individual picture,
        but generally makes comparison between pictures harder.
    :param majorData: (expects one of 'counts', 'fits') Identifies the data to appear in the big plot, the other gets
        shown in the small plot.
    ### GLOBAL Data Manipulation Options
    * these manipulate all the data that is analyzed and plotted.
    :param manualAccumulation: manually add together "accumulations" pictures together to form accumulated pictures for
        pictures for analysis.
    :param cameraType: determines scaling of pixels
    :param loadType: (expects one of 'fits', 'dataray', 'basler') determines the loading function used to load the image
        data.
    :param window: (expects format (xMin, xMax, yMin, xMax)). Specifies a window of the picture to be analyzed in lieu
        of the entire picture. This command overrides the individual arguments (e.g. xMin) if both are present. The
        rest of the picture is completely discarded and not used for any data analysis, e.g. fitting. This defaults to
        cover the entire picture
    :param xMin:
    :param yMax:
    :param yMin:
    :param xMax: Specify a particular bound on the image to be analyzed; other parameters are left
        untouched. See above description of the window parameter. These default to cover
        the entire picture.
    :param accumulations: If using accumulation mode on the Andor, set this parameter to the number of accumulations per
        image so that the data can be properly normalized for comparison to other pictures, backgrounds, etc.
        Defaults to 1.
    :param key: give the function a custom key. By default, the function looks for a key in the raw data file, but
        sometimes scans are done without the master code specifying the key.
    :param zeroCorners: If this is true, average the four corners of the picture and subtract this average from every
        picture. This applies to the raw, -bg, and -avg data. for the latter two, the average is calculated and
        subtracted after the bg or avg is subtracted.
    :param dataRange: Specify which key values to analyze. By default, analyze all of them. (0,-1) will drop the last
        key value, etc.
    :param smartWindow: Not properly implemented at the moment.
    ### LOCAL Data Manipulation Parameters
    * These add extra analysis or change certain parts of the data analysis while leaving other parts intact.
    :param fitWidthGuess: a manual guess for the threshold fit.
    :param plottedData: (can include "raw", "-bg", and or "-avg") An array of strings which tells the function which
        data to plot. Can be used to plot multiple sets of data simultaneously, if needed. Defaults to raw. If only
        a single set is plotted, then that set is also shown in the pictures. In the case that multiple are
        shown, it's a bit funny right now.
    :param bg: A background picture, or a constant value, which is subtracted from each picture.
        defaults to subtracting nothing.
    :param location: Specify a specific pixel to be analyzed instead of the whole picture.
    :param fitPics: Attempt a 2D gaussian fit to each picture.
    :param fitBeamWaist: Don't think this works yet. The idea is that if gaussianFitPics is also true, then you can
    use this        to fit a gaussian beam profile to the expanding gaussian fits. This could be useful, e.g. when
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
        a = consts.opBeamDacToVoltageConversionConstants
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
            raise ValueError('Loadtype of "dataray" has become deprecated and needs to be reimplemented.')
            # rawData = [[] for _ in range(data)]
            # assume user inputted an array of ints.
            # for _ in data:
            #    rawData[keyInc][repInc] = loadDataRay(data)
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

    if showPlot:
        plotPoints(key, majorPlotData, minorPlot=minorPlotData, picture=avgPic, picTitle="Average Picture")

    if showPlot and showPictures:
        if allThePics:
            data = []

            for inc in range(len(rawData)):
                data.append([rawData[inc], dataMinusBg[inc], dataMinusAvg[inc]])
            showPicComparisons(arr(data), key, fitParameters=pictureFitParams)
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


def plotMotTemperature(data, xLabel="", plotTitle="", window=(0, 0, 0, 0), xMin=0, xMax=0, yMin=0, yMax=0,
                       accumulations=1, key=arr([]), dataRange=(0, 0), fitWidthGuess=100):
    """
    Calculate the mot temperature, and plot the data that led to this.

    :param data:
    :param xLabel:
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
     pictureFitParams) = standardImages(data, showPictures=False, showPlot=False, scanType="Time(ms)", xLabel=xLabel,
                                        plotTitle=plotTitle, majorData='fits', loadType='scout',
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
    print("MOT Temperature (simple (don't trust?)):", simpleTemp * 1e6, 'uK')


def plotMotNumberAnalysis(dataSetNumber, motKey, exposureTime, window=(0, 0, 0, 0), cameraType='scout',
                          showStandardImages=False, sidemotPower=2.05, diagonalPower=8, motRadius=8 * 8e-6,
                          imagingLoss=0.8, detuning=10e6):
    """
    Calculate the MOT number and plot the data that resulted in the #.

    :param dataSetNumber: the number corresponding to the data set you want to analyze.
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
                                            window=window, loadType=cameraType, showPlot=showStandardImages)
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
    motNum = computeMotNumber(sidemotPower, diagonalPower, motRadius, exposureTime, imagingLoss, -popt[0],
                              detuning=detuning)
    print('MOT Number:', motNum)
    return motNum


def atomHist(key, atomLocs, pic1Data, bins, binData, fitVals, thresholds, avgPic, atomCount, variationNumber):
    """
    Makes a standard atom histogram-centric plot.

    :param key:
    :param atomLocs: list of coordinate pairs where atoms are. element 0 is row#, element 1 is column#
    :param pic1Data: list (for each location) of ordered lists of the counts on a pixel for each experiment
    :param bins: list (for each location) of the centers of the bins used for the histrogram.
    :param binData: list (for each location) of the accumulations each bin, whose center is stored in "bins" argument.
    :param fitVals: the fitted values of the 2D Gaussian fit of the fit bins and data
    :param thresholds: the found (or set) atom detection thresholds for each pixel.
    :param avgPic: the average of all the pics in this series.
    :param atomCount:
    :param variationNumber:
    """
    # Make colormap. really only need len(locs) + 1 rgbs, but adding an extra makes the spacing of the colors
    # on this colormap more sensible.
    colors = getColors(len(atomLocs))
    # Setup grid
    figure()
    grid1 = mpl.gridspec.GridSpec(12, 16)
    grid1.update(left=0.05, right=0.95, wspace=1.2, hspace=1000)
    gridLeft = mpl.gridspec.GridSpec(12, 16)
    gridLeft.update(left=0.001, right=0.95, hspace=1000)
    gridRight = mpl.gridspec.GridSpec(12, 16)
    gridRight.update(left=0.2, right=0.946, wspace=0, hspace=1000)
    # ### Main Plot
    mainPlt = subplot(grid1[:, :12])
    # plot the fit on top of the histogram
    fineXData = np.linspace(min(list([item for sublist in pic1Data for item in sublist])),
                            max(list([item for sublist in pic1Data for item in sublist])), 500)
    alphaVal = 1.0 / np.sqrt(len(atomLocs))
    for i, loc in enumerate(atomLocs):
        legendEntry = (str(loc) + ' T=' + str(round_sig(thresholds[i])) + ', L = '
                       + str(round_sig(atomCount[i] / len(pic1Data[0]))))
        mainPlt.bar(bins[i], binData[i], 10, color=colors[i],
                    label=legendEntry, alpha=alphaVal)
        mainPlt.plot(fineXData, fitFunc.doubleGaussian(fineXData, *fitVals[i], 0), color=colors[i], linestyle=':',
                     alpha=alphaVal)
        mainPlt.axvline(thresholds[i], color=colors[i], alpha=alphaVal)
    legend()
    if key is None:
        keyVal = 'Average'
    else:
        keyVal = str(key[variationNumber])
    mainPlt.set_title("Key Value = " + keyVal + "\nAvg Loading =" + str(np.mean(atomCount / len(pic1Data[0]))))
    mainPlt.set_ylabel("Occurrence Count")
    mainPlt.set_xlabel("Pixel Counts")
    # ### Pixel Count Data Over Time
    countsPlt = subplot(grid1[0:6, 12:16])
    for i in range(len(atomLocs)):
        countsPlt.plot(pic1Data[i], ".", markersize=1, color=colors[i])
        countsPlt.axhline(thresholds[i], color=colors[i])
    countsPlt.set_title("Count Data")
    countsPlt.set_ylabel("Counts on Pixel")
    countsPlt.set_xlabel("Picture Number")
    # ### Average Picture
    avgPlt = subplot(gridRight[7:12, 12:16])
    avgPlt.imshow(avgPic)
    avgPlt.set_title("Average Image")
    avgPlt.grid('off')
    # draw circles designating the analyzed locations
    for loc in atomLocs:
        circ = Circle([loc[1], loc[0]], 0.2, color='r')
        avgPlt.add_artist(circ)


def singleImage(data, accumulations=1, loadType='andor', bg=arr([0]), title='Single Picture', window=(0, 0, 0, 0),
                xMin=0, xMax=0, yMin=0, yMax=0, zeroCorners=False, smartWindow=False, findMax=False,
                manualAccumulation=False, maxColor=None, key=arr([])):
    """
    :return rawData, dataMinusBg

    ################
    ### Parameters:
    ###
    Required parameters
    :param data: the number of the fits file and (by default) the number of the key file. The function knows where to
        look for the file with the corresponding name. Alternatively, an array of pictures to be used directly.
    ### Cosmetic parameters
    :param title: The title of the image shown.
    :param show: if false, no picture is shown. This can be used if, e.g. you just want to extract the rawData or
        dataMinusBg data, or if you just want to see the coordinates of the maxima of the picture.
    ### LOCAL Data Manipulation Options
    these manipulate the data in just one part of the analysis. Oftentimes, they just add extra info on top of the
    normal output.
    :param findMax:=False
    :param bg:
    ### GLOBAL Data Manipulation Options
    # these manipulate all the data that is analyzed and plotted.
    :param loadType='andor'
    :param zeroCorners=False
    :param accumulations: the number of accumulated pictures in a given picture. The code normalizes the picture based
        on this data so that the numbers reported are "per single picture". If manuallyAccumulate=True, then this number
        is used to average pictures together.

    :param window: (expects format (xMin, xMax, yMin, xMax)). Specifies a window of the picture to be analyzed in lieu
        of the entire picture. This command overrides the individual arguments (e.g. xMin) if both are present. The rest
        of the picture is completely discarded and not used for any data analysis, e.g. fitting. This defaults to cover
        the entire picture

    ::param xMin, xMax, yMin, and yMax: Specify a particular bound on the image to be analyzed; other parameters are
        left untouched. See above description of the window parameter. These default to cover the entire picture.

    :param smartWindow: not fully implemented yet. Supposed to automatically select a window based on where the maxima
        of the picture is.

    :param key give the function a custom key. By default, the function looks for a key in the raw data file, but
        sometimes scans are done without the master code specifying the key.

    :param manualAccumulation: if true, the code will average "accumulated" pictures together. The number to average is
        the "accumulations" parameter.

    """
    # if integer or 1D array
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
        rawData = data

    rawData, dataMinusBg, xPts, yPts = processSingleImage(rawData, bg, window, xMin, xMax, yMin, yMax,
                                                          accumulations, zeroCorners, smartWindow,
                                                          manualAccumulation)

    if not bg == arr(0):
        if findMax:
            coords = np.unravel_index(np.argmax(rawData), rawData.shape)
            print('Coordinates of maximum:', xPts[coords[0]], yPts[coords[1]])
        if maxColor is None:
            maxColor = max(dataMinusBg.flatten())
        imshow(dataMinusBg, extent=(min(xPts), max(xPts), max(yPts), min(yPts)), vmax=maxColor)
    else:
        if findMax:
            coords = np.unravel_index(np.argmax(rawData), rawData.shape)
            print('Coordinates of maximum:', xPts[coords[1]], yPts[coords[0]])
            axvline(xPts[coords[1]], linewidth=0.5)
            axhline(yPts[coords[0]], linewidth=0.5)
        if maxColor is None:
            maxColor = max(rawData.flatten())
        imshow(rawData, extent=(min(xPts), max(xPts), max(yPts), min(yPts)), vmax=maxColor)
    colorbar()
    grid('off')
    return rawData, dataMinusBg


def Survival(fileNumber, atomLocs, **TransferArgs):
    """

    :param fileNumber:
    :param atomLocs:
    :param TransferArgs: See corresponding transfer function for valid TransferArgs.
    :return:
    """
    return Transfer(fileNumber, atomLocs, atomLocs, **TransferArgs)


def Transfer(fileNumber, atomLocs1, atomLocs2, show=True, plotTogether=True, plotLoadingRate=False, legendOption=None,
             fitModule=None, showFitDetails=False, **standardTransferArgs):
    """
    Standard data analysis package for looking at survival rates throughout an experiment.

    :return key, survivalData, survivalErrors
    """
    res = standardTransferAnalysis(fileNumber, atomLocs1, atomLocs2, fitModule=fitModule, 
                                   **standardTransferArgs)
    (atomLocs1, atomLocs2, atomCounts, survivalData, survivalErrs, loadingRate, pic1Data, keyName, key,
     repetitions, thresholds, fits, avgSurvivalData, avgSurvivalErr, avgFit, avgPics, otherDimValues,
     locsList, genAvgs, genErrs) = res
    if not show:
        return key, survivalData, survivalErrs, loadingRate
    if legendOption is None and len(atomLocs1) < 100:
        legendOption = True
    else:
        legendOption = False
    # get the colors for the plot.
    colors, colors2 = getColors(len(atomLocs1) + 1)
    figure()
    # Setup grid
    grid1 = mpl.gridspec.GridSpec(12, 16)
    grid1.update(left=0.05, right=0.95, wspace=1.2, hspace=1000)
    gridLeft = mpl.gridspec.GridSpec(12, 16)
    gridLeft.update(left=0.001, right=0.95, hspace=1000)
    gridRight = mpl.gridspec.GridSpec(12, 16)
    gridRight.update(left=0.2, right=0.946, wspace=0, hspace=1000)
    # Main Plot
    if atomLocs1 == atomLocs2:
        typeName = "Survival"
    else:
        typeName = "Transfer"
    mainPlot = subplot(grid1[:, :12])
    centers = []
    for i, (atomLoc, fit) in enumerate(zip(atomLocs1, fits)):
        if typeName == "Survival":
            leg = r"[%d,%d] " % (atomLocs1[i][0], atomLocs1[i][1])
        else:
            leg = r"[%d,%d]$\rightarrow$[%d,%d] " % (atomLocs1[i][0], atomLocs1[i][1],
                                                     atomLocs2[i][0], atomLocs2[i][1])
        if len(survivalData[i]) == 1:
            leg += (typeName + " % = " + str(round_sig(survivalData[i][0])) + "$\pm$ "
                    + str(round_sig(survivalErrs[i][0])))
        mainPlot.errorbar(key, survivalData[i], yerr=survivalErrs[i], color=colors[i], ls='',
                          marker='o', capsize=6, elinewidth=3, label=leg, alpha=0.3)
        if fitModule is not None:
            if fit['vals'] is None:
                print(loc, 'Fit Failed!')
                continue
            centerIndex = fitModule.center()
            if centerIndex is not None:
                centers.append(fit['vals'][centerIndex])
            mainPlot.plot(fit['x'], fit['nom'], color=colors[i], alpha = 0.5)
            #mainPlot.append(go.Scatter(x=fit['x'], y=fit['nom'], line={'color': color},
            #                           legendgroup=legend, showlegend=False, opacity=alphaVal))
            #if fit['std'] is not None:
                #mainPlot.plot(fit['x'], fit['nom'], color=)
                #mainPlot.append(go.Scatter(x=fit['x'], y=fit['nom'] + fit['std'],
                #                           opacity=alphaVal / 2, line={'color': color},
                #                           legendgroup=legend, showlegend=False, hoverinfo='none'))
                #mainPlot.append(fit['x'], y=fit['nom'] - fit['std'], label=legend)

    mainPlot.set_ylim({-0.02, 1.01})
    if not min(key) == max(key):
        mainPlot.set_xlim(left=min(key) - (max(key) - min(key)) / len(key), right=max(key)
                          + (max(key) - min(key)) / len(key))
    mainPlot.set_xticks(key)

    titletxt = keyName + " Atom " + typeName + " Scan"

    if len(survivalData[0]) == 1:
        titletxt = keyName + " Atom " + typeName + " Point. " + typeName + " % = \n"
        for atomData in survivalData:
            titletxt += ''  # str(round_sig(atomData,4) + ", "

    mainPlot.set_title(titletxt, fontsize=30)
    mainPlot.set_ylabel("Survival Probability", fontsize=20)
    mainPlot.set_xlabel(keyName, fontsize=20)
    mainPlot.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=8, prop={'size': 12})
    #legend()
    # Loading Plot
    loadingPlot = subplot(grid1[0:3, 12:16])
    for i, loc in enumerate(atomLocs1):
        loadingPlot.plot(key, loadingRate[i], ls='', marker='o', color=colors[i], alpha=0.3)
        loadingPlot.axhline(np.mean(loadingRate[i]), color=colors[i], alpha=0.3)
    loadingPlot.set_ylim({0, 1})
    if not min(key) == max(key):
        loadingPlot.set_xlim(left=min(key) - (max(key) - min(key)) / len(key), right=max(key) 
                             + (max(key) - min(key)) / len(key))
    loadingPlot.set_xlabel("Key Values")
    loadingPlot.set_ylabel("Capture %")
    loadingPlot.set_xticks(key)
    loadingPlot.set_title("Loading: Avg$ = " + str(round_sig(np.mean(loadingRate[0]))) + '$')
    for item in ([loadingPlot.title, loadingPlot.xaxis.label, loadingPlot.yaxis.label] +
                     loadingPlot.get_xticklabels() + loadingPlot.get_yticklabels()):
        item.set_fontsize(10)
        # ### Count Series Plot
    countPlot = subplot(gridRight[4:8, 12:15])
    for i, loc in enumerate(atomLocs1):
        countPlot.plot(pic1Data[i], color=colors[i], ls='', marker='.', markersize=1, alpha=0.3)
        #countPlot.plot(pic2Data[i], color=colors2[i], ls='', marker='.', markersize=1, alpha=0.8)
        countPlot.axhline(thresholds[i], color=colors[i], alpha=0.3)
    countPlot.set_xlabel("Picture #")
    countPlot.set_ylabel("Camera Signal")
    countPlot.set_title("Thresh.=" + str(round_sig(thresholds[i])), fontsize=10) #", Fid.="
                        #+ str(round_sig(thresholdFid)), )
    ticksForVis = countPlot.xaxis.get_major_ticks()
    ticksForVis[-1].label1.set_visible(False)
    for item in ([countPlot.title, countPlot.xaxis.label, countPlot.yaxis.label] +
                     countPlot.get_xticklabels() + countPlot.get_yticklabels()):
        item.set_fontsize(10)
    countPlot.set_xlim((0, len(pic1Data[0])))
    tickVals = np.linspace(0, len(pic1Data[0]), len(key) + 1)
    countPlot.set_xticks(tickVals)
    # Count Histogram Plot
    countHist = subplot(gridLeft[4:8, 15:16], sharey=countPlot)
    for i, atomLoc in enumerate(atomLocs1):
        countHist.hist(pic1Data[i], 50, color=colors[i], orientation='horizontal', alpha=0.3)
        #countHist.hist(pic2Data[i], 50, color=colors2[i], orientation='horizontal', alpha=0.3)
        countHist.axhline(thresholds[i], color=colors[i], alpha=1)
    for item in ([countHist.title, countHist.xaxis.label, countHist.yaxis.label] +
                     countHist.get_xticklabels() + countHist.get_yticklabels()):
        item.set_fontsize(10)
    ticks = countHist.get_xticklabels()
    for tickInc in range(len(ticks)):
        ticks[tickInc].set_rotation(-45)
    setp(countHist.get_yticklabels(), visible=False)
    # average image
    avgPlt = subplot(gridRight[9:12, 12:15])
    avgPlt.imshow(avgPics[0]);
    avgPlt.set_title("Average Image")
    avgPlt.grid('off')
    for loc in atomLocs1:
        circ = Circle((loc[1], loc[0]), 0.2, color='r')
        avgPlt.add_artist(circ)
    for loc in atomLocs2:
        circ = Circle((loc[1], loc[0]), 0.1, color='g')
        avgPlt.add_artist(circ)
    mainPlot.errorbar(key, avgSurvivalData, yerr=avgSurvivalErr, color="#FFFFFFFF", ls='',
             marker='o', capsize=6, elinewidth=3, label='Avg')
    if fitModule is not None and showFitDetails:
        mainPlot.plot(avgFit['x'], avgFit['nom'], color='#FFFFFFFF', ls=':')
        fits_df = getFitsDataFrame(fits, fitModule, avgFit)
        display(fits_df)

    return key, survivalData, survivalErrs, loadingRate, fits, avgFit, genAvgs, genErrs


def Loading(fileNum, atomLocations, plotLoadingRate=True, plotCounts=False, **StandardArgs):
    """
    Standard data analysis package for looking at loading rates throughout an experiment.

    return key, loadingRateList, loadingRateErr

    This routine is designed for analyzing experiments with only one picture per cycle. Typically
    These are loading exeriments, for example. There's no survival calculation.
    """
    (pic1Data, thresholds, avgPic, key, loadingRateErr, loadingRateList, allLoadingRate, allLoadingErr, loadFits,
     loadingFitType, keyName, totalPic1AtomData, rawData, showTotalHist, atomLocations,
     avgFits) = standardLoadingAnalysis(fileNum, atomLocations, **StandardArgs)
    pltColors, _ = getColors(len(atomLocations) + 1)
    if showTotalHist:
        pass
        # atomHist(key, atomLocations, pic1Data, bins, binData, fitVals, thresholds, avgPic, atomCount, None)
    if plotLoadingRate:
        # get colors
        figure()
        title('Loading Rate')
        for i, atomLoc in enumerate(atomLocations):
            errorbar(key, loadingRateList[i], yerr=loadingRateErr[i], marker='o', linestyle='', label=atomLoc,
                     color=pltColors[i])
        xlabel("key value")
        ylabel("loading rate")
        fitInfo, loadingFitType = handleFitting(loadingFitType, key, loadingRateList)
        if loadingFitType is not None:
            plot(fitInfo['x'], fitInfo['nom'], ':', label='Fit', linewidth=3)
            center = fitInfo['center']
            fill_between(fitInfo['x'], fitInfo['nom'] - fitInfo['std'], fitInfo['nom'] + fitInfo['std'], alpha=0.1,
                         label=r'$\pm\sigma$ band', color='b')
            axvspan(fitInfo['vals'][center ] - fitInfo['err'][center],
                    fitInfo['vals'][center ] + fitInfo['err'][center],
                    color='b', alpha=0.1)
            axvline(fitInfo['vals'][center ], color='b', linestyle='-.', alpha=0.5,
                    label='fit center $= ' + str(round_sig(fitInfo['vals'][center ])) + '$')
        keyRange = max(key) - min(key)
        xlim(min(key) - keyRange / (2 * len(key)), max(key)
             + keyRange / (2 * len(key)))
        ylim(0, 1)
        legend()
    if plotCounts:
        figure()
        title('Plot Counts')
        for i, atomLoc in enumerate(atomLocations):
            scatter(list(range(pic1Data[i].flatten().size)), pic1Data[i].flatten(), marker='.',
                    label=atomLoc, color=pltColors[i], s=1)
        legend()
    return key, loadingRateList, loadingRateErr


def Assembly(fileNumber, atomLocs1, pic1Num, atomLocs2=None, pic2Num=None, keyOffset=0, window=None,
             picsPerRep=2, dataRange=None, histSecondPeakGuess=None, manualThreshold=None, fitType=None):
    """
    This function checks the efficiency of generating a picture
    I.e. finding atoms at multiple locations at the same time.
    """
    if type(atomLocs1[0]) == int:
        # assume atom grid format.
        topLeftRow = atomLocs1[0]
        topLeftColumn = atomLocs1[1]
        spacing = atomLocs1[2]
        width = atomLocs1[3]
        height = atomLocs1[4]
        atomLocs1 = []
        for widthInc in range(width):
            for heightInc in range(height):
                atomLocs1.append([topLeftRow + spacing * heightInc, topLeftColumn + spacing * widthInc])

    # make it the right size
    if len(arr(atomLocs1).shape) == 1 and len(atomLocs1) is not 4:
        atomLocs1 = [atomLocs1]
    if atomLocs2 is None:
        atomLocs2 = atomLocs1
    # report

    # ### Load Fits File & Get Dimensions
    # Get the array from the fits file. That's all I care about.
    rawData, keyName, key, repetitions = loadHDF5(fileNumber)
    key -= keyOffset
    print("Key Values, in Time Order: ", key)
    # window the images images.
    if window is not None:
        xMin, xMax, yMin, yMax = window
    else:
        xMin, yMin, xMax, yMax = [0, 0] + list(reversed(list(arr(rawData[0]).shape)))
    rawData = np.copy(arr(rawData[:, yMin:yMax, xMin:xMax]))
    # gather some info about the run
    numberOfPictures = int(rawData.shape[0])
    numberOfRuns = int(numberOfPictures / picsPerRep)
    numberOfVariations = int(numberOfPictures / (repetitions * picsPerRep))
    print('Total # of Pictures:', numberOfPictures)
    print('Number of Variations:', numberOfVariations)
    if not len(key) == numberOfVariations:
        raise RuntimeError("The Length of the key doesn't match the data found.")
        ### Initial Data Analysis
    # Group data into variations.
    newShape = (numberOfVariations, repetitions * picsPerRep, rawData.shape[1], rawData.shape[2])
    groupedDataRaw = rawData.reshape(newShape)
    groupedDataRaw, key = orderData(groupedDataRaw, key)
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

    (pic1Data, pic2Data, atomCounts, bins, binnedData, thresholds,
     captureArray, fitNom, fitStd, center, pic1Atoms, pic2Atoms) = arr([[None] * len(atomLocs1)] * 12)
    for i, (loc1, loc2) in enumerate(zip(atomLocs1, atomLocs2)):
        # grab the first picture of each repetition
        pic1Data[i] = normalizeData(groupedData, loc1, 0, picsPerRep)
        pic2Data[i] = normalizeData(groupedData, loc2, 1, picsPerRep)
        atomCounts[i] = []
        for pic1, pic2 in zip(pic1Data[i], pic2Data[i]):
            atomCounts[i].append(pic1)
            atomCounts[i].append(pic2)
        atomCounts[i] = arr(atomCounts[i])
        ### Calculate Atom Threshold. Default binning for this is 10-count-wide bins
        bins[i], binnedData[i] = getBinData(10, pic1Data[i])
        guess1, guess2 = guessGaussianPeaks(bins[i], binnedData[i])
        if histSecondPeakGuess is None:
            guess = arr([max(binnedData[i]), guess1, 30, max(binnedData[i]) * 0.75, 200, 10])
        else:
            guess = arr([max(binnedData[i]), guess1, 30, max(binnedData[i]) * 0.75, histSecondPeakGuess, 10])
        gaussianFitVals = fitDoubleGaussian(bins[i], binnedData[i], guess)
        thresholds[i], thresholdFid = calculateAtomThreshold(gaussianFitVals)
        if manualThreshold is not None:
            thresholds[i] = manualThreshold
        # Calculate survival events
        atomCount = 0
        pic1Atoms[i] = []
        pic2Atoms[i] = []
        for point in pic1Data[i]:
            if point > thresholds[i]:
                pic1Atoms[i].append(True)
                atomCount += 1
            else:
                pic1Atoms[i].append(False)
        for point in pic2Data[i]:
            if point > thresholds[i]:
                pic2Atoms[i].append(True)
                atomCount += 1
            else:
                pic2Atoms[i].append(False)

    ensembleHits = getEnsembleHits(pic1Atoms) if pic1Num == 1 else getEnsembleHits(pic2Atoms)
    ensembleAvgs, ensembleErrs = getEnsembleStatistics(ensembleHits, repetitions)
    (fitNom, fitStd, center,
     fitValues, fitErrs, fitCovs) = handleFitting(fitType, key, ensembleAvgs)
    avgPic = getAvgPic(rawData)
    if not show:
        return key, survivalData, survivalErrs

    # #########################################
    #      Plotting
    # #########################################
    # get the colors for the plot.
    cmapRGB = mpl.cm.get_cmap('gist_rainbow', 100)
    colors, colors2 = [[None] * len(atomLocs1)] * 2
    for i, atomLoc in enumerate(atomLocs1):
        colors[i] = cmapRGB(randint(0, 100))[:-1]
        # the negative of the first color
        colors2[i] = tuple(arr((1, 1, 1)) - arr(colors[i]))
    figure()
    # Setup grid
    grid1 = mpl.gridspec.GridSpec(12, 16)
    grid1.update(left=0.05, right=0.95, wspace=1.2, hspace=1000)
    gridLeft = mpl.gridspec.GridSpec(12, 16)
    gridLeft.update(left=0.001, right=0.95, hspace=1000)
    gridRight = mpl.gridspec.GridSpec(12, 16)
    gridRight.update(left=0.2, right=0.946, wspace=0, hspace=1000)
    # Main Plot
    mainPlot = subplot(grid1[:, :12])
    mainPlot.errorbar(key, ensembleAvgs, yerr=ensembleErrs, color=colors[0], ls='',
                      marker='o', capsize=6, elinewidth=3, label='Raw Data ' + str(atomLoc))
    if fitType is None or fitNom is None:
        pass
    elif fitType == 'Exponential-Decay' or fitType == 'Exponential-Saturation':
        mainPlot.plot(xFit, fitNom[i], ':', color=colors[i], label=r'Fit; decay constant $= '
                                                                   + str(round_sig(fitValues[1])) + '\pm '
                                                                   + str(round_sig(fitErrs[1])) + '$', linewidth=3)
        mainPlot.fill_between(xFit, fitNom[i] - fitStd[i], fitNom[i] + fitStd[i], alpha=0.1, label=r'$\pm\sigma$ band',
                              color=colors[i])
    else:
        # assuming gaussian?
        xFit = np.linspace(min(key), max(key), 1000)
        mainPlot.plot(xFit, fitNom, ':', label='Fit', linewidth=3, color=colors[0])
        mainPlot.fill_between(xFit, fitNom - fitStd, fitNom + fitStd, alpha=0.1,
                              label='2-sigma band', color=colors[0])
        mainPlot.axvspan(fitValues[1] - np.sqrt(fitCovs[1, 1]), fitValues[1] + np.sqrt(fitCovs[1, 1]),
                         color=colors[0], alpha=0.1)
        mainPlot.axvline(fitValues[1], color=colors[0], linestyle='-.',
                         alpha=0.5, label='fit center $= ' + str(round_sig(fitValues[1], 4))
                                          + '\pm ' + str(round_sig(fitErrs[1], 4)) + '$')
    mainPlot.set_ylim({-0.02, 1.01})
    if not min(key) == max(key):
        mainPlot.set_xlim(left=min(key) - (max(key) - min(key)) / len(key), right=max(key)
                                                                                  + (max(key) - min(key)) / len(key))
    mainPlot.set_xticks(key)
    if atomLocs1 == atomLocs2:
        typeName = "Ensemble"
    else:
        typeName = "Ensemble"
    titletxt = keyName + " Atom " + typeName + " Scan"
    if len(ensembleAvgs) == 1:
        titletxt = keyName + " Atom " + typeName + " Point. " + typeName + " % = \n"
        for atomData in ensembleAvgs:
            titletxt += str(atomData) + ", "
    mainPlot.set_title(titletxt, fontsize=30)
    mainPlot.set_ylabel("Ensemble Probability", fontsize=20)
    mainPlot.set_xlabel(keyName, fontsize=20)
    mainPlot.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=4)
    legend()
    # Loading Plot
    """
    loadingPlot = subplot(grid1[0:3, 12:16])
    for i, loc in enumerate(atomLocs1):
        loadingPlot.plot(key, captureArray[i], ls='', marker='o', color=colors[i])
        loadingPlot.axhline(np.mean(captureArray[i]), color=colors[i])
    loadingPlot.set_ylim({0, 1})
    if not min(key) == max(key):
        loadingPlot.set_xlim(left=min(key) - (max(key) - min(key)) / len(key), right=max(key) 
                             + (max(key) - min(key)) / len(key))
    loadingPlot.set_xlabel("Key Values")
    loadingPlot.set_ylabel("Capture %")
    loadingPlot.set_xticks(key)
    loadingPlot.set_title("Loading: Avg$ = " + str(round_sig(np.mean(captureArray[0]))) + '$')
    for item in ([loadingPlot.title, loadingPlot.xaxis.label, loadingPlot.yaxis.label] +
             loadingPlot.get_xticklabels() + loadingPlot.get_yticklabels()):
        item.set_fontsize(10)
    """
    # ### Count Series Plot
    countPlot = subplot(gridRight[4:8, 12:15])
    for i, loc in enumerate(atomLocs1):
        countPlot.plot(pic1Data[i], color=colors[i], ls='', marker='.', markersize=1, alpha=1)
        countPlot.plot(pic2Data[i], color=colors2[i], ls='', marker='.', markersize=1, alpha=0.8)
        countPlot.axhline(thresholds[i], color='w')
    countPlot.set_xlabel("Picture #")
    countPlot.set_ylabel("Camera Signal")
    countPlot.set_title("Thresh.=" + str(round_sig(thresholds[i])) + ", Fid.=" + str(round_sig(thresholdFid)),
                        fontsize=10)
    ticksForVis = countPlot.xaxis.get_major_ticks()
    ticksForVis[-1].label1.set_visible(False)
    for item in ([countPlot.title, countPlot.xaxis.label, countPlot.yaxis.label] + countPlot.get_xticklabels()
                     + countPlot.get_yticklabels()):
        item.set_fontsize(10)
    countPlot.set_xlim((0, len(pic1Data[0])))
    tickVals = np.linspace(0, len(pic1Data[0]), len(key) + 1)
    countPlot.set_xticks(tickVals)
    # Count Histogram Plot
    countHist = subplot(gridLeft[4:8, 15:16], sharey=countPlot)
    for i, atomLoc in enumerate(atomLocs1):
        countHist.hist(pic1Data[i], 50, color=colors[i], orientation='horizontal', alpha=0.5)
        countHist.hist(pic2Data[i], 50, color=colors2[i], orientation='horizontal', alpha=0.3)
        countHist.axhline(thresholds[i], color='w')
    for item in ([countHist.title, countHist.xaxis.label, countHist.yaxis.label] + countHist.get_xticklabels()
                  + countHist.get_yticklabels()):
        item.set_fontsize(10)
    ticks = countHist.get_xticklabels()
    for tickInc in range(len(ticks)):
        ticks[tickInc].set_rotation(-45)
    setp(countHist.get_yticklabels(), visible=False)
    # average image
    avgPlt = subplot(gridRight[9:12, 12:15])
    avgPlt.imshow(avgPic);
    avgPlt.set_title("Average Image")
    avgPlt.grid('off')
    for loc in atomLocs1:
        circ = Circle((loc[1], loc[0]), 0.2, color='r')
        avgPlt.add_artist(circ)
    for loc in atomLocs2:
        circ = Circle((loc[1], loc[0]), 0.1, color='g')
        avgPlt.add_artist(circ)
    return key


def plotPoints(key, majorPlot, minorPlot=None, circleLoc=(-1, -1), picture=arr([[]]), picTitle=""):
    """
    This plotter plots a scatter plot on the main axis from majorPlot,
    a scatter on the minor axis using minorPlot, and a picture using picture.
    :param picture:
    :param circleLoc:
    :param picTitle: optional title to display above plot.
    :param key: this is used for the x values of both the major plot and the minor plot.
    :param majorPlot: a dictionary with the following elements containing info for the main plot of the figure (REQUIRED):
        - "ax1": A dictionary containing info for plotting on the first axis (main axis)
            with the following elements (REQUIRED):
            - "data": a 1 or 2-dimensional array with data to be plotted with key (REQUIRED)
            - "ylabel": label for axis 1 (on the left)
            - "legendLabels": an array with labels for each data set
        - "ax2": An optional dictionary containing info for plotting on the second axis, elements similar to ax1
        - "xlabel": label for the x axis
        - "title": title
        - "altKey": another key to plot on the other x-axis (above the plot).
    :param minorPlot: a dictionary with info for the minor plot. Same structure as majorPlotData, but optional.
    """
    if minorPlot is None:
        minorPlot = {}
    # Setup grid
    figure()
    grid1 = GridSpec(12, 16)
    grid1.update(left=0.05, right=0.95, wspace=1.2, hspace=1000)
    # ### Main Plot
    fillPlotDataDefaults(majorPlot)
    majorAx1 = subplot(grid1[:, :11])
    pltColors = ['r', 'c', 'g', '#FFFFFF', 'y', 'm', 'b']
    majorLineCount = 0
    lines = []
    if majorPlot['ax1']['data'].ndim == 1:
        lines += majorAx1.plot(key, majorPlot['ax1']['data'], 'o', label=majorPlot['ax1']['legendLabels'],
                               color=pltColors[majorLineCount])
        majorLineCount += 1
    else:
        for data in majorPlot['ax1']['data']:
            lines += majorAx1.plot(key, data, 'o', label=majorPlot['ax1']['legendLabels'][majorLineCount],
                                   color=pltColors[majorLineCount])
            majorLineCount += 1
    if 'fitYData' in majorPlot['ax1']:
        for fitCount in range(len(majorPlot['ax1']['fitYData'])):
            lines += majorAx1.plot(majorPlot['ax1']['fitXData'][fitCount], majorPlot['ax1']['fitYData'][fitCount],
                                   label=majorPlot['ax1']['legendLabels'][majorLineCount], color=pltColors[fitCount])
            majorLineCount += 1

    majorAx1.set_ylabel(majorPlot['ax1']['ylabel'])
    majorAx1.grid(which='both', color='#FFFFFF')

    if 'ax2' in majorPlot:
        majorAx2 = majorAx1.twinx()
        if majorPlot['ax1']['data'].ndim == 1:
            lines += majorAx2.plot(key, majorPlot['ax2']['data'], 'o', label=majorPlot['ax2']['legendLabels'],
                                   color=pltColors[majorLineCount])
            majorLineCount += 1
        else:
            count = 0
            for data in majorPlot['ax2']['data']:
                lines += majorAx2.plot(key, data, 'o', label=majorPlot['ax2']['legendLabels'][count],
                                       color=pltColors[majorLineCount])
                count += 1
                majorLineCount += 1
        majorAx2.set_ylabel(majorPlot['ax2']['ylabel'])
        majorAx2.spines['right'].set_color('m')
        majorAx2.spines['top'].set_color('m')
        majorAx2.yaxis.label.set_color('m')
        majorAx2.tick_params(axis='y', colors='m')
        majorAx2.grid(which='both', color='m')

    majorAx1.legend(lines, [l.get_label() for l in lines], loc='best', fancybox=True, framealpha=0.4)
    majorAx1.set_xlabel(majorPlot['xlabel'])
    majorAx1.set_title(majorPlot['title'])
    # todo, maybe modify this if I want to use it in the future.
    if 'majorAltKey' in majorPlot:
        ax2 = majorAx1.twiny()
        ax2.plot(majorPlot['majorAltKey'], majorPlot['ax1']['data'], 'o')
        ax2.grid(color='c')
        ax2.spines['top'].set_color('c')
        ax2.xaxis.label.set_color('c')
        ax2.tick_params(axis='x', colors='c')
        ax2.set_xlabel("DAC Values")
        ax2.set_xlim(reversed(ax2.get_xlim()))
    # Minor Data Plot
    lines = []
    if 'ax1' in minorPlot:
        fillPlotDataDefaults(minorPlot)
        minorAx1 = subplot(grid1[0:6, 12:16])
        minorLineCount = 0
        if minorPlot['ax1']['data'].ndim == 1:
            lines += minorAx1.plot(key, minorPlot['ax1']['data'], 'o', label=minorPlot['ax1']['legendLabels'],
                                   color=pltColors[minorLineCount])
            minorLineCount += 1
        else:
            count = 0
            for data in minorPlot['ax1']['data']:
                lines += minorAx1.plot(key, data, 'o', label=minorPlot['ax1']['legendLabels'][count],
                                       color=pltColors[minorLineCount])
                count += 1
                minorLineCount += 1
        minorAx1.set_ylabel(minorPlot['ax1']['ylabel'])

        if 'ax2' in minorPlot:
            minorAx2 = twinx(minorAx1)
            if minorPlot['ax1']['data'].ndim == 1:
                lines += minorAx2.plot(key, minorPlot['ax2']['data'], 'o', label=minorPlot['ax2']['legendLabels'],
                                       color=pltColors[minorLineCount])
                minorLineCount += 1
            else:
                count = 0
                for data in minorPlot['ax2']['data']:
                    lines += minorAx2.plot(key, data, 'o', label=minorPlot['ax2']['legendLabels'][count],
                                           color=pltColors[minorLineCount], alpha=0.4)
                    count += 1
                    minorLineCount += 1
            minorAx2.set_ylabel(minorPlot['ax2']['ylabel'])
            minorAx2.spines['right'].set_color('m')
            minorAx2.spines['top'].set_color('m')
            minorAx2.yaxis.label.set_color('m')
            minorAx2.tick_params(axis='y', colors='m')
            minorAx2.grid(color='m')
        minorAx1.legend(lines, [l.get_label() for l in lines], loc='best', fancybox=True, framealpha=0.4)
        minorAx1.set_xlabel(minorPlot['xlabel'])
        minorAx1.set_title(minorPlot['title'])
    # Image
    if picture.size != 0:
        image = subplot(grid1[7:12, 12:16])
        im = image.pcolormesh(picture)
        image.grid(0)
        image.axis('off')
        image.set_title(picTitle)
        colorbar(im, ax=image)
        if circleLoc != (-1, -1):
            circ = Circle(circleLoc, 0.2, color='r')
            image.add_artist(circ)
