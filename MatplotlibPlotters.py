__version__ = "1.1"

import time
from pandas import DataFrame
from MainAnalysis import standardPopulationAnalysis, analyzeNiawgWave, standardTransferAnalysis, standardAssemblyAnalysis, AnalyzeRearrangeMoves
from numpy import array as arr
from random import randint
from Miscellaneous import getColors, round_sig, round_sig_str, getMarkers
from matplotlib.pyplot import *
import matplotlib as mpl
from scipy.optimize import curve_fit as fit
from AnalysisHelpers import (loadDataRay, loadCompoundBasler, processSingleImage, orderData,
                             normalizeData, getBinData, getSurvivalData, getSurvivalEvents, fitDoubleGaussian,
                             guessGaussianPeaks, calculateAtomThreshold, getAvgPic, getEnsembleHits,
                             getEnsembleStatistics, handleFitting, getLoadingData, loadDetailedKey, processImageData,
                             fitPictures, fitGaussianBeamWaist, assemblePlotData, showPics, showBigPics,
                             showPicComparisons, ballisticMotExpansion, simpleMotExpansion, calcMotTemperature,
                             integrateData, computeMotNumber, getFitsDataFrame, genAvgDiscrepancyImage, getGridDims)
import MarksConstants as consts
import FittingFunctions as fitFunc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from TimeTracker import TimeTracker


def indvHists(dat, thresh, colors, extra=None):
    f, axs = subplots(10,10)
    for i, (d,t,c) in enumerate(zip(dat, thresh, colors[1:])):
        ax = axs[len(axs[0]) - i%len(axs[0]) - 1][int(i/len(axs))]
        ax.hist(d, 50, color=c, histtype='stepfilled')
        ax.axvline(t, color=c, ls=':')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(False)
        if extra is not None:
            t = ax.text(0.25, 10, 'L%:' + round_sig_str(np.mean(extra[i])), fontsize=8)
            t.set_bbox(dict(facecolor='k', alpha=0.5))

    f.subplots_adjust(wspace=0, hspace=0)


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
    avgPlt.grid(False)
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
    grid(False)
    return rawData, dataMinusBg


def Survival(fileNumber, atomLocs, **TransferArgs):
    """

    :param fileNumber:
    :param atomLocs:
    :param TransferArgs: See corresponding transfer function for valid TransferArgs.
    :return:
    """
    return Transfer(fileNumber, atomLocs, atomLocs, **TransferArgs)


def Transfer(fileNumber, atomLocs1_orig, atomLocs2_orig, show=True, plotLoadingRate=False, legendOption=None,
             fitModule=None, showFitDetails=False, showFitCenterPlot=False, showImagePlots=True, plotIndvHists=False, 
             timeit=False, **standardTransferArgs):
    """
    Standard data analysis package for looking at survival rates throughout an experiment.

    :return key, survivalData, survivalErrors
    """
    tt = TimeTracker()
    res = standardTransferAnalysis(fileNumber, atomLocs1_orig, atomLocs2_orig, fitModule=fitModule, tt=tt,
                                   **standardTransferArgs)
    tt.clock('After-Standard-Analysis')
    (atomLocs1, atomLocs2, atomCounts, survivalData, survivalErrs, loadingRate, pic1Data, keyName, key,
     repetitions, thresholds, fits, avgSurvivalData, avgSurvivalErr, avgFit, avgPics, otherDimValues,
     locsList, genAvgs, genErrs, gaussianFitVals, tt) = res
    if not show:
        return key, survivalData, survivalErrs, loadingRate
    if legendOption is None and len(atomLocs1) < 50:
        legendOption = True
    else:
        legendOption = False
    # get the colors for the plot.
    colors, colors2 = getColors(len(atomLocs1) + 1)
    f = figure()
    # Setup grid
    grid1 = mpl.gridspec.GridSpec(12, 16)
    grid1.update(left=0.05, right=0.95, wspace=1.2, hspace=1000)
    gridLeft = mpl.gridspec.GridSpec(12, 16)
    gridLeft.update(left=0.001, right=0.95, hspace=1000)
    gridRight = mpl.gridspec.GridSpec(12, 16)
    gridRight.update(left=0.1, right=0.95, wspace=0, hspace=1000)
    # Main Plot
    if atomLocs1 == atomLocs2:
        typeName = "S"
    else:
        typeName = "T"
    mainPlot = subplot(grid1[:, :11])
    centers = []
    longLegend = len(survivalData[0]) == 1
    markers = getMarkers()
    for i, (atomLoc, fit) in enumerate(zip(atomLocs1, fits)):
        if typeName == "S":
            leg = r"[%d,%d] " % (atomLocs1[i][0], atomLocs1[i][1])
        else:
            leg = r"[%d,%d]$\rightarrow$[%d,%d] " % (atomLocs1[i][0], atomLocs1[i][1],
                                                     atomLocs2[i][0], atomLocs2[i][1])
        if longLegend:
            leg += (typeName + " % = " + str(round_sig(survivalData[i][0])) + "$\pm$ "
                    + str(round_sig(survivalErrs[i][0])))
        mainPlot.errorbar(key, survivalData[i], yerr=survivalErrs[i], color=colors[i], ls='',
                          capsize=6, elinewidth=3, label=leg, alpha=0.3, marker=markers[i%len(markers)])
        if fitModule is not None and showFitDetails:
            if fit['vals'] is None:
                print(loc, 'Fit Failed!')
                continue
            centerIndex = fitModule.center()
            if centerIndex is not None:
                centers.append(fit['vals'][centerIndex])
            mainPlot.plot(fit['x'], fit['nom'], color=colors[i], alpha = 0.5)
    mainPlot.set_ylim({-0.02, 1.01})
    if not min(key) == max(key):
        r = max(key) - min(key)
        mainPlot.set_xlim(left=min(key) - r / len(key), right=max(key)+ r / len(key))
    mainPlot.set_xticks(key)
    rotateTicks(mainPlot)

    titletxt = keyName + " " + typeName + " Scan"
    if len(survivalData[0]) == 1:
        titletxt = keyName + " " + typeName + " Point.\n Avg " + typeName + "% = " + round_sig_str(np.mean(avgSurvivalData))        

    mainPlot.set_title(titletxt, fontsize=30)
    mainPlot.set_ylabel("S %", fontsize=20)
    mainPlot.set_xlabel(keyName, fontsize=20)
    mainPlot.xaxis.set_label_coords(0.95, -0.1)
    cols = 4 if longLegend else 10
    if legendOption:
        mainPlot.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=cols, prop={'size': 12})
    # Loading Plot
    loadingPlot = subplot(grid1[0:3, 11:16])
    for i, loc in enumerate(atomLocs1):
        loadingPlot.plot(key, loadingRate[i], ls='', marker='o', color=colors[i], alpha=0.3)
        loadingPlot.axhline(np.mean(loadingRate[i]), color=colors[i], alpha=0.3)
    loadingPlot.set_ylim({0, 1})
    if not min(key) == max(key):
        loadingPlot.set_xlim(left=min(key) - (max(key) - min(key)) / len(key), right=max(key) 
                             + (max(key) - min(key)) / len(key))
    loadingPlot.set_xlabel("Key Values")
    loadingPlot.set_ylabel("Loading %")
    loadingPlot.set_xticks(key[0:-1:2])
    rotateTicks(loadingPlot)

    loadingPlot.set_title("Loading: Avg$ = " + str(round_sig(np.mean(arr(loadingRate.tolist())))) + '$')
    for item in ([loadingPlot.title, loadingPlot.xaxis.label, loadingPlot.yaxis.label] +
                     loadingPlot.get_xticklabels() + loadingPlot.get_yticklabels()):
        item.set_fontsize(10)
    
    # ### Count Series Plot
    countPlot = subplot(gridRight[4:8, 11:15])    
    for i, loc in enumerate(atomLocs1):
        countPlot.plot(pic1Data[i], color=colors[i], ls='', marker='.', markersize=1, alpha=0.3)
        countPlot.axhline(thresholds[i], color=colors[i], alpha=0.3)

    countPlot.set_xlabel("Picture #")
    countPlot.set_ylabel("Camera Signal")
    countPlot.set_title("Thresh.=" + str(round_sig(thresholds[i])), fontsize=10)
    
    ticksForVis = countPlot.xaxis.get_major_ticks()
    ticksForVis[-1].label1.set_visible(False)
    for item in ([countPlot.title, countPlot.xaxis.label, countPlot.yaxis.label] +
                     countPlot.get_xticklabels() + countPlot.get_yticklabels()):
        item.set_fontsize(10)
    countPlot.set_xlim((0, len(pic1Data[0])))
    tickVals = np.linspace(0, len(pic1Data[0]), len(key) + 1)
    countPlot.set_xticks(tickVals[0:-1:2])
    rotateTicks(countPlot)
    
    # Count Histogram Plot
    countHist = subplot(gridLeft[4:8, 15:16], sharey=countPlot)
    for i, atomLoc in enumerate(atomLocs1):
        countHist.hist(pic1Data[i], 50, color=colors[i], orientation='horizontal', alpha=0.3, histtype='stepfilled')
        countHist.axhline(thresholds[i], color=colors[i], alpha=0.3)
    for item in ([countHist.title, countHist.xaxis.label, countHist.yaxis.label] +
                     countHist.get_xticklabels() + countHist.get_yticklabels()):
        item.set_fontsize(10)
    rotateTicks(countHist)

    setp(countHist.get_yticklabels(), visible=False)
    
    # average image
    avgPlt1 = subplot(gridRight[8:12, 11:13])
    
    avgPlt2 = subplot(gridRight[8:12, 13:15])
    for plt, dat, locs in zip([avgPlt1, avgPlt2], avgPics, [atomLocs1, atomLocs2]):
        plt.imshow(dat, origin='lower', cmap='Greys_r');
        plt.set_xticks([])
        plt.set_yticks([])
        plt.grid(False)
        for loc, c in zip(locs, colors):
            circ = Circle((loc[1], loc[0]), 0.2, color=c)
            plt.add_artist(circ)
    mainPlot.errorbar(key, avgSurvivalData, yerr=avgSurvivalErr, color="#FFFFFFFF", ls='',
             marker='o', capsize=6, elinewidth=3, label='Avg')
    if fitModule is not None:
        mainPlot.plot(avgFit['x'], avgFit['nom'], color='#FFFFFFFF', ls=':')
        fits_df = getFitsDataFrame(fits, fitModule, avgFit)
        display(fits_df)
    if fitModule is not None and showFitCenterPlot:
        figure()
        fitCenterPic, vmin, vmax = genAvgDiscrepancyImage(centers, avgPics[0].shape, atomLocs1)
        imshow(fitCenterPic, cmap=cm.get_cmap('RdBu'), vmin=vmin, vmax=vmax, origin='lower')
        title('Fit-Centers (white is average)')
        colorbar()
    tt.clock('After-Main-Plots')
    if showImagePlots:
        ims = []
        f, axs = subplots(1,4)

        ims.append(axs[0].imshow(avgPics[0], origin='lower'))
        axs[0].set_title('Avg 1st Pic')

        ims.append(axs[1].imshow(avgPics[1],origin='lower'))
        axs[1].set_title('Avg 2nd Pic')
        
        avgSurvivals = []
        for s in survivalData:
            avgSurvivals.append(np.mean(s))
        avgSurvivalPic, vmin, vmax = genAvgDiscrepancyImage(avgSurvivals, avgPics[0].shape, atomLocs1)
        ims.append(axs[2].imshow(avgSurvivalPic, cmap=cm.get_cmap('RdBu'), vmin=vmin, vmax=vmax, origin='lower'))
        axs[2].set_title('Avg Surv')
        
        avgLoads = []
        for l in loadingRate:
            avgLoads.append(np.mean(l))
        avgLoadPic, vmin, vmax = genAvgDiscrepancyImage(avgLoads, avgPics[0].shape, atomLocs1)
        ims.append(axs[3].imshow(avgLoadPic, cmap=cm.get_cmap('RdBu'), vmin=vmin, vmax=vmax, origin='lower'))
        axs[3].set_title('Avg Load')
        
        for ax, im in zip(axs, ims):
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.grid(False)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            f.colorbar(im, cax, orientation='vertical')
        tt.clock('After-Image-Plots')
    if plotIndvHists:
        indvHists(pic1Data, thresholds, colors, extra=avgSurvivals)
        tt.clock('After-Indv-Hists')
    if timeit:
        tt.display()
    avgPlt1.set_position([0.58,0,0.3,0.3])
    avgPlt2.set_position([0.73,0,0.3,0.3])
    return (key, survivalData, survivalErrs, loadingRate, fits, avgFit, genAvgs, genErrs, pic1Data, 
            gaussianFitVals, centers, thresholds, avgSurvivalPic)


def rotateTicks(plot):
    ticks = plot.get_xticklabels()
    for tickInc in range(len(ticks)):
        ticks[tickInc].set_rotation(-45)


def Loading(fileNum, atomLocations, **PopulationArgs):
    """
    A small wrapper, partially for the extra defaults in this case partially for consistency with old function definitions.
    """
    return Population(fileNum, atomLocations, 0, 1, **PopulationArgs)


def Population(fileNum, atomLocations, whichPic, picsPerRep, plotLoadingRate=True, plotCounts=False, legendOption=None, showImagePlots=True, 
               plotIndvHists=False, showFitDetails=False, showFitCenterPlot=True, show=True, **StandardArgs):
    """
    Standard data analysis package for looking at population %s throughout an experiment.

    return key, loadingRateList, loadingRateErr

    This routine is designed for analyzing experiments with only one picture per cycle. Typically
    These are loading exeriments, for example. There's no survival calculation.
    """
    (pic1Data, thresholds, avgPic, key, loadRateErr, loadRate, avgLoadRate, avgLoadErr, fits,
     fitModule, keyName, totalPic1AtomData, rawData, atomLocations, 
     avgFits, atomImages) = standardPopulationAnalysis(fileNum, atomLocations, whichPic, picsPerRep, **StandardArgs)
    colors, _ = getColors(len(atomLocations) + 1)
    if not show:
        return key, loadRate, loadRateErr, pic1Data, atomImages, thresholds
    if legendOption is None and len(atomLocations) < 50:
        legendOption = True
    else:
        legendOption = False
    # get the colors for the plot.
    markers = ['o','^','<','>','v']
    f = figure()
    # Setup grid
    grid1 = mpl.gridspec.GridSpec(12, 16)
    grid1.update(left=0.05, right=0.95, wspace=1.2, hspace=1000)
    gridLeft = mpl.gridspec.GridSpec(12, 16)
    gridLeft.update(left=0.001, right=0.95, hspace=1000)
    gridRight = mpl.gridspec.GridSpec(12, 16)
    gridRight.update(left=0.2, right=0.946, wspace=0, hspace=1000)
    # Main Plot
    typeName = "L"
    mainPlot = subplot(grid1[:, :12])
    centers = []
    longLegend = len(loadRate[0]) == 1
    for i, (atomLoc, fit) in enumerate(zip(atomLocations, fits)):
        leg = r"[%d,%d] " % (atomLoc[0], atomLoc[1])
        if longLegend:
            leg += (typeName + " % = " + str(round_sig(loadRate[i][0])) + "$\pm$ "
                    + str(round_sig(loadRateErr[i][0])))
        mainPlot.errorbar(key, loadRate[i], yerr=loadRateErr[i], color=colors[i], ls='',
                          capsize=6, elinewidth=3, label=leg, alpha=0.2, marker=markers[i%len(markers)])
        if fitModule is not None:
            if fit == [] or fit['vals'] is None:
                continue
            centerIndex = fitModule.center()
            if centerIndex is not None:
                centers.append(fit['vals'][centerIndex])
            mainPlot.plot(fit['x'], fit['nom'], color=colors[i], alpha = 0.5)
    if fitModule is not None:
        if avgFits['vals'] is None:
            print('Avg Fit Failed!')
        else:
            centerIndex = fitModule.center()
            mainPlot.plot(avgFits['x'], avgFits['nom'], color='w', alpha = 1)

    mainPlot.set_ylim({-0.02, 1.01})
    if not min(key) == max(key):
        mainPlot.set_xlim(left=min(key) - (max(key) - min(key)) / len(key), right=max(key)
                          + (max(key) - min(key)) / len(key))
    mainPlot.set_xticks(key)
    rotateTicks(mainPlot)

    titletxt = keyName + " Atom " + typeName + " Scan"
    if len(loadRate[0]) == 1:
        titletxt = keyName + " Atom " + typeName + " Point.\n Avg " + typeName + "% = " + round_sig_str(np.mean(avgLoadRate))
    
    mainPlot.set_title(titletxt, fontsize=30)
    mainPlot.set_ylabel("S %", fontsize=20)
    mainPlot.set_xlabel(keyName, fontsize=20)
    if legendOption == True:
        cols = 4 if longLegend else 10
        mainPlot.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=cols, prop={'size': 12})
    # Loading Plot
    loadingPlot = subplot(grid1[0:3, 12:16])
    for i, loc in enumerate(atomLocations):
        loadingPlot.plot(key, loadRate[i], ls='', marker='o', color=colors[i], alpha=0.3)
        loadingPlot.axhline(np.mean(loadRate[i]), color=colors[i], alpha=0.3)
    loadingPlot.set_ylim({0, 1})
    if not min(key) == max(key):
        loadingPlot.set_xlim(left=min(key) - (max(key) - min(key)) / len(key), right=max(key) 
                             + (max(key) - min(key)) / len(key))
    loadingPlot.set_xlabel("Key Values")
    loadingPlot.set_ylabel("Loading %")
    loadingPlot.set_xticks(key)
    loadingPlot.set_title("Loading: Avg$ = " + str(round_sig(np.mean(arr(loadRate)))) + '$')
    for item in ([loadingPlot.title, loadingPlot.xaxis.label, loadingPlot.yaxis.label] +
                     loadingPlot.get_xticklabels() + loadingPlot.get_yticklabels()):
        item.set_fontsize(10)
        # ### Count Series Plot
    countPlot = subplot(gridRight[4:8, 12:15])    
    for i, loc in enumerate(atomLocations):
        countPlot.plot(pic1Data[i], color=colors[i], ls='', marker='.', markersize=1, alpha=0.3)
        # countPlot.plot(pic2Data[i], color=colors2[i], ls='', marker='.', markersize=1, alpha=0.8)
        countPlot.axhline(thresholds[i], color=colors[i], alpha=0.3)


    countPlot.set_xlabel("Picture #")
    countPlot.set_ylabel("Camera Signal")
    countPlot.set_title("Thresh.=" + str(round_sig(thresholds[i])), fontsize=10) #", Fid.="
                        # + str(round_sig(thresholdFid)), )
    ticksForVis = countPlot.xaxis.get_major_ticks()
    ticksForVis[-1].label1.set_visible(False)
    for item in ([countPlot.title, countPlot.xaxis.label, countPlot.yaxis.label] +
                     countPlot.get_xticklabels() + countPlot.get_yticklabels()):
        item.set_fontsize(10)
    countPlot.set_xlim((0, len(pic1Data[0])))
    tickVals = np.linspace(0, len(pic1Data[0]), len(key) + 1)
    countPlot.set_xticks(tickVals[0:-1:2])
    # Count Histogram Plot
    countHist = subplot(gridLeft[4:8, 15:16], sharey=countPlot)
    for i, atomLoc in enumerate(atomLocations):
        countHist.hist(pic1Data[i], 50, color=colors[i], orientation='horizontal', alpha=0.3, histtype='stepfilled')
        countHist.axhline(thresholds[i], color=colors[i], alpha=0.3)
    for item in ([countHist.title, countHist.xaxis.label, countHist.yaxis.label] +
                     countHist.get_xticklabels() + countHist.get_yticklabels()):
        item.set_fontsize(10)
    rotateTicks(countHist)
    setp(countHist.get_yticklabels(), visible=False)
    # average image
    avgPlt = subplot(gridRight[8:12, 12:15])
    avgPlt.imshow(avgPic, origin='lower');
    avgPlt.set_xticks([])
    avgPlt.set_yticks([])
    avgPlt.grid(False)
    for loc in atomLocations:
        circ = Circle((loc[1], loc[0]), 0.2, color='r')
        avgPlt.add_artist(circ)
    mainPlot.errorbar(key, avgLoadRate, yerr=avgLoadErr, color="#FFFFFFFF", ls='',
             marker='o', capsize=6, elinewidth=3, label='Avg')
    if fitModule is not None and showFitDetails:
        mainPlot.plot(avgFit['x'], avgFit['nom'], color='#FFFFFFFF', ls=':')
        fits_df = getFitsDataFrame(fits, fitModule, avgFit)
        display(fits_df)
    elif fitModule is not None:
        for val, name in zip(avgFits['vals'], fitModule.args()):
            print(name, val)
    if fitModule is not None and showFitCenterPlot and fits[0] != []:
        figure()
        fitCenterPic, vmin, vmax = genAvgDiscrepancyImage(centers, avgPic.shape, atomLocations)
        imshow(fitCenterPic, cmap=cm.get_cmap('RdBu'), vmin=vmin, vmax=vmax, origin='lower')
        title('Fit-Centers (white is average)')
        colorbar()
    if showImagePlots:
        ims = []
        f, axs = subplots(1,2)

        ims.append(axs[0].imshow(avgPic, origin='lower'))
        axs[0].set_title('Avg 1st Pic')

        avgLoads = []
        for l in loadRate:
            avgLoads.append(np.mean(l))
        avgLoadPic, vmin, vmax = genAvgDiscrepancyImage(avgLoads, avgPic.shape, atomLocations)
        ims.append(axs[1].imshow(avgLoadPic, cmap=cm.get_cmap('RdBu'), vmin=vmin, vmax=vmax, origin='lower'))
        axs[1].set_title('Avg Load')

        for ax, im in zip(axs, ims):
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.grid(False)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            f.colorbar(im, cax, orientation='vertical')
    if plotIndvHists:
        indvHists(pic1Data, thresholds, colors, extra=thresholds)
    # output thresholds
    """    thresholds = np.flip(np.reshape(thresholds, (10,10)),1)
    with open('J:/Code-Files/T-File.txt','w') as f:
        for row in thresholds:
            for thresh in row:
                f.write(str(thresh) + ' ') """
    return key, loadRate, loadRateErr, pic1Data, atomImages, thresholds


def Assembly(fileNumber, atomLocs1, pic1Num, partialCredit=False, **standardAssemblyArgs):
    """
    This function checks the efficiency of generating a picture
    I.e. finding atoms at multiple locations at the same time.
    """
    (atomLocs1, atomLocs2, key, thresholds, pic1Data, pic2Data, fit, ensembleStats, avgPic, atomCounts, keyName,
     indvStatistics, lossAvg, lossErr, fitModule,
     enhancementStats) = standardAssemblyAnalysis(fileNumber, atomLocs1, pic1Num, partialCredit=partialCredit, **standardAssemblyArgs)

    if not show:
        return key, survivalData, survivalErrs

    # #########################################
    #      Plotting
    colors, colors2 = getColors(len(atomLocs1)+1)
    f = figure()
    # Setup grid
    grid1 = mpl.gridspec.GridSpec(12, 16)
    grid1.update(left=0.05, right=0.95, wspace=1.2, hspace=1000)
    gridLeft = mpl.gridspec.GridSpec(12, 16)
    gridLeft.update(left=0.001, right=0.95, hspace=1000)
    gridRight = mpl.gridspec.GridSpec(12, 16)
    gridRight.update(left=0.2, right=0.946, wspace=0, hspace=1000)
    # Main Plot
    mainPlot = subplot(grid1[:, :12])
    for stats, label, c in zip((ensembleStats, enhancementStats), ('Assembly', 'Enhancement'), colors):
        mainPlot.errorbar(key, stats['avg'], yerr=stats['err'], color=c, ls='',
                          marker='o', capsize=6, elinewidth=3, label=label)
    
    if partialCredit:
        mainPlot.set_ylim({-0.02, len(atomLocs1)+0.01})
    else:
        mainPlot.set_ylim({-0.02, 1.01})
    if not min(key) == max(key):
        mainPlot.set_xlim(left=min(key) - (max(key) - min(key)) / len(key), right=max(key) + (max(key) - min(key)) / len(key))
    mainPlot.set_xticks(key)
    rotateTicks(mainPlot)
    titletxt = keyName + " Atom Ensemble Scan"
    if len(ensembleStats['avg']) == 1:
        titletxt = keyName + " Atom Ensemble Point. Ensemble % = \n"
        for atomData in ensembleStats['avg']:
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
    countPlot.set_title("Thresh.=" + str(round_sig(thresholds[i]))) 
    ticksForVis = countPlot.xaxis.get_major_ticks()
    ticksForVis[-1].label1.set_visible(False)
    for item in ([countPlot.title, countPlot.xaxis.label, countPlot.yaxis.label] + countPlot.get_xticklabels()
                     + countPlot.get_yticklabels()):
        item.set_fontsize(10)
    countPlot.set_xlim((0, len(pic1Data[0])))
    tickVals = np.linspace(0, len(pic1Data[0]), len(key) + 1)
    countPlot.set_xticks(tickVals[::2])
    rotateTicks(countPlot)
    # Count Histogram Plot
    countHist = subplot(gridLeft[4:8, 15:16], sharey=countPlot)
    for i, atomLoc in enumerate(atomLocs1):
        countHist.hist(pic1Data[i], 50, color=colors[i], orientation='horizontal', alpha=0.5,histtype='stepfilled')
        countHist.hist(pic2Data[i], 50, color=colors2[i], orientation='horizontal', alpha=0.3,histtype='stepfilled')
        countHist.axhline(thresholds[i], color='w')
    for item in ([countHist.title, countHist.xaxis.label, countHist.yaxis.label] + countHist.get_xticklabels()
                  + countHist.get_yticklabels()):
        item.set_fontsize(10)
    rotateTicks(countHist)
    setp(countHist.get_yticklabels(), visible=False)
    # average image
    avgPlt = subplot(gridRight[8:12, 12:15])
    avgPlt.imshow(avgPic, origin='lower');
    avgPlt.grid(False)
    for loc in atomLocs1:
        circ = Circle((loc[1], loc[0]), 0.2, color='r')
        avgPlt.add_artist(circ)
    for loc in atomLocs2:
        circ = Circle((loc[1], loc[0]), 0.1, color='g')
        avgPlt.add_artist(circ)
    avgPlt.set_xticks([])
    avgPlt.set_yticks([])

    f.subplots_adjust(bottom=0, top=1, hspace=0)
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
        image.axis(False)
        image.set_title(picTitle)
        colorbar(im, ax=image)
        if circleLoc != (-1, -1):
            circ = Circle(circleLoc, 0.2, color='r')
            image.add_artist(circ)


def Rearrange(rerngInfoAddress, fileNumber, locations,splitByNumberOfMoves=False, **rearrangeArgs):
    """

    :param rerngInfoAddress:
    :param fileNumber:
    :param locations:
    :param rearrangeArgs:
    :return:
    """
    res = AnalyzeRearrangeMoves(rerngInfoAddress, fileNumber, locations, splitByNumberOfMoves=splitByNumberOfMoves, **rearrangeArgs)
    allData, fits, pics, moves = res
    f, ax = subplots(1)
    for loc in allData:
        ax.errorbar( allData[loc].transpose().columns, allData[loc]['success'], yerr=allData[loc]['error'], 
                     marker='o', ls='', capsize=6, elinewidth=3, color='b' )
        if splitByNumberOfMoves:
            for i,h,t in zip(allData[loc].transpose().columns, allData[loc]['success'], allData[loc]['occurances']):
                ax.text(i+0.25, h, int(t))            
        else:
            for i, (h, t) in enumerate(zip(allData[loc]['success'], allData[loc]['occurances'])):
                ax.text(i+0.25, h, int(t))
    for _, d in allData.items():
        display(d)
    ax.set_xlabel('Moves Involved in Assembly')
    ax.set_ylabel('Assembly Success Rate')
    ax.set_title('Rearranging Individual Move Fidelity Analysis')
    rotateTicks(ax)
    return allData, pics, moves
            