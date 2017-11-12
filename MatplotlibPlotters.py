__version__ = "1.0"

from numpy import array as arr
from random import randint
from Miscellaneous import getColors, round_sig
from matplotlib.pyplot import *
import matplotlib as mpl
from AnalysisHelpers import (loadHDF5, loadDataRay, loadCompoundBasler, processSingleImage, orderData,
                             normalizeData, getBinData, getSurvivalData, getSurvivalEvents, fitDoubleGaussian,
                             guessGaussianPeaks, calculateAtomThreshold, getAvgPic, getEnsembleHits,
                             getEnsembleStatistics, handleFitting, getLoadingData)
import FittingFunctions as fitFunc


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
                manualAccumulation=False, maxColor=None):
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
            rawRawData = [[[] for y in range(accumulations)] for x in range(len(key))]
            dataInc = 0
            for keyInc in range(len(key)):
                for repInc in range(accumulations):
                    rawRawData[keyInc][repInc] = loadBasler(initDataNum + dataInc)
                    dataInc += 1
            rawRawData = arr(rawRawData)
            # average all the pics for a given key value
            rawData = [[] for x in range(len(key))]
            variationInc = 0
            singleAvgPic = 0
            for variationPics in rawRawData:
                avgPic = 0;
                for pic in variationPics:
                    avgPic += pic
                avgPic /= accumulations
                rawData[variationInc] = avgPic
                singleAvgPic += avgPic
                variationInc += 1
            rawData = arr(rawData)
        elif loadType == 'ace':
            rawData = loadCompoundBasler(data)
        elif loadType == 'dataray':
            rawData = loadDataRay(data)
        else:
            raise ValueError("Bad argument for LoadType.")
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


def Transfer(fileNumber, atomLocs1, atomLocs2, show=True, accumulations=1, key=None,
             picsPerRep=2, plotTogether=True, plotLoadingRate=False, manualThreshold=None,
             fitType=None, window=None, xMin=None, xMax=None, yMin=None, yMax=None, dataRange=None,
             histSecondPeakGuess=None, keyOffset=0, sumAtoms=False):
    """
    Standard data analysis package for looking at survival rates throughout an experiment.

    :return key, survivalData, survivalErrors
    """
    if type(atomLocs1[0]) == int:
        # assume atom grid format.
        topLeftRow = atomLocs1[0]
        topLeftColumn = atomLocs1[1]
        spacing = atomLocs1[2]
        width = atomLocs1[3]  # meaning the atoms array width number x height number, say 5 by 5
        height = atomLocs1[4]
        atomLocs1 = []
        for widthInc in range(width):
            for heightInc in range(height):
                atomLocs1.append([topLeftRow + spacing * heightInc, topLeftColumn + spacing * widthInc])

    # make it the right size
    if len(arr(atomLocs1).shape) == 1 and len(atomLocs1) is not 5:
        atomLocs1 = [atomLocs1]
    if atomLocs2 is None:
        atomLocs2 = atomLocs1
    # report

    #### Load Fits File & Get Dimensions
    # Get the array from the fits file. That's all I care about.
    rawData, keyName, key, repetitions = loadHDF5(fileNumber)
    key -= keyOffset
    if len(key) < 100:
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
    avgPic = getAvgPic(groupedData);

    (pic1Data, pic2Data, atomCounts, bins, binnedData, thresholds, survivalData, survivalErrs,
     captureArray, fitNom, fitStd, center, fitValues, fitErrs, fitCovs) = arr([[None] * len(atomLocs1)] * 15)
    for i, (loc1, loc2) in enumerate(zip(atomLocs1, atomLocs2)):
        # grab the first picture of each repetition
        pic1Data[i] = normalizeData(groupedData, loc1, 0, picsPerRep);
        pic2Data[i] = normalizeData(groupedData, loc2, 1, picsPerRep);
        atomCounts[i] = []
        for pic1, pic2 in zip(pic1Data[i], pic2Data[i]):
            atomCounts[i].append(pic1)
            atomCounts[i].append(pic2)
        atomCounts[i] = arr(atomCounts[i])
        ### Calculate Atom Threshold. Default binning for this is 10-count-wide bins
        bins[i], binnedData[i] = getBinData(10, pic1Data[i]);
        guess1, guess2 = guessGaussianPeaks(bins[i], binnedData[i]);
        if histSecondPeakGuess is None:
            guess = arr([max(binnedData[i]), guess1, 30, max(binnedData[i]) * 0.75, 200, 10]);
        else:
            guess = arr([max(binnedData[i]), guess1, 30, max(binnedData[i]) * 0.75, histSecondPeakGuess, 10]);
        gaussianFitVals = fitDoubleGaussian(bins[i], binnedData[i], guess);
        thresholds[i], thresholdFid = calculateAtomThreshold(gaussianFitVals);
        if manualThreshold is not None:
            thresholds[i] = manualThreshold
        # Calculate survival events
        if picsPerRep > 1:
            # Get Data in final form for exporting
            survivalList = getSurvivalEvents(atomCounts[i], thresholds[i], numberOfRuns)
            survivalData[i], survivalErrs[i], captureArray[i] = getSurvivalData(survivalList, repetitions);
        atomCount = 0;
        for point in pic1Data[i]:
            if point > thresholds[i]:
                atomCount += 1
        xFit = np.linspace(min(key), max(key), 1000)
        fitNom[i] = fitStd[i] = center[i] = fitValues[i] = fitErrs[i] = fitCovs[i] = None
        if sumAtoms == False:
            fitNom[i], fitStd[i], center[i], fitValues[i], fitErrs[i], fitCovs[i] = handleFitting(fitType, key,
                                                                                                  survivalData[i])

    if sumAtoms:
        # survivalDataSum = np.sum(survivalData)/len(atomLocs1)
        survivalDataSum = sum(survivalData * captureArray) / sum(captureArray)  # weighted sum with loading
        survivalErrsSum = np.sqrt(np.sum(survivalErrs ** 2)) / len(atomLocs1)
        fitNomSum = fitStdSum = centerSum = fitValuesSum = fitErrsSum = fitCovsSum = None
        fitNomSum, fitStdSum, centerSum, fitValuesSum, fitErrsSum, fitCovsSum = handleFitting(fitType, key,
                                                                                              survivalDataSum)

    if show:
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
        if atomLocs1 == atomLocs2:
            typeName = "Survival"
        else:
            typeName = "Transfer"
        mainPlot = subplot(grid1[:, :12])
        for i, atomLoc in enumerate(atomLocs1):
            leg = r"[%d,%d]$\rightarrow$[%d,%d] " % (atomLocs1[i][0], atomLocs1[i][1],
                                                     atomLocs2[i][0], atomLocs2[i][1])
            if len(survivalData[i]) == 1:
                leg += (typeName + " % = " + str(round_sig(survivalData[i][0])) + "$\pm$ "
                        + str(round_sig(survivalErrs[i][0])))
            mainPlot.errorbar(key, survivalData[i], yerr=survivalErrs[i], color=colors[i], ls='',
                              marker='o', capsize=6, elinewidth=3,
                              label=leg)
            if fitType is None or fitNom[i] == None or sumAtoms == True:
                pass
            elif fitType == 'Exponential-Decay' or fitType == 'Exponential-Saturation':
                mainPlot.plot(xFit, fitNom[i], ':', color=colors[i], label=r'Fit; decay constant $= '
                                                                           + str(round_sig(fitValues[1])) + '\pm '
                                                                           + str(round_sig(fitErrs[1])) + '$',
                              linewidth=3)
                mainPlot.fill_between(xFit, fitNom[i] - fitStd[i], fitNom[i] + fitStd[i],
                                      alpha=0.1, label=r'$\pm\sigma$ band',
                                      color=colors[i])
            elif (fitType == 'RabiFlop'):
                mainPlot.plot(xFit, fitNom[i], ':', color=colors[i], label=r'eff Rabi rate $= '
                                                                           + str(round_sig(fitValues[i][1])) + '\pm '
                                                                           + str(round_sig(fitErrs[i][1])) + '$\n'
                                                                           + '$\pi-time='
                                                                           + str(
                    round_sig(1 / fitValues[i][1] / 2)) + '\pm '
                                                                           + str(
                    round_sig(fitErrs[i][1]) / fitValues[i][1] ** 2 / 2) + '$\n'
                                                                           + 'amplitude $='
                                                                           + str(round_sig(fitValues[i][0])) + '\pm '
                                                                           + str(round_sig(fitErrs[i][0])) + '$',
                              linewidth=3)
                mainPlot.fill_between(xFit, fitNom[i] - fitStd[i], fitNom[i] + fitStd[i],
                                      alpha=0.1, label=r'$\pm\sigma$ band',
                                      color=colors[i])
            elif (fitType == 'Gaussian-Dip') or (fitType == 'Gaussian-Bump'):
                # assuming gaussian?
                mainPlot.plot(xFit, fitNom[i], ':', label='Fit', linewidth=3, color=colors[i])
                mainPlot.fill_between(xFit, fitNom[i] - fitStd[i], fitNom[i] + fitStd[i], alpha=0.1,
                                      label='2-sigma band', color=colors[i])
                mainPlot.axvspan(fitValues[i][1] - np.sqrt(fitCovs[i][1, 1]),
                                 fitValues[i][1] + np.sqrt(fitCovs[i][1, 1]),
                                 color=colors[i], alpha=0.1)
                mainPlot.axvline(fitValues[i][1], color=colors[i], linestyle='-.',
                                 alpha=0.5, label='fit center $= ' + str(round_sig(fitValues[i][2], 6))
                                                  + '\pm ' + str(round_sig(fitErrs[i][2], 4)) + '$')
            elif fitType is not None:
                plot(xFit, fitNom[i], ':', label='Fit', linewidth=3)
                fill_between(xFit, fitNom[i] - fitStd[i], fitNom[i] + fitStd[i], alpha=0.1,
                             label=r'$\pm\sigma$ band', color=colors[i])
                axvspan(fitValues[i][center[i]] - np.sqrt(fitCovs[i][center[i], center[i]]),
                        fitValues[i][center[i]] + np.sqrt(fitCovs[i][center[i], center[i]]),
                        color=colors[i], alpha=0.1)
                axvline(fitValues[i][center[i]], color=colors[i], linestyle='-.', alpha=0.5,
                        label='fit center $= ' + str(round_sig(fitValues[i][center[i]])) + '$')
            else:
                pass
        mainPlot.set_ylim({-0.02, 1.01})
        if not min(key) == max(key):
            mainPlot.set_xlim(left=min(key) - (max(key) - min(key)) / len(key), right=max(key)
                                                                                      + (max(key) - min(key)) / len(
                key))
        mainPlot.set_xticks(key)

        titletxt = keyName + " Atom " + typeName + " Scan"

        if len(survivalData[0]) == 1:
            titletxt = keyName + " Atom " + typeName + " Point. " + typeName + " % = \n"
            for atomData in survivalData:
                titletxt += ''  # str(round_sig(atomData,4) + ", "

        mainPlot.set_title(titletxt, fontsize=30)
        mainPlot.set_ylabel("Survival Probability", fontsize=20)
        mainPlot.set_xlabel(keyName, fontsize=20)
        mainPlot.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=4)
        legend()
        # Loading Plot
        loadingPlot = subplot(grid1[0:3, 12:16])
        for i, loc in enumerate(atomLocs1):
            loadingPlot.plot(key, captureArray[i], ls='', marker='o', color=colors[i])
            loadingPlot.axhline(np.mean(captureArray[i]), color=colors[i])
        loadingPlot.set_ylim({0, 1})
        if not min(key) == max(key):
            loadingPlot.set_xlim(left=min(key) - (max(key) - min(key)) / len(key), right=max(key)
                                                                                         + (max(key) - min(key)) / len(
                key))
        loadingPlot.set_xlabel("Key Values")
        loadingPlot.set_ylabel("Capture %")
        loadingPlot.set_xticks(key)
        loadingPlot.set_title("Loading: Avg$ = " + str(round_sig(np.mean(captureArray[0]))) + '$')
        for item in ([loadingPlot.title, loadingPlot.xaxis.label, loadingPlot.yaxis.label] +
                         loadingPlot.get_xticklabels() + loadingPlot.get_yticklabels()):
            item.set_fontsize(10)
            # ### Count Series Plot
        countPlot = subplot(gridRight[4:8, 12:15])
        for i, loc in enumerate(atomLocs1):
            countPlot.plot(pic1Data[i], color=colors[i], ls='', marker='.', markersize=1, alpha=1)
            countPlot.plot(pic2Data[i], color=colors2[i], ls='', marker='.', markersize=1, alpha=0.8)
            countPlot.axhline(thresholds[i], color='w')
        countPlot.set_xlabel("Picture #")
        countPlot.set_ylabel("Camera Signal")
        countPlot.set_title("Thresh.=" + str(round_sig(thresholds[i])) + ", Fid.="
                            + str(round_sig(thresholdFid)), fontsize=10)
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
            countHist.hist(pic1Data[i], 50, color=colors[i], orientation='horizontal', alpha=0.5)
            countHist.hist(pic2Data[i], 50, color=colors2[i], orientation='horizontal', alpha=0.3)
            countHist.axhline(thresholds[i], color='w')
        for item in ([countHist.title, countHist.xaxis.label, countHist.yaxis.label] +
                         countHist.get_xticklabels() + countHist.get_yticklabels()):
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
        if sumAtoms:
            figure()
            colorsLoc = cmapRGB(randint(0, 100))[:-1]
            errorbar(key, survivalDataSum, yerr=survivalErrsSum, color=colorsLoc, ls='',
                     marker='o', capsize=6, elinewidth=3, label='sum Data ' + str(atomLoc))
            if fitNomSum == None:
                pass
            elif fitType == "RabiFlop":
                sumatomplot = plot(xFit, fitNomSum, ':', label='eff Rabi rate $= '
                                                               + str(round_sig(fitValuesSum[1], 3)) + '\pm '
                                                               + str(round_sig(fitErrsSum[1], 3)) + '$\n'
                                                               + '$\pi-time='
                                                               + str(round_sig(1 / fitValuesSum[1] / 2)) + '\pm '
                                                               + str(
                    round_sig(fitErrsSum[1]) / fitValuesSum[1] ** 2 / 2) + '$\n'
                                                               + 'amplitude $='
                                                               + str(round_sig(fitValuesSum[0])) + '\pm '
                                                               + str(round_sig(fitErrsSum[0])) + '$', linewidth=3,
                                   color=colorsLoc)
                fill_between(xFit, fitNomSum - fitStdSum, fitNomSum + fitStdSum, color=colorsLoc,
                             alpha=0.1, label=r'$\pm\sigma$ band')
            elif (fitType == 'Gaussian-Dip') or (fitType == 'Gaussian-Bump'):
                sumatomplot = plot(xFit, fitNomSum, ':', label='freq center $= '
                                                               + str(round_sig(fitValuesSum[1], 7)) + '\pm '
                                                               + str(round_sig(fitErrsSum[1], 7)) + '$\n'
                                                               + 'width=$'
                                                               + str(round_sig(fitValuesSum[2])) + '\pm '
                                                               + str(round_sig(fitErrsSum[2])) + '$\n', linewidth=3,
                                   color=colorsLoc)
                fill_between(xFit, fitNomSum - fitStdSum, fitNomSum + fitStdSum, color=colorsLoc,
                             alpha=0.1, label=r'$\pm\sigma$ band')
            elif fitType is not None:
                plot(xFit, fitNomSum, ':', linewidth=3, color=colorsLoc, label=r'decay constant = '
                                                                               + str(
                    round_sig(fitValuesSum[1], 7)) + '$\pm $' + str(round_sig(fitErrsSum[1], 7)))
                # fill_between(xFit, fitNomSum - fitStdSum, fitNomSum + fitStdSum, alpha=0.1, label=r'$\pm\sigma$ band',
                #             color='b')
                # axvspan(fitValuesSum - np.sqrt(fitCovsSum[1,1]),
                #    fitValuesSum + np.sqrt(fitCovsSum[1,1]),
                #    color=colorsLoc, alpha=0.1)
                # axvline(fitValuesSum, color=colorsLoc,linestyle='-.', alpha=0.5,
                #    label='fit center $= '+ str(round_sig(fitValuesSum))+'$')
                # pass
            legend()
            title(titletxt + str(survivalDataSum) + '$\pm$' + str(survivalErrsSum), fontsize=30)
            ylabel("sum Survival Probability", fontsize=20)
            xlabel(keyName, fontsize=20)
            ylim(-0.02, 1.01)
    return key, survivalData, survivalErrs, captureArray


def Loading(fileNum, atomLocations, accumulations=1, key=None, picsPerExperiment=1,
            analyzeTogether=False, plotLoadingRate=False, picture=0, manualThreshold=None,
            loadingFitType=None, showIndividualHist=True, showTotalHist="def"):
    """
    Standard data analysis package for looking at loading rates throughout an experiment.

    return key, loadingRateList, loadingRateErr

    This routine is designed for analyzing experiments with only one picture per cycle. Typically
    These are loading exeriments, for example. There's no survival calculation.
    """
    if type(atomLocations[0]) == int:
        # assume atom grid format.
        topLeftRow = atomLocations[0]
        topLeftColumn = atomLocations[1]
        spacing = atomLocations[2]
        width = atomLocations[3]
        height = atomLocations[4]
        atomLocations = []
        for widthInc in range(width):
            for heightInc in range(height):
                atomLocations.append([topLeftRow + spacing * heightInc, topLeftColumn + spacing * widthInc])

    # ### Load Fits File & Get Dimensions
    # Get the array from the fits file. That's all I care about.
    rawData, keyName, key, repetitions = loadHDF5(fileNum)
    # the .shape member of an array gives an array of the dimesnions of the array.
    numOfPictures = rawData.shape[0]
    numOfVariations = int(numOfPictures / repetitions)
    # handle defaults.
    if numOfVariations == 1:
        if showTotalHist == "def":
            showTotalHist = False
        if key is None:
            key = arr([0])
    else:
        if showTotalHist == 'def':
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
        newShape = (numOfVariations, repetitions, s[1], s[2])
    groupedData = rawData.reshape(newShape)
    groupedData, key = orderData(groupedData, key)
    print('Data Shape:', groupedData.shape)
    loadingRateList, loadingRateErr = [[[] for x in range(len(atomLocations))] for x in range(2)]
    print('Analyzing Variation... ', end='')
    for dataInc, data in enumerate(groupedData):
        print(str(dataInc) + ', ', end='')
        avgPic = getAvgPic(data)
        # initialize empty lists
        (pic1Data, pic1Atom, thresholds, thresholdFid, fitVals, bins, binData,
         atomCount) = arr([[None] * len(atomLocations)] * 8)

        # fill lists with data
        for i, atomLoc in enumerate(atomLocations):
            (pic1Data[i], pic1Atom[i], thresholds[i], thresholdFid[i],
             fitVals[i], bins[i], binData[i], atomCount[i]) = getLoadingData(data, atomLoc, picture, picsPerExperiment,
                                                                             manualThreshold, 10)
            if plotLoadingRate:
                loadingRateList[i].append(np.mean(pic1Atom[i]))
                loadingRateErr[i].append(np.std(pic1Atom[i]) / np.sqrt(len(pic1Atom[i])))
        if showIndividualHist:
            atomHist(key, atomLocations, pic1Data, bins, binData, fitVals, thresholds, avgPic, atomCount,
                     dataInc)
    avgPic = getAvgPic(rawData)
    (pic1Data, pic1Atom, thresholds, thresholdFid, fitVals, bins, binData,
     atomCount) = arr([[None] * len(atomLocations)] * 8)
    # loadingRateList, loadingRateErr = [[[] for x in range(len(atomLocations))]] * 2
    # fill lists with data
    for i, atomLoc in enumerate(atomLocations):
        (pic1Data[i], pic1Atom[i], thresholds[i], thresholdFid[i],
         fitVals[i], bins[i], binData[i], atomCount[i]) = getLoadingData(rawData, atomLoc, picture,
                                                                  picsPerExperiment, manualThreshold, 5)
    if showTotalHist:
        atomHist(key, atomLocations, pic1Data, bins, binData, fitVals, thresholds, avgPic, atomCount, None)
    if plotLoadingRate:
        # get colors
        colors = getColors(atomLocations)
        figure()
        title('Loading Rate')
        for i, atomLoc in enumerate(atomLocations):
            errorbar(key, loadingRateList[i], yerr=loadingRateErr[i], marker='o', linestyle='', label=atomLoc,
                     color=colors[i])
        xlabel("key value")
        ylabel("loading rate")
        fitInfo, loadingFitType = handleFitting(loadingFitType, key, loadingRateList)
        if loadingFitType is not None:
            plot(fitInfo['x'], fitInfo['nom'], ':', label='Fit', linewidth=3)
            center = fitInfo['center']
            fill_between(fitInfo['x'], fitInfo['nom'] - fitInfo['std'], fitInfo['nom'] + fitInfo['std'], alpha=0.1,
                         label=r'$\pm\sigma$ band', color='b')
            axvspan(fitInfo['vals'][center ] - fitInfo['err'][center ],
                    fitInfo['vals'][center ] + fitInfo['err'][center ],
                    color='b', alpha=0.1)
            axvline(fitInfo['vals'][center ], color='b', linestyle='-.', alpha=0.5,
                    label='fit center $= ' + str(round_sig(fitInfo['vals'][center ])) + '$')
        keyRange = max(key) - min(key)
        xlim(min(key) - keyRange / (2 * len(key)), max(key)
             + keyRange / (2 * len(key)))
        ylim(0, 1)
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
