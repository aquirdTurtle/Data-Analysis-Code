__version__ = "1.1"

import time
from pandas import DataFrame
from MainAnalysis import standardPopulationAnalysis, analyzeNiawgWave, standardTransferAnalysis, standardAssemblyAnalysis, AnalyzeRearrangeMoves
from numpy import array as arr
from random import randint
from Miscellaneous import getColors, round_sig, round_sig_str, getMarkers, errString
import Miscellaneous as misc
from matplotlib.pyplot import *
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable


from scipy.optimize import curve_fit as fit
from AnalysisHelpers import (loadDataRay, loadCompoundBasler, processSingleImage, orderData,
                             normalizeData, getBinData, getSurvivalData, getSurvivalEvents, fitDoubleGaussian,
                             guessGaussianPeaks, calculateAtomThreshold, getAvgPic, getEnsembleHits,
                             getEnsembleStatistics, handleFitting, 
                             loadDetailedKey, processImageData,
                             fitPictures, fitGaussianBeamWaist, assemblePlotData, ballisticMotExpansion, simpleMotExpansion, 
                             calcMotTemperature,integrateData, computeMotNumber, getFitsDataFrame, genAvgDiscrepancyImage, 
                             getGridDims, newCalcMotTemperature)
import AnalysisHelpers as ah
import MarksConstants as consts 
from matplotlib.patches import Ellipse
from TimeTracker import TimeTracker
from fitters import gaussian_2d, LargeBeamMotExpansion, exponential_saturation, double_gaussian


def rotateTicks(plot):
    ticks = plot.get_xticklabels()
    for tickInc in range(len(ticks)):
        ticks[tickInc].set_rotation(-45)


def indvHists(dat, thresh, colors, extra=None, extraname=None, extra2=None, extra2Name=None, gaussianFitVals=None):
    f, axs = subplots(10,10, figsize=(25,12.5))
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
            txt = extraname + round_sig_str( np.mean(extra[i])) if extraname is not None else round_sig_str(np.mean(extra[i]))
            t = ax.text( 0.25, max(heights)-5, txt, fontsize=12 )
            t.set_bbox(dict(facecolor='k', alpha=0.3))
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

    
def plotMotTemperature(data, key=None, magnification=3, **standardImagesArgs):
    """
    Calculate the mot temperature, and plot the data that led to this.

    :param data:
    :param standardImagesArgs: see the standardImages function to see the acceptable arguments here.
    :return:
    """
    temp, fitVals, fitCov, times, waists, rawData, pictureFitParams = ah.temperatureAnalysis(data, magnification, key=key, **standardImagesArgs)
    errs = np.sqrt(np.diag(fitCov))
    f, ax = subplots()
    ax.plot(times, waists, 'bo', label='Raw Data Waist')
    ax.plot(times, 2 * LargeBeamMotExpansion.f(times, *fitVals), 'c:', label='balistic MOT expansion Fit Waist')
    ax.yaxis.label.set_color('c')
    ax.grid(True,color='b')
    ax2 = ax.twinx()
    ax2.plot(times, pictureFitParams[:, 0], 'ro:', marker='*', label='Fit Amplitude (counts)')
    ax2.yaxis.label.set_color('r')
    ax.set_title('Measured atom cloud size over time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Gaussian fit waist (m)')
    ax.legend()
    ax2.legend()
    ax2.grid(True,color='r')
    showPics(rawData, key, fitParams=pictureFitParams)
    print("\nTemperture in the Large Laser Beam Approximation:", misc.errString(temp * 1e6, errs[2]*1e6), 'uK')
    print('Fit-Parameters:', fitVals)
    return pictureFitParams,rawData

    
def plotMotNumberAnalysis(dataSetNumber, motKey, exposureTime,  *fillAnalysisArgs):
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
    rawData, intRawData, motnumber, fitParams, fluorescence = ah.motFillAnalysis(dataSetNumber, motKey, exposureTime, *fillAnalysisArgs)
    figure()
    plot(motKey, intRawData, 'bo', label='data', color='b')
    xfitPts = np.linspace(min(motKey), max(motKey), 1000)
    plot(xfitPts, exponential_saturation.f(xfitPts, *fitParams), 'b-', label='fit', color='r', linestyle=':')
    xlabel('loading time (s)')
    ylabel('integrated counts')
    title('Mot Fill Curve')
    print("integrated saturated counts subtracting background =", -fitParams[0])
    print("loading time 1/e =", fitParams[1], "s")
    print('MOT Number:', motnumber)
    print('Light Scattered off of full MOT:', fluorescence * consts.h * consts.Rb87_D2LineFrequency * 1e9, "nW")
    return motnumber


def singleImage(data, accumulations=1, loadType='andor', bg=arr([0]), title='Single Picture', window=(0, 0, 0, 0),
                xMin=0, xMax=0, yMin=0, yMax=0, zeroCorners=False, smartWindow=False, findMax=False,
                manualAccumulation=False, maxColor=None, key=arr([])):
    """
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

    res = processSingleImage(rawData, bg, window, xMin, xMax, yMin, yMax, accumulations, zeroCorners, smartWindow, manualAccumulation)
    rawData, dataMinusBg, xPts, yPts = res
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
    :return key, transferData, survivalErrors
    """
    tt = TimeTracker()
    res = standardTransferAnalysis( fileNumber, atomLocs1_orig, atomLocs2_orig, fitModule=fitModule, tt=tt,
                                    **standardTransferArgs )
    tt.clock('After-Standard-Analysis')
    (atomLocs1, atomLocs2, atomCounts, transferData, transferErrs, loadingRate, pic1Data, keyName, key,
     repetitions, thresholds, fits, avgTransferData, avgTransferErr, avgFit, avgPics, otherDimValues,
     locsList, genAvgs, genErrs, gaussianFitVals, tt, threshFids, rmsResiduals) = res
    if not show:
        return (key, transferData, transferErrs, loadingRate, fits, avgFit, genAvgs, genErrs, pic1Data, 
            gaussianFitVals, [None], thresholds, [None])
    legendOption = True if legendOption is None and len(atomLocs1) < 50 else False
    # set locations of plots.
    f = figure()
    typeName = "S." if atomLocs1 == atomLocs2 else "T."
    grid1 = mpl.gridspec.GridSpec(12, 16,left=0.05, right=0.95, wspace=1.2, hspace=1000)
    mainPlot = subplot(grid1[:, :11])
    loadingPlot = subplot(grid1[0:3, 11:16])
    grid1.update( left=0.1, right=0.95, wspace=0, hspace=1000 )
    countPlot = subplot(grid1[4:8, 11:15])    
    grid1.update( left=0.001, right=0.95, hspace=1000 )
    countHist = subplot(grid1[4:8, 15:16], sharey=countPlot)
    avgPlt1 = subplot(grid1[8:12, 11:13])
    avgPlt2 = subplot(grid1[8:12, 13:15])

    centers = []
    colors, colors2 = getColors(len(atomLocs1) + 1)
    longLegend = len(transferData[0]) == 1
    markers = getMarkers()
    # Main Plot
    for i, (atomLoc, fit) in enumerate(zip(atomLocs1, fits)):
        leg = (r"[%d,%d] " % (atomLocs1[i][0], atomLocs1[i][1]) if typeName == "S" 
               else r"[%d,%d]$\rightarrow$[%d,%d] " % (atomLocs1[i][0], atomLocs1[i][1], atomLocs2[i][0], atomLocs2[i][1]))
        if longLegend:
            leg += (typeName + " % = " + str(round_sig(transferData[i][0])) + "$\pm$ " + str(round_sig(transferErrs[i][0])))
        mainPlot.errorbar(key, transferData[i], yerr=transferErrs[i], color=colors[i], ls='',
                          capsize=6, elinewidth=3, label=leg, alpha=0.3, marker=markers[i%len(markers)])
        if fitModule is not None and showFitDetails:
            if fit['vals'] is None:
                print(loc, 'Fit Failed!')
                continue
            if fitModule.center() is not None:
                centers.append(fit['vals'][fitModule.center()])
            mainPlot.plot(fit['x'], fit['nom'], color=colors[i], alpha = 0.5)
    if type(keyName) is not type("a string"):
        keyn = ''
        for kn in keyName:
            keyn += kn + ', '
        keyName = keyn
    titletxt = keyName + " " + typeName + " Scan"
    if len(transferData[0]) == 1:
        titletxt = keyName + " " + typeName + " Point.\n Avg " + typeName + "% = " + round_sig_str(np.mean(avgTransferData))        
    mainPlot.set_title(titletxt, fontsize=30)
    mainPlot.set_ylabel("S %", fontsize=20)
    mainPlot.set_xlabel(keyName, fontsize=20)
    mainPlot.set_yticks(np.arange(0, 1, 0.1 ))
    mainPlot.set_yticks(np.arange(0, 1, 0.05), minor=True)
    mainPlot.xaxis.set_label_coords(0.95, -0.1)
    cols = 4 if longLegend else 10
    if legendOption:
        mainPlot.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=cols, prop={'size': 12})
    # Loading Plot
    for i, loc in enumerate(atomLocs1):
        loadingPlot.plot(key, loadingRate[i], ls='', marker='o', color=colors[i], alpha=0.3)
        loadingPlot.axhline(np.mean(loadingRate[i]), color=colors[i], alpha=0.3)
    loadingPlot.set_xlabel(keyName)
    loadingPlot.set_ylabel("Loading %")
    loadingPlot.set_yticks(np.arange(0,1,0.2))
    loadingPlot.set_yticks(np.arange(0,1,0.1), minor=True)

    loadingPlot.set_title("Loading: Avg$ = " + str(round_sig(np.mean(arr(loadingRate.tolist())))) + '$')
    for item in ([loadingPlot.title, loadingPlot.xaxis.label, loadingPlot.yaxis.label] +
                     loadingPlot.get_xticklabels() + loadingPlot.get_yticklabels()):
        item.set_fontsize(10)
    # shared
    for plot in [mainPlot, loadingPlot]:
        if not min(key) == max(key):
            r = max(key) - min(key)
            plot.set_xlim(left=min(key) - r / len(key), right=max(key)+ r / len(key))
        plot.grid(True, color='#AAAAAA', which='Major')
        plot.grid(True, color='#090909', which='Minor')
        plot.set_ylim({0, 1})
        rotateTicks(plot)
        plot.set_xticks(key)
    # ### Count Series Plot
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
    rotateTicks(countPlot)
    # Count Histogram Plot
    for i, atomLoc in enumerate(atomLocs1):
        countHist.hist(pic1Data[i], 50, color=colors[i], orientation='horizontal', alpha=0.3, histtype='stepfilled')
        countHist.axhline(thresholds[i], color=colors[i], alpha=0.3)
    for item in ([countHist.title, countHist.xaxis.label, countHist.yaxis.label] + 
                 countHist.get_xticklabels() + countHist.get_yticklabels()):
        item.set_fontsize(10)
    rotateTicks(countHist)
    setp(countHist.get_yticklabels(), visible=False)
    
    # average image
    for plt, dat, locs in zip([avgPlt1, avgPlt2], avgPics, [atomLocs1, atomLocs2]):
        plt.imshow(dat, origin='lower', cmap='Greys_r');
        plt.set_xticks([])
        plt.set_yticks([])
        plt.grid(False)
        for loc, c in zip(locs, colors):
            circ = Circle((loc[1], loc[0]), 0.2, color=c)
            plt.add_artist(circ)
    mainPlot.errorbar( key, avgTransferData, yerr=avgTransferErr, color="#FFFFFFFF", ls='',
                       marker='o', capsize=6, elinewidth=3, label='Avg' )
    if fitModule is not None:
        mainPlot.plot(avgFit['x'], avgFit['nom'], color='#FFFFFFFF', ls=':')
        for label, fitVal, err in zip(fitModule.args(), avgFit['vals'], avgFit['errs']):
            print( label,':', errString(fitVal, err) )
        print(avgFit['errs'])
        if showFitDetails:
            display(getFitsDataFrame(fits, fitModule, avgFit))
    if fitModule is not None and showFitCenterPlot:
        figure()
        fitCenterPic, vmin, vmax = genAvgDiscrepancyImage(centers, avgPics[0].shape, atomLocs1)
        imshow(fitCenterPic, cmap=cm.get_cmap('seismic_r'), vmin=vmin, vmax=vmax, origin='lower')
        title('Fit-Centers (white is average)')
        colorbar()
    tt.clock('After-Main-Plots')
    avgTransferPic = None
    if showImagePlots:
        ims = []
        f, axs = subplots(1,9, figsize=(30,12))

        ims.append(axs[0].imshow(avgPics[0], origin='lower'))
        axs[0].set_title('Avg 1st Pic')

        ims.append(axs[1].imshow(avgPics[1],origin='lower'))
        axs[1].set_title('Avg 2nd Pic')
        
        avgTransfers = []
        for s in transferData:
            avgTransfers.append(np.mean(s))
        avgTransferPic, vmin, vmax = genAvgDiscrepancyImage(avgTransfers, avgPics[0].shape, atomLocs1)
        ims.append(axs[2].imshow(avgTransferPic, cmap=cm.get_cmap('seismic_r'), vmin=vmin, vmax=vmax, origin='lower'))
        axs[2].set_title('Avg Trans:' + str(misc.round_sig(np.mean(avgTransfers))), fontsize=12)
        
        avgLoads = []
        for l in loadingRate:
            avgLoads.append(np.mean(l))
        avgLoadPic, vmin, vmax = genAvgDiscrepancyImage(avgLoads, avgPics[0].shape, atomLocs1)
        ims.append(axs[3].imshow(avgLoadPic, cmap=cm.get_cmap('seismic_r'), vmin=vmin, vmax=vmax, origin='lower'))
        axs[3].set_title('Avg Load:' + str(misc.round_sig(np.mean(avgLoads))), fontsize=12)
        
        thresholdPic, vmin, vmax = genAvgDiscrepancyImage(thresholds, avgPics[0].shape, atomLocs1)
        ims.append(axs[4].imshow(thresholdPic, cmap=cm.get_cmap('seismic_r'), vmin=vmin, vmax=vmax, origin='lower'))
        axs[4].set_title('Thresholds:' + str(misc.round_sig(np.mean(thresholds))), fontsize=12)
        
        thresholdFidPic, vmin, vmax = genAvgDiscrepancyImage(threshFids, avgPics[0].shape, atomLocs1)
        ims.append(axs[5].imshow(thresholdFidPic, cmap=cm.get_cmap('seismic_r'), vmin=vmin, vmax=vmax, origin='lower'))
        axs[5].set_title('Thresholds Fidelities:' + str(misc.round_sig(np.mean(threshFids))), fontsize=12)
        
        imagePeakDiff = []
        for g in gaussianFitVals:
            if g is not None:
                imagePeakDiff.append(abs(g[1] - g[4]))
            else:
                imagePeakDiff.append(0)
        peakDiffImage, vmin, vmax = genAvgDiscrepancyImage(imagePeakDiff, avgPics[0].shape, atomLocs1)
        ims.append(axs[6].imshow(peakDiffImage, cmap=cm.get_cmap('seismic_r'), vmin=vmin, vmax=vmax, origin='lower'))
        axs[6].set_title('Imaging-Signal:' + str(misc.round_sig(np.mean(imagePeakDiff))), fontsize=12)

        residualImage, _, vmax = genAvgDiscrepancyImage(rmsResiduals, avgPics[0].shape, atomLocs1)
        ims.append(axs[7].imshow(residualImage, cmap=cm.get_cmap('inferno'), vmin=0, vmax=vmax, origin='lower'))
        axs[7].set_title('Fit Rms Residuals:' + str(misc.round_sig(np.mean(rmsResiduals))), fontsize=12)
        
        if genAvgs is not None:
            if genAvgs[0] is not None:
                genAtomAvgs = [np.mean(dp) for dp in genAvgs]
                print('what',genAvgs)
                genImage, _, vmax = genAvgDiscrepancyImage(genAtomAvgs, avgPics[0].shape, atomLocs1)
            else:
                genAtomAvgs = [0]
                genImage = np.zeros(avgPics[0].shape)
                vmax=1
        ims.append(axs[8].imshow(genImage, cmap=cm.get_cmap('inferno'), vmin=0, vmax=vmax, origin='lower'))
        axs[8].set_title('Atom-Generation: ' + str(misc.round_sig(np.mean(genAtomAvgs))), fontsize=12)
        
        for ax, im in zip(axs, ims):
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.grid(False)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cb = f.colorbar(im, cax, orientation='vertical')
            cb.ax.tick_params(labelsize=8) 
        tt.clock('After-Image-Plots')
    if plotIndvHists:
        indvHists(pic1Data, thresholds, colors, extra=avgTransfers, extraname='S:', gaussianFitVals=gaussianFitVals)
        tt.clock('After-Indv-Hists')
    if timeit:
        tt.display()
    avgPlt1.set_position([0.58,0,0.3,0.3])
    avgPlt2.set_position([0.73,0,0.3,0.3])
    return (key, transferData, transferErrs, loadingRate, fits, avgFit, genAvgs, genErrs, pic1Data, 
            gaussianFitVals, centers, thresholds, avgTransferPic)


def Loading(fileNum, atomLocations, **PopulationArgs):
    """
    A small wrapper, partially for the extra defaults in this case partially for consistency with old function definitions.
    """
    return Population(fileNum, atomLocations, 0, 1, **PopulationArgs)


def Population(fileNum, atomLocations, whichPic, picsPerRep, plotLoadingRate=True, plotCounts=False, legendOption=None,
               showImagePlots=True, plotIndvHists=False, showFitDetails=False, showFitCenterPlot=True, show=True, histMain=False,
               mainAlpha=0.2, avgColor='w', **StandardArgs):
    """
    Standard data analysis package for looking at population %s throughout an experiment.

    return key, loadingRateList, loadingRateErr

    This routine is designed for analyzing experiments with only one picture per cycle. Typically
    These are loading exeriments, for example. There's no survival calculation.
    """
    res = standardPopulationAnalysis(fileNum, atomLocations, whichPic, picsPerRep, **StandardArgs)
    (locCounts, thresholds, avgPic, key, loadRateErr, loadRate, avgLoadRate, avgLoadErr, fits,
     fitModule, keyName, atomData, rawData, atomLocations,  avgFits, atomImages,
     gaussFitVals, totalAvg, totalErr, threshFids) = res
    colors, _ = getColors(len(atomLocations) + 1)
    if not show:
        return key, loadRate, loadRateErr, locCounts, atomImages, thresholds, avgLoadRate
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
    loadingPlot = subplot(grid1[0:3, 12:16])
    countPlot = subplot(gridRight[4:8, 12:15])    
    if not histMain:
        mainPlot = subplot(grid1[:, :12])
        countHist = subplot(gridLeft[4:8, 15:16], sharey=countPlot)
    else:
        countHist = subplot(grid1[:, :12])
        mainPlot = subplot(gridLeft[4:8, 15:16], sharey=countPlot)
    centers = []
    longLegend = len(loadRate[0]) == 1
    if len(arr(key).shape) == 2:
        # 2d scan: no normal plot possible, so make colormap plot of avg
        key1, key2 = key[:,0], key[:,1]
        key1 = np.sort(key1)
        key2 = np.sort(key2)
    else:
        for i, (atomLoc, fit) in enumerate(zip(atomLocations, fits)):
            leg = r"[%d,%d] " % (atomLoc[0], atomLoc[1])
            if longLegend:
                leg += (typeName + " % = " + str(round_sig(loadRate[i][0])) + "$\pm$ "
                        + str(round_sig(loadRateErr[i][0])))
            mainPlot.errorbar(key, loadRate[i], yerr=loadRateErr[i], color=colors[i], ls='',
                              capsize=6, elinewidth=3, label=leg, alpha=mainAlpha, marker=markers[i%len(markers)])
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
                mainPlot.plot(avgFits['x'], avgFits['nom'], color=avgColor, alpha = 1)
        mainPlot.grid(True, color='#AAAAAA', which='Major')
        mainPlot.grid(True, color='#090909', which='Minor')
        mainPlot.set_yticks(np.arange(0,1,0.1))
        mainPlot.set_yticks(np.arange(0,1,0.05), minor=True)
        mainPlot.set_ylim({-0.02, 1.01})
        if not min(key) == max(key):
            mainPlot.set_xlim(left=min(key) - (max(key) - min(key)) / len(key), right=max(key)
                              + (max(key) - min(key)) / len(key))
        mainPlot.set_xticks(key)
        rotateTicks(mainPlot)
        titletxt = keyName + " Atom " + typeName + " Scan"
        if len(loadRate[0]) == 1:
            print(avgLoadErr)
            titletxt = keyName + " Atom " + typeName + " Point.\n Avg " + typeName + "% = " + errString(totalAvg, totalErr ) 
        mainPlot.set_title(titletxt, fontsize=30)
        mainPlot.set_ylabel("S %", fontsize=20)
        mainPlot.set_xlabel(keyName, fontsize=20)
        if legendOption == True:
            cols = 4 if longLegend else 10
            mainPlot.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=cols, prop={'size': 12})
    # Loading Plot
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
    loadingPlot.set_yticks(np.arange(0,1,0.1), minor=True)
    loadingPlot.set_yticks(np.arange(0,1,0.2))
    loadingPlot.grid(True, color='#AAAAAA', which='Major')
    loadingPlot.grid(True, color='#090909', which='Minor')
    loadingPlot.set_title("Loading: Avg$ = " +  str(round_sig(np.mean(arr(loadRate)))) + '$')
    for item in ([loadingPlot.title, loadingPlot.xaxis.label, loadingPlot.yaxis.label] +
                     loadingPlot.get_xticklabels() + loadingPlot.get_yticklabels()):
        item.set_fontsize(10)
    # ### Count Series Plot
    for i, loc in enumerate(atomLocations):
        countPlot.plot(locCounts[i], color=colors[i], ls='', marker='.', markersize=1, alpha=0.3)
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
    countPlot.set_xlim((0, len(locCounts[0])))
    tickVals = np.linspace(0, len(locCounts[0]), len(key) + 1)
    countPlot.set_xticks(tickVals[0:-1:2])
    # Count Histogram Plot
    for i, atomLoc in enumerate(atomLocations):
        if histMain:
            countHist.hist(locCounts[i], 50, color=colors[i], orientation='vertical', alpha=0.3, histtype='stepfilled')
            countHist.axvline(thresholds[i], color=colors[i], alpha=0.3)            
        else:
            countHist.hist(locCounts[i], 50, color=colors[i], orientation='horizontal', alpha=0.3, histtype='stepfilled')
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
    mainPlot.errorbar(key, avgLoadRate, yerr=avgLoadErr, color=avgColor, ls='',
             marker='o', capsize=6, elinewidth=3, label='Avg')
    if fitModule is not None and showFitDetails:
        mainPlot.plot(avgFit['x'], avgFit['nom'], color=avgColor, ls=':')
        fits_df = getFitsDataFrame(fits, fitModule, avgFit)
        display(fits_df)
    elif fitModule is not None:
        for val, name in zip(avgFits['vals'], fitModule.args()):
            print(name, val)
    if fitModule is not None and showFitCenterPlot and fits[0] != []:
        figure()
        fitCenterPic, vmin, vmax = genAvgDiscrepancyImage(centers, avgPic.shape, atomLocations)
        imshow(fitCenterPic, cmap=cm.get_cmap('seismic_r'), vmin=vmin, vmax=vmax, origin='lower')
        title('Fit-Centers (white is average)')
        colorbar()
    if showImagePlots:
        ims = []
        f, axs = subplots(1,4)
        
        ims.append(axs[0].imshow(avgPic, origin='lower'))
        axs[0].set_title('Avg 1st Pic')

        avgLoads = []
        for l in loadRate:
            avgLoads.append(np.mean(l))
        avgLoadPic, vmin, vmax = genAvgDiscrepancyImage(avgLoads, avgPic.shape, atomLocations)
        ims.append(axs[1].imshow(avgLoadPic, cmap=cm.get_cmap('seismic_r'), vmin=vmin, vmax=vmax, origin='lower'))
        axs[1].set_title('Avg Load')
        
        thresholdPic, vmin, vmax = genAvgDiscrepancyImage(thresholds, avgPic.shape, atomLocations)
        ims.append(axs[2].imshow(thresholdPic, cmap=cm.get_cmap('seismic_r'), vmin=vmin, vmax=vmax, origin='lower'))
        axs[2].set_title('Thresholds')
        
        thresholdFidPic, vmin, vmax = genAvgDiscrepancyImage(threshFids, avgPic.shape, atomLocations)
        ims.append(axs[3].imshow(thresholdFidPic, cmap=cm.get_cmap('seismic_r'), vmin=vmin, vmax=vmax, origin='lower'))
        axs[3].set_title('Thresholds Fidelities')
        
        for ax, im in zip(axs, ims):
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.grid(False)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            f.colorbar(im, cax, orientation='vertical')
    avgLoads = []
    for s in loadRate:
        avgLoads.append(np.mean(s))
    if plotIndvHists:
        indvHists(locCounts, thresholds, colors, extra=avgLoads, extraname='L:')
    """
    # output thresholds
    thresholds = np.flip(np.reshape(thresholds, (10,10)),1)
    with open('J:/Code-Files/T-File.txt','w') as f:
        for row in thresholds:
            for thresh in row:
                f.write(str(thresh) + ' ') 
    """
    return {'Key': key, 'Loading': loadRate, 'Loading_Error': loadRateErr, 'Pixel_Counts':locCounts, 'Atom_Images':atomImages, 
            'Thresholds':thresholds, 'Threshold_Gaussian_Fits:':gaussFitVals, 'Atom_Data':atomData, 'Raw_Data':rawData}


def Assembly(fileNumber, atomLocs1, pic1Num, partialCredit=False, **standardAssemblyArgs):
    """
    This function checks the efficiency of generating a picture
    I.e. finding atoms at multiple locations at the same time.
    """
    res = standardAssemblyAnalysis(fileNumber, atomLocs1, pic1Num, partialCredit=partialCredit, **standardAssemblyArgs)
    (atomLocs1, atomLocs2, key, thresholds, pic1Data, pic2Data, fit, ensembleStats, avgPic, atomCounts, keyName,
     indvStatistics, lossAvg, lossErr, fitModule, enhancementStats) = res

    if not show:
        return key, survivalData, survivalErrs
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
    return key, survivalData, survivalErrs


def Rearrange(rerngInfoAddress, fileNumber, locations,splitByNumberOfMoves=False, **rearrangeArgs):
    """
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
        

def showPics(data, key, fitParams=np.array([]), indvColorBars=False, colorMax=-1):
    num = len(data)
    gridsize1, gridsize2 = (0, 0)
    for i in range(100):
        if i*i >= num:
            gridsize1 = i
            gridsize2 = i-1 if i*(i-1) >= num else i
            break
    fig, plts = subplots(gridsize2, gridsize1, figsize=(15, 10))
    fig2, plts2 = subplots(gridsize2, gridsize1, figsize=(15, 10))
    rowCount, picCount, count = 0,0,0
    maximum, minimum = sorted(data.flatten())[colorMax], min(data.flatten())
    # get picture fits & plots
    for row in plts:
        for _ in row:
            plts[rowCount, picCount].grid(0)
            if count >= len(data):
                count += 1
                picCount += 1
                continue
            pic = data[count]
            if indvColorBars:
                maximum, minimum = max(pic.flatten()), min(pic.flatten())
            y, x = [np.linspace(1, pic.shape[i], pic.shape[i]) for i in range(2)]
            x, y = np.meshgrid(x, y)
            im = plts[rowCount, picCount].imshow( pic, origin='bottom', extent=(x.min(), x.max(), y.min(), y.max()),
                                                  vmin=minimum, vmax=maximum )
            plts[rowCount, picCount].axis('off')
            plts[rowCount, picCount].set_title(str(round_sig(key[count], 4)), fontsize=8)
            if fitParams.size != 0:
                if (fitParams[count] != np.zeros(len(fitParams[count]))).all():
                    try:
                        ellipse = Ellipse(xy=(fitParams[count][1], fitParams[count][2]),
                                          width=2*fitParams[count][3], height=2*fitParams[count][4],
                                          angle=-fitParams[count][5]*180/np.pi, edgecolor='r', fc='None', lw=2, alpha=0.2)
                        plts[rowCount, picCount].add_patch(ellipse)
                    except ValueError:
                        pass
                plts2[rowCount,picCount].grid(0)
                x, y = np.arange(0,len(pic[0])), np.arange(0,len(pic))
                X, Y = np.meshgrid(x,y)
                vals = np.reshape(gaussian_2d.f((X,Y), *fitParams[count]), pic.shape)
                im2 = plts2[rowCount,picCount].imshow(pic-vals, vmin=-2, vmax=2, origin='bottom',
                                                     extent=(x.min(), x.max(), y.min(), y.max()))
                plts2[rowCount,picCount].axis('off')
                plts2[rowCount,picCount].set_title(str(round_sig(key[count], 4)), fontsize=8)
            count += 1
            picCount += 1
        picCount = 0
        rowCount += 1
    # final touches
    for f in [fig, fig2]:
        cax = f.add_axes([0.95, 0.1, 0.03, 0.8])
        f.colorbar(im, cax=cax)
