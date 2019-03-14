
import time
from pandas import DataFrame
from MainAnalysis import standardPopulationAnalysis, analyzeNiawgWave, standardTransferAnalysis, standardAssemblyAnalysis, AnalyzeRearrangeMoves
from numpy import array as arr
from random import randint
from Miscellaneous import getColors, round_sig, round_sig_str, getMarkers, errString
import Miscellaneous as misc
from matplotlib.pyplot import *
import matplotlib as mpl
import mpl_toolkits.axes_grid1 as axesTool


from scipy.optimize import curve_fit as fit
from LoadingFunctions import loadDataRay, loadCompoundBasler, loadDetailedKey
from AnalysisHelpers import (processSingleImage, orderData,
                             normalizeData, getBinData, getTransferStats, getTransferEvents, fitDoubleGaussian,
                             guessGaussianPeaks, calculateAtomThreshold, getAvgPic, getEnsembleHits,
                             getEnsembleStatistics, handleFitting, processImageData,
                             fitPictures, fitGaussianBeamWaist, assemblePlotData, ballisticMotExpansion, simpleMotExpansion, 
                             calcMotTemperature,integrateData, computeMotNumber, getFitsDataFrame, genAvgDiscrepancyImage, 
                             getGridDims, newCalcMotTemperature)
import AnalysisHelpers as ah
import MarksConstants as consts 
from matplotlib.patches import Ellipse
from TimeTracker import TimeTracker
from fitters import LargeBeamMotExpansion, exponential_saturation
from fitters.Gaussian import gaussian_2d, double as double_gaussian

def rotateTicks(plot):
    ticks = plot.get_xticklabels()
    for tickInc in range(len(ticks)):
        ticks[tickInc].set_rotation(-45)
        

def makeThresholdStatsImages(ax, thresholds, locs, shape, ims, lims):
    thresholdList = [thresh.t for thresh in thresholds]
    thresholdPic, lims[0][0], lims[0][1] = genAvgDiscrepancyImage(thresholdList, shape, locs)
    ims.append(ax[0].imshow(thresholdPic, cmap=cm.get_cmap('seismic_r'), vmin=lims[0][0], vmax=lims[0][1], origin='lower'))
    ax[0].set_title('Thresholds:' + str(misc.round_sig(np.mean(thresholdList))), fontsize=12)
    
    fidList = [thresh.fidelity for thresh in thresholds]
    thresholdFidPic, lims[1][0], lims[1][1] = genAvgDiscrepancyImage(fidList, shape, locs)
    ims.append(ax[1].imshow(thresholdFidPic, cmap=cm.get_cmap('seismic_r'), vmin=lims[1][0], vmax=lims[1][1], origin='lower'))
    ax[1].set_title('Threshold Fidelities:' + str(misc.round_sig(np.mean(fidList))), fontsize=12)
        
    imagePeakDiff = []
    gaussFitList = [thresh.fitVals for thresh in thresholds]
    for g in gaussFitList:
        if g is not None:
            imagePeakDiff.append(abs(g[1] - g[4]))
        else:
            imagePeakDiff.append(0)
    peakDiffImage, lims[2][0], lims[2][1] = genAvgDiscrepancyImage(imagePeakDiff, shape, locs)
    ims.append(ax[2].imshow(peakDiffImage, cmap=cm.get_cmap('seismic_r'), vmin=lims[2][0], vmax=lims[2][1], origin='lower'))
    ax[2].set_title('Imaging-Signal:' + str(misc.round_sig(np.mean(imagePeakDiff))), fontsize=12)

    residualList = [thresh.rmsResidual for thresh in thresholds]
    residualImage, _, lims[3][1] = genAvgDiscrepancyImage(residualList, shape, locs)
    lims[3][0] = 0
    ims.append(ax[3].imshow(residualImage, cmap=cm.get_cmap('inferno'), vmin=lims[3][0], vmax=lims[3][1], origin='lower'))
    ax[3].set_title('Fit Rms Residuals:' + str(misc.round_sig(np.mean(residualList))), fontsize=12)
    
    for i, a in enumerate(ax[4:]):
        noData = np.zeros((25,23))
        lims[4+i][0], lims[4+i][1] = [0, 0]
        ims.append(a.imshow(noData, cmap=cm.get_cmap('Greys'), vmin=-1, vmax=0))
        a.set_title('(Nothing)')
    

def plotThresholdHists(thresholds, colors, extra=None, extraname=None, thresholds_2=None, shape=(10,10)):
    f, axs = subplots(shape[0], shape[1], figsize=(34.0, 16.0))
    if thresholds_2 is None:
        thresholds_2 = [None for _ in thresholds]
    for i, (t, t2, c) in enumerate(zip(thresholds, thresholds_2, colors[1:])):
        ax = axs[len(axs[0]) - i%len(axs[0]) - 1][int(i/len(axs))]
        ax.bar(t.binCenters, t.binHeights, align='center', width=t.binCenters[1] - t.binCenters[0], color=c)
        ax.axvline(t.t, color='w', ls=':')
        minx, maxx = min(t.binCenters), max(t.binCenters)
        maxy = max(t.binHeights)
        if t2 is not None:
            ax.plot(t2.binEdges(), t2.binEdgeHeights(), color='r', ls='steps')
            ax.axvline(t2.t, color='r', ls='-.')
            minx, maxx = min(list(t2.binCenters) + [minx]), max(list(t2.binCenters) + [maxx])
            maxy = max(list(t2.binHeights) + [maxy])
            if t2.fitVals is not None:
                xpts = np.linspace(min(t2.binCenters), max(t2.binCenters), 1000)
                ax.plot(xpts, double_gaussian.f(xpts, *t2.fitVals), 'r', ls='-.')
        if t.fitVals is not None:
            xpts = np.linspace(min(t.binCenters), max(t.binCenters), 1000)
            ax.plot(xpts, double_gaussian.f(xpts, *t.fitVals), 'w')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlim(minx, maxx)
        ax.set_ylim(0, maxy)
        ax.grid(False)
        if extra is not None:
            txt = extraname + round_sig_str( np.mean(extra[i])) if extraname is not None else round_sig_str(np.mean(extra[i]))
            t = ax.text( (maxx + minx) / 2, maxy / 2, txt, fontsize=12 )
            t.set_bbox(dict(facecolor='k', alpha=0.3))
    f.subplots_adjust(wspace=0, hspace=0)
    
        
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
    res = ah.temperatureAnalysis(data, magnification, key=key, loadType='basler',**standardImagesArgs)
    temp, fitVals, fitCov, times, waists, rawData, pictureFitParams, key = res
    errs = np.sqrt(np.diag(fitCov))
    f, ax = subplots(figsize=(20,3))
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
    
    
def plotMotNumberAnalysis(data, motKey, exposureTime,  *fillAnalysisArgs):
    """
    Calculate the MOT number and plot the data that resulted in the #.

    :param data: the number corresponding to the data set you want to analyze.
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
    rawData, intRawData, motnumber, fitParams, fluorescence, motKey = ah.motFillAnalysis(data, motKey, exposureTime, loadType='basler', *fillAnalysisArgs)
    figure(figsize=(20,3))
    plot(motKey, intRawData, 'bo', label='data', color='b')
    xfitPts = np.linspace(min(motKey), max(motKey), 1000)
    plot(xfitPts, exponential_saturation.f(xfitPts, *fitParams), 'b-', label='fit', color='r', linestyle=':')
    xlabel('loading time (s)')
    ylabel('integrated counts')
    title('Mot Fill Curve: MOT Number:' + str(motnumber))
    print("integrated saturated counts subtracting background =", -fitParams[0])
    print("loading time 1/e =", fitParams[1], "s")
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


def Transfer( fileNumber, atomLocs1_orig, atomLocs2_orig, show=True, legendOption=None,
              fitModules=[None], showFitDetails=False, showFitCenterPlot=False, showImagePlots=None, plotIndvHists=False, 
              timeit=False, transThresholdSame=False, outputThresholds=False, **standardTransferArgs ):
    """
    Standard data analysis package for looking at survival rates throughout an experiment.
    :return key, transferData, survivalErrors
    """
    avgColor='k'
    tt = TimeTracker()
    res = standardTransferAnalysis( fileNumber, atomLocs1_orig, atomLocs2_orig, fitModules=fitModules, tt=tt,
                                    transThresholdSame=transThresholdSame, **standardTransferArgs )
    tt.clock('After-Standard-Analysis')
    (atomLocs1, atomLocs2, transferData, transferErrs, initPopulation, pic1Data, keyName, key,
     repetitions, initThresholds, fits, avgTransferData, avgTransferErr, avgFit, avgPics, otherDimValues,
     locsList, genAvgs, genErrs, tt, transVarAvg, transVarErr, initAtomImages, transAtomImages,
     pic2Data, transPixelCounts, transThresholds, fitModules) = res
    if not show:
        return (key, transferData, transferErrs, initPopulation, fits, avgFit, genAvgs, genErrs, pic1Data, 
            gaussianFitVals, [None], initThresholds, [None])
    showImagePlots = showImagePlots if showImagePlots is not None else (False if len(atomLocs1) == 1 else True)
    legendOption = True if legendOption is None and len(atomLocs1) < 50 else False
    # set locations of plots.
    f = figure(figsize=(25.0, 8.0))
    typeName = "Survival" if atomLocs1 == atomLocs2 else "Transfer"
    grid1 = mpl.gridspec.GridSpec(12, 16,left=0.05, right=0.95, wspace=1.2, hspace=1000)
    mainPlot = subplot(grid1[:, :11])
    initPopPlot = subplot(grid1[0:3, 11:16])
    grid1.update( left=0.1, right=0.95, wspace=0, hspace=1000 )
    countPlot = subplot(grid1[4:8, 11:15])    
    grid1.update( left=0.001, right=0.95, hspace=1000 )
    countHist = subplot(grid1[4:8, 15:16], sharey=countPlot)
    avgPlt1 = subplot(grid1[8:12, 11:13])
    avgPlt2 = subplot(grid1[8:12, 13:15])
    if type(keyName) is not type("a string"):
        keyName = sum([kn+',' for kn in keyname])
    titletxt = (keyName + " " + typeName + " Point.\n Avg " + typeName + "% = " 
                + misc.dblErrString(np.mean(transVarAvg), np.sqrt(np.sum(arr(avgTransferErr)**2)/len(avgTransferErr)), 
                                    np.sqrt(np.sum(arr(transVarErr)**2)/len(transVarErr))))
    # some easily parallelizable stuff
    plotList = [mainPlot, initPopPlot, countPlot, avgPlt1, avgPlt2]
    xlabels = [keyName,keyName,'Picture #','','']
    ylabels = [typeName + " %","Initial Population %", "Camera Signal",'','']
    titles = [titletxt, "Initial Population: Avg$ = " + str(round_sig(np.mean(arr(initPopulation.tolist())))) + '$', "Thresh.=" + str(round_sig(np.mean([initThresholds[i].t for i in range(len(initThresholds))]))), '', '']
    majorYTicks = [np.arange(0,1,0.1),np.arange(0,1,0.2),np.linspace(min(pic1Data[0]),max(pic1Data[0]),5),[],[]]
    minorYTicks = [np.arange(0,1,0.05),np.arange(0,1,0.1),np.linspace(min(pic1Data[0]),max(pic1Data[0]),10),[],[]]
    majorXTicks = [key, key, np.linspace(0,len(pic1Data[0]),10), [],[]]
    grid_options = [True,True,True,False,False]
    fontsizes = [20,10,10,10,10]
    for subplt, xlbl, ylbl, title, yTickMaj, yTickMin, xTickMaj, fs, grid in zip(plotList, xlabels, ylabels, titles, majorYTicks, minorYTicks, majorXTicks, fontsizes, grid_options):
        subplt.set_xlabel(xlbl, fontsize=fs)
        subplt.set_ylabel(ylbl, fontsize=fs)
        subplt.set_title(title, fontsize=fs)
        subplt.set_yticks(yTickMaj)
        subplt.set_yticks(yTickMin, minor=True)
        subplt.set_xticks(xTickMaj)
        rotateTicks(subplt)
        subplt.grid(grid, color='#AAAAAA', which='Major')
        subplt.grid(grid, color='#090909', which='Minor')
        for item in ([subplt.title, subplt.xaxis.label, subplt.yaxis.label] + subplt.get_xticklabels() + subplt.get_yticklabels()):
            item.set_fontsize(fs)
    
    centers = []
    colors, colors2 = getColors(len(atomLocs1) + 1)
    longLegend = len(transferData[0]) == 1
    markers = getMarkers()
    
    # Main Plot
    for i, (atomLoc, fit, module) in enumerate(zip(atomLocs1, fits, fitModules)):
        leg = (r"[%d,%d] " % (atomLocs1[i][0], atomLocs1[i][1]) if typeName == "Survival" 
               else r"[%d,%d]$\rightarrow$[%d,%d] " % (atomLocs1[i][0], atomLocs1[i][1], atomLocs2[i][0], atomLocs2[i][1]))
        if longLegend:
            leg += (typeName + " % = " + str(round_sig(transferData[i][0])) + "$\pm$ " + str(round_sig(transferErrs[i][0])))
        unevenErrs = [[err[0] for err in transferErrs[i]], [err[1] for err in transferErrs[i]]]
        print(arr(unevenErrs).shape)
        print(transferData[i].shape)
        print(key.shape)
        mainPlot.errorbar(key, transferData[i], yerr=unevenErrs, color=colors[i], ls='',
                          capsize=6, elinewidth=3, label=leg, alpha=0.3, marker=markers[i%len(markers)], markersize=10)
        if module is not None and showFitDetails and fit['vals'] is not None:
            #if fitModules.center() is not None:
            centers.append(module.getCenter(fit['vals']))
            mainPlot.plot(fit['x'], fit['nom'], color=colors[i], alpha=0.5)
    print('hi')
    mainPlot.xaxis.set_label_coords(0.95, -0.1)
    if legendOption:
        mainPlot.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol = 4 if longLegend else 10, prop={'size': 12})
    # Init Population Plot
    for i, loc in enumerate(atomLocs1):
        initPopPlot.plot(key, initPopulation[i], ls='', marker='o', color=colors[i], alpha=0.3)
        initPopPlot.axhline(np.mean(initPopulation[i]), color=colors[i], alpha=0.3)
    # shared
    for plot in [mainPlot, initPopPlot]:
        if not min(key) == max(key):
            r = max(key) - min(key)
            plot.set_xlim(left = min(key) - r / len(key), right = max(key) + r / len(key))
        plot.set_ylim({0, 1})
    # ### Count Series Plot
    for i, loc in enumerate(atomLocs1):
        countPlot.plot(pic1Data[i], color=colors[i], ls='', marker='.', markersize=1, alpha=0.3)
        countPlot.axhline(initThresholds[i].t, color=colors[i], alpha=0.3)
    ticksForVis = countPlot.xaxis.get_major_ticks()
    ticksForVis[-1].label1.set_visible(False)
    # Count Histogram Plot
    for i, atomLoc in enumerate(atomLocs1):
        countHist.hist(pic1Data[i], 50, color=colors[i], orientation='horizontal', alpha=0.3, histtype='stepfilled')
        countHist.axhline(initThresholds[i].t, color=colors[i], alpha=0.3)
    setp(countHist.get_yticklabels(), visible=False)
    
    # average images
    for plt, dat, locs in zip([avgPlt1, avgPlt2], avgPics, [atomLocs1, atomLocs2]):
        plt.imshow(dat, origin='lower', cmap='Greys_r');
        for loc, c in zip(locs, colors):
            circ = Circle((loc[1], loc[0]), 0.2, color=c)
            plt.add_artist(circ)
    
    unevenErrs = [[err[0] for err in avgTransferErr], [err[1] for err in avgTransferErr]]
    (_, caps, _) = mainPlot.errorbar( key, avgTransferData, yerr=unevenErrs, color="#BBBBBB", ls='',
                       marker='o', capsize=12, elinewidth=5, label='Atom-Avg', markersize=10 )
    for cap in caps:
        cap.set_markeredgewidth(1.5)
    unevenErrs = [[err[0] for err in transVarErr], [err[1] for err in transVarErr]]
    (_, caps, _) = mainPlot.errorbar( key, transVarAvg, yerr=unevenErrs, color=avgColor, ls='',
                       marker='o', capsize=12, elinewidth=5, label='Event-Avg', markersize=10 )
    for cap in caps:
        cap.set_markeredgewidth(1.5)
    if fitModules[-1] is not None:
        mainPlot.plot(avgFit['x'], avgFit['nom'], color=avgColor, ls=':')
        for label, fitVal, err in zip(fitModules[-1].args(), avgFit['vals'], avgFit['errs']):
            print( label,':', errString(fitVal, err) )
        if showFitDetails:
            for f in getFitsDataFrame(fits, fitModules, avgFit):
                display(f)
    if fitModules[0] is not None and showFitCenterPlot:
        f, ax = subplots()
        fitCenterPic, vmin, vmax = genAvgDiscrepancyImage(centers, avgPics[0].shape, atomLocs1)
        im = ax.imshow(fitCenterPic, cmap=cm.get_cmap('seismic_r'), vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title('Fit-Centers (white is average)')
        ax.grid(False)
        f.colorbar(im)
    tt.clock('After-Main-Plots')
    avgTransferPic = None
    if showImagePlots:
        f, axs = subplots(2 if transThresholdSame else 3, 5, figsize=(36.0, 16.0))
        lims = [[None, None] for _ in range(10 if transThresholdSame else 15)]
        ims = [None for _ in range(5)]
        makeThresholdStatsImages(axs[1,:], initThresholds, atomLocs1, avgPics[0].shape, ims, lims[5:10])
        if not transThresholdSame:
            makeThresholdStatsImages(axs[2,:], transThresholds, atomLocs2, avgPics[0].shape, ims, lims[10:15])
              
        avgTransfers = [np.mean(s) for s in transferData]
        avgTransferPic, l20, l21 = genAvgDiscrepancyImage(avgTransfers, avgPics[0].shape, atomLocs1)
                
        avgPops = [np.mean(l) for l in initPopulation]
        avgInitPopPic, l30, l31 = genAvgDiscrepancyImage(avgPops, avgPics[0].shape, atomLocs1)
        
        if genAvgs is not None:
            genAtomAvgs = [np.mean(dp) for dp in genAvgs] if genAvgs[0] is not None else [0]
            genImage, _, l41 = genAvgDiscrepancyImage(genAtomAvgs, avgPics[0].shape, atomLocs1) if genAvgs[0] is not None else (np.zeros(avgPics[0].shape), 0, 1)

        images = [avgPics[0], avgPics[1], avgTransferPic, avgInitPopPic, genImage]
        lims[0:5] = [[min(avgPics[0].flatten()), max(avgPics[0].flatten())], [min(avgPics[1].flatten()), max(avgPics[1].flatten())], [l20,l21],[l30,l31],[0,l41]]
        cmaps = ['viridis', 'viridis', 'seismic_r','seismic_r','inferno']
        titles = ['Avg 1st Pic', 'Avg 2nd Pic', 'Avg Trans:' + str(misc.round_sig(np.mean(avgTransfers))), 'Avg Load:' + str(misc.round_sig(np.mean(avgPops))),'Atom-Generation: ' + str(misc.round_sig(np.mean(genAtomAvgs)))]
        for i, (ax, lim, image, cmap_) in enumerate(zip(axs.flatten(), lims, images, cmaps)):
            ims[i] = ax.imshow(image, vmin=lim[0], vmax=lim[1], origin='lower', cmap=cm.get_cmap(cmap_))
        for ax, lim, title, im in zip(axs.flatten(), lims, titles, ims):
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.grid(False)
            ax.set_title(title, fontsize=12)
            divider = axesTool.make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cb = f.colorbar(im, cax, orientation='vertical')
            cb.ax.tick_params(labelsize=8)
            for d in  im.get_array().flatten():
                p = (d - lim[0]) / (lim[1] - lim[0])
                cb.ax.plot( [0, 0.25], [p, p], color='w' )
            cb.outline.set_visible(False)
        tt.clock('After-Image-Plots')
    if plotIndvHists:
        if type(atomLocs1_orig[-1]) == int:
            print('hiiiii')
            shape = (atomLocs1_orig[-1], atomLocs1_orig[-2])
        else:
            print('what')
            shape = (10,10)
        plotThresholdHists(initThresholds, colors, extra=avgTransfers, extraname=r"$\rightarrow$:", thresholds_2=transThresholds, shape=shape)
        tt.clock('After-Indv-Hists')
    if timeit:
        tt.display()
    avgPlt1.set_position([0.58,0,0.3,0.3])
    avgPlt2.set_position([0.73,0,0.3,0.3])
    
    if outputThresholds:
        thresholdList = np.flip(np.reshape([t.t for t in initThresholds], (10,10)),1)
        with open('J:/Code-Files/T-File.txt','w') as f:
            for row in thresholdList:
                for thresh in row:
                    f.write(str(thresh) + ' ')
    
    return { 'Key':key, 'All_Transfer':transferData, 'All_Transfer_Errs':transferErrs, 'Initial_Populations':initPopulation, 'Transfer_Fits':fits, 'Average_Transfer_Fit':avgFit,
            'Average_Atom_Generation':genAvgs, 'Average_Atom_Generation_Err':genErrs, 'Picture_1_Data':pic1Data, 'Fit_Centers':centers, 
            'Average_Transfer_Pic':avgTransferPic, 'Transfer_Averaged_Over_Variations':transVarAvg, 'Transfer_Averaged_Over_Variations_Err':transVarErr, 'Average_Transfer':avgTransferData,
            'Average_Transfer_Err':avgTransferErr, 'Initial_Atom_Images':initAtomImages, 'Transfer_Atom_Images':transAtomImages, 'Picture_2_Data':pic2Data, 'Initial_Thresholds':initThresholds,
            'Transfer_Thresholds':transThresholds, 'Fit_Modules':fitModules }


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
    atomLocs_orig = atomLocations
    avgColor='w'
    res = standardPopulationAnalysis(fileNum, atomLocations, whichPic, picsPerRep, **StandardArgs)
    (locCounts, thresholds, avgPic, key, allPopsErr, allPops, avgPop, avgPopErr, fits,
     fitModules, keyName, atomData, rawData, atomLocations, avgFits, atomImages,
     totalAvg, totalErr) = res
    colors, _ = getColors(len(atomLocations) + 1)
    
    if not show:
        return key, allPops, allPopsErr, locCounts, atomImages, thresholds, avgPop
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
    popPlot = subplot(grid1[0:3, 12:16])
    countPlot = subplot(gridRight[4:8, 12:15])    
    if not histMain:
        mainPlot = subplot(grid1[:, :12])
        countHist = subplot(gridLeft[4:8, 15:16], sharey=countPlot)
    else:
        countHist = subplot(grid1[:, :12])
        mainPlot = subplot(gridLeft[4:8, 15:16], sharey=countPlot)
    centers = []
    longLegend = len(allPops[0]) == 1
    if len(arr(key).shape) == 2:
        # 2d scan: no normal plot possible, so make colormap plot of avg
        key1, key2 = key[:,0], key[:,1]
        key1 = np.sort(key1)
        key2 = np.sort(key2)
    else:
        for i, (atomLoc, fit, module) in enumerate(zip(atomLocations, fits, fitModules)):
            leg = r"[%d,%d] " % (atomLoc[0], atomLoc[1])
            if longLegend:
                pass
                #leg += (typeName + " % = " + str(round_sig(allPops[i][0])) + "$\pm$ "
                #        + str(round_sig(allPopsErr[i][0])))
                
            unevenErrs = [[err[0] for err in allPopsErr[i]], [err[1] for err in allPopsErr[i]]]
            mainPlot.errorbar(key, allPops[i], yerr=unevenErrs, color=colors[i], ls='',
                              capsize=6, elinewidth=3, label=leg, alpha=mainAlpha, marker=markers[i%len(markers)],markersize=5)
            if module is not None:
                if fit == [] or fit['vals'] is None:
                    continue
                centerIndex = module.center()
                if centerIndex is not None:
                    centers.append(fit['vals'][centerIndex])
                mainPlot.plot(fit['x'], fit['nom'], color=colors[i], alpha = 0.5)
        if fitModules[-1] is not None:
            if avgFits['vals'] is None:
                print('Avg Fit Failed!')
            else:
                centerIndex = fitModules[-1].center()
                mainPlot.plot(avgFits['x'], avgFits['nom'], color=avgColor, alpha = 1,markersize=5)
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
        if len(allPops[0]) == 1:
            titletxt = keyName + " Atom " + typeName + " Point.\n Avg " + typeName + "% = " + errString(totalAvg, totalErr) 
        mainPlot.set_title(titletxt, fontsize=30)
        mainPlot.set_ylabel("S %", fontsize=20)
        mainPlot.set_xlabel(keyName, fontsize=20)
        if legendOption == True:
            cols = 4 if longLegend else 10
            mainPlot.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=cols, prop={'size': 12})
    # Population Plot
    for i, loc in enumerate(atomLocations):
        popPlot.plot(key, allPops[i], ls='', marker='o', color=colors[i], alpha=0.3)
        popPlot.axhline(np.mean(allPops[i]), color=colors[i], alpha=0.3)
    popPlot.set_ylim({0, 1})
    if not min(key) == max(key):
        popPlot.set_xlim(left=min(key) - (max(key) - min(key)) / len(key), right=max(key) 
                             + (max(key) - min(key)) / len(key))
    popPlot.set_xlabel("Key Values")
    popPlot.set_ylabel("Population %")
    popPlot.set_xticks(key)
    popPlot.set_yticks(np.arange(0,1,0.1), minor=True)
    popPlot.set_yticks(np.arange(0,1,0.2))
    popPlot.grid(True, color='#AAAAAA', which='Major')
    popPlot.grid(True, color='#090909', which='Minor')
    popPlot.set_title("Population: Avg$ = " +  str(round_sig(np.mean(arr(allPops)))) + '$')
    for item in ([popPlot.title, popPlot.xaxis.label, popPlot.yaxis.label] +
                     popPlot.get_xticklabels() + popPlot.get_yticklabels()):
        item.set_fontsize(10)
    # ### Count Series Plot
    for i, loc in enumerate(atomLocations):
        countPlot.plot(locCounts[i], color=colors[i], ls='', marker='.', markersize=1, alpha=0.3)
        countPlot.axhline(thresholds[i].t, color=colors[i], alpha=0.3)

    countPlot.set_xlabel("Picture #")
    countPlot.set_ylabel("Camera Signal")
    countPlot.set_title("Thresh.=" + str(round_sig(thresholds[i].t)), fontsize=10) #", Fid.="
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
            countHist.axvline(thresholds[i].t, color=colors[i], alpha=0.3)            
        else:
            countHist.hist(locCounts[i], 50, color=colors[i], orientation='horizontal', alpha=0.3, histtype='stepfilled')
            countHist.axhline(thresholds[i].t, color=colors[i], alpha=0.3)
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
    avgPopErr = [[err[0] for err in avgPopErr], [err[1] for err in avgPopErr]]
    mainPlot.errorbar(key, avgPop, yerr=avgPopErr, color=avgColor, ls='',
             marker='o', capsize=6, elinewidth=3, label='Avg', markersize=5)
    if fitModules is not [None] and showFitDetails:
        mainPlot.plot(avgFit['x'], avgFit['nom'], color=avgColor, ls=':')
        fits_df = getFitsDataFrame(fits, fitModules, avgFit,markersize=5)
        display(fits_df)
    #elif fitModules is not [None]:
    #    for val, name in zip(avgFits['vals'], fitModule.args()):
    #        print(name, val)
    if fitModules is not [None] and showFitCenterPlot and fits[0] != []:
        figure()
        fitCenterPic, vmin, vmax = genAvgDiscrepancyImage(centers, avgPic.shape, atomLocations)
        imshow(fitCenterPic, cmap=cm.get_cmap('seismic_r'), vmin=vmin, vmax=vmax, origin='lower')
        title('Fit-Centers (white is average)')
        colorbar()
    if showImagePlots:
        ims = []
        lims = [[0,0] for _ in range(5)]
        f, axs = subplots(1,6)
        
        ims.append(axs[0].imshow(avgPic, origin='lower'))
        axs[0].set_title('Avg 1st Pic')

        avgPops = []
        for l in allPops:
            avgPops.append(np.mean(l))
        avgPopPic, vmin, vmax = genAvgDiscrepancyImage(avgPops, avgPic.shape, atomLocations)
        ims.append(axs[1].imshow(avgPopPic, cmap=cm.get_cmap('seismic_r'), vmin=vmin, vmax=vmax, origin='lower'))
        axs[1].set_title('Avg Population')
        
        makeThresholdStatsImages(axs[2:], thresholds, atomLocations, avgPic.shape, ims, lims)

        for ax, im in zip(axs, ims):
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.grid(False)
            divider = axesTool.make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            f.colorbar(im, cax, orientation='vertical')
    avgPops = []
    for s in allPops:
        avgPops.append(np.mean(s))
    if plotIndvHists:
        if type(atomLocs_orig[-1]) == int:
            shape = (atomLocs_orig[-1], atomLocs_orig[-2])
        else:
            shape = (10,10)
        plotThresholdHists(thresholds, colors, extra=avgPops, extraname="L:", shape=shape)
    """
    # output thresholds
    thresholds = np.flip(np.reshape(thresholds, (10,10)),1)
    with open('J:/Code-Files/T-File.txt','w') as f:
        for row in thresholds:
            for thresh in row:
                f.write(str(thresh) + ' ') 
    """
    return { 'Key': key, 'All_Populations': allPops, 'All_Populations_Error': allPopsErr, 'Pixel_Counts':locCounts, 'Atom_Images':atomImages, 
             'Thresholds':thresholds, 'Atom_Data':atomData, 'Raw_Data':rawData, 'Average_Population': avgPop, 'Average_Population_Error': avgPopErr }


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
        mainPlot.errorbar( key, stats['avg'], yerr=stats['err'], color=c, ls='',
                           marker='o', capsize=6, elinewidth=3, label=label )
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
        if i*(i-2) >= num:
            gridsize1 = i
            gridsize2 = i-2# if i*(i-1) >= num else i
            break
    fig = figure(figsize=(20,20))
    grid = axesTool.AxesGrid( fig, 111, nrows_ncols=(2, num), axes_pad=0.0, share_all=True,
                              label_mode="L", cbar_location="right", cbar_mode="single" )
    rowCount, picCount, count = 0,0,0
    maximum, minimum = sorted(data.flatten())[colorMax], min(data.flatten())
    # get picture fits & plots
    for picNum in range(num):
        pl = grid[count]
        pl.grid(0)
        if count >= len(data):
            count += 1
            picCount += 1
            continue
        pic = data[count]
        if indvColorBars:
            maximum, minimum = max(pic.flatten()), min(pic.flatten())
        y, x = [np.linspace(1, pic.shape[i], pic.shape[i]) for i in range(2)]
        x, y = np.meshgrid(x, y)
        im1 = pl.imshow( pic, origin='bottom', extent=(x.min(), x.max(), y.min(), y.max()),
                                              vmin=minimum, vmax=maximum )
        pl.axis('off')
        pl.set_title(str(round_sig(key[count], 4)), fontsize=8)
        if fitParams.size != 0:
            if (fitParams[count] != np.zeros(len(fitParams[count]))).all():
                try:
                    ellipse = Ellipse(xy=(fitParams[count][1], fitParams[count][2]),
                                      width=2*fitParams[count][3], height=2*fitParams[count][4],
                                      angle=-fitParams[count][5]*180/np.pi, edgecolor='r', fc='None', lw=2, alpha=0.2)
                    pl.add_patch(ellipse)
                except ValueError:
                    pass
            pl2 = grid[count+num]
            pl2.grid(0)
            x, y = np.arange(0,len(pic[0])), np.arange(0,len(pic))
            X, Y = np.meshgrid(x,y)
            vals = np.reshape(gaussian_2d.f((X,Y), *fitParams[count]), pic.shape)
            im2 = pl2.imshow(pic-vals, vmin=-2, vmax=2, origin='bottom',
                                                 extent=(x.min(), x.max(), y.min(), y.max()))
            pl2.axis('off')
        count += 1
        picCount += 1
    grid.cbar_axes[0].colorbar(im1);
