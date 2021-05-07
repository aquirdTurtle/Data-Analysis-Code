import time
from pandas import DataFrame
from numpy import array as arr
from random import randint
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import mpl_toolkits.axes_grid1 as axesTool
import mpl_toolkits.axes_grid1
from scipy.optimize import curve_fit as fit
from matplotlib.patches import Ellipse
import IPython
import IPython.display as disp

from . import MainAnalysis as ma
from . import TransferAnalysis
from . import Miscellaneous as misc

from .MainAnalysis import analyzeNiawgWave, standardAssemblyAnalysis, AnalyzeRearrangeMoves
from .LoadingFunctions import loadDataRay, loadCompoundBasler, loadDetailedKey
from .AnalysisHelpers import (processSingleImage, orderData,
                              normalizeData, getBinData, fitDoubleGaussian,
                              guessGaussianPeaks, calculateAtomThreshold, getAvgPic, getEnsembleHits,
                              getEnsembleStatistics, processImageData,
                              fitPictures, fitGaussianBeamWaist, integrateData, 
                              computeMotNumber, getFitsDataFrame, genAvgDiscrepancyImage, getGridDims)

from . import TransferAnalysisOptions as tao
from . import ThresholdOptions as to
from . import AnalysisHelpers as ah
from . import MarksConstants as mc 
from . import PopulationAnalysis as pa 
from .TimeTracker import TimeTracker
from .fitters import LargeBeamMotExpansion, exponential_saturation
from .fitters.Gaussian import gaussian_2d, double as double_gaussian, bump
from . import ExpFile as exp

def addAxColorbar(fig, ax, im):
    cax = mpl_toolkits.axes_grid1.make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

def makeAvgPlts(avgPlt1, avgPlt2, avgPics, analysisOpts, colors):        
    print("Making Avg Plots...")
    # Average Images
    for plt, dat, locs in zip([avgPlt1, avgPlt2], avgPics, [analysisOpts.initLocs(), analysisOpts.tferLocs()]):
        plt.imshow(dat, origin='lower', cmap='Greys_r');
    # first, create the total list
    markerTotalList = []
    for psc_s, prc, color in zip(analysisOpts.postSelectionConditions, analysisOpts.positiveResultConditions, colors ):    
        for psc in psc_s:
            assert(len(psc.markerWhichPicList) == len(psc.markerLocList))
            for markernum, pic in enumerate(psc.markerWhichPicList):
                markerTotalList.append((pic, psc.markerLocList[markernum]))
        if prc is not None:
            for markernum, pic in enumerate(prc.markerWhichPicList):
                markerTotalList.append((pic, prc.markerLocList[markernum]))
    markerRunningList = []
    markerLocModList = ((0.25,0.25),(0.25,-0.25),(-0.25,-0.25),(-0.25,0.25))
    for psc_s, prc, color in zip(analysisOpts.postSelectionConditions, analysisOpts.positiveResultConditions, colors ):    
        for psc in psc_s:
            for markernum, pic in enumerate(psc.markerWhichPicList):
                markerRunningList.append((pic, psc.markerLocList[markernum]))
                loc = locs[psc.markerLocList[markernum]]
                if markerTotalList.count((pic, psc.markerLocList[markernum])) == 1:                    
                    circ = Circle((loc[1], loc[0]), 0.2, color=color)
                else:                    
                    circNum = markerTotalList.count((pic, psc.markerLocList[markernum])) \
                              - markerRunningList.count((pic, psc.markerLocList[markernum]))
                    circ = Circle((loc[1]+markerLocModList[circNum][0], loc[0]+markerLocModList[circNum][1]), 0.2, color=color)
                [avgPlt1,avgPlt2][pic].add_artist(circ)
        if prc is not None:
            for markernum, pic in enumerate(prc.markerWhichPicList):
                markerRunningList.append((pic, prc.markerLocList[markernum]))
                loc = locs[prc.markerLocList[markernum]]
                circ = Circle((loc[1], loc[0]), 0.2, color=color)
                [avgPlt1,avgPlt2][pic].add_artist(circ)
    if avgPics[0].shape[0]/avgPics[0].shape[1] < 0.3:
        avgPlt1.set_position([0.7,0.15,0.22,0.12])
        avgPlt2.set_position([0.7,0,0.22,0.12])
    else:
        avgPlt1.set_position([0.68,0,0.14,0.3])
        avgPlt2.set_position([0.83,0,0.14,0.3])
    avgPlt1.set_title('Avg Pic #1')
    avgPlt2.set_title('Avg Pic #2')

    
def fancyImshow( fig, ax, image, avgSize='20%', pad_=0, cb=True, imageArgs={}, hAvgArgs={'color':'orange'}, vAvgArgs={'color':'orange'}, 
                 ticklabels=True,do_vavg=True, do_havg=True, hFitParams=None, vFitParams=None, 
                 subplotsAdjustArgs=dict(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0), 
                 fitModule=bump, flipVAx = False, fitParams2D=None):
    """
    Expand a normal image plot of the image input, giving it a colorbar, a horizontal-averaged step plot to the left
    and a vertically-averaged step plot below.
    @param fig:
    @param ax:
    @param image:
    @param avgSize: size of the averaged step plots
    @param pad_: recommended values: 0 or 0.3 to see the ticks on the image as well as the step plots.
    """
    fig.subplots_adjust(**subplotsAdjustArgs)
    im = ax.imshow(image, **imageArgs)
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    #ax.axis('off')
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = hax = vax = None
    hAvg, vAvg = ah.collapseImage(image)
    if cb:
        cax = divider.append_axes('right', size='5%', pad=0)
        fig, colorbar(im, cax, orientation='vertical')
    if do_vavg:
        vAvg = [vAvg[0]] + list(vAvg)
        vax = divider.append_axes('bottom', size=avgSize, pad=pad_)
        vline = vax.step(np.arange(len(vAvg)), vAvg, **vAvgArgs)
        vax.set_yticks([])
        if not ticklabels:
            vax.set_xticks([])
        if vFitParams is not None:
            fxpts = np.linspace(0, len(vAvg), 1000)
            fypts = fitModule.f(fxpts, *vFitParams)
            vax.plot(fxpts+0.5,fypts)
        vax.set_xlim(0,len(vAvg)-1)
    if do_havg:
        # subtle difference here from the vAvg case
        hAvg = list(hAvg) + [hAvg[-1]]
        hax = divider.append_axes('left', size=avgSize, pad=pad_)
        hline = hax.step(hAvg, np.arange(len(hAvg)),**hAvgArgs)
        hax.set_ylim(0,len(hAvg)-1)
        hax.set_xticks([])
        if not ticklabels:
            hax.set_yticks([])
        if hFitParams is not None:
            fxpts = np.linspace(0, len(hAvg), 1000)
            fypts = fitModule.f(fxpts, *hFitParams)
            hax.plot(fypts, fxpts)
        if flipVAx:
            hax.invert_yaxis()
    if fitParams2D is not None:
        x = np.arange(len(image[0]))
        y = np.arange(len(image))
        X, Y = np.meshgrid(x,y)
        data_fitted = gaussian_2d.f_notheta((X,Y), *fitParams2D)
        ax.contour(x, y, data_fitted.reshape(image.shape[0],image.shape[1]), 
                   levels=np.linspace(min(data_fitted),max(data_fitted),4), colors='w', alpha=0.2)
    return ax, cax, hax, vax, hAvg, vAvg, im, vline, hline

def rotateTicks(plot):
    ticks = plot.get_xticklabels()
    for tickInc in range(len(ticks)):
        ticks[tickInc].set_rotation(-45)
        
def imageTickedColorbar(f, im, ax, lim):
    divider = axesTool.make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = f.colorbar(im, cax, orientation='vertical')
    cb.ax.tick_params(labelsize=8)
    if lim[1] - lim[0] == 0:
        return
    for d in  im.get_array().flatten():
        p = (d - lim[0]) / (lim[1] - lim[0])
        cb.ax.plot( [0, 0.25], [p, p], color='w' )
    cb.outline.set_visible(False)
        
def makeThresholdStatsImages(ax, thresholds, locs, shape, ims, lims, fig):
    thresholdList = [thresh.t for thresh in thresholds]
    thresholdPic, lims[0][0], lims[0][1] = genAvgDiscrepancyImage(thresholdList, shape, locs)
    ims.append(ax[0].imshow(thresholdPic, cmap=cm.get_cmap('seismic_r'), vmin=lims[0][0], vmax=lims[0][1], origin='lower'))
    ax[0].set_title('Thresholds:' + str(misc.round_sig(np.mean(thresholdList))), fontsize=12)
    imageTickedColorbar(fig, ims[-1], ax[0], lims[0])
    
    fidList = [thresh.fidelity for thresh in thresholds]
    thresholdFidPic, lims[1][0], lims[1][1] = genAvgDiscrepancyImage(fidList, shape, locs)
    ims.append(ax[1].imshow(thresholdFidPic, cmap=cm.get_cmap('seismic_r'), vmin=lims[1][0], vmax=lims[1][1], origin='lower'))
    ax[1].set_title('Threshold Fidelities:' + str(misc.round_sig(np.mean(fidList))), fontsize=12)
    imageTickedColorbar(fig, ims[-1], ax[1], lims[1])
    
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
    imageTickedColorbar(fig, ims[-1], ax[2], lims[2])

    residualList = [thresh.rmsResidual for thresh in thresholds]
    residualImage, _, lims[3][1] = genAvgDiscrepancyImage(residualList, shape, locs)
    lims[3][0] = 0
    ims.append(ax[3].imshow(residualImage, cmap=cm.get_cmap('inferno'), vmin=lims[3][0], vmax=lims[3][1], origin='lower'))
    ax[3].set_title('Fit Rms Residuals:' + str(misc.round_sig(np.mean(residualList))), fontsize=12)
    imageTickedColorbar(fig, ims[-1], ax[3], lims[3])
    for a in ax:
        a.tick_params(axis='both', which='major', labelsize=8)
    return imagePeakDiff


def plotThresholdHists( thresholds, colors, extra=None, extraname=None, thresholds_2=None, shape=(10,10), title='', minx=None, maxx=None,
                        localMinMax=False, detailColor='k' ):
    fig, axs = subplots(shape[0], shape[1], figsize=(25.0, 3.0))
    if thresholds_2 is None:
        thresholds_2 = [None for _ in thresholds]
    binCenterList = [th.binCenters for th in thresholds] + [th.binCenters if th is not None else [] for th in thresholds_2]
    binCenterList = [item for sublist in binCenterList for item in sublist]
    binCenterList = [item for item in binCenterList if item] 
    if minx is None:
        minx = min(binCenterList)
    if maxx is None:
        maxx = max(binCenterList)
    for tnum, (th, th2, color) in enumerate(zip(thresholds, thresholds_2, colors[1:])):
        if len(axs.shape) == 2:
            ax = axs[len(axs[0]) - tnum%len(axs[0]) - 1][int(tnum/len(axs))]
        else:
            ax = axs[tnum]
        ax.bar(th.binCenters, th.binHeights, align='center', width=th.binCenters[1] - th.binCenters[0], color=color)
        #ax.semilogy(th.binCenters, th.binHeights, color=c)
        ax.axvline(th.t, color=detailColor, ls=':')
        if localMinMax:
            minx, maxx = min(th.binCenters), max(th.binCenters)
        maxy = max(th.binHeights)
        if th2 is not None:
            ax.plot(th2.binEdges(), th2.binEdgeHeights(), color='r')
            ax.axvline(th2.t, color='r', ls='-.')
            if localMinMax:
                minx, maxx = min(list(th2.binCenters) + [minx]), max(list(th2.binCenters) + [maxx])
            maxy = max(list(th2.binHeights) + [maxy])
            if th2.fitVals is not None:
                xpts = np.linspace(min(th2.binCenters), max(th2.binCenters), 1000)
                ax.plot(xpts, double_gaussian.f(xpts, *th2.fitVals), color='r', ls='-.')
        if th.fitVals is not None:
            xpts = np.linspace(min(th.binCenters), max(th.binCenters), 1000)
            ax.plot(xpts, double_gaussian.f(xpts, *th.fitVals), detailColor)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlim(minx, maxx)
        ax.set_ylim(0.1, maxy)
        ax.grid(False)
        if extra is not None:
            txt = extraname + misc.round_sig_str( np.mean(extra[tnum])) if extraname is not None else misc.round_sig_str(np.mean(extra[tnum]))
            axtxt = ax.text( (maxx + minx) / 2, maxy / 2, txt, fontsize=12 )
            axtxt.set_bbox(dict(facecolor=detailColor, alpha=0.3))
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.suptitle(title, fontsize=24)
    return fig

def plotNiawg(fileIndicator, points=300, plotTogether=True, plotVolts=False):
    """
    I have not used this function in a while. August 25th, 2019
    
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


def plotImages( data, mainPlot='fits', key=None, magnification=3, showAllPics=True, plottedData=None, loadType='basler', fitPics=True, **standardImagesArgs ):
    res = ma.standardImages( data, loadType=loadType, fitPics=fitPics, manualAccumulation=True, quiet=True, key=key, plottedData=plottedData, 
                             **standardImagesArgs )
    (key, rawData, dataMinusBg, dataMinusAvg, avgPic, pictureFitParams, pictureFitErrors, plottedData, v_params, v_errs, h_params, h_errs, intRawData) = res
    # convert to meters
    if fitPics:
        waists = 2 * mc.baslerScoutPixelSize * np.sqrt((pictureFitParams[:, 3]**2+pictureFitParams[:, 4]**2)/2) * magnification
        # convert to s
        errs = np.sqrt(np.diag(pictureFitErrors))
    f, ax = subplots(figsize=(20,3))
    if mainPlot=='fits':
        ax.plot(key, waists, 'bo', label='Fit Waist')
        ax.yaxis.label.set_color('c')
        ax.grid( True, color='b' )
        ax2 = ax.twinx()
        ax2.plot(key, pictureFitParams[:, 0], 'ro:', marker='*', label='Fit Amp (counts)')
        ax2.yaxis.label.set_color( 'r' )
        ax.set_title('Measured atom cloud size over time')
        ax.set_ylabel('Gaussian fit waist (m)')
        ax.legend(loc='right')
        ax2.legend(loc='lower center')
        ax2.grid(True,color='r')
    elif mainPlot=='counts':
        ax.plot(key, intRawData, 'bo', label='Integrated Counts')
        ax.yaxis.label.set_color('c')
        ax.grid( True, color='b' )
        ax.set_ylabel('Integrated Counts')
        ax.legend(loc='right')        
    if showAllPics:
        if 'raw' in plottedData:
            picFig = showPics(rawData, key, fitParams=pictureFitParams, hFitParams=h_params, vFitParams=v_params)
        if '-bg' in plottedData:
            picFig = showPics(dataMinusBg, key, fitParams=pictureFitParams, hFitParams=h_params, vFitParams=v_params)
    return pictureFitParams, rawData, intRawData
    
    
def plotMotTemperature(data, key=None, magnification=3, showAllPics=True, temperatureGuess=100e-6, plot1D=False, **standardImagesArgs):
    """
    Calculate the mot temperature, and plot the data that led to this.
    :param data:
    :param standardImagesArgs: see the standardImages function to see the acceptable arguments here.
    :return:
    """
    res = ah.temperatureAnalysis(data, magnification, key=key, loadType='basler',temperatureGuess=temperatureGuess, **standardImagesArgs)
    (temp, fitVals, fitCov, times, waists, rawData, pictureFitParams, key, plottedData, dataMinusBg, 
     v_params, v_errs, h_params, h_errs, waists_1D, temp_1D, fitVals_1D, fitCov_1D) = res
    errs = np.sqrt(np.diag(fitCov))
    errs_1D = np.sqrt(np.diag(fitCov_1D))
    fig, ax = plt.subplots(figsize=(20,3))
    ax.plot(times, waists, 'bo', label='Fit Waist 2D')
    ax.plot(times, 2 * LargeBeamMotExpansion.f(times, *fitVals), 'c:', label='Balistic Expansion Waist 2D')
    
    if plot1D:
        ax.plot(times, waists_1D, 'go', label='x-Fit Waist 1D')
        ax.plot(times, 2 * LargeBeamMotExpansion.f(times, *fitVals_1D), 'w:', label='X-Balistic Expansion Waist 1D')
    
    ax.yaxis.label.set_color('c')
    ax.grid(True,color='b')

    ax2 = ax.twinx()
    ax2.plot(times, pictureFitParams[:, 0], 'ro:', marker='*', label='Fit Amp (counts)')
    ax2.yaxis.label.set_color('r')
    ax.set_title('Measured atom cloud size over time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Gaussian fit waist (m)')
    ax.legend(loc='right')
    ax2.legend(loc='lower center')
    ax2.grid(True,color='r')
    if showAllPics:
        if 'raw' in plottedData:
            picFig = showPics(rawData, key, fitParams=pictureFitParams, hFitParams=h_params, vFitParams=v_params)
        if '-bg' in plottedData:
            picFig = showPics(dataMinusBg, key, fitParams=pictureFitParams, hFitParams=h_params, vFitParams=v_params)    
    print("\nTemperture in the Large Laser Beam Approximation (2D Fits):", misc.errString(temp * 1e6, errs[2]*1e6),    'uK')
    print("\nTemperture in the Large Laser Beam Approximation (1D Fits):", misc.errString(temp_1D * 1e6, errs_1D[2]*1e6), 'uK')
    print('2D Fit-Parameters:', fitVals)
    return pictureFitParams, rawData, temp * 1e6, errs[2]*1e6, [fig, picFig]
    
    
def plotMotNumberAnalysis(data, motKey, exposureTime,  **fillAnalysisArgs):
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
    (rawData, intRawData, motnumber, fitParams, fluorescence, motKey, fitErr)\
        = ah.motFillAnalysis(data, motKey, exposureTime, loadType='basler', **fillAnalysisArgs)
    fig = plt.figure(figsize=(20,5))
    ax1 = subplot2grid((1, 4), (0, 0), colspan=3)
    ax2 = subplot2grid((1, 4), (0, 3), colspan=1)
    ax1.plot(motKey, intRawData, 'bo', label='data', color='b')
    xfitPts = np.linspace(min(motKey), max(motKey), 1000)
    ax1.plot(xfitPts, exponential_saturation.f(xfitPts, *fitParams), 'b-', label='fit', color='r', linestyle=':')
    ax1.set_xlabel('loading time (s)')
    ax1.set_ylabel('integrated counts')
    ax1.set_title('Mot Fill Curve: MOT Number:' + str(motnumber))
    res = fancyImshow(fig, ax2, rawData[-1])
    res[0].set_title('Final Image')
    print("integrated saturated counts subtracting background =", -fitParams[0])
    print("loading time 1/e =", fitParams[1], "s")
    print('Light Scattered off of full MOT:', fluorescence * mc.h * mc.Rb87_D2LineFrequency * 1e9, "nW")
    return motnumber, fitParams[1], rawData[-1], fitErr[1], rawData, fig


def singleImage(data, accumulations=1, loadType='andor', bg=arr([0]), title='Single Picture', window=(0, 0, 0, 0),
                xMin=0, xMax=0, yMin=0, yMax=0, zeroCorners=False, smartWindow=False, findMax=False,
                manualAccumulation=False, maxColor=None, key=arr([])):
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


def Survival(fileID, atomLocs, **TransferArgs):
    """
    Survival is a special case of transfer where the "transfer" is to the original location.
    
    :param fileNumber:
    :param atomLocs:
    :param TransferArgs: See corresponding transfer function for valid TransferArgs.
    :return: see Transfer()
    """
    return Transfer(fileID, tao.getStandardSurvivalOptions(atomLocs), atomLocs, **TransferArgs)

def Tunneling( fileID, tunnelPair1Locs, tunnelPair2Locs, dataColor=['#FF0000','#FFAAAA','#AAAAFF','#0000AA', 'k', '#00FF00', '#00AA00'], 
               plotAvg=False, postSelectOnSurvival=True, includeSurvival=True, 
               showFitDetails=True, includeHOM=True, **transferArgs ):
    """
    A small wrapper for doing tunneling analysis. 
    :return: see Transfer()
    """
    t1locs = ah.unpackAtomLocations(tunnelPair1Locs)
    t2locs = ah.unpackAtomLocations(tunnelPair2Locs)
    assert(len(t1locs)==len(t2locs))
    numPairs = len(t1locs)
    
    initLocs = t1locs + t2locs
    tferLocs = t1locs + t2locs
    numConditions = 4*numPairs + int(includeHOM) + 2*int(includeSurvival)
    psConditions = [[] for _ in range(numConditions)]
    prConditions = [None for _ in range(numConditions)]
    for pairNum in range(numPairs):
        singleLoad = tao.condition(name="LoadL", whichPic=[0,0], whichAtoms=[pairNum,pairNum+numPairs],
                                           conditions=[True,False],numRequired=-1, markerWhichPicList=(0,), markerLocList=(pairNum,))
        otherSingleLoad = tao.condition(name="LoadR", whichPic=[0,0],whichAtoms=[pairNum, pairNum+numPairs],
                                                conditions=[False,True], numRequired=-1, markerWhichPicList=(0,), markerLocList=(pairNum+numPairs,))      
        survival = tao.condition(name="Sv.", whichPic=[1,1],whichAtoms=[pairNum,pairNum+numPairs],
                                         conditions=[True,True], numRequired=1)
        loadBoth = tao.condition(name="LoadB", whichPic=[0,0],whichAtoms=[pairNum,pairNum+numPairs],
                                         conditions=[True,True], numRequired=2, markerWhichPicList=(0,0), markerLocList=(pairNum,pairNum+numPairs,))
        bothOrNoneSurvive = tao.condition(name="evenPSv",whichPic=[1,1],whichAtoms=[pairNum,pairNum+numPairs],
                                         conditions=[True,True],numRequired=[0,2])
        psConditions[pairNum].append(singleLoad)
        psConditions[pairNum+numPairs].append(singleLoad)
        psConditions[pairNum+2*numPairs].append(otherSingleLoad)
        psConditions[pairNum+3*numPairs].append(otherSingleLoad)
        if postSelectOnSurvival:
            psConditions[pairNum].append(survival)
            psConditions[pairNum+numPairs].append(survival)
            psConditions[pairNum+2*numPairs].append(survival)        
            psConditions[pairNum+3*numPairs].append(survival)
            if includeHOM:
                psConditions[pairNum+4*numPairs].append(bothOrNoneSurvive)
        if includeHOM:
            psConditions[pairNum+4*numPairs].append(loadBoth)
        if includeSurvival:
            psConditions[pairNum+5*numPairs].append(singleLoad)
            psConditions[pairNum+6*numPairs].append(otherSingleLoad)
        prc1 = tao.condition(name="FinL", whichPic=[1],whichAtoms=[pairNum],conditions=[True],numRequired=-1,
                            markerWhichPicList=(1,), markerLocList=(pairNum,))
        prc2 = tao.condition(name="FinR", whichPic=[1], whichAtoms=[pairNum+numPairs],conditions=[True],numRequired=-1,
                            markerWhichPicList=(1,), markerLocList=(pairNum+numPairs,))        
        prcHom = tao.condition(name="P11",whichPic=[1,1],whichAtoms=[pairNum,pairNum+numPairs],conditions=[True,True],numRequired=2,
                              markerWhichPicList=(1,1), markerLocList=(pairNum, pairNum+numPairs,))
        assert(pairNum <= 1 and pairNum+numPairs <= 1)
        prConditions[pairNum] = prc1
        prConditions[pairNum+numPairs] = prc2
        prConditions[pairNum+2*numPairs] = prc1
        prConditions[pairNum+3*numPairs] = prc2
        if includeHOM:
            prConditions[pairNum+4*numPairs] = prcHom
        if includeSurvival:
            prConditions[pairNum+5*numPairs] = survival
            prConditions[pairNum+6*numPairs] = survival
    res = Transfer(fileID, tao.TransferAnalysisOptions(initLocs, tferLocs, postSelectionConditions=psConditions,
                                                       positiveResultConditions=prConditions), 
                    dataColor=dataColor, plotAvg=plotAvg, showFitDetails=showFitDetails, **transferArgs)
    # currently no smarter way of handling this other than doing it manually like this.
    if includeSurvival and includeHOM:
        ax = res['Main_Axis']
        data = [np.array(dset) for dset in res['All_Transfer']]
        SP1 = data[0]
        SP2 = data[3]
        S1 = data[5]
        S2 = data[6]

        term1 = SP1*SP2 + (1-SP1)*(1-SP2)
        fraction2 = S1*S2/(S1*S2+(1-S1)*(1-S2))

        hompredict = term1 * fraction2

        dpdsp1 = (SP2-(1-SP2))*fraction2
        dpdsp2 = (SP1-(1-SP1))*fraction2
        dpdp1 = term1*((S1*S2+(1-S1)*(1-S2))*S2-S1*S2*(S2-(1-S2)))/(S1*S2+(1-S1)*(1-S2))**2
        dpdp2 = term1*((S1*S2+(1-S1)*(1-S2))*S1-S1*S2*(S1-(1-S1)))/(S1*S2+(1-S1)*(1-S2))**2
        homErrs = []
        for errnum in range(2):
            errs = [np.array([err[errnum] for err in dset]) for dset in res['All_Transfer_Errs']]    
            homErrs.append(np.sqrt(dpdsp1**2*errs[0]**2+dpdsp2**2*errs[3]**2+dpdp1**2*errs[5]**2+dpdp2**2*errs[6]**2))

        ax.errorbar(res['Key'], hompredict,yerr=[homErrs[0],homErrs[1]], marker='o', linestyle='', 
                    color='purple', markersize=15, label='P11 Pred', capsize=5)
        #ax.plot(res['Key'], hompredict, marker='o', linestyle='', 
        #            color='purple', markersize=15, label='P11 Prediction')

    return res
    
def Transfer( fileNumber, anaylsisOpts, show=True, legendOption=None, fitModules=[None], 
              showFitDetails=False, showFitCharacterPlot=False, showImagePlots=None, plotIndvHists=False, 
              timeit=False, outputThresholds=False, plotFitGuess=False, newAnnotation=False, 
              plotImagingSignal=False, expFile_version=4, plotAvg=True, countMain=False, histMain=False,
              flattenKeyDim=None, forceNoAnnotation=False, cleanOutput=True, dataColor='gist_rainbow', dataEdgeColors=None,
              tOptions=[to.ThresholdOptions()], resInput=None, countRunningAvg=None, **standardTransferArgs ):
    """
    Standard data analysis function for looking at survival rates throughout an experiment. I'm very bad at keeping the 
    function argument descriptions up to date.
    """
    avgColor='k'
    tt = TimeTracker()
    if resInput is None:
        try:
            res = TransferAnalysis.standardTransferAnalysis( fileNumber, anaylsisOpts, fitModules=fitModules, 
                                                             expFile_version=expFile_version, tOptions=tOptions, 
                                                            **standardTransferArgs )
        except OSError as err:
            if (str(err) == "Unable to open file (bad object header version number)"):
                print( "Unable to open file! (bad object header version number). This is usually a sign that the experiment "
                       "is still in progress, or that the experiment hasn't closed the hdf5 file.")
            else:
                print("OSError! Exception: " + str(err))
            return
    else: 
        res = resInput
    tt.clock('After-Standard-Analysis')
    (transferData, transferErrs, initPopulation, pic1Data, keyName, key, repetitions, initThresholds, 
     fits, avgTransferData, avgTransferErr, avgFit, avgPics, genAvgs, genErrs, transVarAvg, transVarErr, 
     initAtomImages, transAtomImages, pic2Data, transThresholds, fitModules, basicInfoStr, ensembleHits, 
     tOptions, analysisOpts, initAtoms, tferAtoms, tferList) = res
    print('key:',key)
    if flattenKeyDim != None:
        key = key[:,flattenKeyDim]
    showImagePlots = showImagePlots if showImagePlots is not None else (False if analysisOpts.numAtoms() == 1 else True)
    legendOption = True if legendOption is None and analysisOpts.numAtoms() < 50 else False
    # set locations of plots.
    fig = figure(figsize=(25.0, 8.0))
    typeName = "Survival" if analysisOpts.initLocs() == analysisOpts.tferLocs() else "Transfer"
    grid1 = mpl.gridspec.GridSpec(12, 16,left=0.05, right=0.95, wspace=1.2, hspace=1)
    if countMain:
        countPlot = subplot(grid1[:, :11])
    elif histMain:
        countHist = subplot(grid1[:, :11])
    else:
        mainPlot = subplot(grid1[:, :11])
    initPopPlot = subplot(grid1[0:3, 12:16])
    grid1.update( left=0.1, right=0.95, wspace=0, hspace=1000 )
    if countMain:
        mainPlot = subplot(grid1[4:8, 12:15])
        countHist = subplot(grid1[4:8, 15:16], sharey=mainPlot)
    elif histMain:        
        countPlot = subplot(grid1[4:8, 12:15])
        mainPlot = subplot(grid1[4:8, 15:16], sharey=countPlot)
    else:
        countPlot = subplot(grid1[4:8, 12:15])
        countHist = subplot(grid1[4:8, 15:16], sharey=countPlot)
    grid1.update( left=0.001, right=0.95, hspace=1000 )
    
    avgPlt1 = subplot(grid1[8:12, 11:13])
    avgPlt2 = subplot(grid1[8:12, 13:15])
    if type(keyName) is not type("a string"):
        keyName = ' '.join([kn+',' for kn in keyName])
    titletxt = (keyName + " " + typeName + "; Avg. " + typeName + "% = " 
                + (misc.dblAsymErrString(np.mean(transVarAvg), *avgTransferErr[0], *transVarErr[0]) 
                   if len( transVarAvg ) == 1 else
                   misc.dblErrString(np.mean(transVarAvg),  
                                     np.sqrt(np.sum(arr(avgTransferErr)**2)/len(avgTransferErr)), 
                                     np.sqrt(np.sum(arr(transVarErr)**2)/len(transVarErr)))))
    # some easily parallelizable stuff
    plotList = [mainPlot, initPopPlot, countPlot, avgPlt1, avgPlt2]
    xlabels = [keyName,keyName,'Picture #','','']
    ylabels = [typeName + " %","Initial Pop %", "Camera Signal",'','']
    titles = [titletxt, "Initial Pop: Avg$ = " + str(misc.round_sig(np.mean(arr(initPopulation)))) + '$',
              "Thresh.=" + str(misc.round_sig(np.mean( [initThresholds[i][j].t for i in range(len(initThresholds)) 
                                                        for j in range(len(initThresholds[i]))]))) , '', '']
    majorYTicks = [np.arange(0,1,0.1),np.arange(0,1,0.2),np.linspace(min(pic1Data[0]),max(pic1Data[0]),5),[],[]]
    minorYTicks = [np.arange(0,1,0.05),np.arange(0,1,0.1),np.linspace(min(pic1Data[0]),max(pic1Data[0]),10),[],[]]
    if len(key.shape) == 1:
        xtickKey = key if len(key) < 30 else np.linspace(min(key),max(key),30)
    if len(key.shape) == 2:
        xtickKey = key[:,0] if len(key[:,0]) < 30 else np.linspace(min(key[:,0]),max(key[:,0]),30)
    majorXTicks = [xtickKey, xtickKey, np.linspace(0,len(pic1Data[0]),10), [],[]]
    grid_options = [True,True,True,False,False]
    fontsizes = [20,10,10,10,10]
    for pltNum, (subplt, xlbl, ylbl, title, yTickMaj, yTickMin, xTickMaj, fs, grid) in \
                enumerate(zip(plotList, xlabels, ylabels, titles, majorYTicks, minorYTicks, majorXTicks, fontsizes, grid_options)):
        subplt.set_xlabel(xlbl, fontsize=fs)
        subplt.set_ylabel(ylbl, fontsize=fs)
        subplt.set_title(title, fontsize=fs, loc='left', pad=50 if pltNum==0 else 0)
        subplt.set_yticks(yTickMaj)
        subplt.set_yticks(yTickMin, minor=True)
        subplt.set_xticks(xTickMaj)
        rotateTicks(subplt)
        subplt.grid(grid, color='#909090', which='Major', linewidth=2)
        #subplt.grid(grid, color='#AAAAAA', which='Minor')
        for item in ([subplt.title, subplt.xaxis.label, subplt.yaxis.label] + subplt.get_xticklabels() + subplt.get_yticklabels()):
            item.set_fontsize(fs)
    
    fitCharacters = []
    
    if type(dataColor) == str:
        colors, colors2 = misc.getColors(analysisOpts.numDataSets() + 1, cmStr=dataColor)
    else:
        colors = dataColor
    longLegend = len(transferData[0]) == 1
    markers = misc.getMarkers()
    
    # Main Plot
    if dataEdgeColors is None:
        dataEdgeColors = colors
    for dataSetInc, (atomLoc, fit, module, color, edgecolor) in enumerate(zip(range(analysisOpts.numDataSets()), fits, fitModules, colors, dataEdgeColors)):
        #leg = (r"[%d,%d] " % (analysisOpts.initLocs()[atomInc][0], analysisOpts.initLocs()[atomInc][1]) if typeName == "Survival"
        #       else r"[%d,%d]$\rightarrow$[%d,%d] " % (analysisOpts.initLocs()[atomInc][0], analysisOpts.initLocs()[atomInc][1], 
        #                                               analysisOpts.tferLocs()[atomInc][0], analysisOpts.tferLocs()[atomInc][1]))
        #if longLegend:
            #leg += (typeName + " % = " + misc.asymErrString(transferData[atomInc][0], *reversed(transferErrs[atomInc][0])))
        leg = anaylsisOpts.positiveResultConditions[dataSetInc].name + "\n("
        for ps in anaylsisOpts.postSelectionConditions[dataSetInc]:
            leg += ps.name +','
        leg += ")"
        unevenErrs = [[err[0] for err in transferErrs[dataSetInc]], [err[1] for err in transferErrs[dataSetInc]]]
        print('!!!!', key, transferData[dataSetInc], unevenErrs)
        mainPlot.errorbar ( key, transferData[dataSetInc], yerr=unevenErrs, color=color, ls='',
                            capsize=6, elinewidth=3, label=leg, 
                           alpha=0.3 if plotAvg else 0.9, marker=markers[dataSetInc%len(markers)], markersize=15,
                          markerEdgeColor=edgecolor, markerEdgeWidth=2)
        if module is not None and showFitDetails and fit['vals'] is not None:
            if module.fitCharacter(fit['vals']) is not None:
                fitCharacters.append(module.fitCharacter(fit['vals']))
            mainPlot.plot(fit['x'], fit['nom'], color=color, alpha=0.5)
            if plotFitGuess:
                mainPlot.plot(fit['x'], fit['guess'], color='r', alpha=1)

    mainPlot.xaxis.set_label_coords(0.95, -0.15)
    if legendOption:
        mainPlot.legend(loc="upper right", bbox_to_anchor=(1, 1.1), fancybox=True, 
                        ncol = 4 if longLegend else 10, prop={'size': 14}, frameon=False)
    # ### Init Population Plot
    for datasetNum, _ in enumerate(analysisOpts.postSelectionConditions):
        print(np.array(initPopulation).shape)
        initPopPlot.plot(key, initPopulation[datasetNum], ls='', marker='o', color=colors[datasetNum], alpha=0.3)
        initPopPlot.axhline(np.mean(initPopulation[datasetNum]), color=colors[datasetNum], alpha=0.3)
    # some shared properties 
    for plot in [mainPlot, initPopPlot]:
        if not min(key) == max(key):
            r = max(key) - min(key)
            plot.set_xlim(left = min(key) - r / len(key), right = max(key) + r / len(key))
        plot.set_ylim({0, 1})
    # ### Count Series Plot
    for locNum, loc in enumerate(analysisOpts.initLocs()):
        countPlot.plot(pic1Data[locNum], color=colors[locNum], ls='', marker='.', 
                       markersize=2 if countMain else 1, alpha=1 if countMain else 0.05)
        if countRunningAvg is not None:
            countPlot.plot(np.convolve(pic1Data[locNum], np.ones(countRunningAvg)/countRunningAvg, mode='valid'),
                           color=colors[locNum], alpha=1 if countMain else 0.5)
        for threshInc, thresh in enumerate(initThresholds[locNum]):
            picsPerVar = int(len(pic1Data[locNum])/len(initThresholds[locNum]))
            countPlot.plot([picsPerVar*threshInc, picsPerVar*(threshInc+1)], [thresh.t, thresh.t], color=colors[locNum], alpha=0.3)
    ticksForVis = countPlot.xaxis.get_major_ticks()
    ticksForVis[-1].label1.set_visible(False)
    # Count Histogram Plot
    for i, atomLoc in enumerate(analysisOpts.initLocs()):
        countHist.hist(pic1Data[i], 50, color=colors[i], orientation='horizontal', alpha=0.3, histtype='stepfilled')
        countHist.axhline(initThresholds[i][0].t, color=colors[i], alpha=0.3)
    setp(countHist.get_yticklabels(), visible=False)
    makeAvgPlts(avgPlt1, avgPlt2, avgPics, analysisOpts, colors)

    avgFitCharacter = None
    if plotAvg:
        unevenErrs = [[err[0] for err in avgTransferErr], [err[1] for err in avgTransferErr]]
        (_, caps, _) = mainPlot.errorbar( key, avgTransferData, yerr=unevenErrs, color="#BBBBBB", ls='',
                           marker='o', capsize=12, elinewidth=5, label='Atom-Avg', markersize=10,   )
        for cap in caps:
            cap.set_markeredgewidth(1.5)
        unevenErrs = [[err[0] for err in transVarErr], [err[1] for err in transVarErr]]
        (_, caps, _) = mainPlot.errorbar( key, transVarAvg, yerr=unevenErrs, color=avgColor, ls='',
                           marker='o', capsize=12, elinewidth=5, label='Event-Avg', markersize=10 )
        for cap in caps:
            cap.set_markeredgewidth(1.5)
        if fitModules[-1] is not None:
            mainPlot.plot(avgFit['x'], avgFit['nom'], color=avgColor, ls=':')
            avgFitCharacter = fitModules[-1].fitCharacter(avgFit['vals'])
            if plotFitGuess:
                mainPlot.plot(fit['x'], fit['guess'], color='r', alpha=0.5)
    if fitModules[0] is not None and showFitCharacterPlot:
        f, ax = subplots()
        fitCharacterPic, vmin, vmax = genAvgDiscrepancyImage(fitCharacters, avgPics[0].shape, analysisOpts.initLocs())
        im = ax.imshow(fitCharacterPic, cmap=cm.get_cmap('seismic_r'), vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title('Fit-Character (white is average)')
        #ax.grid(False)
        f.colorbar(im)
    tt.clock('After-Main-Plots')
    avgTransferPic = None
    if showImagePlots:
        imagingSignal = []
        f_imgPlots = []
        if tOptions[0].indvVariationThresholds:
            for varInc in range(len(initThresholds[0])):
                f_img, axs = subplots(1, 9 if tOptions[0].tferThresholdSame else 13, 
                                      figsize = (36.0, 2.0) if tOptions[0].tferThresholdSame else (36.0,3))
                lims = [[None, None] for _ in range(9 if tOptions[0].tferThresholdSame else 13)]
                ims = [None for _ in range(5)]
                imagingSignal.append(np.mean(
                    makeThresholdStatsImages(axs[5:9], [atomThresholds[varInc] for atomThresholds in initThresholds], 
                                             analysisOpts.initLocs(), avgPics[0].shape, ims, lims[5:9], f_img)))
                if not tOptions[0].tferThresholdSame:
                    makeThresholdStatsImages(axs[9:13], [atomThresholds[varInc] for atomThresholds in transThresholds], 
                                             analysisOpts.tferLocs(), avgPics[0].shape, ims, lims[9:13], f_img)

                avgTransfers = [np.mean(s) for s in transferData]
                avgTransferPic, l20, l21 = genAvgDiscrepancyImage(avgTransfers, avgPics[0].shape, analysisOpts.initLocs())

                avgPops = [np.mean(l) for l in initPopulation]
                avgInitPopPic, l30, l31 = genAvgDiscrepancyImage(avgPops, avgPics[0].shape, analysisOpts.initLocs())
                
                if genAvgs is not None:
                    print('genavgs',genAvgs)
                    genAtomAvgs = [np.mean(dp) for dp in genAvgs] if genAvgs[0] is not None else [0]
                    genImage, _, l41 = genAvgDiscrepancyImage(genAtomAvgs, avgPics[0].shape, 
                                                              analysisOpts.initLocs()) if genAvgs[0] is not None else (np.zeros(avgPics[0].shape), 0, 1)
                else:
                    genImage, l41, genAtomAvgs = (np.zeros(avgInitPopPic.shape), 1, [0])
                images = [avgPics[0], avgPics[1], avgTransferPic, avgInitPopPic, genImage]
                lims[0:5] = [[min(avgPics[0].flatten()), max(avgPics[0].flatten())], 
                             [min(avgPics[1].flatten()), max(avgPics[1].flatten())], 
                             [l20,l21],[l30,l31],[0,l41]]
                cmaps = ['viridis', 'viridis', 'seismic_r','seismic_r','inferno']
                titles = ['Avg 1st Pic', 'Avg 2nd Pic', 'Avg Trans:' + str(misc.round_sig(np.mean(avgTransfers))), 
                          'Avg Load:' + str(misc.round_sig(np.mean(avgPops))),'Atom-Generation: ' + str(misc.round_sig(np.mean(genAtomAvgs)))]

                for i, (ax, lim, image, cmap_) in enumerate(zip(axs.flatten(), lims, images, cmaps)):
                    ims[i] = ax.imshow(image, vmin=lim[0], vmax=lim[1], origin='lower', cmap=cm.get_cmap(cmap_))
                for ax, lim, title, im in zip(axs.flatten(), lims, titles, ims):
                    ax.set_yticklabels([])
                    ax.set_xticklabels([])
                    #ax.grid(False)
                    ax.set_title(title, fontsize=12)
                    imageTickedColorbar(fig, im, ax, lim)
                tt.clock('After-Image-Plots')
                f_img.suptitle(keyName + ': ' + str(key[varInc]))
                f_imgPlots.append(f_img)
        else:
            f_img, axs = subplots(1, 9 if tOptions[0].tferThresholdSame else 13, figsize = (36.0, 4.0) if tOptions[0].tferThresholdSame else (36.0,3))
            lims = [[None, None] for _ in range(9 if tOptions[0].tferThresholdSame else 13)]
            ims = [None for _ in range(5)]
            imagingSignal.append(np.mean(
                makeThresholdStatsImages(axs[5:9], [atomThresholds[0] for atomThresholds in initThresholds], 
                                         analysisOpts.initLocs(), avgPics[0].shape, ims, lims[5:9], f_img)))
            imagingSignal = [imagingSignal[0] for _ in range(len(initThresholds[0]))]
            if not tOptions[0].tferThresholdSame:
                makeThresholdStatsImages(axs[9:13], [atomThresholds[0] for atomThresholds in transThresholds], 
                                         analysisOpts.tferLocs(), avgPics[0].shape, ims, lims[9:13], f_img)

            avgTransfers = [np.mean(s) for s in transferData]
            avgTransferPic, l20, l21 = genAvgDiscrepancyImage(avgTransfers, avgPics[0].shape, analysisOpts.initLocs())

            avgPops = [np.mean(l) for l in initPopulation]
            avgInitPopPic, l30, l31 = genAvgDiscrepancyImage(avgPops, avgPics[0].shape, analysisOpts.initLocs())

            if genAvgs is not None:
                print('genavgs',genAvgs)
                genAtomAvgs = [np.mean(dp) for dp in genAvgs] if genAvgs[0] is not None else [0]
                genImage, _, l41 = genAvgDiscrepancyImage(genAtomAvgs, avgPics[0].shape, 
                                                          analysisOpts.initLocs()) if genAvgs[0] is not None else (np.zeros(avgPics[0].shape), 0, 1)
            else:
                genImage, l41, genAtomAvgs = ([[0]], 1, [0])
            images = [avgPics[0], avgPics[1], avgTransferPic, avgInitPopPic, genImage]
            lims[0:5] = [[min(avgPics[0].flatten()), max(avgPics[0].flatten())], [min(avgPics[1].flatten()), max(avgPics[1].flatten())], [l20,l21],[l30,l31],[0,l41]]
            cmaps = ['viridis', 'viridis', 'seismic_r','seismic_r','inferno']
            titles = ['Avg 1st Pic', 'Avg 2nd Pic', 'Avg Trans:' + str(misc.round_sig(np.mean(avgTransfers))), 
                      'Avg Load:' + str(misc.round_sig(np.mean(avgPops))),'Atom-Generation: ' + str(misc.round_sig(np.mean(genAtomAvgs)))]

            for i, (ax, lim, image, cmap_) in enumerate(zip(axs.flatten(), lims, images, cmaps)):
                ims[i] = ax.imshow(image, vmin=lim[0], vmax=lim[1], origin='lower', cmap=cm.get_cmap(cmap_))
            for ax, lim, title, im in zip(axs.flatten(), lims, titles, ims):
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                #ax.grid(False)
                ax.set_title(title, fontsize=12)
                imageTickedColorbar(fig, im, ax, lim)
            tt.clock('After-Image-Plots')
            f_img.suptitle('All Data')
            f_imgPlots.append(f_img)
    if plotImagingSignal:
        mainTwinx = mainPlot.twinx();
        mainTwinx.plot(key, imagingSignal, 'ro');
        #mainTwinx.grid(False)
        mainTwinx.set_ylabel('Imaging Signal (Counts)',color='r');
        mainTwinx.spines['right'].set_color('r')
        mainTwinx.yaxis.label.set_color('r')
        mainTwinx.tick_params(axis='y', colors='r')
    if plotIndvHists:
        if type(analysisOpts.initLocsIn[-1]) == int:
            shape = (analysisOpts.initLocsIn[-1], analysisOpts.initLocsIn[-2])
        else:
            raise ValueError("Can't currently plot indv hists if giving an explicit atom list instead of a grid.")
        thresholdFigs = []
        if tOptions[0].indvVariationThresholds:
            print('Plotting individual variation threshold data...')
            binCenterList = ([th.binCenters for thresholds in initThresholds for th in thresholds] 
                            + [th.binCenters if th is not None else [] for thresholds_2 in transThresholds for th in thresholds_2])
            binCenterList = [item for sublist in binCenterList for item in sublist]
            binCenterList = [item for item in binCenterList if item] 
            minx, maxx = min(binCenterList), max(binCenterList)
            for varInc in range(len(initThresholds[0])):
                thresholdFigs.append(plotThresholdHists([atomThresholds[varInc] for atomThresholds in initThresholds], 
                                                        colors, extra=avgTransfers, extraname=r"$\rightarrow$:", 
                                                        thresholds_2=[atomThresholds[varInc] for atomThresholds in transThresholds], 
                                                        shape=shape, title= keyName + ': ' + str(key[varInc]), minx=minx,maxx=maxx))
        else:
            print('Plotting all threshold data...')
            thresholdFigs.append(plotThresholdHists([atomThresholds[0] for atomThresholds in initThresholds], 
                                                    colors, extra=avgTransfers, extraname=r"$\rightarrow$:", 
                                                    thresholds_2=[atomThresholds[0] for atomThresholds in transThresholds], 
                                                    shape=shape, title='All Data'))
        tt.clock('After-Indv-Hists')
        for thresholdFig in thresholdFigs:
            display(thresholdFig)
    if timeit:
        tt.display()
    
    if (newAnnotation or not exp.checkAnnotation(fileNumber, force=False, quiet=True, expFile_version=expFile_version)) and not forceNoAnnotation :
        disp.display(fig)
        if fitModules[-1] is not None:
            print("Avg Fit R-Squared: " + misc.round_sig_str(avgFit["R-Squared"]))
            fitInfoString = ""
            for label, fitVal, err in zip(fitModules[-1].args(), avgFit['vals'], avgFit['errs']):
                fitInfoString += label+': '+misc.errString(fitVal, err) + "<br>  "
            fitInfoString += (fitModules[-1].getFitCharacterString() + ': ' 
                                       + misc.errString(fitModules[-1].fitCharacter(avgFit['vals']), 
                                                        fitModules[-1].fitCharacterErr(avgFit['vals'], avgFit['errs'])))
            disp.display(disp.Markdown(fitInfoString))
            if showFitDetails:
                for f in getFitsDataFrame(fits, fitModules, avgFit):
                    display(f)

        exp.annotate(fileNumber,expFile_version)
    configName = exp.getConfiguration(fileNumber,expFile_version=expFile_version)
    
    if not forceNoAnnotation:
        rawTitle, _, lev = exp.getAnnotation(fileNumber,expFile_version=expFile_version)
        expTitle = ''.join('#' for _ in range(lev)) + ' File ' + str(fileNumber) + " (" + configName +"): " + rawTitle
    else:
        expTitle = ''.join('#' for _ in range(3)) + ' File ' + str(fileNumber) + " (" + configName +")"
        
    if cleanOutput:
        disp.clear_output()
    disp.display(disp.Markdown(expTitle))
    with exp.ExpFile(fileNumber,expFile_version=expFile_version) as fid:
        fid.get_basic_info()
    
    if fitModules[-1] is not None and avgFit['errs'] is not None:
        print(avgFit['errs'])
        print("Avg Fit R-Squared: " + misc.round_sig_str(avgFit["R-Squared"]))
        fitInfoString = ""
        for label, fitVal, err in zip(fitModules[-1].args(), avgFit['vals'], avgFit['errs']):
            fitInfoString += label+': '+misc.errString(fitVal, err) + "<br>  "
        fitInfoString += (fitModules[-1].getFitCharacterString() + ': ' 
                                   + misc.errString(fitModules[-1].fitCharacter(avgFit['vals']), 
                                                    fitModules[-1].fitCharacterErr(avgFit['vals'], avgFit['errs'])))
        disp.display(disp.Markdown(fitInfoString))
    if fitModules[-1] is not None and showFitDetails:
            for f in getFitsDataFrame(fits, fitModules, avgFit):
                display(f)
    
    if outputThresholds:
        thresholdList = np.flip(np.reshape([t.t for t in initThresholds], (10,10)),1)
        with open('J:/Code-Files/T-File.txt','w') as f:
            for row in thresholdList:
                for thresh in row:
                    f.write(str(thresh) + ' ')
    return {'Key':key, 'All_Transfer':transferData, 'All_Transfer_Errs':transferErrs, 'Initial_Populations':initPopulation, 
            'Transfer_Fits':fits, 'Average_Transfer_Fit':avgFit, 'Average_Atom_Generation':genAvgs, 
            'Average_Atom_Generation_Err':genErrs, 'Picture_1_Data':pic1Data, 'Fit_Character':fitCharacters, 
            'Average_Transfer_Pic':avgTransferPic, 'Transfer_Averaged_Over_Variations':transVarAvg, 
            'Transfer_Averaged_Over_Variations_Err':transVarErr, 'Average_Transfer':avgTransferData,
            'Average_Transfer_Err':avgTransferErr, 'Initial_Atom_Images':initAtomImages, 
            'Transfer_Atom_Images':transAtomImages, 'Picture_2_Data':pic2Data, 'Initial_Thresholds':initThresholds,
            'Transfer_Thresholds':transThresholds, 'Fit_Modules':fitModules, 'Average_Fit_Character':avgFitCharacter,
            'Ensemble_Hits':ensembleHits, 'InitAtoms':initAtoms, 'TferAtoms':tferAtoms, 'tferList':tferList, 'Main_Axis':mainPlot,
            'Figures':[fig, *f_imgPlots]}


def Loading(fileID, atomLocs, **TransferArgs):
    """
    A small wrapper, partially for the extra defaults in this case partially for consistency with old function definitions.
    """
    return Transfer(fileID, tao.getStandardLoadingOptions(atomLocs), atomLocs, **TransferArgs)


def Population(fileNum, atomLocations, whichPic, picsPerRep, plotLoadingRate=True, legendOption=None, showImagePlots=True,
               plotIndvHists=False, showFitDetails=False, showFitCharacterPlot=True, show=True, histMain=False,
               mainAlpha=0.2, avgColor='w', newAnnotation=False, thresholdOptions=to.ThresholdOptions(), clearOutput=True,
               dataCmap='gist_rainbow', countMain=False,
               **StandardArgs):
    """
    Standard data analysis package for looking at population %s throughout an experiment.

    return key, loadingRateList, loadingRateErr

    This routine is designed for analyzing experiments with only one picture per cycle. Typically
    These are loading exeriments, for example. There's no survival calculation.
    """
    atomLocs_orig = atomLocations
    avgColor='w'
    res = pa.standardPopulationAnalysis(fileNum, atomLocations, whichPic, picsPerRep, thresholdOptions=thresholdOptions, **StandardArgs)
    (locCounts, thresholds, avgPic, key, allPopsErr, allPops, avgPop, avgPopErr, fits,
     fitModules, keyName, atomData, rawData, atomLocations, avgFits, atomImages,
     totalAvg, totalErr, variationCountData, variationAtomData, varThresholds) = res
    colors, _ = misc.getColors(len(atomLocations) + 1, cmStr=dataCmap)
    
    if not show:
        return key, allPops, allPopsErr, locCounts, atomImages, thresholds, avgPop
    if legendOption is None and len(atomLocations) < 50:
        legendOption = True
    else:
        legendOption = False
    # get the colors for the plot.
    markers = ['o','^','<','>','v']
    f_main = figure(figsize=(20,7))
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
    if countMain:
        countPlot = subplot(gridRight[:, :11])
        mainPlot = subplot(gridLeft[4:8, 15:16], sharey=countPlot)
        countHist = subplot(gridRight[:, 11:12])
    elif histMain:
        countPlot = subplot(gridRight[:, :1])
        countHist = subplot(grid1[:, 1:12])
        mainPlot = subplot(gridLeft[4:8, 15:16], sharey=countPlot)
    else:
        mainPlot = subplot(grid1[:, :12])
        countPlot = subplot(gridRight[:, 12:15])
        countHist = subplot(gridLeft[4:8, 15:16], sharey=countPlot)    
        
    fitCharacters = []
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
            print(unevenErrs)
            mainPlot.errorbar(key, allPops[i], yerr=unevenErrs, color=colors[i], ls='',
                              capsize=6, elinewidth=3, label=leg, alpha=mainAlpha, marker=markers[i%len(markers)],markersize=5)
            if module is not None:
                if fit == [] or fit['vals'] is None:
                    continue
                fitCharacters.append(module.fitCharacter(fit['vals']))
                mainPlot.plot(fit['x'], fit['nom'], color=colors[i], alpha = 0.5)
        if fitModules[-1] is not None:
            if avgFits['vals'] is None:
                print('Avg Fit Failed!')
            else:
                mainPlot.plot(avgFits['x'], avgFits['nom'], color=avgColor, alpha = 1,markersize=5)
        mainPlot.grid(True, color='#AAAAAA', which='Major')
        mainPlot.grid(True, color='#090909', which='Minor')
        mainPlot.set_yticks(np.arange(0,1,0.1))
        mainPlot.set_yticks(np.arange(0,1,0.05), minor=True)
        mainPlot.set_ylim({-0.02, 1.01})
        if not min(key) == max(key):
            mainPlot.set_xlim(left=min(key) - (max(key) - min(key)) / len(key), right=max(key)
                              + (max(key) - min(key)) / len(key))
        if len(key) < 30:
            mainPlot.set_xticks(key)
        else:
            mainPlot.set_xticks(np.linspace(min(key),max(key),30))
        rotateTicks(mainPlot)
        titletxt = keyName + " Atom " + typeName + " Scan"
        if len(allPops[0]) == 1:
            titletxt = keyName + " Atom " + typeName + " Point.\n Avg " + typeName + "% = " + misc.errString(totalAvg, totalErr) 
        
        mainPlot.set_title(titletxt, fontsize=30 if not histMain else 12 )
        mainPlot.set_ylabel("S %", fontsize=20 if not histMain else 9 )
        mainPlot.set_xlabel(keyName, fontsize=20 if not histMain else 9 )
        if legendOption == True:
            if histMain:
                cols = 4 if longLegend else 10
                countHist.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=cols, prop={'size': 12})
            else:
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
    popPlot.set_title("Population: Avg$ = " +  str(misc.round_sig(np.mean(arr(allPops)))) + '$')
    for item in ([popPlot.title, popPlot.xaxis.label, popPlot.yaxis.label] +
                     popPlot.get_xticklabels() + popPlot.get_yticklabels()):
        item.set_fontsize(10)
    # ### Count Series Plot
    for i, loc in enumerate(atomLocations):
        countPlot.plot(locCounts[i], color=colors[i], ls='', marker='.', markersize=1, alpha=0.3)
        countPlot.axhline(thresholds[i].t, color=colors[i], alpha=0.3)

    countPlot.set_xlabel("Picture #")
    countPlot.set_ylabel("Camera Signal")
    countPlot.set_title("Thresh.=" + str(misc.round_sig(thresholds[i].t)), fontsize=10) #", Fid.="
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
            countHist.hist(locCounts[i], 50, color=colors[i], orientation='vertical', alpha=mainAlpha, histtype='stepfilled')
            countHist.axvline(thresholds[i].t, color=colors[i], alpha=0.3)
        else:
            countHist.hist(locCounts[i], 50, color=colors[i], orientation='horizontal', alpha=mainAlpha, histtype='stepfilled')
            countHist.axhline(thresholds[i].t, color=colors[i], alpha=0.3)
    for item in ([countHist.title, countHist.xaxis.label, countHist.yaxis.label] +
                     countHist.get_xticklabels() + countHist.get_yticklabels()):
        item.set_fontsize(10 if not histMain else 20)
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
    mainPlot.errorbar(key, avgPop, yerr=avgPopErr, color=avgColor, ls='', marker='o', capsize=6, elinewidth=3, label='Avg', markersize=5)
    if fitModules is not [None] and showFitDetails:
        mainPlot.plot(avgFits['x'], avgFits['nom'], color=avgColor, ls=':')

    if fitModules is not [None] and showFitCharacterPlot and fits[0] != []:
        figure()
        print('fitCharacter',fitCharacters)
        fitCharacterPic, vmin, vmax = genAvgDiscrepancyImage(fitCharacters, avgPic.shape, atomLocations)
        imshow(fitCharacterPic, cmap=cm.get_cmap('seismic_r'), vmin=vmin, vmax=vmax, origin='lower')
        title('Fit-Character (white is average)')
        colorbar()
    if showImagePlots:
        ims = []
        lims = [[0,0] for _ in range(5)]
        f_im, axs = subplots(1,6, figsize=(20,5))
        
        ims.append(axs[0].imshow(avgPic, origin='lower'))
        axs[0].set_title('Avg 1st Pic')

        avgPops = []
        for l in allPops:
            avgPops.append(np.mean(l))
        avgPopPic, vmin, vmax = genAvgDiscrepancyImage(avgPops, avgPic.shape, atomLocations)
        ims.append(axs[1].imshow(avgPopPic, cmap=cm.get_cmap('seismic_r'), vmin=vmin, vmax=vmax, origin='lower'))
        axs[1].set_title('Avg Population')
        
        makeThresholdStatsImages(axs[2:], thresholds, atomLocations, avgPic.shape, ims, lims, f_im)

    avgPops = []
    for s in allPops:
        avgPops.append(np.mean(s))
    if plotIndvHists:
        shape = (atomLocs_orig[-1], atomLocs_orig[-2]) if type(atomLocs_orig[-1]) == int else (10,10)
        if thresholdOptions.indvVariationThresholds:
            for varInc in range(len(key)):
                #misc.transpose(varThresholds)[varInc]
                #print(len([atomTh[varInc] for atomTh in varThresholds]), len(avgPops), len(colors))
                #print([atomTh[varInc] for atomTh in varThresholds])
                plotThresholdHists(varThresholds[varInc], colors, extra=avgPops, extraname=r"L:", 
                                   shape=shape, title= keyName + ': ' + str(key[varInc]))
        else:
            plotThresholdHists(thresholds, colors, extra=avgPops, extraname="L:", shape=shape)
            """
            thresholdFigs.append(plotThresholdHists([atomThresholds[0] for atomThresholds in initThresholds], 
                                                    colors, extra=avgTransfers, extraname=r"$\rightarrow$:", 
                                                    thresholds_2=[atomThresholds[0] for atomThresholds in transThresholds], 
                                                    shape=shape, title='All Data'))"""

            

    
    disp.display(f_main)
    if newAnnotation or not exp.checkAnnotation(fileNum, force=False, quiet=True):
        exp.annotate(fileNum)
    if clearOutput:
        disp.clear_output()
    
    rawTitle, notes, lev = exp.getAnnotation(fileNum)
    expTitle = ''.join('#' for _ in range(lev)) + ' File ' + str(fileNum) + ': ' + rawTitle
    disp.display(disp.Markdown(expTitle))
    disp.display(disp.Markdown(notes))
    with exp.ExpFile(fileNum) as f:
        f.get_basic_info()
    
    if fitModules[-1] is not None:
        for label, fitVal, err in zip(fitModules[-1].args(), avgFits['vals'], avgFits['errs']):
            print( label,':', misc.errString(fitVal, err) )
        if showFitDetails:
            fits_df = getFitsDataFrame(fits, fitModules, avgFits, markersize=5)
            display(fits_df)
    return { 'Key': key, 'All_Populations': allPops, 'All_Populations_Error': allPopsErr, 'Pixel_Counts':locCounts, 
            'Atom_Images':atomImages, 'Thresholds':thresholds, 'Atom_Data':atomData, 'Raw_Data':rawData, 
            'Average_Population': avgPop, 'Average_Population_Error': avgPopErr }

def showPics(data, key, fitParams=None, indvColorBars=False, colorMax=-1, fancy=True, hFitParams=None, vFitParams=None):
    """
    A little function for making a nice display of an array of pics and fits to the pics
    """
    num = len(data)
    gridsize1, gridsize2 = (0, 0)
    if hFitParams is None:
        hFitParams = [None for _ in range(num)]
    if vFitParams is None:
        vFitParams = [None for _ in range(num)]
    for i in range(100):
        if i*(i-2) >= num:
            gridsize1 = i
            gridsize2 = i-2
            break
    if fancy:
        fig, axs = subplots( 2 if fitParams is not None else 1, num, figsize=(20,5))
        grid = axs.flatten()
    else:
        fig = figure(figsize=(20,20))
        grid = axesTool.AxesGrid( fig, 111, nrows_ncols=( 2 if fitParams is not None else 1, num), axes_pad=0.0, share_all=True,
                                  label_mode="L", cbar_location="right", cbar_mode="single" )
    rowCount, picCount, count = 0,0,0
    maximum, minimum = sorted(data.flatten())[colorMax], min(data.flatten())
    # get picture fits & plots
    for picNum in range(num):
        pl = grid[count]
        if count >= len(data):
            count += 1
            picCount += 1
            continue
        pic = data[count]
        if indvColorBars:
            maximum, minimum = max(pic.flatten()), min(pic.flatten())
        y, x = [np.linspace(1, pic.shape[i], pic.shape[i]) for i in range(2)]
        x, y = np.meshgrid(x, y)
        if fancy:            
            res =fancyImshow(fig, pl, pic, imageArgs={'extent':(x.min(), x.max(), y.min(), y.max()),
                                                  'vmin':minimum, 'vmax':maximum}, cb=False, ticklabels=False,
                                  hAvgArgs={'linewidth':0.5}, vAvgArgs={'linewidth':0.5}, 
                                  hFitParams=hFitParams[count], vFitParams=vFitParams[count])
            pl = res[0]
        else:
            im1 = pl.imshow( pic, origin='bottom', extent=(x.min(), x.max(), y.min(), y.max()),
                                                  vmin=minimum, vmax=maximum )
            pl.axis('off')
        pl.set_title(str(misc.round_sig(key[count], 4)), fontsize=8)
        if fitParams is not None:
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
            v_ = gaussian_2d.f_notheta((X,Y), *fitParams[count])
            vals = np.reshape(v_, pic.shape)
            if fancy:            
                res=fancyImshow(fig, pl2, pic-vals, imageArgs={'extent':(x.min(), x.max(), y.min(), y.max())},
                                       cb=False, ticklabels=False, hAvgArgs={'linewidth':0.5}, vAvgArgs={'linewidth':0.5})
                pl2 = res[0]
            else:
                im2 = pl2.imshow(pic-vals, vmin=-2, vmax=2, origin='bottom',
                                                     extent=(x.min(), x.max(), y.min(), y.max()))
            pl2.axis('off')
        count += 1
        picCount += 1
    return fig
