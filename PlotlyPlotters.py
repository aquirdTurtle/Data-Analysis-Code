__version__ = "1.2"

import numpy as np
from numpy import array as arr
from IPython.display import display
from pandas import DataFrame
from Miscellaneous import round_sig, getStats, transpose, getColors
import plotly.graph_objs as go
from plotly.offline import iplot
from plotly.tools import make_subplots
from AnalysisHelpers import (unpackAtomLocations, getEnsembleHits, getNetLoss, getNetLossStats,
                             normalizeData, loadHDF5)
from MainAnalysis import standardAssemblyAnalysis, standardLoadingAnalysis, standardTransferAnalysis


def Survival(fileNumber, atomLocs, **TransferArgs):
    """See corresponding transfer function for valid TransferArgs."""
    (key, survival, survivalerr, captureArray) = Transfer(fileNumber, atomLocs, atomLocs, **TransferArgs)
    return key, survival, survivalerr, captureArray


def Transfer(fileNumber, atomLocs1, atomLocs2, show=True, accumulations=1, key=None, manualThreshold=None,
             fitType=None, window=None, xMin=None, xMax=None, yMin=None, yMax=None, dataRange=None,
             histSecondPeakGuess=None, keyOffset=0, sumAtoms=True, outputMma=False, dimSlice=None,
             varyingDim=None, showCounts=False):
    """
    Standard data analysis package for looking at survival rates throughout an experiment.

    Returns key, survivalData, survivalErrors
    """
    (atomLocs1, atomLocs2, atomCounts, survivalData, survivalErrs, loadingRate, pic1Data, keyName, key,
     repetitions, thresholds, fits, avgSurvivalData, avgSurvivalErr, avgFit, avgPic, otherDimValues,
     locsList) = standardTransferAnalysis(fileNumber, atomLocs1, atomLocs2, accumulations=accumulations, key=key,
                                          picsPerRep=2, manualThreshold=manualThreshold, fitType=fitType, window=window,
                                          xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, dataRange=dataRange,
                                          histSecondPeakGuess=histSecondPeakGuess, keyOffset=keyOffset,
                                          sumAtoms=sumAtoms, outputMma=outputMma, dimSlice=dimSlice,
                                          varyingDim=varyingDim)
    if not show:
        return key, survivalData, survivalErrs, loadingRate

    # get the colors for the plots.
    pltColors, pltColors2 = getColors(len(locsList) + 1)
    scanType = "Survival" if atomLocs1 == atomLocs2 else "Transfer"
    if otherDimValues[0] is not None:
        legends = [r"%d,%d>%d,%d @%d" % (loc1[0], loc1[1], loc2[0], loc2[1], other) +
                   (scanType + " % = " + str(round_sig(d[0])) + "+-" + str(round_sig(e[0])) if len(d) == 1 else "")
                   for loc1, loc2, d, e, other in zip(locsList, locsList, survivalData, survivalErrs,
                                                      otherDimValues)]
    else:
        legends = [r"%d,%d>%d,%d" % (loc1[0], loc1[1], loc2[0], loc2[1]) +
                   (scanType + " % = " + str(round_sig(d[0])) + "+-" + str(round_sig(e[0])) if len(d) == 1 else "")
                   for loc1, loc2, d, e in zip(locsList, locsList, survivalData, survivalErrs)]
    survivalErrs = list(survivalErrs)
    # Make the plots
    alphaVal = 1.0 / (len(atomLocs1) ** 0.5)
    mainPlot, countsHist, countsFig, loadingPlot, avgFig = [[] for _ in range(5)]
    avgFig.append(go.Heatmap(z=avgPic, colorscale='Viridis', colorbar=go.ColorBar(x=1, y=0.15, len=0.3)))
    centers = []
    for data, err, loc, color, legend, fitData in zip(survivalData, survivalErrs, locsList, pltColors,
                                                      legends, fits):
        mainPlot.append(go.Scatter(x=key, y=data, opacity=alphaVal, mode="markers", name=legend,
                                   error_y={"type": 'data', "array": err, 'color': color},
                                   marker={'color': color}, legendgroup=legend))
        if fitType is not None:
            if fitData['vals'] is None:
                print(loc, 'Fit Failed!')
                continue
            print(loc, round_sig(fitData['vals'][2], 4), '+-', round_sig(fitData['errs'][2], 2))
            centers.append(fitData['vals'][2])
            mainPlot.append(go.Scatter(x=fitData['x'], y=fitData['nom'], line={'color': color},
                                       legendgroup=legend, showlegend=False, opacity=alphaVal))
            mainPlot.append(go.Scatter(x=fitData['x'], y=fitData['nom'] + fitData['std'],
                                       opacity=alphaVal / 2, line={'color': color},
                                       legendgroup=legend, showlegend=False, hoverinfo='none'))
            mainPlot.append(go.Scatter(x=fitData['x'], y=fitData['nom'] - fitData['std'],
                                       opacity=alphaVal / 2, line={'color': color},
                                       legendgroup=legend, fill='tonexty', showlegend=False,
                                       hoverinfo='none'))
    if fitType is not None:
        print('Fit Center Statistics:')
        display(getStats(centers))
    # countsFig.append(go.Scatter(y=atomCounts, mode='markers', opacity=0.1, marker={'color':color, 'size':1},
    #            legendgroup='avg', showlegend=False))
    # countsHist.append(go.Histogram(y=atomCounts, nbinsy=100, legendgroup='avg', showlegend=False, opacity=0.1,
    #                                   xbins=dict(start=min(pic1Data.flatten()), end=max(pic1Data.flatten())),
    #                                   marker=dict(color=color)))

    for data, load, loc, color, threshold, legend in zip(pic1Data, loadingRate, atomLocs1, pltColors,
                                                         thresholds, legends):
        # countsHist.append(go.Histogram(y=data, nbinsy=100, legendgroup=str(loc), showlegend=False, opacity=alphaVal,
        #                               xbins=dict(start=min(pic1Data.flatten()), end=max(pic1Data.flatten())),
        #                               marker=dict(color=color)))
        # countsFig.append(go.Scatter(y=data, mode='markers', marker={'color':color, 'size':1},
        #                            legendgroup=str(loc), showlegend=False))
        # countsFig.append(go.Scatter(x=[0,len(pic1Data[0].flatten())], y=[threshold,threshold], showlegend=False,
        #                             mode='lines', line={'color':color, 'width':1}, hoverinfo='none',
        #                             legendgroup=str(loc)))
        loadingPlot.append(go.Scatter(x=key, y=load, mode="markers", name=str(loc),
                                      marker={'color': color}, legendgroup=legend, showlegend=False,
                                      opacity=alphaVal))
        avgFig.append(go.Scatter(x=[loc[1]], y=[loc[0]], mode='markers', hoverinfo='none',
                                 showlegend=False, legendgroup=legend, marker={'size': 5, 'color': '#FF0000'}))

    # average stuff

    mainPlot.append(go.Scatter(x=key, y=avgSurvivalData, mode="markers", name='avg',
                               error_y={"type": 'data', "array": avgSurvivalErr, 'color': '#000000'},
                               marker={'color': '#000000'}, legendgroup='avg'))
    if fitType is not None:
        mainPlot.append(go.Scatter(x=avgFit['x'], y=avgFit['nom'], line={'color': '#000000'},
                                   legendgroup='avg', showlegend=False, opacity=alphaVal))
        mainPlot.append(go.Scatter(x=avgFit['x'], y=avgFit['nom'] + avgFit['std'],
                                   opacity=alphaVal / 2, line={'color': '#000000'},
                                   legendgroup='avg', showlegend=False, hoverinfo='none'))
        mainPlot.append(go.Scatter(x=avgFit['x'], y=avgFit['nom'] - avgFit['std'],
                                   opacity=alphaVal / 2, line={'color': '#000000'},
                                   legendgroup='avg', fill='tonexty', showlegend=False,
                                   hoverinfo='none'))
    if showCounts:
        avgOnly = True
        if avgOnly:
            countsFig.append(go.Scatter(y=arr(transpose(pic1Data)).flatten(), mode='markers',
                                        marker={'color': '#000000', 'size': 1}, legendgroup='avg', showlegend=False))
        countsHist.append(go.Histogram(y=atomCounts.flatten(), nbinsy=200, legendgroup='avg',
                                       showlegend=False, opacity=0.1, marker=dict(color="#000000"),
                                       xbins=dict(start=min(pic1Data.flatten()), end=max(pic1Data.flatten()))))
        d, _ = np.histogram(atomCounts.flatten(), bins=200)
        countsHist.append(go.Scatter(x=[0, max(d) / 2], y=[np.mean(thresholds), np.mean(thresholds)],
                                     showlegend=False, mode='lines', line={'color': "#000000", 'width': 1},
                                     hoverinfo='none', legendgroup='avg'))
        # format and arrange plots. large grid is mostly to precisely place the histogram.
        fig = make_subplots(
            rows=3, cols=12, print_grid=False, horizontal_spacing=0.1, vertical_spacing=0.05,
            specs=[[{'rowspan': 3, 'colspan': 9}, None, None, None, None, None, None, None, None,
                    {'colspan': 2}, None, {}],
                   [None, None, None, None, None, None, None, None, None, {'colspan': 3}, None, None],
                   [None, None, None, None, None, None, None, None, None, {'colspan': 3}, None, None]])
        for mainLine in mainPlot:
            fig.append_trace(mainLine, 1, 1)
        for avgPart in avgFig:
            fig.append_trace(avgPart, 3, 10)
        for load in loadingPlot:
            fig.append_trace(load, 2, 10)
        for counts in countsFig:
            fig.append_trace(counts, 1, 10)
        for hist in countsHist:
            fig.append_trace(hist, 1, 12)
        fig['layout']['yaxis2'].update(title="Count", range=[min(pic1Data.flatten()), max(pic1Data.flatten())])
        fig['layout']['xaxis2'].update(range=[0, len(pic1Data[0].flatten())])
        fig['layout']['yaxis3'].update(range=[min(pic1Data.flatten()), max(pic1Data.flatten())],
                                       showticklabels=False)
        fig['layout']['xaxis3'].update(showticklabels=False)
        fig['layout']['yaxis4'].update(title="Loading %", range=[0, 1])
        fig['layout']['yaxis5'].update(title="Average Image")
    else:
        # not show counts
        countsHist.append(go.Histogram(x=atomCounts.flatten(), nbinsx=200, legendgroup='avg',
                                       showlegend=False, opacity=0.9,
                                       xbins=dict(start=min(pic1Data.flatten()), end=max(pic1Data.flatten())),
                                       marker=dict(color="#000000")))
        d, _ = np.histogram(atomCounts.flatten(), bins=200)
        countsHist.append(go.Scatter(x=[np.mean(thresholds), np.mean(thresholds)], y=[0, max(d) / 2],
                                     showlegend=False, mode='lines', line={'color': "#000000", 'width': 1},
                                     hoverinfo='none', legendgroup='avg'))
        # format and arrange plots. large grid is mostly to precisely place the histogram.
        fig = make_subplots(
            rows=3, cols=12, print_grid=False, horizontal_spacing=0.1, vertical_spacing=0.05,
            specs=[[{'rowspan': 3, 'colspan': 9}, None, None, None, None, None, None, None, None,
                    {'colspan': 3}, None, None],
                   [None, None, None, None, None, None, None, None, None, {'colspan': 3}, None, None],
                   [None, None, None, None, None, None, None, None, None, {'colspan': 3}, None, None]])
        for mainLine in mainPlot:
            fig.append_trace(mainLine, 1, 1)
        for avgPart in avgFig:
            fig.append_trace(avgPart, 3, 10)
        for load in loadingPlot:
            fig.append_trace(load, 2, 10)
        for hist in countsHist:
            fig.append_trace(hist, 1, 10)
        # fig['layout']['yaxis2'].update(range=[min(pic1Data.flatten()), max(pic1Data.flatten())], showticklabels=False)
        # fig['layout']['xaxis2'].update(showticklabels=False)
        fig['layout']['yaxis3'].update(title="Loading %", range=[0, 1])
        fig['layout']['yaxis4'].update(title="Average Image")
    fig['layout'].update(barmode='overlay')
    fig['layout']['yaxis1'].update(title=scanType + " %", range=[0, 1])
    fig['layout']['xaxis1'].update(title=str(keyName))
    iplot(fig)
    return key, survivalData, survivalErrs, loadingRate


def Loading(fileNum, atomLocations, showTotalHist=True, showIndividualHist=False, showLoadingRate=True,
            showLoadingPic=False, **StandardArgs):
    """
    Standard data analysis package for looking at loading rates throughout an experiment.
    return key, loadingRateList, loadingRateErr

    See standardLoadingAnalysis for valid standardArgs

    This routine is designed for analyzing experiments with only one picture per cycle. Typically
    These are loading exeriments, for example. There's no survival calculation.

    :param fileNum:
    :param atomLocations:
    :param showTotalHist:
    :param showIndividualHist:
    :param showLoadingRate:
    :param showLoadingPic:
    :param StandardArgs:
    :return:
    """

    (pic1Data, thresholds, avgPic, key, loadingRateErr, loadingRateList, allLoadingRate, allLoadingErr, loadFits,
     loadingFitType, keyName, totalPic1AtomData, rawData, showTotalHist, atomLocations,
     avgFits) = standardLoadingAnalysis(fileNum, atomLocations, **StandardArgs)

    countsHist = []
    if showTotalHist:
        print(pic1Data.flatten())
        d, _ = np.histogram(pic1Data.flatten(), bins=100)
        countsHist.append(go.Histogram(x=pic1Data.flatten(), nbinsx=100, legendgroup='avg',
                                       showlegend=False, xbins=dict(start=min(pic1Data.flatten()),
                                                                    end=max(pic1Data.flatten())),
                                       marker=dict(color='#000000')))
        countsHist.append(go.Scatter(x=[0, max(d)], y=[np.mean(thresholds), np.mean(thresholds)],
                                     showlegend=False, mode='lines', line={'color': '#000000', 'width': 1},
                                     hoverinfo='none', legendgroup='avg'))
    if showIndividualHist:
        pass
        # atomHistWithBokeh( key, atomLocations, pic1Data, bins, binData, fitVals, thresholds,
        #                  avgPic, atomCount, dataInc, outputFileAddr=outputFileAddr,
        #                  interactive=interactive)

    if showLoadingPic:
        loadingPic = np.zeros(avgPic.shape)
        locFromKey = []
        minHor = min(transpose(key)[0])
        minVert = min(transpose(key)[1])
        for keyItem in key:
            locFromKey.append([int((keyItem[0] - minHor) / 9 * 2 + 2), int((keyItem[1] - minVert) / 9 * 2 + 2)])
        for i, loc in enumerate(locFromKey):
            loadingPic[loc[1]][loc[0]] = max(loadingRateList[i])
    if showLoadingRate:
        colors, _ = getColors(len(atomLocations) + 1)
        alphaVal = 1.0 / (len(atomLocations) ** 0.7)
        avgFig, mainPlot = [[] for _ in range(2)]
        avgFig.append(go.Heatmap(z=avgPic, colorscale='Viridis', colorbar=go.ColorBar(x=1, y=0.15, len=0.3)))
        for err, loc, color, load, fitData in zip(loadingRateErr, atomLocations, colors, loadingRateList, loadFits):
            print('hi!', loc)
            mainPlot.append(go.Scatter(x=key, y=load, error_y={'type': 'data', 'array': err, 'color': color},
                                       mode='markers', name=str(loc), legendgroup=str(loc),
                                       marker={'color': color}, opacity=alphaVal))
            if loadingFitType is not None:
                print(loc, round_sig(fitData['vals'][1], 4), '+-', round_sig(fitData['errs'][1], 2))
                mainPlot.append(go.Scatter(x=fitData['x'], y=fitData['nom'], line={'color': color},
                                           legendgroup=str(loc), showlegend=False, opacity=alphaVal))
                mainPlot.append(go.Scatter(x=fitData['x'], y=fitData['nom'] + fitData['std'],
                                           opacity=alphaVal / 2, line={'color': color},
                                           legendgroup=str(loc), showlegend=False, hoverinfo='none'))
                mainPlot.append(go.Scatter(x=fitData['x'], y=fitData['nom'] - fitData['std'],
                                           opacity=alphaVal / 2, line={'color': color},
                                           legendgroup=str(loc), fill='tonexty', showlegend=False,
                                           hoverinfo='none'))
        mainPlot.append(go.Scatter(x=key, y=allLoadingRate, marker={'color': '#000000'},
                                   error_y={'type': 'data', 'array': allLoadingErr, 'color': "#000000"},
                                   mode='markers', name='avg', legendgroup='avg'))
        if loadingFitType is not None:
            print('avg fit:', round_sig(avgFits['vals'][1], 4), '+-', round_sig(avgFits['errs'][1], 2))
            mainPlot.append(go.Scatter(x=avgFits['x'], y=avgFits['nom'], line={'color': '#000000'},
                                       legendgroup='avg', showlegend=False, opacity=alphaVal))
            mainPlot.append(go.Scatter(x=avgFits['x'], y=avgFits['nom'] + avgFits['std'],
                                       opacity=alphaVal / 2, line={'color': '#000000'},
                                       legendgroup='avg', showlegend=False, hoverinfo='none'))
            mainPlot.append(go.Scatter(x=avgFits['x'], y=avgFits['nom'] - avgFits['std'],
                                       opacity=alphaVal / 2, line={'color': '#000000'},
                                       legendgroup='avg', fill='tonexty', showlegend=False,
                                       hoverinfo='none'))
        fig = make_subplots(
            rows=3, cols=12, print_grid=False, horizontal_spacing=0, vertical_spacing=0.05,
            specs=[[{'rowspan': 3, 'colspan': 9}, None, None, None, None, None, None, None, None, {'colspan': 2}, None,
                    {}],
                   [None, None, None, None, None, None, None, None, None, {'colspan': 3}, None, None],
                   [None, None, None, None, None, None, None, None, None, {'colspan': 3}, None, None]])
        for mainLine in mainPlot:
            fig.append_trace(mainLine, 1, 1)
        for avgPart in avgFig:
            fig.append_trace(avgPart, 3, 10)
        # for load in loadingPlot:
        #    fig.append_trace(load, 2, 10)
        # for counts in countsFig:
        #    fig.append_trace(counts, 1, 10)
        for hist in countsHist:
            fig.append_trace(hist, 2, 10)
        fig['layout'].update(barmode='overlay')
        fig['layout']['yaxis1'].update(title="Loading %", range=[0, 1])
        fig['layout']['xaxis1'].update(title=str(keyName))
        fig['layout']['yaxis2'].update(title="Count", range=[min(pic1Data.flatten()), max(pic1Data.flatten())])
        fig['layout']['xaxis2'].update(range=[0, len(pic1Data.flatten())])
        fig['layout']['yaxis3'].update(range=[min(pic1Data.flatten()), max(pic1Data.flatten())], showticklabels=False)
        fig['layout']['xaxis3'].update(showticklabels=False)
        # fig['layout']['yaxis4'].update(title="Loading %", range=[0,1])
        # fig['layout']['yaxis5'].update(title="Average Image")
        print('plotting figure...')
        iplot(fig)
        """
        legends = [str(loc) for loc in atomLocations]
        mainPlot = pointsWithBokeh(key, loadingRateList, colors, atomLocations,
                            scanType='Loading', width=900, height=500, keyName=keyName, errs=loadingRateErr, 
                            legends=legends, avgData=allLoadingRate, avgErrs=allLoadingErr)
        mainPlot.legend.click_policy = "mute"
        mainPlot.xaxis.axis_label = keyName
        mainPlot.yaxis.axis_label = "Loading Rate"
        setDefaultColors(mainPlot)
        """
    return key, loadingRateList, loadingRateErr, totalPic1AtomData, rawData


def Assembly(fileNumber, atomLocs1, pic1Num, atomLocs2=None, pic2Num=None, keyOffset=0, window=None,
             picsPerRep=2, histSecondPeakGuess=None, manualThreshold=None, fitType=None, allAtomLocs1=None,
             allAtomLocs2=None):
    """
    This function checks the efficiency of generating a picture;
    I.e. finding atoms at multiple locations at the same time.
    """
    (atomLocs1, atomLocs2, key, thresholds, pic1Data, pic2Data, fit, ensembleStats, avgPic, atomCounts, keyName,
     indvStatistics, lossAvg,
     lossErr) = standardAssemblyAnalysis(fileNumber, atomLocs1, pic1Num, atomLocs2=atomLocs2,
                                         pic2Num=pic2Num, keyOffset=keyOffset, window=window,
                                         picsPerRep=picsPerRep, histSecondPeakGuess=histSecondPeakGuess,
                                         manualThreshold=manualThreshold, fitType=fitType,
                                         allAtomLocs1=allAtomLocs1, allAtomLocs2=allAtomLocs2)
    # ######################## Plotting
    # get the colors for the plot.
    colors, colors2 = getColors(len(atomLocs1) + 1)
    mainPlot = [go.Scatter(x=key, y=ensembleStats['avg'], mode="markers", name='Ensemble',
                           error_y={"type": 'data', "array": ensembleStats['err'], 'color': '#000000'},
                           marker={'color': '#000000'}, legendgroup='ensemble'),
                go.Scatter(x=key, y=lossAvg, mode="markers", name='Loss',
                           error_y={"type": 'data', "array": lossErr, 'color': '#000000'},
                           marker={'color': '#000000', 'symbol': 'x', 'size': 10}, legendgroup='ensemble')
                ]
    # loss is the loss %, but for these plots it's the % of atoms lost regardless location. i.e. it looks at
    # number in first picture & number in second picture.
    for atomStats, loc, color in zip(indvStatistics, atomLocs1, colors):
        mainPlot.append(go.Scatter(x=key, y=atomStats['avg'], mode="markers", name=str(loc),
                                   error_y={"type": 'data', "array": atomStats['err'], 'color': color},
                                   marker={'color': color}, legendgroup=str(loc)))

    countsHist, countsFig, loadingPlot, avgFig = [[] for _ in range(4)]
    avgFig.append(go.Heatmap(z=avgPic, colorscale='Viridis', colorbar=go.ColorBar(x=1, y=0.15, len=0.3)))

    bins = []
    for data in pic1Data:
        b, _ = np.histogram(data, bins=100)
        bins.append(b)
    maxHistHeight = max(arr(bins).flatten())
    for data, loc, color, threshold in zip(pic1Data, atomLocs1, colors, thresholds):
        countsHist.append(go.Histogram(y=data, nbinsy=100, legendgroup=str(loc), showlegend=False, opacity=0.3,
                                       xbins=dict(start=min(pic1Data.flatten()), end=max(pic1Data.flatten())),
                                       marker=dict(color=color)))
        countsHist.append(go.Scatter(x=[0, maxHistHeight], y=[threshold, threshold],
                                     showlegend=False, mode='lines', line={'color': color, 'width': 1},
                                     hoverinfo='none', legendgroup=str(loc)))
        # countsFig.append(go.Scatter(y=data, mode='markers', marker={'color':color, 'size':1},
        #                            legendgroup=str(loc), showlegend=False))
        # countsFig.append(go.Scatter(x=[0,len(pic1Data[0].flatten())], y=[threshold,threshold], showlegend=False,
        #                             mode='lines', line={'color':color, 'width':1}, hoverinfo='none',
        #                             legendgroup=str(loc)))
        # loadingPlot.append(go.Scatter(x=key, y=load, mode="markers", name=str(loc),
        #                              marker ={'color' : color}, legendgroup=str(loc), showlegend=False))
        avgFig.append(go.Scatter(x=[loc[1]], y=[loc[0]], mode='markers', hoverinfo='none',
                                 showlegend=False, legendgroup=str(loc), marker={'size': 5, 'color': '#FF0000'}))
    """
    """
    fig = make_subplots(rows=3, cols=12, print_grid=False, horizontal_spacing=0, vertical_spacing=0,
                        specs=[[{'rowspan': 3, 'colspan': 9}, None, None, None, None, None, None, None, None,
                                {'colspan': 2}, None, {}],
                               [None, None, None, None, None, None, None, None, None, {'colspan': 3}, None, None],
                               [None, None, None, None, None, None, None, None, None, {'colspan': 3}, None, None]])
    for mainLine in mainPlot:
        fig.append_trace(mainLine, 1, 1)
    for avgPart in avgFig:
        fig.append_trace(avgPart, 3, 10)
    # for load in loadingPlot:
    #    fig.append_trace(load, 2, 10)
    for counts in countsFig:
        fig.append_trace(counts, 1, 10)
    for hist in countsHist:
        fig.append_trace(hist, 1, 12)
    fig['layout'].update(barmode='overlay')
    fig['layout']['yaxis1'].update(title="Ensemble %", range=[0, 1])
    fig['layout']['xaxis1'].update(title=str(keyName))
    fig['layout']['yaxis2'].update(title="Count", range=[min(pic1Data.flatten()), max(pic1Data.flatten())])
    fig['layout']['xaxis2'].update(range=[0, len(pic1Data[0].flatten())])
    fig['layout']['yaxis3'].update(range=[min(pic1Data.flatten()), max(pic1Data.flatten())], showticklabels=False)
    fig['layout']['xaxis3'].update(showticklabels=False)
    fig['layout']['yaxis4'].update(title="Loading %", range=[0, 1])
    fig['layout']['yaxis5'].update(title="Average Image")
    iplot(fig)
    return key, fig


def AnalyzeRearrangeMoves(rerngInfoAddress, fileNumber, locations, picNumber=2, threshold=300,
                          splitByNumberOfMoves=False, plotByNumberOfMoves=False, allLocsList=None):
    """
    Analyzes the rearrangement move log file and displays statistics for different types of moves.

    :param rerngInfoAddress:
    :param fileNumber:
    :param locations:
    :param picNumber:
    :param threshold:
    :param splitByNumberOfMoves:
    :param plotByNumberOfMoves:
    :param allLocsList:
    :return:
    """
    if plotByNumberOfMoves and not splitByNumberOfMoves:
        raise RuntimeError("Can't plot by number of moves if not splitting by number of moves.")
    locations = unpackAtomLocations(locations)
    if allLocsList is not None:
        allLocsList = unpackAtomLocations(allLocsList)
    # Open file and create list of moves.
    moveList = []
    with open(rerngInfoAddress) as centerLog:
        for i, line in enumerate(centerLog):
            if i < 5:
                continue
            txt = line.split(' ')
            if txt[0] == 'Rep' or txt[0] == 'Source:':
                continue
            if txt[0] == 'Moves:\n':
                moveList.append([])
                continue
            txt[4] = txt[4][:-1]
            move = '(' + str(txt[1]) + ',' + str(txt[3]) + ')->(' + str(txt[2]) + ',' + str(txt[4] + ')')
            moveList[-1].append(move)
    pics, _, _, repetitions = loadHDF5(fileNumber)
    moveData = {}
    if splitByNumberOfMoves:
        for i, move in enumerate(moveList):
            moveName = len(move)
            if moveName not in moveData:
                moveData[moveName] = [pics[2*i], pics[2*i+1]]
            else:
                moveData[moveName].append(pics[2*i])
                moveData[moveName].append(pics[2*i + 1])
    else:
        for i, move in enumerate(moveList):
            if len(move) == 0:
                moveName = 'No-Move'
            else:
                moveName = '{'
                for m in move:
                    moveName += m + ','
                moveName = moveName[:-2] + ')}'
            if moveName not in moveData:
                moveData[moveName] = [pics[2*i], pics[2*i+1]]
            else:
                moveData[moveName].append(pics[2*i])
                moveData[moveName].append(pics[2*i + 1])
    """
    netLossList = getNetLoss(pic1Atoms, pic2Atoms)
    lossAvg, lossErr = getNetLossStats(netLossList, repetitions)
    """
    (allPic1Data, allPic2Data, allPic1Atoms, allPic2Atoms,
     allLocsPic1Data, allLocsPic2Data, allLocsPic1Atoms, allLocsPic2Atoms) = [[] for _ in range(8)]
    for loc in locations:
        allPic1Data.append(normalizeData(pics, loc, 0, 2))
        allPic2Data.append(normalizeData(pics, loc, 1, 2))
    for point1, point2 in zip(allPic1Data, allPic2Data):
        allPic1Atoms.append(point1 > threshold)
        allPic2Atoms.append(point2 > threshold)
    if allLocsList is not None:
        for loc in allLocsList:
            allLocsPic1Data.append(normalizeData(pics, loc, 0, 2))
            allLocsPic2Data.append(normalizeData(pics, loc, 1, 2))
        for point1, point2 in zip(allLocsPic1Data, allLocsPic2Data):
            allLocsPic1Atoms.append(point1 > threshold)
            allLocsPic2Atoms.append(point2 > threshold)
    else:
        (allLocsPic1Data, allLocsPic2Data, allLocsPic1Atoms,
         allLocsPic2Atoms) = allPic1Data, allPic2Data, allPic1Atoms, allPic2Atoms
    allEvents = (getEnsembleHits(allPic2Atoms) if picNumber == 2 else getEnsembleHits(allPic1Atoms))
    allLossList = getNetLoss(allLocsPic1Atoms, allLocsPic2Atoms)
    allLossAvg, allLossErr = getNetLossStats(allLossList, len(allLossList))
    print('Average Loss:', allLossAvg, '\nLoss Error:', allLossErr)
    print('Total Average Assembly:', round_sig(np.mean(allEvents)),
          round_sig(np.std(allEvents)/np.sqrt(len(allEvents))))
    d = DataFrame()
    allLossAvg, allLossErr = [[], []]
    for keyName, data in moveData.items():
        data = arr(data)
        (pic1Data, pic1Atoms, pic2Data, pic2Atoms, pic1AllLocsData, pic1AllLocsAtoms, pic2AllLocsData,
         pic2AllLocsAtoms) = [[] for _ in range(8)]
        for loc in locations:
            pic1Data.append(normalizeData(data, loc, 0, 2).tolist())
            pic2Data.append(normalizeData(data, loc, 1, 2).tolist())
            pic1Atoms.append([])
            pic2Atoms.append([])
            for (point1, point2) in zip(pic1Data[-1], pic2Data[-1]):
                pic1Atoms[-1].append(point1 > threshold)
                pic2Atoms[-1].append(point2 > threshold)
        if allLocsList is not None:
            for loc in allLocsList:
                pic1AllLocsData.append(normalizeData(data, loc, 0, 2).tolist())
                pic2AllLocsData.append(normalizeData(data, loc, 1, 2).tolist())
                pic1AllLocsAtoms.append([])
                pic2AllLocsAtoms.append([])
                for (point1, point2) in zip(pic1AllLocsData[-1], pic2AllLocsData[-1]):
                    pic1AllLocsAtoms[-1].append(point1 > threshold)
                    pic2AllLocsAtoms[-1].append(point2 > threshold)
            lossList = getNetLoss(pic1AllLocsAtoms, pic2AllLocsAtoms)
            a, e = getNetLossStats(lossList, len(lossList))
            allLossAvg.append(a[0])
            allLossErr.append(e[0])
        atomEvents = (getEnsembleHits(pic2Atoms) if picNumber == 2 else getEnsembleHits(pic1Data))
        d[keyName] = [int(len(data)/2), np.mean(atomEvents), np.std(atomEvents)/np.sqrt(len(atomEvents))]
    allLossAvg = arr(allLossAvg)
    d = d.transpose()
    d.columns = ['occurances', 'success', 'error']
    d = d.sort_values('occurances', ascending=False)
    if plotByNumberOfMoves:
        # calculate the expected values
        maxNum = max(d.transpose().columns)
        singleProb = d.transpose()[1]['success']
        predictedNumMoves = np.arange(1, maxNum + 1)
        predictedProbs = singleProb**predictedNumMoves
        # plot stuff
        fig = [go.Scatter(x=d.transpose().columns, y=d['success'], error_y={'array': d['error']}, mode='markers',
                          name='Observed Data'),
               go.Scatter(x=predictedNumMoves, y=predictedProbs, mode='Markers', name='Expected'),
               go.Scatter(x=d.transpose().columns, y=allLossAvg, mode='markers', name='Loss')]
        tempLayout = go.Layout(xaxis={'title': 'Moves Made'}, yaxis={'title': 'Success Probability'})
        finalFig = go.Figure(data=fig, layout=tempLayout)
        iplot(finalFig)
    return d
