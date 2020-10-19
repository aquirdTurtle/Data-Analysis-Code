__version__ = "1.6"

"""
Most recent changes:
- separated loading and survival pics in the average pics in transfer
"""

import numpy as np
from numpy import array as arr
from IPython.display import display
from .Miscellaneous import transpose, getColors, errString
import plotly.graph_objs as go
from plotly.offline import iplot, plot as plotlyplot
from plotly.tools import make_subplots
from .MainAnalysis import standardAssemblyAnalysis, standardPopulationAnalysis, AnalyzeRearrangeMoves, analyzeScatterData
from . import TransferAnalysis
from pandas import DataFrame
from .fitters import linear
from . import AnalysisHelpers as ah
from .AnalysisHelpers import getFitsDataFrame


def ScatterData(fileNumber, atomLocs1, plotfit=True, **scatterOptions):
    (key, psSurvivals, psErrors, fitData, fitFin, survivalData, survivalErrs,
     survivalFits, atomLocs1) = analyzeScatterData(fileNumber, atomLocs1, **scatterOptions)
    surv = arr(psSurvivals).flatten()
    err = arr(psErrors).flatten()
    mainPlot = []
    color = '#000000'
    legend = 'Site-Avg'
    mainPlot.append(go.Scatter(x=key, y=surv, mode='markers',
                               error_y=dict(type='data', array=err, visible=True, color=color),
                               marker={'color': color}, name=legend, legendgroup=legend))
    if plotfit:
        if fitData['vals'] is not None:
            mainPlot.append(go.Scatter(x=fitData['x'], y=arr(linear.f(fitData['x'], *fitData['vals'])), mode='line',
                                       line={'color': color}, legendgroup=legend, showlegend=False))
            alphaVal = 0.5
            mainPlot.append(go.Scatter(x=fitData['x'], y=fitData['nom'], line={'color': color},
                                       legendgroup=legend, showlegend=False))
            mainPlot.append(go.Scatter(x=fitData['x'], y=fitData['nom'] + fitData['std'],
                                       opacity=alphaVal / 2, line={'color': color},
                                       legendgroup=legend, showlegend=False, hoverinfo='none'))
            mainPlot.append(go.Scatter(x=fitData['x'], y=fitData['nom'] - fitData['std'],
                                       opacity=alphaVal / 2, line={'color': color},
                                       legendgroup=legend, fill='tonexty', showlegend=False,
                                       hoverinfo='none', fillcolor='rgba(0, 0, 0, ' + str(alphaVal / 2) + ')'))
    legends = [str(loc) for loc in atomLocs1]
    pltColors, pltColors2 = getColors(len(atomLocs1) + 1)
    for atomData, color, fit, legend in zip(survivalData, pltColors, survivalFits, legends):
        mainPlot.append(go.Scatter(x=key, y=atomData, mode='markers',
                                   error_y=dict(type='data', array=err, visible=True, color=color),
                                   marker={'color': color}, legendgroup=legend, name=legend))
        if plotfit:
            if fit['vals'] is not None:
                mainPlot.append(go.Scatter(x=fit['x'], y=fit['nom'], line={'color': color},
                                           legendgroup=legend, showlegend=False))
                mainPlot.append(go.Scatter(x=fit['x'], y=fit['nom'] + fit['std'],
                                           opacity=alphaVal / 2, line={'color': color},
                                           legendgroup=legend, showlegend=False, hoverinfo='none'))
                mainPlot.append(go.Scatter(x=fit['x'], y=fit['nom'] - fit['std'],
                                           opacity=alphaVal / 2, line={'color': color},
                                           legendgroup=legend, fill='tonexty', showlegend=False,
                                           hoverinfo='none', fillcolor='rgba(0, 0, 0, ' + str(alphaVal / 2) + ')'))

    l = go.Layout(title='Survival Vs. Atoms Loaded', xaxis={'title': 'Atoms loaded'}, yaxis=dict(title='Survival %'))
    f = go.Figure(data=mainPlot, layout=l)
    iplot(f)
    print('Fit Vals:', fitData['vals'])
    print('Fit errs:', fitData['errs'])


def Survival(fileNumber, atomLocs, **TransferArgs):
    """See corresponding transfer function for valid TransferArgs."""
    return Transfer(fileNumber, atomLocs, atomLocs, **TransferArgs)


def Transfer(fileNumber, atomLocs1, atomLocs2, show=True, fitModules=[None], showCounts=False, showGenerationData=False,
             histBins=150, showFitDetails=False, savePlotlyFig=False, leanPlot=False, **standardTransferArgs):
    """
    Standard data analysis package for looking at survival rates throughout an experiment.
    """
    avgColor = '#FFFFFF'
    res = TransferAnalysis.standardTransferAnalysis(fileNumber, atomLocs1, atomLocs2, fitModules=fitModules,
                                   **standardTransferArgs)
    (atomLocs1, atomLocs2, transferData, transferErrs, initPopulation, pic1Data, keyName, key,
     repetitions, initThresholds, fits, avgTransferData, avgTransferErr, avgFit, avgPics, otherDimValues,
     locsList, genAvgs, genErrs, tt, transVarAvg, transVarErr, initAtomImages, transAtomImages,
     pic2Data, transThresholds, fitModules, transThresholdSame, basicInfoStr) = res
    if not show:
        return key, transferData, transferErrs, initPopulation
    
    # get the colors for the plots.
    pltColors, pltColors2 = getColors(len(locsList) + 1)
    scanType = "S." if atomLocs1 == atomLocs2 else "T."
    if scanType == "S.":
        legends = [r"%d,%d " % (loc1[0], loc1[1]) + (scanType + "% = " + str(errString(d[0], e[0])) if len(d) == 1
                   else "") for loc1, d, e in zip(locsList, transferData, transferErrs)]

    elif otherDimValues[0] is not None:
        legends = [r"%d,%d>%d,%d @%d " % (loc1[0], loc1[1], loc2[0], loc2[1], other) +
                   (scanType + "%=" + str(errString(d[0]), e[0]) if len(d) == 1 else "")
                   for loc1, loc2, d, e, other in zip(locsList, locsList, transferData, transferErrs,
                                                      otherDimValues)]
    else:
        legends = [r"%d,%d>%d,%d " % (loc1[0], loc1[1], loc2[0], loc2[1]) +
                   (scanType + "%=" + errString(d[0], e[0]) if len(d) == 1 else "")
                   for loc1, loc2, d, e in zip(locsList, locsList, transferData, transferErrs)]
    transferErrs = list(transferErrs)
    # Make the plots
    alphaVal = 0.5
    mainPlot, countsHist, countsFig, initPopPlot = [[] for _ in range(4)]
    avgFigs = [[] for _ in range(2)]
    avgFigs[0].append(go.Heatmap(z=avgPics[0], colorscale='Viridis'))
    avgFigs[1].append(go.Heatmap(z=avgPics[1], colorscale='Viridis'))
    fitCharacters = []
    
    if fitModules[0] is not None and showFitDetails:
        frames = getFitsDataFrame(fits, fitModules, avgFit)
        for frame in frames:
            display(frame)
    for data, err, loc, color, legend, fitData, gen, genErr, module in zip(transferData, transferErrs, locsList, pltColors,
                                                      legends, fits, genAvgs, genErrs, fitModules):
        mainPlot.append(go.Scatter(x=key, y=data, opacity=alphaVal, mode="markers", name=legend,
                                   error_y={"type": 'data', 'symmetric':False, "array": [e[1] for e in err], 
                                            'arrayminus':[e[0] for e in err],
                                            'color': color, 'visible':True},
                                   marker={'color': color}, legendgroup=legend))
        if showGenerationData:
            mainPlot.append(go.Scatter(x=key, y=gen, opacity=alphaVal, mode="markers", name=legend,
                                       error_y={"type": 'data', "array": genErr, 'color': color, 'visible':True},
                                       marker={'color': color, 'symbol': 'star'}, legendgroup=legend, showlegend=False))
        if fitModules[0] is not None:
            if fitData['vals'] is None:
                print(loc, 'Fit Failed!')
                continue
            if showFitDetails:
                fitCharacters.append(module.fitCharacter(fitData['vals']))
                mainPlot.append(go.Scatter(x=[0], y=[0.5], mode='text', text=[module.getFitCharacterString(fitData['vals'])], 
                                           textposition="top center", legendgroup=legend, showlegend=False))
                
            mainPlot.append(go.Scatter(x=fitData['x'], y=fitData['nom'], line={'color': color},
                                       legendgroup=legend, showlegend=False, opacity=alphaVal))
            if fitData['std'] is not None:
                rgbcolor = color.strip('#')
                rgbColor = tuple(int(rgbcolor[i:i+2], 16) for i in (0, 2, 4))
                mainPlot.append(go.Scatter(x=fitData['x'], y=fitData['nom'] + fitData['std'],
                                           opacity=alphaVal / 2, line={'color': color},
                                           legendgroup=legend, showlegend=False, hoverinfo='none'))
                mainPlot.append(go.Scatter(x=fitData['x'], y=fitData['nom'] - fitData['std'],
                                           opacity=alphaVal / 2, line={'color': color},
                                           legendgroup=legend, fill='tonexty', showlegend=False,
                                           hoverinfo='none', 
                                           fillcolor='rgba('+str(rgbColor[0])+', '+str(rgbColor[1])+', '+str(rgbColor[2])+', '+str(alphaVal/2) + ')'))
    transferPic = np.zeros(avgPics[0].shape)
    for i, loc in enumerate(atomLocs1):
        transferPic[loc[0], loc[1]] = np.mean(transferData[i])
    if fitModules[0] is not None and showFitDetails:
        print('Fit Character:', fitCharacters)
        print('Average Fit Character:', np.mean(fitCharacters), np.median(fitCharacters))
        fitCharacterPic, vmin, vmax = ah.genAvgDiscrepancyImage(fitCharacters, avgPics[0].shape, atomLocs1)
        fitCharacterFig = [go.Heatmap(z=fitCharacterPic, colorscale='Viridis', colorbar=go.heatmap.ColorBar(x=1, y=0.15, len=0.3))]
        layout = go.Layout(title='Fit-Character Pic')
        iplot(go.Figure(data=fitCharacterFig, layout=layout))
    maxHeight = np.max(arr([np.histogram(data.flatten(), bins=histBins)[0] for data in pic1Data]).flatten())
    for data, pop, loc1, loc2, color, threshold, legend in zip(pic1Data, initPopulation, atomLocs1, atomLocs2, pltColors,
                                                         initThresholds, legends):
        countsHist.append(go.Histogram(x=data, nbinsx=histBins, legendgroup=legend, showlegend=False, opacity=alphaVal/2,
                                       marker=dict(color=color)))
        countsHist.append(go.Scatter(y=[0, maxHeight], x=[threshold[0].t, threshold[0].t], showlegend=False,
                                     mode='lines', line={'color': color, 'width': 1}, hoverinfo='none',
                                     legendgroup=legend))
        initPopPlot.append(go.Scatter(x=key, y=pop, mode="markers", name=str(loc1),
                                      marker={'color': color}, legendgroup=legend, showlegend=False,
                                      opacity=alphaVal))
        
        avgFigs[0].append(go.Scatter(x=[loc1[1]], y=[loc1[0]], mode='markers', hoverinfo='none',
                                 showlegend=False, legendgroup=legend, marker={'size': 2, 'color': '#FF0000'}))
        avgFigs[1].append(go.Scatter(x=[loc2[1]], y=[loc2[0]], mode='markers', hoverinfo='none',
                                 showlegend=False, legendgroup=legend, marker={'size': 2, 'color': '#FF0000'}))
    # average stuff
    mainPlot.append(go.Scatter(x=key, y=avgTransferData, mode="markers", name='avg',
                               
                               error_y={"type": 'data', 'color': avgColor, 'symmetric':False, 
                                        "array": [e[1] for e in avgTransferErr], 'arrayminus':[e[0] for e in avgTransferErr]},
                               marker={'color': avgColor}, legendgroup='avg'))
    if fitModules[0] is not None:
        mainPlot.append(go.Scatter(x=avgFit['x'], y=avgFit['nom'], line={'color': avgColor},
                                   legendgroup='avg', showlegend=False, opacity=alphaVal))
        mainPlot.append(go.Scatter(x=[0], y=[0.5], mode='text', text=[fitModules[-1].getFitCharacterString(avgFit['vals'])], 
                                   textposition="top center", legendgroup='avg', showlegend=False))
        if avgFit['std'] is not None:
            mainPlot.append(go.Scatter(x=avgFit['x'], y=avgFit['nom'] + avgFit['std'],
                                       opacity=alphaVal / 2, line={'color': avgColor},
                                       legendgroup='avg', showlegend=False, hoverinfo='none'))
            mainPlot.append(go.Scatter(x=avgFit['x'], y=avgFit['nom'] - avgFit['std'],
                                       opacity=alphaVal / 2, line={'color': avgColor},
                                       legendgroup='avg', fill='tonexty', showlegend=False,
                                       hoverinfo='none'))
    if showCounts:
        avgOnly = True
        if avgOnly:
            countsFig.append(go.Scatter(y=arr(transpose(pic1Data)).flatten(), mode='markers',
                                        marker={'color': avgColor, 'size': 1}, legendgroup='avg', showlegend=False))
        countsHist.append(go.Histogram(x=atomCounts.flatten(), nbinsx=200, legendgroup='avg',
                                       showlegend=False, opacity=0.1, marker=dict(color=avgColor),
                                       xbins=dict(start=min(pic1Data.flatten()), end=max(pic1Data.flatten()))))
        d, _ = np.histogram(atomCounts.flatten(), bins=200)
        countsHist.append(go.Scatter(x=[0, max(d) / 2], y=[np.mean([thresh.t for thresh in initThresholds]), np.mean([thresh.t for thresh in initThresholds])],
                                     showlegend=False, mode='lines', line={'color': avgColor, 'width': 1},
                                     hoverinfo='none', legendgroup='avg'))
        # format and arrange plots. large grid is mostly to precisely place the histogram.
        n = None
        r = 'rowspan'
        c = 'colspan'
        fig = make_subplots(
            rows=3, cols=24, print_grid=False, horizontal_spacing=0.03, vertical_spacing=0.05,
            specs=[[{r: 3, c: 18}, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, {c: 4}, n, n, n,      {}, n],
                   [n,             n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, {c: 6}, n, n, n,      n,  n],
                   [n,             n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, {c: 3}, n, n, {c: 3}, n,  n]])
        traceSets = [mainPlot, avgFigs[0], avgFigs[1], initPopPlot, countsFig, countsHist]
        
        for mainLine in mainPlot:
            fig.add_trace(mainLine, 1, 1)
        for avgPart in avgFigs[0]:
            fig.add_trace(avgPart, 3, 19)
        for avgPart in avgFigs[1]:
            fig.add_trace(avgPart, 3, 22)
        for pop in initPopPlot:
            fig.add_trace(pop, 2, 19)
        for counts in countsFig:
            if not leanPlot:
                fig.add_trace(counts, 1, 19)
        for hist in countsHist:
            if not leanPlot:
                fig.add_trace(hist, 1, 23)
        fig['layout']['yaxis2'].update(title="Count", range=[min(pic1Data.flatten()), max(pic1Data.flatten())])
        fig['layout']['xaxis2'].update(range=[0, len(pic1Data[0].flatten())])
        fig['layout']['yaxis3'].update(range=[min(pic1Data.flatten()), max(pic1Data.flatten())],
                                       showticklabels=False)
        fig['layout']['xaxis3'].update(showticklabels=False)
        fig['layout']['yaxis4'].update(title="Initial Population %", range=[0, 1])
        fig['layout']['yaxis5'].update(title="Average Image")
    else:
        # format and arrange plots. large grid is mostly to precisely place the histogram.
        n = None
        r = 'rowspan'
        c = 'colspan'
        fig = make_subplots(
            rows=3, cols=24, print_grid=False, horizontal_spacing=0.03, vertical_spacing=0.05,
            specs=[[{r: 3, c: 18}, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, {c: 4}, n, n, n,      {}, n],
                   [n,             n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, {c: 6}, n, n, n,      n,  n],
                   [n,             n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, {c: 3}, n, n, {c: 3}, n,  n]])
        for mainLine in mainPlot:
            fig.add_trace(mainLine, 1, 1)
        for avgPart in avgFigs[0]:
            fig.add_trace(avgPart, 3, 19)
        for avgPart in avgFigs[1]:
            fig.add_trace(avgPart, 3, 22)
        for pop in initPopPlot:
            fig.add_trace(pop, 2, 19)
        for hist in countsHist:
            if not leanPlot:
                fig.add_trace(hist, 1, 19)
        fig['layout']['yaxis3'].update(title="Initial Population %", range=[0, 1])
        fig['layout']['yaxis5'].update(title="Average Image")
    fig['layout'].update(barmode='overlay', plot_bgcolor="rgb(10,10,10)", paper_bgcolor="rgb(0,0,0)")
    fig['layout']['yaxis1'].update(title=scanType + " %", range=[0, 1])
    fig['layout']['xaxis1'].update(title=str(keyName))
    iplot(fig)
    if savePlotlyFig:
        plotlyplot(fig, filename='plotly_plot.html')
    return key, transferData, transferErrs, initPopulation, fits, avgFit, genAvgs, genErrs, fitCharacters



def Loading(fileNum, atomLocations, **TransferArgs):
    return Population(fileNum, atomLocations, whichPic=0, **TransferArgs)

def Population(fileNum, atomLocations, showLoadingRate=True, showLoadingPic=False, plotCounts=False, countsMain=False,
            indvHist=True, histMain=False, simplePlot=False, showTotalHist=False, histBins=100, picsPerRep=1, whichPic=0, **StandardArgs):
    """
    Standard data analysis package for looking at loading rates throughout an experiment.
    return key, loadingRateList, loadingRateErr

    See standardLoadingAnalysis for valid standardArgs

    This routine is designed for analyzing experiments with only one picture per cycle. Typically
    These are loading exeriments, for example. There's no survival calculation.

    :param fileNum:
    :param atomLocations:
    :param showIndividualHist:
    :param showLoadingRate:
    :param showLoadingPic:
    :param StandardArgs:
    :param countsMain:
    :param plotCounts:
    :return:
    """

    res = standardPopulationAnalysis(fileNum, atomLocations, whichPic, picsPerRep, **StandardArgs)
    (pic1Data, thresholds, avgPic, key, loadingRateErr, loadingRateList, allLoadingRate, allLoadingErr, loadFits,
            fitModules, keyName, totalPic1AtomData, rawData, atomLocations, avgFits, atomImages, totalAvg, totalErr) = res 
    maxHeight = np.max(arr([np.histogram(data.flatten(), bins=histBins)[0] for data in pic1Data]).flatten())

    totalHist = []
    if showTotalHist:
        d, _ = np.histogram(pic1Data.flatten(), bins=100)
        totalHist.append(go.Histogram(x=pic1Data.flatten(), nbinsx=100, legendgroup='avg',
                                      showlegend=False, xbins=dict(start=min(pic1Data.flatten()),
                                                                   end=max(pic1Data.flatten())),
                                      marker=dict(color='#000000')))
        totalHist.append(go.Scatter(x=[np.mean(thresholds), np.mean(thresholds)], y=[0, max(d)],
                                    showlegend=False, mode='lines', line={'color': '#000000', 'width': 1},
                                    hoverinfo='none', legendgroup='avg'))
    colors, _ = getColors(len(atomLocations) + 1)
    countsFig = []
    if plotCounts:
        for atom, color in zip(atomLocations, colors):
            countsFig.append(go.Scatter(x=list(range(pic1Data[atom].flatten().size)), y=pic1Data[atom].flatten(),
                                        showlegend=False, mode='markers', line={'color': '#000000', 'width': 1},
                                        hoverinfo='none', legendgroup=str(atom), marker={'color': color, 'size':1}))
    indvHistFig = []
    alphaVal = 0.5
    if indvHist:
        for i, (atom, color, threshold) in enumerate(zip(atomLocations, colors, thresholds)):
            indvHistFig.append(go.Histogram(x=pic1Data[i].flatten(), legendgroup=str(atom), name=str(atom),
                                            nbinsx=histBins, showlegend=simplePlot, marker=dict(color=color),
                                            opacity=alphaVal))
            indvHistFig.append(go.Scatter(y=[0, maxHeight], x=[threshold, threshold], showlegend=False,
                                          mode='lines', line={'color': color, 'width': 1}, hoverinfo='none',
                                          legendgroup=str(atom)))
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
        avgFig, mainPlot = [[] for _ in range(2)]
        avgFig.append(go.Heatmap(z=avgPic, colorscale='Viridis', colorbar=go.heatmap.ColorBar(x=1, y=0.15, len=0.3)))
        for err, loc, color, load, fitData in zip(loadingRateErr, atomLocations, colors, loadingRateList, loadFits):
            mainPlot.append(go.Scatter(x=key, y=load, error_y={'type': 'data', 'array': err, 'color': color},
                                       mode='markers', name=str(loc), legendgroup=str(loc),
                                       marker={'color': color}, opacity=alphaVal))
            avgFig.append(go.Scatter(x=[loc[1]], y=[loc[0]], mode='markers', hoverinfo='none',
                              showlegend=False, legendgroup=str(loc), marker={'size': 2, 'color': '#FF0000'}))
            if fitModules is not None:
                print(loc, errString(fitData['vals'][1], fitData['errs'][1], 4))
                mainPlot.append(go.Scatter(x=fitData['x'], y=fitData['nom'], line={'color': color},
                                           legendgroup=str(loc), showlegend=False, opacity=alphaVal))
                if fitData['std'] is not None:
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
        if fitModules is not None:
            print('avg fit:', errString(avgFits['vals'][1], avgFits['errs'][1], 4))
            mainPlot.append(go.Scatter(x=avgFits['x'], y=avgFits['nom'], line={'color': '#000000'},
                                       legendgroup='avg', showlegend=False, opacity=alphaVal))
            if avgFits['std'] is not None:
                mainPlot.append(go.Scatter(x=avgFits['x'], y=avgFits['nom'] + avgFits['std'],
                                           opacity=alphaVal / 2, line={'color': '#000000'},
                                           legendgroup='avg', showlegend=False, hoverinfo='none'))
                mainPlot.append(go.Scatter(x=avgFits['x'], y=avgFits['nom'] - avgFits['std'],
                                           opacity=alphaVal / 2, line={'color': '#000000'},
                                           legendgroup='avg', fill='tonexty', showlegend=False,
                                           hoverinfo='none'))
        if simplePlot:
            if countsMain:
                plotData = countsFig
                layout = go.Layout(xaxis={'title': 'Pic #'}, yaxis={'title': 'Count #'})
            elif histMain:
                if showTotalHist:
                    histToShow = totalHist
                elif indvHist:
                    histToShow = indvHistFig
                else:
                    histToShow = []
                plotData = histToShow
                layout = go.Layout(xaxis={'title': 'Pic #'}, yaxis={'title': 'Count #'}, barmode='overlay')
            else:
                plotData = mainPlot
                layout = go.Layout(xaxis={'title': keyName}, yaxis={'title': 'Loading %', 'range': [0,1]})
            fig = go.Figure(data=plotData, layout=layout)
        else:
            fig = make_subplots(
                rows=3, cols=12, print_grid=False, horizontal_spacing=0, vertical_spacing=0.05,
                specs=[[{'rowspan': 3, 'colspan': 9}, None, None, None, None, None, None, None, None, {'colspan': 2},
                        None, {}],
                       [None, None, None, None, None, None, None, None, None, {'colspan': 3}, None, None],
                       [None, None, None, None, None, None, None, None, None, {'colspan': 3}, None, None]])
            if countsMain:
                mainLoc = (1, 10)
                mainNum = '2'
                countsNum = '1'
                countsLoc = (1, 1)
            else:
                mainLoc = (1, 1)
                mainNum = '1'
                countsNum = '2'
                countsLoc = (1, 10)
            for mainLine in mainPlot:
                fig.add_trace(mainLine, mainLoc[0], mainLoc[1])
            for avgPart in avgFig:
                fig.add_trace(avgPart, 3, 10)
            for counts in countsFig:
                fig.add_trace(counts, countsLoc[0], countsLoc[1])
            if showTotalHist:
                histToShow = totalHist
            elif indvHist:
                histToShow = indvHistFig
            else:
                histToShow = []
            for hist in histToShow:
                fig.add_trace(hist, 2, 10)
            layout = go.Layout(plot_bgcolor="rgb(182, 215, 168)", paper_bgcolor="rgb(182, 215, 168)")
            fig['layout'] = layout
            fig['layout'].update(barmode='overlay', plot_bgcolor="rgb(182, 215, 168)", paper_bgcolor="rgb(182, 215, 168)")
            fig['layout']['yaxis' + mainNum].update(title="Loading %", range=[0, 1])
            fig['layout']['xaxis' + mainNum].update(title=str(keyName))
            fig['layout']['yaxis' + countsNum].update(title="Count", range=[min(pic1Data.flatten()), max(pic1Data.flatten())])
            fig['layout']['xaxis' + countsNum].update(range=[0, len(pic1Data[0].flatten())])
            fig['layout']['yaxis3'].update(range=[min(pic1Data.flatten()), max(pic1Data.flatten())], showticklabels=False)
            fig['layout']['xaxis3'].update(showticklabels=False)
            # fig['layout']['yaxis4'].update(title="Loading %", range=[0,1])
            # fig['layout']['yaxis5'].update(title="Average Image")
        print('plotting figure...')
        iplot(fig)
    return key, loadingRateList, loadingRateErr, totalPic1AtomData, rawData, allLoadingRate


def Assembly(fileNumber, atomLocs1, pic1Num, **standardAssemblyArgs):
    """
    This function checks the efficiency of generating a picture;
    I.e. finding atoms at multiple locations at the same time.
    """
    (atomLocs1, atomLocs2, key, thresholds, pic1Data, pic2Data, fit, ensembleStats, avgPic, atomCounts, keyName,
     indvStatistics, lossAvg,
     lossErr) = standardAssemblyAnalysis(fileNumber, atomLocs1, pic1Num, **standardAssemblyArgs)
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
    avgFig.append(go.Heatmap(z=avgPic, colorscale='Viridis', colorbar=go.heatmap.ColorBar(x=1, y=0.15, len=0.3)))

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
        fig.add_trace(mainLine, 1, 1)
    for avgPart in avgFig:
        fig.add_trace(avgPart, 3, 10)
    # for load in loadingPlot:
    #    fig.add_trace(load, 2, 10)
    for counts in countsFig:
        fig.add_trace(counts, 1, 10)
    for hist in countsHist:
        fig.add_trace(hist, 1, 12)
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


def Rearrange(rerngInfoAddress, fileNumber, locations, **rearrangeArgs):
    """

    :param rerngInfoAddress:
    :param fileNumber:
    :param locations:
    :param rearrangeArgs:
    :return:
    """
    allData, fits, pics, moves = AnalyzeRearrangeMoves(rerngInfoAddress, fileNumber, locations, **rearrangeArgs)
    for loc in allData:
        fig = [go.Scatter(x=allData[loc].transpose().columns, y=allData[loc]['success'],
                          error_y={'array': allData[loc]['error']}, mode='markers', name='Observed Data')]
        if fits['No-Target-Split'] is not None:
            xpts = np.linspace(0, len(allData[loc].transpose().columns), 100)
            fig.append(go.Scatter(x=xpts, y=fitFunc.exponentialDecay(xpts, *fits[loc]), name='Fit-Values:' + str(fits[loc])))
        tempLayout = go.Layout(xaxis={'title': 'Moves Made'}, yaxis={'title': 'Success Probability'},
                               title=loc)
        finalFig = go.Figure(data=fig, layout=tempLayout)
        iplot(finalFig)
    for _, d in allData.items():
        display(d)
    return allData, pics, moves
