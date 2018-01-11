__version__ = "1.0"

from matplotlib.cm import get_cmap

from bokeh.plotting import figure, ColumnDataSource, show
from bokeh.io import export_png, output_notebook, output_file, reset_output
from bokeh.layouts import gridplot, column
from bokeh.models import LinearColorMapper, ColorBar
from bokeh.models import ColorBar as bok_ColorBar
import bokeh.models as bokModels

import numpy as np
from numpy import array as arr

import FittingFunctions as fitFunc
from Miscellaneous import round_sig, transpose
import MarksConstants as consts

from IPython.display import display, Image


def atomHistWithBokeh(key, atomLocs, pic1Data, bins, binData, fitVals, thresholds, avgPic,
                      atomCount, variationNumber, outputFileAddr=None, interactive=False):
    """
    Makes a standard atom histogram-centric plot. 
    key:
    :param atomLocs: list of coordinate pairs where atoms are. element 0 is row#, element 1 is column#
    :param pic1Data: list (for each location) of ordered lists of the counts on a pixel for each experiment
    :param bins: list (for each location) of the centers of the bins used for the histrogram.
    :param binData: list (for each location) of the accumulations each bin, whose center is stored in "bins" argument.
    :param fitVals: the fitted values of the 2D Gaussian fit of the fit bins and data
    :param thresholds: the found (or set) atom detection thresholds for each pixel.
    :param avgPic: the average of all the pics in this series.
    """
    # Make colormap. really only need len(locs) + 1 rgbs, but adding an extra makes the spacing of the colors
    # on this colormap more sensible.
    cmapRGB = get_cmap('gist_rainbow', len(atomLocs))
    colors = [cmapRGB(i)[:-1] for i in range(len(atomLocs))]
    colors = ['#%02x%02x%02x' % tuple(int(255 * color[i]) for i in range(len(color))) for color in colors]
    # Setup grid
    if key is None:
        keyVal = 'Average'
    else:
        keyVal = str(key[variationNumber])

    bokehFig = figure(
        title="Key Value = " + keyVal + "\nAvg Loading =" + str(np.mean(atomCount / len(pic1Data[0]))),
        plot_width=650, plot_height=450)
    fineXData = np.linspace(min(list([item for sublist in pic1Data for item in sublist])),
                            max(list([item for sublist in pic1Data for item in sublist])), 500);
    alphaVal = 1.0 / (len(atomLocs) ** 0.75)
    for i, loc in enumerate(atomLocs):
        legendEntry = (str(loc) + ' T=' + str(round_sig(thresholds[i])) + ', L = '
                       + str(round_sig(atomCount[i] / len(pic1Data[0]))))
        renderer = bokehFig.quad(top=binData[i], bottom=0, left=bins[i] - 5, right=bins[i] + 5,
                                 fill_color=colors[i], line_color=None, alpha=alphaVal, legend=legendEntry,
                                 muted_alpha=1, muted_line_color="#FFFFFF", muted_color=colors[i])
        bokehFig.line(fineXData, fitFunc.doubleGaussian(fineXData, *fitVals[i], 0),
                      line_color=colors[i], line_width=2, alpha=alphaVal)
        bokehFig.ray(x=[thresholds[i]], y=[0], length=0, angle=consts.pi / 2, line_width=1, color=colors[i],
                     alpha=alphaVal)
    bokehFig.xaxis.axis_label = "Pixel Counts"
    bokehFig.yaxis.axis_label = "Occurrence Count"
    # I'm using mute in reverse, making it much brighter when "muted".
    bokehFig.legend.click_policy = "mute"
    setDefaultColors(bokehFig)
    countsFig = CountsPlotWithBokeh(pic1Data, thresholds, atomLocs, colors)
    avgFig = AvgImgWithBokeh(avgPic, atomLocs)
    #
    l = gridplot([[bokehFig, column(countsFig, avgFig)]])
    bokehOutput(l, interactive)


def pointsWithBokeh(key, data, colors, atomLocations, keyName='', legends=None, errs=None, scanType="Data", width=650,
                    height=450, small=False, avgData=None, avgErrs=None):
    """
    Make an error-bar plot of the inputted points.
    """
    if len(keyName) > 1:
        varying = ""
        for name in keyName:
            varying += name + ", "
        titletxt = varying + " Atom " + scanType + " Scan"
    else:
        titletxt = keyName + " Atom " + scanType + " Scan"
    if type(data[0]) is np.float64 or len(data[0]) == 1:
        titletxt = keyName + " Atom " + scanType + " Point. " + scanType + " % = \n"
        for atomData in data:
            titletxt += str(atomData) + ", "
    if len(key.shape) > 1:
        xlim = (min(key[:][0]) - (max(key[:][0]) - min(key[:][0])) / len(key[:][0]), max(key[:][0])
                          + (max(key[:][0]) - min(key[:][0])) / len(key[:][0]))
    elif not min(key) == max(key):
        xlim = (min(key) - (max(key) - min(key)) / len(key), max(key)
                          + (max(key) - min(key)) / len(key))
    else:
        xlim = (key[0] - 1, key[0] + 1)
    if errs is None:
        errs = [None] * len(atomLocations)
    else:
        errs = list(errs)
    hoverData = bokModels.HoverTool(tooltips=[("Location", "@atomLocation"),
                                              ("Key Value", "@xval"),
                                              (scanType + " %", "@yval")])
    alphaVal = 1.0 / (len(atomLocations)**0.75)
    pointPlt = figure(title=titletxt, y_range=(0, 1), x_range=xlim, plot_width=width, plot_height=height,
                      tools=[hoverData, 'pan'])
    source = [dict(xval=key, yval=data[i], atomLocation=[atomLocations[i]]*len(key))
              for i in range(len(atomLocations))]
    if legends is None:
        legends = [""] * len(atomLocations)
    for i, atomLoc in enumerate(atomLocations):
        bokErrbar(pointPlt,source=source[i], yerr=errs[i], legend=legends[i], color=colors[i],
                  hoverData=hoverData, muted_color=colors[i], alpha=alphaVal)
    if avgData is not None:
        avgSource = dict(xval=key, yval=avgData, atomLocation=['avg']*len(key) )
        bokErrbar(pointPlt, source=avgSource, yerr=avgErrs, legend="Average Value", color="#000000",
                  hoverData=hoverData, muted_color="#000000", alpha=1, muted_alpha=0.1)
    setDefaultColors(pointPlt)
    pointPlt.xaxis.ticker = key
    pointPlt.xaxis.major_label_orientation = np.pi/4
    pointPlt.legend.click_policy = "mute"
    if small:
        pointPlt.xaxis.major_label_text_font_size = '5pt'
        pointPlt.yaxis.major_label_text_font_size = '5pt'
        pointPlt.title.text_font_size = '6pt'
        pointPlt.xaxis.axis_label_text_font_size = '6pt'
        pointPlt.yaxis.axis_label_text_font_size = '6pt'
    else:
        pointPlt.yaxis.axis_label = scanType + " Probability"
        pointPlt.xaxis.axis_label = keyName
    return pointPlt


def AvgImgWithBokeh(avgPic, atomLocs, width=225, height=225, titleTxt='Avg Image', small=True):
    yvals, xvals = np.meshgrid(np.arange(avgPic.shape[0]), np.arange(avgPic.shape[1]))
    source = ColumnDataSource(data=dict(
        left=(xvals - 0.5).flatten(),
        right=(xvals + 0.5).flatten(),
        top=(yvals + 0.5).flatten(),
        bottom=(yvals - 0.5).flatten(),
        x=xvals.flatten(),
        y=yvals.flatten(),
        imgVals=arr(transpose(avgPic)).flatten()))

    hoverData = bokModels.HoverTool(tooltips=[("(row,col)", "(@y, @x)"),
                                              ("Avg", "@imgVals")])

    avgFig = figure(title=titleTxt, x_range=(-0.5, avgPic.shape[1] - 0.5),
                        y_range=(-0.5, avgPic.shape[0] - 0.5), plot_width=width, plot_height=height,
                        tools=[hoverData, 'pan'])

    cmap = LinearColorMapper(palette="Inferno256",
                             low=min(avgPic.flatten()), high=max(avgPic.flatten()))
    # create squares for hover
    boxes = avgFig.quad(left='left', right='right', top='top', bottom='bottom', alpha=0, source=source)

    avgFig.image(image=[avgPic], x=-0.5, y=-0.5, dw=avgPic.shape[1], dh=avgPic.shape[0],
                 color_mapper=cmap)

    solarized = False;

    if solarized:
        backgroundColor = "#001E26"
        textColor = "#aaaaaa"
    else:
        backgroundColor = "FFFFFF"
        textColor = "#000000"

    color_bar = bok_ColorBar(color_mapper=cmap,
                             location=(0, 0), padding=0, width=int(width / 10),
                             background_fill_color=backgroundColor, major_label_text_color=textColor,
                             major_label_text_font_size='6pt')
    avgFig.add_layout(color_bar, 'right')
    # create circles for atom locations
    for loc in atomLocs:
        avgFig.circle(loc[1], loc[0], size=2)
    hoverData.renderers.append(boxes)
    avgFig.xaxis.ticker = []
    avgFig.yaxis.ticker = []
    setDefaultColors(avgFig)
    if small:
        avgFig.xaxis.major_label_text_font_size = '5pt'
        avgFig.yaxis.major_label_text_font_size = '5pt'
        avgFig.title.text_font_size = '6pt'
        avgFig.xaxis.axis_label_text_font_size = '6pt'
        avgFig.yaxis.axis_label_text_font_size = '6pt'
    else:
        avgFig.yaxis.axis_label = "Counts"
        avgFig.xaxis.axis_label = "Pic#"
    return avgFig


def CountsPlotWithBokeh(picData, thresholds, atomLocs, colors, width=225, height=225, small=True):
    countsFig = figure(title="Count Data", plot_width=width, plot_height=height)
    for i in range(len(atomLocs)):
        countsFig.circle(np.arange(len(picData[i])), picData[i], size=0.1, color=colors[i])
        countsFig.ray(x=[0], y=[thresholds[i]], length=0, angle=0, line_width=1, color=colors[i])

    countsFig.xaxis.major_label_orientation = np.pi/4
    setDefaultColors(countsFig)
    if small:
        countsFig.xaxis.major_label_text_font_size = '5pt'
        countsFig.yaxis.major_label_text_font_size = '5pt'
        countsFig.title.text_font_size = '6pt'
        countsFig.xaxis.axis_label_text_font_size = '6pt'
        countsFig.yaxis.axis_label_text_font_size = '6pt'
    else:
        countsFig.yaxis.axis_label = "Counts"
        countsFig.xaxis.axis_label = "Pic#"
    return countsFig



def plotFits(fig, fitType, xFit, fitNom, fitStd, fitVals, fitCorr, color='b', alpha=1, plotCenter=True ):
    if fitType is None or fitNom == None:# or sumAtoms == True:
        pass
    elif fitType is not None:
        # Default handling
        fig.line( xFit, fitNom, color=color, alpha=alpha)
        bokFillBetween( fig, xFit, fitNom - fitStd, fitNom + fitStd, color=color, alpha=alpha / 2)
        # def gaussian(x, A1, x01, sig1, offset):
        if plotCenter:
            bokVSpan(fig, fitVals[1], np.sqrt(fitCorr[1,1]), color=color, alpha=alpha/2)
            fig.ray( fitVals[1], y=[0], length=0, angle=consts.pi/2, line_width=1,color=color, alpha=alpha)
    return fig


def setOutput(interactive, outputFileAddress):
    reset_output()
    if interactive:
        print('outputting interactive to notebook.')
        output_notebook()
    else:
        if outputFileAddress is None:
            # bokio.output_file('TempBokeh.png')
            print('outputting static to notebook.')
        else:
            output_file(outputFileAddress)
            print('outputting to file.')


def bokehOutput(fig, interactive):
    if interactive:
        show(fig)
    else:
        export_png(fig, filename="TempBokeh.png")
        display(Image(filename='TempBokeh.png'))


def setDefaultColors(bokehFig):
    """
    call right before show.
    """
    solarized = False
    if solarized:
        bokehFig.border_fill_color = "#001E26"
        bokehFig.background_fill_color = "#00202A"
        bokehFig.xgrid.grid_line_color = "#666666"
        bokehFig.ygrid.grid_line_color = "#666666"
        bokehFig.xaxis.major_tick_line_color = "#aaaaaa"
        bokehFig.xaxis.minor_tick_line_color = "#aaaaaa"
        bokehFig.xaxis.major_label_text_color = "#aaaaaa"
        bokehFig.xaxis.axis_label_text_color = "#aaaaaa"
        bokehFig.yaxis.major_tick_line_color = "#aaaaaa"
        bokehFig.yaxis.minor_tick_line_color = "#aaaaaa"
        bokehFig.yaxis.major_label_text_color = "#aaaaaa"
        bokehFig.yaxis.axis_label_text_color = "#aaaaaa"
        bokehFig.legend.background_fill_alpha = 0
        bokehFig.legend.label_text_color = "#FFFFFF"
        bokehFig.title.text_color = "#aaaaaa"
    else:
        bokehFig.background_fill_color = "#EEEEEE"
    bokehFig.legend.label_text_font_size = "7pt"
    bokehFig.legend.glyph_height = 15
    bokehFig.legend.label_height = 7
    bokehFig.legend.spacing = 0


def bokErrbar(fig, source, xerr=None, yerr=None, color='red', legend=None, alpha=1, muted_alpha=1, muted_color='red',
              hoverData=None, point_kwargs={}, error_kwargs={}):
    x = source['xval']
    y = source['yval']
    if len(x.shape) > 1:
        barsize = 0.02
    if min(x) == max(x):
        barsize = 0.02
    else:
        xrange = max(x)-min(x)
        barsize = xrange/100
    circ = fig.circle("xval", "yval", color=color, legend=legend, alpha=alpha, muted_alpha = muted_alpha,
               muted_color=muted_color, source=ColumnDataSource(source), **point_kwargs)
    if hoverData is not None:
        hoverData.renderers.append(circ)
    if xerr is not None:
        x_err_x = []
        x_err_y = []
        for px, py, err in zip(x, y, xerr):
            x_err_x.append((px - err, px + err))
            x_err_y.append((py, py))
        fig.multi_line( x_err_x, x_err_y, color=color, alpha = alpha,
                        muted_alpha = muted_alpha,
                        muted_color=muted_color,**error_kwargs)
    if yerr is not None:
        y_err_x = []
        y_err_y = []
        for px, py, err in zip(x, y, yerr):
            y_err_x.append((px, px))
            y_err_y.append((py - err, py + err))
        fig.rect(x, arr(y_err_y)[:,1], barsize, 0.001, line_color=color, alpha = alpha, muted_alpha = muted_alpha,
                 muted_color=muted_color)
        fig.rect(x, arr(y_err_y)[:,0], barsize, 0.001, line_color=color, alpha = alpha, muted_alpha = muted_alpha,
                 muted_color=muted_color)
        fig.multi_line(y_err_x, y_err_y, color=color, alpha = alpha, muted_alpha = muted_alpha, muted_color=muted_color,
                       **error_kwargs)


def bokFillBetween(fig, xvals, line1, line2, alpha=1, color='red', legend=None):
    patchX = list(xvals) + list(reversed(xvals))
    patchY = list(line1) + list(reversed(line2))
    fig.patch(patchX, patchY, color=color, legend=legend, alpha=alpha)


def bokVSpan(fig, center, sigma, color='b', alpha=1, lowerYLim=0, upperYLim=1):
    patchX = [center - sigma, center - sigma, center + sigma, center + sigma]
    patchY = [lowerYLim, upperYLim, upperYLim, lowerYLim]
    fig.patch(patchX, patchY, color=color, alpha=alpha)


def SurvivalWithBokeh( fileNumber, atomLocs, **TransferArgs):
    """See corresponding transfer function for valid TransferArgs."""
    (key, survival, survivalerr,
     captureArray) = TransferWithBokeh(fileNumber, atomLocs, atomLocs, **TransferArgs)


def TransferWithBokeh(fileNumber, atomLocs1, atomLocs2, show=True, accumulations=1, key=None,
                      picsPerRep=2, plotTogether=True, plotLoadingRate=False, manualThreshold=None,
                      fitType=None, window=None, xMin=None, xMax=None, yMin=None, yMax=None, dataRange=None,
                      histSecondPeakGuess=None, keyOffset=0, sumAtoms=False, interactive=True,
                      outputFileAddr=None, outputMma=False, dimSlice=None):
    """
    Standard data analysis package for looking at survival rates throughout an experiment.

    Returns key, survivalData, survivalErrors
    """
    (atomLocs1, atomLocs2, survivalData, survivalErrs, loadingRate, pic1Data, keyName, key, repetitions, thresholds,
     fits,
     avgSurvivalData, avgSurvivalErr, avgFit,
     avgPic) = standardTransferAnalysis(fileNumber, atomLocs1, atomLocs2, accumulations=accumulations, key=key,
                                        picsPerRep=2,
                                        manualThreshold=manualThreshold, fitType=fitType, window=window, xMin=xMin,
                                        xMax=xMax, yMin=yMin,
                                        yMax=yMax, dataRange=dataRange, histSecondPeakGuess=histSecondPeakGuess,
                                        keyOffset=keyOffset,
                                        sumAtoms=sumAtoms, outputMma=outputMma, dimSlice=dimSlice)

    if show:
        # #########################################
        #      Plotting
        # #########################################
        # get the colors for the plot.
        # bokio.output_notebook()
        setOutput(interactive, outputFileAddr)
        cmapRGB = mpl.cm.get_cmap('gist_rainbow', len(atomLocs1) + 1)
        colors = [cmapRGB(i)[:-1] for i in range(len(atomLocs1) + 1)]
        # the negative of the first color
        colors2 = [tuple(arr((1, 1, 1)) - arr(color)) for color in colors]
        colors = ['#%02x%02x%02x' % tuple(int(255 * color[i]) for i in range(len(color))) for color in colors]
        colors2 = ['#%02x%02x%02x' % tuple(int(255 * color[i]) for i in range(len(color))) for color in colors2]
        if atomLocs1 == atomLocs2:
            scanType = "Survival"
        else:
            scanType = "Transfer"
        legends = [None] * len(atomLocs1)
        for i, atomLoc in enumerate(atomLocs1):
            legends[i] = r"[%d,%d]>>[%d,%d] " % (atomLocs1[i][0], atomLocs1[i][1], atomLocs2[i][0], atomLocs2[i][1])
            if len(survivalData[i]) == 1:
                legends[i] += (scanType + " % = " + str(round_sig(survivalData[i][0]))
                               + "+- " + str(round_sig(survivalErrs[i][0])))
        survivalErrs = list(survivalErrs)
        mainPlot = pointsWithBokeh(key, survivalData, colors, atomLocs1,
                                   scanType=scanType, width=600, height=450, keyName=keyName, errs=survivalErrs,
                                   legends=legends, avgData=avgSurvivalData, avgErrs=avgSurvivalErr)
        alphaVal = 1.0 / (len(atomLocs1) ** 0.75)
        for i, atomLoc in enumerate(atomLocs1):
            if fits[i] is not None:
                mainPlot = plotFits(mainPlot, fitType, fits[i]['x'], fits[i]['nom'], fits[i]['std'], fits[i]['vals'],
                                    fits[i]['cov'],
                                    color=colors[i], alpha=alphaVal)
        if avgFit is not None:
            mainPlot = plotFits(mainPlot, fitType, avgFit['x'], avgFit['nom'], avgFit['std'], avgFit['vals'],
                                avgFit['cov'],
                                color='#000000', alpha=0.5)
        loadingPlot = pointsWithBokeh(key, loadingRate, colors, atomLocs1, keyName=keyName, width=250,
                                      height=150, small=True)
        countsFig = CountsPlotWithBokeh(pic1Data, thresholds, atomLocs1, colors, width=250, height=150)
        avgFig = AvgImgWithBokeh(avgPic, atomLocs1, width=125, height=150)

        if fits[0] is not None:
            fitPic = np.zeros(avgPic.shape)
            """
                if type(atomLocs1[0]) == int:
                # assume atom grid format.
                topLeftRow = atomLocs1[0]
                topLeftColumn = atomLocs1[1]
                spacing = atomLocs1[2]
                width = atomLocs1[3] # meaning the atoms array width number x height number, say 5 by 5
                height = atomLocs1[4]
                atomLocs1 = []
                for widthInc in range(width):
                    for heightInc in range(height):
                        atomLocs1.append([topLeftRow + spacing * heightInc, topLeftColumn + spacing * widthInc])
            """
            # print('Average-Fit parameters:', fitValuesSum)
            # print('Average-Fit errors:', np.sqrt(np.diag(fitCovsSum)))
            """
            centers = []            
            rowData = [[] for row in range(gridSettings[4])]
            colData = [[] for col in range(gridSettings[3])]
            for i, loc in enumerate(atomLocs1):
                rowData[int((loc[0] - gridSettings[1]) / gridSettings[2])].append(fitValues[i][1])
                colData[int((loc[1] - gridSettings[0]) / gridSettings[2])].append(fitValues[i][1])
            for i, loc in enumerate(atomLocs1):
                if fitValues[i] is not None:
                    fitPic[loc[0], loc[1]] = fitValues[i][1]
                    centers.append(fitValues[i][1])
                else:
                    fitPic[loc[0], loc[1]] = 0
            vals = [v[1] for v in fitValues]
            fitImg = AvgImgWithBokeh( fitPic, atomLocs1, width=300, height=300 )
            bok.show(fitImg)

            print('Row Averages:')
            for data in rowData:
                print(np.mean(data), end='     ')
            print('')
            print('Columm Averages:')
            for data in colData:
                print(np.mean(data), end='     ')
            print('')

            print('Average Fit Center:', np.mean(centers))
            print('Fit Center Range:', np.max(centers)- np.min(centers))
            print('Fit Center STD:', np.std(centers))
            """

        transferPic = np.zeros(avgPic.shape)
        for i, loc in enumerate(atomLocs1):
            transferPic[loc[0], loc[1]] = np.mean(survivalData[i])
        transferFig = AvgImgWithBokeh(transferPic, atomLocs1, width=115, height=150, titleTxt='Avg Transfer')
        l = bokeh.layouts.gridplot([[mainPlot, bokeh.layouts.column(loadingPlot, countsFig,
                                                                    bokeh.layouts.row(avgFig, transferFig))]])
        bokehOutput(l, interactive)
        # bok.show(l)
    return key, survivalData, survivalErrs, loadingRate


def LoadingWithBokeh(fileNum, atomLocations, accumulations=1, key=None, picsPerExperiment=1,
                     analyzeTogether=False, plotLoadingRate=False, picture=0, manualThreshold=None,
                     loadingFitType=None, showIndividualHist=True, showTotalHist=None, outputFileAddr=None,
                     interactive=True, showLoadingPic=False):
    """
    Standard data analysis package for looking at loading rates throughout an experiment.

    return key, loadingRateList, loadingRateErr

    This routine is designed for analyzing experiments with only one picture per cycle. Typically
    These are loading exeriments, for example. There's no survival calculation.
    """
    atomLocations = unpackAtomLocations(atomLocations)
    #### Load Fits File & Get Dimensions
    # Get the array from the fits file. That's all I care about.
    rawData, keyName, key, repetitions = loadHDF5(fileNum)
    # the .shape member of an array gives an array of the dimesnions of the array.
    numOfPictures = rawData.shape[0];
    numOfVariations = int(numOfPictures / repetitions)
    # handle defaults.
    if numOfVariations == 1:
        if showTotalHist == None:
            showTotalHist = False
        if key is None:
            key = arr([0])
    else:
        if showTotalHist == None:
            showTotalHist = True
        if key is None:
            key = arr([])
    # make it the right size
    if len(arr(atomLocations).shape) == 1:
        atomLocations = [atomLocations]
    print("Key Values, in experiment's order: ", key)
    print('Total # of Pictures:', numOfPictures)
    print('# of repetitions per variation:', repetitions)
    print('Number of Variations:', numOfVariations)
    if not len(key) == numOfVariations:
        raise ValueError("ERROR: The Length of the key (" + str(len(key)) + ") doesn't match the data found ("
                         + str(numOfVariations) + ").")
    ### Initial Data Analysis
    s = rawData.shape
    if analyzeTogether:
        newShape = (1, s[0], s[1], s[2])
    else:
        newShape = (numOfVariations, repetitions, s[1], s[2])
    groupedData = rawData.reshape(newShape)
    groupedData, key, _ = orderData(groupedData, key)
    print('Data Shape:', groupedData.shape)
    loadingRateList, loadingRateErr = [[[] for x in range(len(atomLocations))] for x in range(2)]
    print('Analyzing Variation... ', end='')
    allLoadingRate, allLoadingErr = [[[]] * len(groupedData) for _ in range(2)]
    totalPic1AtomData = []
    # loop through variations
    for variationInc, data in enumerate(groupedData):
        avgPic = getAvgPic(data)
        # initialize empty lists
        (pic1Data, pic1Atom, thresholds, thresholdFid, fitVals, bins, binData,
         atomCount) = arr([[None] * len(atomLocations)] * 8)
        # fill lists with data
        allAtomPicData = []
        for i, atomLoc in enumerate(atomLocations):
            (pic1Data[i], pic1Atom[i], thresholds[i], thresholdFid[i],
             fitVals[i], bins[i], binData[i], atomCount[i]) = getData(data, atomLoc, picture,
                                                                      picsPerExperiment, manualThreshold, 10)
            totalPic1AtomData.append(pic1Atom[i])
            allAtomPicData.append(np.mean(pic1Atom[i]))
            loadingRateList[i].append(np.mean(pic1Atom[i]))
            loadingRateErr[i].append(np.std(pic1Atom[i]) / np.sqrt(len(pic1Atom[i])))
        allLoadingRate[variationInc] = np.mean(allAtomPicData)
        allLoadingErr[variationInc] = np.std(allAtomPicData) / np.sqrt(len(allAtomPicData))
        if showIndividualHist:
            atomHistWithBokeh(key, atomLocations, pic1Data, bins, binData, fitVals, thresholds,
                              avgPic, atomCount, variationInc, outputFileAddr=outputFileAddr, interactive=interactive)
    avgPic = getAvgPic(rawData)
    # get averages across all variations
    (pic1Data, pic1Atom, thresholds, thresholdFid, fitVals, bins, binData,
     atomCount) = arr([[None] * len(atomLocations)] * 8)
    for i, atomLoc in enumerate(atomLocations):
        (pic1Data[i], pic1Atom[i], thresholds[i], thresholdFid[i],
         fitVals[i], bins[i], binData[i], atomCount[i]) = getData(rawData, atomLoc, picture,
                                                                  picsPerExperiment, manualThreshold, 5)
    if showTotalHist:
        atomHistWithBokeh(key, atomLocations, pic1Data, bins, binData, fitVals, thresholds, avgPic, atomCount,
                          None, outputFileAddr=outputFileAddr, interactive=interactive)
    if showLoadingPic:
        loadingPic = np.zeros(avgPic.shape)

        minHor = min(transpose(key)[0])
        minVert = min(transpose(key)[1])
        locFromKey = [[int((keyItem[0] - minHor) / 9 * 2 + 2), int((keyItem[1] - minVert) / 9 * 2 + 2)] for keyItem in
                      key]
        for i, loc in enumerate(locFromKey):
            loadingPic[loc[1]][loc[0]] = max(loadingRateList[i])
        loadingFig = AvgImgWithBokeh(loadingPic, atomLocations, width=300, height=300,
                                     titleTxt='Loading when selected')
        bok.show(loadingFig)
    if plotLoadingRate:
        fitData = handleFitting(loadingFitType, key, loadingRateList)
        setOutput(interactive, outputFileAddr)
        xvals, yvals = np.meshgrid(np.arange(avgPic.shape[0]), np.arange(avgPic.shape[1]))
        hoverData = bokModels.HoverTool(tooltips=[("Location", "@atomLocation"),
                                                  ("Key Value", "@xval"),
                                                  ("Loading", "@yval")])
        # get colors
        cmapRGB = get_cmap('gist_rainbow', len(atomLocations))
        colors = [cmapRGB(i)[:-1] for i in range(len(atomLocations))]
        colors = ['#%02x%02x%02x' % tuple(int(255 * color[i]) for i in range(len(color))) for color in colors]
        keyRange = max(key[:][0]) - min(key[:][0]) if len(key.shape) > 1 else max(key) - min(key)
        alphaVal = 1.0 / (len(atomLocations) ** 0.7)
        legends = [str(loc) for loc in atomLocations]
        mainPlot = pointsWithBokeh(key, loadingRateList, colors, atomLocations,
                                   scanType='Loading', width=900, height=500, keyName=keyName, errs=loadingRateErr,
                                   legends=legends, avgData=allLoadingRate, avgErrs=allLoadingErr)
        mainPlot.legend.click_policy = "mute"
        mainPlot.xaxis.axis_label = keyName
        mainPlot.yaxis.axis_label = "Loading Rate"
        setDefaultColors(mainPlot)
        bokehOutput(mainPlot, interactive)
        if loadingFitType is not None:
            plot(fitData['x'], fitData['nom'], ':', label='Fit', linewidth=3)
            fill_between(fitData['x'], fitData['nom'] - fitData['std'],
                         fitData['nom'] + fitData['std'], alpha=0.1, label=r'$\pm\sigma$ band',
                         color='b')
            axvspan(fitData['vals'][fitData['center']]
                    - np.sqrt(fitData['cov'][fitData['center'], fitData['center']]),
                    fitData['vals'][fitData['center']] + np.sqrt(fitCovs[fitData['center'], fitData['center']]),
                    color='b', alpha=0.1)
            axvline(fitData['vals'][fitData['center']], color='b', linestyle='-.', alpha=0.5,
                    label='fit center $= ' + str(round_sig(fitData['vals'][fitData['center']])) + '$')
    return key, loadingRateList, loadingRateErr, totalPic1AtomData, rawData


def AssemblyWithBokeh(fileNumber, atomLocs1, pic1Num, atomLocs2=None, pic2Num=None, keyOffset=0, window=None,
                      picsPerRep=2, dataRange=None, histSecondPeakGuess=None, manualThreshold=None,
                      fitType=None):
    """
    This function checks the efficiency of generating a picture;
    I.e. finding atoms at multiple locations at the same time.
    """
    (atomLocs1, atomLocs2, key, thresholds, pic1Data, pic2Data, fit, ensembleAvgs, ensembleErrs, avgPic,
     atomCounts, keyName) = standardAssemblyAnalysis(fileNumber, atomLocs1, pic1Num, atomLocs2=atomLocs2,
                                                     pic2Num=pic2Num, keyOffset=keyOffset, window=window,
                                                     picsPerRep=picsPerRep, histSecondPeakGuess=histSecondPeakGuess,
                                                     manualThreshold=manualThreshold, fitType=fitType)
    if not show:
        return key, survivalData, survivalErrs;
        # #########################################
    #      Plotting
    bokio.output_notebook()
    # get the colors for the plot.
    cmapRGB = mpl.cm.get_cmap('gist_rainbow', len(atomLocs1) + 1)
    colors = [cmapRGB(i)[:-1] for i in range(len(atomLocs1) + 1)]
    # the negative of the first color
    colors2 = [tuple(arr((1, 1, 1)) - arr(color)) for color in colors]
    colors = ['#%02x%02x%02x' % tuple(int(255 * color[i]) for i in range(len(color))) for color in colors]
    colors2 = ['#%02x%02x%02x' % tuple(int(255 * color[i]) for i in range(len(color))) for color in colors2]
    scanType = "Ensemble"
    ensembleErrs = list(ensembleErrs)
    mainPlot = pointsWithBokeh(key, [list(ensembleAvgs)], colors, [[0, 0]],
                               scanType=scanType, width=650, height=450, keyName=keyName, errs=[ensembleErrs])
    countsFig = CountsPlotWithBokeh(pic1Data, thresholds, atomLocs1, colors, width=200, height=200)
    avgFig = AvgImgWithBokeh(avgPic, atomLocs1, width=200, height=220)
    l = bokeh.layouts.gridplot([[mainPlot, bokeh.layouts.column(countsFig, avgFig)]])
    bok.show(l)
    # Main Plot
    """
    if len(ensembleAvgs) == 1:
        titletxt = keyName + " Atom " + typeName + " Point. " + typeName + " % = \n"
        for atomData in ensembleAvgs:
            titletxt += str(atomData) + ", "
    mainPlot = bok.figure(title=titletxt)
    mainPlot.set_ylabel( "Ensemble Probability", fontsize=20)
    mainPlot.set_xlabel( keyName, fontsize=20)


    mainPlot.errorbar(key, ensembleAvgs, yerr=ensembleErrs, color=colors[0], ls='', 
                      marker='o', capsize=6, elinewidth=3, label='Raw Data ' + str(atomLoc) )
    if fitType is None or fitNom is None:
        pass
    elif fitType == 'Exponential-Decay' or fitType == 'Exponential-Saturation':
        mainPlot.plot( xFit, fitNom[i], ':', color=colors[i], label=r'Fit; decay constant $= '
                      + str(round_sig(fitValues[1])) + '\pm '
                      + str(round_sig(fitErrs[1])) + '$', linewidth=3 )
        mainPlot.fill_between(xFit, fitNom[i] - fitStd[i], fitNom[i] + fitStd[i],
                              alpha=0.1, label=r'$\pm\sigma$ band',
                              color=colors[i])
    else:
        # assuming gaussian?
        xFit = np.linspace(min(key), max(key), 1000)
        mainPlot.plot( xFit, fitNom, ':', label='Fit', linewidth=3, color=colors[0] )
        mainPlot.fill_between(xFit, fitNom - fitStd, fitNom + fitStd, alpha=0.1,
                              label='2-sigma band', color=colors[0])
        mainPlot.axvspan(fitValues[1] - np.sqrt(fitCovs[1,1]), fitValues[1] + np.sqrt(fitCovs[1,1]), 
                         color=colors[0], alpha=0.1)
        mainPlot.axvline(fitValues[1], color=colors[0], linestyle='-.', 
                         alpha=0.5, label='fit center $= '+str(round_sig(fitValues[1], 4))
                         + '\pm ' + str(round_sig(fitErrs[1], 4)) + '$')
    mainPlot.set_ylim({-0.02, 1.01})
    if not min(key) == max(key):
        mainPlot.set_xlim( left=min(key) - (max(key) - min(key)) / len(key), right=max(key) 
                          + (max(key) - min(key)) / len(key))
    mainPlot.set_xticks( key )
    mainPlot.legend(loc="upper center", bbox_to_anchor=(0.5,-0.1), fancybox=True, ncol=4)
    legend()
    # Loading Plot
    """
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
    tickVals = np.linspace(0,len(pic1Data[0]), len(key)+1)
    countPlot.set_xticks(tickVals)
    # Count Histogram Plot
    countHist = subplot(gridLeft[4:8, 15:16], sharey=countPlot)
    for i, atomLoc in enumerate(atomLocs1):
        countHist.hist(pic1Data[i], 50, color=colors[i], orientation='horizontal', alpha=0.5)
        countHist.hist(pic2Data[i], 50, color=colors2[i], orientation='horizontal', alpha=0.3)
        countHist.axhline(thresholds[i], color='w')
    for item in ([ countHist.title, countHist.xaxis.label, countHist.yaxis.label] +
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
        circ = Circle((loc[1],loc[0]), 0.2, color='r')
        avgPlt.add_artist(circ)
    for loc in atomLocs2:
        circ = Circle((loc[1],loc[0]), 0.1, color='g')
        avgPlt.add_artist(circ)
    """
    return key


