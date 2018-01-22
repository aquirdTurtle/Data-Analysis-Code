__version__ = "1.4"

from os import linesep
from pandas import read_csv as pd_read_csv
import pandas as pd

from astropy.io import fits

from numpy import array as arr
import h5py as h5
from inspect import signature
import uncertainties as unc
import uncertainties.unumpy as unp
from warnings import warn

from matplotlib.pyplot import *
from matplotlib.patches import Ellipse

from scipy.optimize import minimize, basinhopping, curve_fit as fit
import scipy.special as special
import scipy.interpolate as interp


import MarksConstants as consts
import FittingFunctions as fitFunc

from Miscellaneous import transpose, round_sig
from copy import deepcopy

dataAddress = None


def modFitFunc(hBiasIn, vBiasIn, depthIn, *testBiases):
    newDepths = extrapolateModDepth(hBiasIn, vBiasIn, depthIn, testBiases)
    if newDepths is None:
        return 1e9
    return np.std(newDepths)


def extrapolateEveningBiases(hBiasIn, vBiasIn, depthIn):
    # normalize biases
    hBiasIn /= np.sum(hBiasIn)
    vBiasIn /= np.sum(vBiasIn)
    guess = np.concatenate((hBiasIn, vBiasIn))
    f = lambda g: modFitFunc(hBiasIn, vBiasIn, depthIn, *g)
    result = minimize(f, guess)
    return result, extrapolateModDepth(hBiasIn, vBiasIn, depthIn, result['x'])


def extrapolateModDepth(hBiasIn, vBiasIn, depthIn, testBiases):
    """
    assumes that hBiasIn and vBiasIn are normalized.
    This function extrapolates what the depth of each tweezer should be based on the
    current depths and current biases. Basically, it assumes that if you change the bias by x%,
    then the depth for every atom in that row/column will change by x%.
    """
    hBiasTest = testBiases[:len(hBiasIn)]
    if len(hBiasTest) > 1:
        for b in hBiasTest:
            if b <= 0 or b > 1:
                return None
    vBiasTest = testBiases[len(hBiasIn):len(hBiasIn) + len(vBiasIn)]
    if len(vBiasTest) > 1:
        for b in vBiasTest:
            if b <= 0 or b > 1:
                return None
    # normalize tests
    hBiasTest /= np.sum(hBiasTest)
    vBiasTest /= np.sum(vBiasTest)
    modDepth = deepcopy(depthIn)
    for rowInc, _ in enumerate(depthIn):
        dif = (vBiasTest[rowInc] - vBiasIn[rowInc])/vBiasIn[rowInc]
        modDepth[rowInc] = modDepth[rowInc] * (1-dif)
    for colInc, _ in enumerate(transpose(depthIn)):
        dif = (hBiasTest[colInc] - hBiasIn[colInc])/hBiasIn[colInc]
        modDepth[:, colInc] = modDepth[:, colInc] * (1-dif)
    return modDepth


def setPath(day, month, year):
    """
    This function sets the location of where all of the data files are stored. It is occasionally called more
    than once in a notebook if the user needs to work past midnight.

    :param day: A number string, e.g. '11'.
    :param month: The name of a month, e.g. 'November' (must match file path capitalization).
    :param year: A number string, e.g. '2017'.
    :return:
    """
    dataRepository = "J:\\Data repository\\New Data Repository"
    global dataAddress
    dataAddress = dataRepository + "\\" + year + "\\" + month + "\\" + month + " " + day + "\\Raw Data\\"


def getLabels(plotType):
    """

    :param plotType:
    :return:
    """
    if plotType == "Detuning":
        xlabelText = "Detuning (dac value)"
        titleText = "Detuning Scan"
    elif plotType == "Field":
        xlabelText = "Differential Field Change / 2 (dac value)"
        titleText = "Differential Magnetic Field Scan"
    elif plotType == "Time(ms)":
        xlabelText = "Time(ms)"
        titleText = "Time Scan"
    elif plotType == "Power":
        xlabelText = "Power (dac units)"
        titleText = "Power Scan"
    else:
        xlabelText = plotType
        titleText = ""
    return xlabelText, titleText


def fitWithClass(fitClass, key, vals):
    """
    fitClass is an class object *gasp*
    """
    xFit = (np.linspace(min(key), max(key), 1000) if len(key.shape) == 1 else np.linspace(min(transpose(key)[0]), max(transpose(key)[0]), 1000))
    fitNom = fitStd = centerIndex = fitValues = fitErrs = fitCovs = None
    from numpy.linalg import LinAlgError
    try:
        # 3 parameters
        if len(key) < len(signature(fitClass.f).parameters) - 1:
            print('Not enough data points to constrain a fit!')
            raise RuntimeError()
        fitValues, fitCovs = fit(fitClass.f, key, vals, p0=[fitClass.guess(key, vals)])
        fitErrs = np.sqrt(np.diag(fitCovs))
        corr_vals = unc.correlated_values(fitValues, fitCovs)
        fitUncObject = fitClass.f_unc(xFit, *corr_vals)
        fitNom = unp.nominal_values(fitUncObject)
        fitStd = unp.std_devs(fitUncObject)
        fitFinished = True
    except (RuntimeError, LinAlgError, ValueError):
        warn('Data Fit Failed!')
        fitFinished = False
    fitInfo = {'x': xFit, 'nom': fitNom, 'std': fitStd, 'vals': fitValues, 'errs': fitErrs, 'cov': fitCovs}
    return fitInfo, fitFinished


def combineData(data, key):
    """
    combines similar key value data entries. data will be in order that unique key items appear in key.
    For example, if key = [1,3,5,3,7,1], returned key and corresponding data will be newKey = [1, 3, 5, 7]

    :param data:
    :param key:
    :return:
    """
    items = {}
    newKey = []
    newData = []
    for elem in key:
        if str(elem) not in items:
            indexes = [i for i, x in enumerate(key) if x == elem]
            # don't get it again
            items[str(elem)] = "!"
            newKey.append(elem)
            newItem = np.zeros((data.shape[1], data.shape[2]))
            # average together the corresponding data.
            for index in indexes:
                newItem += data[index]
            newItem /= len(indexes)
            newData.append(newItem)
    return arr(newData), arr(newKey)

# ##############
# ### Data-Loading Functions


def openHDF5(fileId):
    """

    :param fileId:
    :return:
    """
    if type(fileId) == int:
        path = dataAddress + "data_" + str(fileId) + ".h5"
    else:
        # assume a file address itself
        path = fileId
    file = h5.File(path, 'r')
    return file


def getKeyFromHDF5(file):
    """

    :param file:
    :return:
    """
    keyNames = []
    keyValues = []
    foundOne = False
    for var in file['Master-Parameters']['Seq #1 Variables']:
        if not file['Master-Parameters']['Seq #1 Variables'][var].attrs['Constant']:
    # to look at older files...
#    for var in file['Master-Parameters']['Variables']:
#        if not file['Master-Parameters']['Variables'][var].attrs['Constant']:
            foundOne = True
            keyNames.append(var)
            keyValues.append(arr(file['Master-Parameters']['Seq #1 Variables'][var]))
    if foundOne:
        if len(keyNames) > 1:
            return keyNames, arr(transpose(arr(keyValues)))
        else:
            return keyNames[0], arr(keyValues[0])
    else:
        return 'No-Variation', arr([1])


def getPicsFromHDF5(file):
    """
    Need to re-shape the pics.
    :param file:
    :return:
    """
    pics = arr(file['Andor']['Pictures'])
    pics = pics.reshape((pics.shape[0], pics.shape[2], pics.shape[1]))
    return pics


def loadHDF5(fileId):
    """
    Loads the key info from the hdf5 file and returns it.
    returns pics, keyName, key, reps

    :param fileId:
    :return:
    """
    file = openHDF5(fileId)
    keyName, key = getKeyFromHDF5(file)
    reps = file['Master-Parameters']['Repetitions'][0]
    pics = getPicsFromHDF5(file)
    return pics, keyName, key, reps


def loadDataRay(num):
    """

    :param num:
    :return:
    """
    fileName = dataAddress + "dataRay_" + str(num) + ".wct"
    file = pd_read_csv(fileName, header=None, skiprows=[0, 1, 2, 3, 4])
    data = file.as_matrix()
    for i, row in enumerate(data):
        data[i][-1] = float(row[-1][:-2])
        for j, elem in enumerate(data[i]):
            data[i][j] = float(elem)
    return data.astype(float)


def loadCompoundBasler(num, cameraName='ace'):
    if cameraName == 'ace':
        path = dataAddress + "AceData_" + str(num) + ".txt"
    elif cameraName == 'scout':
        path = dataAddress + "ScoutData_" + str(num) + ".txt"
    else:
        raise ValueError('cameraName bad value for Basler camera.')
    with open(path) as file:
        original = file.read()
        pics = original.split(";")
        dummy = linesep.join([s for s in pics[0].splitlines() if s])
        dummy2 = dummy.split('\n')
        dummy2[0] = dummy2[0].replace(' \r', '')
        data = np.zeros((len(pics), len(dummy2), len(arr(dummy2[0].split(' ')))))
        picInc = 0
        for pic in pics:
            # remove extra empty lines
            pic = linesep.join([s for s in pic.splitlines() if s])
            lines = pic.split('\n')
            lineInc = 0
            for line in lines:
                line = line.replace(' \r', '')
                picLine = arr(line.split(' '))
                picLine = arr(list(filter(None, picLine)))
                data[picInc][lineInc] = picLine
                lineInc += 1
            picInc += 1
    return data


def loadFits(num):
    """
    Legacy. We don't use fits files anymore.

    :param num:
    :return:
    """
    # Get the array from the fits file. That's all I care about.
    path = dataAddress + "data_" + str(num) + ".fits"
    with fits.open(path, "append") as fitsFile:
        try:
            rawData = arr(fitsFile[0].data, dtype=float)
            return rawData
        except IndexError:
            fitsFile.info()
            raise RuntimeError("Fits file was empty!")


def loadKey(num):
    """
    Legacy. We don't use dedicated key files anymore, but rather it gets loaded into the hdf5 file.

    :param num:
    :return:
    """
    key = np.array([])
    path = dataAddress + "key_" + str(num) + ".txt"
    with open(path) as keyFile:
        for line in keyFile:
            key = np.append(key, [float(line.strip('\n'))])
        keyFile.close()
    return key


def loadDetailedKey(num):
    """
    Legacy. We don't use dedicated key files anymore, rather it gets loaded from the hdf5 file.

    :param num:
    :return:
    """
    key = np.array([])
    varName = 'None-Variation'
    path = dataAddress + "key_" + str(num) + ".txt"
    with open(path) as keyFile:
        # for simple runs should only be one line.
        count = 0
        for line in keyFile:
            if count == 1:
                print("ERROR! Multiple lines in detailed key file not yet supported.")
            keyline = line.split()
            varName = keyline[0]
            key = arr(keyline[1:], dtype=float)
            count += 1
        keyFile.close()
    return key, varName


def browseh5(runNum=None, printOption=None, fileopen=None, filepath=None):
    """
    input format of a tuple with various depth

        examples of use:
    fileopenloc=h5.File('J:\\Data Repository\\New Data Repository\\2017\\September\\September 13\\Raw Data\\data_1.h5')
    browseh5(fileloc)
    or
    filepathloc='J:\\Data Repository\\New Data Repository\\2017\\September\\September 13\\Raw Data\\data_111.h5'
    browseh5(filepath=filepathloc)
    or
    browseh5(runNum=111) # for the data of the same date
    then it displays a list of possible catagories, for example 'Master-Parameters'
    then input browseh5(file,'Master-Parameters')
    further displays master-scripts
    then input browseh5(file,('Master-Parameters','master-scripts') to see the script. need to input in tuples
    for more than one inputs
    :param runNum:
    :param printOption:
    :param fileopen:
    :param filepath:
    :return
    """
    if fileopen is None:
        if filepath is None:
            if runNum is not None:
                path = dataAddress + "data_" + str(runNum) + ".h5"
                filenametmp = h5.File(path)
            else:
                print('error for input parameters')
                return
        else:
            path = filepath
            filenametmp = h5.File(path)
    else:
        filenametmp = fileopen

    evalstr = 'filenametmp'
    printAll = True  # see if it already output things with strings, otherwise print as normal as True
    if filenametmp is None:
        print("input file is empty")
    elif printOption:
        for m in filenametmp:
            print(m)
            for n in filenametmp[m]:
                print('-', n)
                if type(filenametmp[m][n]) == h5._hl.dataset.Dataset:
                    if type(filenametmp[m][n][0]) == np.bytes_:
                        x = [z.decode('UTF-8') for z in filenametmp[m][n]]
                        print(''.join(x))
    elif printOption is None:
        for m in filenametmp:
            print(m)
    else:
        if type(printOption) == str:
            printOption = (printOption, '')  # convert to a tuple to access strings individually
        for i in printOption:
            if i != '':
                evalstr += '[\'' + i + '\']'
    if evalstr:
        evalstr1 = eval(evalstr)
        try:
            if type(evalstr1[0]) == np.bytes_:
                x = [z.decode('UTF-8') for z in evalstr1]
                print(''.join(x))
                printAll = False
            else:
                for m in evalstr1:
                    print(m)
        except:
            pass
        if printAll:
            for m in evalstr1:
                print(m)


def ballisticMotExpansion(t, sigma_y0, sigma_vy, sigma_I):
    """
    You can see a derivation of this in a different notebook for temperature calculations. I don't know why, but
    this function seems somewhat unstable as a fitting function. It doesn't always work very sensibly.

    :param t:
    :param sigma_y0:
    :param sigma_vy:
    :param sigma_I:
    :return:
    """
    return sigma_I*np.sqrt((sigma_y0**2 + sigma_vy**2 * t**2)/(sigma_y0**2+sigma_vy**2*t**2+sigma_I**2))


def simpleMotExpansion(t, sigma_y0, sigma_vy):
    """
    For some reason calculations involving this tend to be wildly off... I need to look into this more the next time
    I need to calculate a temperature.

    this simpler version ignores the size of the beam waist of the atoms.
    It should generally behave better with noisy data.

    :param t:
    :param sigma_y0:
    :param sigma_vy:
    :return:

    """
    return sigma_y0 + sigma_vy * t


def beamWaistExpansion(z, w0, z0, wavelength):
    """
    assuming gaussian intensity profile of I~exp{-2z^2/w{z}^2}
    :param z:
    :param w0:
    :param z0:
    :param wavelength:
    :return:
    """
    return w0 * np.sqrt(1+(wavelength*(z-z0)/(np.pi*w0**2))**2)


def fitPic(picture, showFit=True, guessSigma_x=1, guessSigma_y=1):
    """

    :param picture:
    :param showFit:
    :param guessSigma_x:
    :param guessSigma_y:
    :return:
    """
    pos = arr(np.unravel_index(np.argmax(picture), picture.shape))
    pic = picture.flatten()
    x = np.linspace(1, picture.shape[1], picture.shape[1])
    y = np.linspace(1, picture.shape[0], picture.shape[0])
    x, y = np.meshgrid(x, y)
    initial_guess = [np.max(pic) - np.min(pic), pos[1], pos[0], guessSigma_x, guessSigma_y, 0, np.min(pic)]
    try:
        popt, pcov = fit(fitFunc.gaussian_2D, (x, y), pic, p0=initial_guess, maxfev=2000)
    except RuntimeError:
        popt = np.zeros(len(initial_guess))
        pcov = np.zeros((len(initial_guess), len(initial_guess)))
        warn('Fit Pic Failed!')
    if showFit:
        data_fitted = fitFunc.gaussian_2D((x, y), *popt)
        fig, ax = subplots(1, 1)
        grid('off')
        im = ax.pcolormesh(picture, extent=(x.min(), x.max(), y.min(), y.max()))
        ax.contour(x, y, data_fitted.reshape(picture.shape[0],picture.shape[1]), 4, colors='w', alpha=0.2)
        fig.colorbar(im)
    return popt, np.sqrt(np.diag(pcov))


def fitPictures(pictures, dataRange, guessSigma_x=1, guessSigma_y=1):
    """

    :param pictures:
    :param dataRange:
    :param guessSigma_x:
    :param guessSigma_y:
    :return:
    """
    fitParameters = []
    fitErrors = []
    count = 0
    warningHasBeenThrown = False
    for picture in pictures:
        if count not in dataRange:
            count += 1
            fitParameters.append(np.zeros(7))
            fitErrors.append(np.zeros(7))
        try:
            parameters, errors = fitPic(picture, showFit=False, guessSigma_x=guessSigma_x, guessSigma_y=guessSigma_y)
        except RuntimeError:
            if not warningHasBeenThrown:
                print("Warning! Not all picture fits were able to fit the picture signal to a 2D Gaussian.\n"
                      "When the fit fails, the fit parameters are all set to zero.")
                warningHasBeenThrown = True
            parameters = np.zeros(7)
            errors = np.zeros(7)
        # append things regardless of whether the fit succeeds or not in order to keep things the right length.
        fitParameters.append(parameters)
        fitErrors.append(errors)
        count += 1
    return np.array(fitParameters), np.array(fitErrors)


def fitDoubleGaussian(binCenters, binnedData, fitGuess):
    from scipy.optimize import curve_fit
    try:
        fitVals, fitCovNotUsed = curve_fit(lambda x, a1, a2, a3, a4, a5, a6:
                                           fitFunc.doubleGaussian(x, a1, a2, a3, a4, a5, a6, 0),
                                           binCenters, binnedData, fitGuess)
    except:
        warn('Double-Gaussian Fit Failed!')
        fitVals = (0, 0, 0, 0, 0, 0)
    return fitVals


def fitGaussianBeamWaist(data, key, wavelength):
    # expects waists as inputs
    initial_guess = [min(data.flatten()), key[int(3*len(key)/4)]]
    try:
        # fix the wavelength
        # beamWaistExpansion(z, w0, wavelength)
        popt, pcov = fit(lambda x, a, b: beamWaistExpansion(x, a, b, wavelength), key, data, p0=initial_guess)
    except RuntimeError:
        popt, pcov = [0, 0]
        warn('Fit Failed!')
    return popt, pcov


# #############################
# ### Analyzing machine outputs


def load_SRS_SR780(fileAddress):
    """
    from a TXT file from the SRS, returns the frequencies (the [0] element) and the powers (the [1] element)
    """
    data = pd.read_csv(fileAddress, delimiter=',', header=None)
    return data[0], data[1]


def load_HP_4395A(fileAddress):
    """
    Analyzing HP 4395A Spectrum & Network Analyzer Data
    """
    data = pd.read_csv(fileAddress, delimiter='\t', header=11)
    return data["Frequency"], data["Data Trace"]


def load_RSA_6114A(fileLocation):
    """
    return xData, yData, yUnits, xUnits
    """
    lines = []
    count = 0
    yUnits = ""
    xUnits = ""
    xPointNum, xStart, xEnd = [0, 0, 0]
    with open(fileLocation) as file:
        for line in iter(file.readline, ''):
            count += 1
            # 18 lines to skip.
            if count == 11:
                yUnits = str(line[line[:].index('>')+1:line[1:].index('<')+1])
                continue
            elif count == 12:
                xUnits = str(line[line[:].index('>')+1:line[1:].index('<')+1])
                continue
            elif count == 16:
                xPointNum = float(line[line[:].index('>')+1:line[1:].index('<')+1])
                continue
            elif count == 17:
                xStart = float(line[line[:].index('>')+1:line[1:].index('<')+1])
                continue
            elif count == 18:
                xEnd = float(line[line[:].index('>')+1:line[1:].index('<')+1])
                continue
            elif count <= 18:
                continue
            try:
                lines.append(line[line[:].index('>')+1:line[1:].index('<')+1])
            except ValueError:
                pass
    yData = np.float64(arr(lines))
    xData = np.linspace(xStart, xEnd, xPointNum)
    return xData, yData, yUnits, xUnits


# ##########################
# ### Some AOM Optimizations

def getOptimalAomBiases(minX, minY, spacing, widthX, widthY):
    """

    :param minX:
    :param minY:
    :param spacing:
    :param widthX:
    :param widthY:
    :return:
    """
    # these calibrations were taken on 9/11/2017\n",
    # At Vertical Frequency = 80 MHz. \n",
    horFreq = [70, 75, 65, 67.5, 72.5, 80, 85, 90, 95, 60, 50, 55, 45, 62.5, 57.5, 52.5]
    powerInRail = [209, 197, 180, 198, 205, 186, 156, 130, 72.5, 181, 109, 179, 43.5, 174, 182, 165]
    relativeHorPowerInRail = arr(powerInRail)/max(powerInRail) * 100
    horAomCurve = interp.interp1d(horFreq, relativeHorPowerInRail)
    # at horizontal freq of 70MHz\n",
    vertFreq = [80, 82.5, 77.5, 75, 85, 90, 95, 100, 105, 70, 65, 60, 55, 50, 52.5, 57.5, 62.5]
    vertPowerInRail = [206, 204, 202, 201, 197, 184, 145, 126, 64, 193, 185, 140, 154, 103, 141, 140, 161]
    relativeVertPowerInRail = arr(vertPowerInRail) / max(vertPowerInRail) * 100
    vertAomCurve = interp.interp1d(vertFreq, relativeVertPowerInRail)
    xFreqs = [minX + num * spacing for num in range(widthX)]
    xAmps = [100 / horAomCurve(xFreq) for xFreq in xFreqs]
    yFreqs = [minY + num * spacing for num in range(widthY)]
    yAmps = [100 / vertAomCurve(yFreq) for yFreq in yFreqs]
    return xFreqs, xAmps, yFreqs, yAmps


def maximizeAomPerformance(minX, minY, spacing, widthX, widthY, iterations=10):
    """
    computes the amplitudes and phases to maximize the AOM performance.
    :param minX:
    :param minY:
    :param spacing:
    :param widthX:
    :param widthY:
    :param iterations:
    :return:
    """

    xFreqs, xAmps, yFreqs, yAmps = getOptimalAomBiases(minX, minY, spacing, widthX, widthY)

    def calcWave(xPts, phases, freqs, amps):
        volts = np.zeros(len(xPts))
        for phase, freq, amp in zip(phases, freqs, amps):
            volts += amp * np.cos(freq * 1e6 * xPts + phase)
        return volts

    def getXMax(phases):
        x = np.linspace(0, 10e-6, 10000)
        return max(calcWave(x, phases, xFreqs, xAmps))

    def getYMax(phases):
        x = np.linspace(0, 10e-6, 10000)
        return max(calcWave(x, phases, yFreqs, yAmps))

    xBounds = [(0, 2 * consts.pi) for _ in range(widthX)]
    xGuess = arr([0 for _ in range(widthX)])
    minimizer_kwargs = dict(method="L-BFGS-B", bounds=xBounds)
    xPhases = basinhopping(getXMax, xGuess, minimizer_kwargs=minimizer_kwargs, niter=iterations, stepsize=0.2)
    print('xFreqs', xFreqs)
    print('xAmps', xAmps)
    print('X-Phases', xPhases.x)

    yGuess = arr([0 for _ in range(widthY)])
    yBounds = [(0, 2 * consts.pi) for _ in range(widthY)]
    minimizer_kwargs = dict(method="L-BFGS-B", bounds=yBounds)
    yPhases = basinhopping(getYMax, yGuess, minimizer_kwargs=minimizer_kwargs, niter=iterations, stepsize=0.2)
    print('yFreqs', yFreqs)
    print('yAmps', yAmps)
    print('Y-Phases', yPhases.x)

    xpts = np.linspace(0, 10e-6, 10000)
    ypts = calcWave(xpts, xPhases.x, xFreqs, xAmps)
    yptsOrig = calcWave(xpts, xGuess, xFreqs, xAmps)
    title('X-Axis')
    plot(xpts, ypts, ':', label='X-Optimization')
    plot(xpts, yptsOrig, ':', label='X-Worst-Case')
    legend()

    figure()
    yptsOrig = calcWave(xpts, yGuess, yFreqs, yAmps)
    ypts = calcWave(xpts, yPhases.x, yFreqs, yAmps)
    title('Y-Axis')
    plot(xpts, ypts, ':', label='Y-Optimization')
    plot(xpts, yptsOrig, ':', label='Y-Worst-Case')
    legend()


def integrateData(pictures):
    """

    :param pictures:
    :return:
    """
    if len(pictures.shape) == 3:
        integratedData = np.zeros(pictures.shape[0])
        picNum = 0
        for pic in pictures:
            for row in pic:
                for elem in row:
                    integratedData[picNum] += elem
            picNum += 1
    else:
        integratedData = 0
        for row in pictures:
            for elem in row:
                integratedData += elem
    return integratedData


def fillPlotDataDefaults(plotData):
    """

    :param plotData:
    :return:
    """
    if 'ax1' not in plotData:
        raise RuntimeError('plot data must have ax1 defined!')
    else:
        if 'data' not in plotData['ax1']:
            raise RuntimeError('ax1 of plot data must contain data!')
        if 'ylabel' not in plotData['ax1']:
            plotData['ax1']['ylabel'] = ''
        if 'legendLabels' not in plotData['ax1']:
            if plotData['ax1']['data'].ndim == 2:
                # create empty labels
                plotData['ax1']['legendLabels'] = ['' for _ in range(plotData['ax1']['data'].shape[0])]
            else:
                plotData['ax1']['legendLabels'] = ''
    if 'ax2' in plotData:
        if 'data' not in plotData['ax2']:
            raise RuntimeError('if ax2 is defined, it must contain data!')
        if 'ylabel' not in plotData['ax2']:
            plotData['ax2']['ylabel'] = ''
        if 'legendLabels' not in plotData['ax2']:
            if plotData['ax2']['data'].ndim == 2:
                # create empty labels
                plotData['ax2']['legendLabels'] = ['' for _ in range(plotData['ax2']['data'].shape[0])]
            else:
                plotData['ax2']['legendLabels'] = ''
    if 'xlabel' not in plotData:
        plotData['xlabel'] = ""
    if 'title' not in plotData:
        plotData['title'] = ""


def assemblePlotData(rawData, dataMinusBg, dataMinusAverage, positions, waists, plottedData, scanType,
                     xLabel, plotTitle, location, waistFits=None, key=None):
    """
    take the data and organize it into the appropriate structures.

    :param rawData:
    :param dataMinusBg:
    :param dataMinusAverage:
    :param positions:
    :param waists:
    :param plottedData:
    :param scanType:
    :param xLabel:
    :param plotTitle:
    :param location:
    :param waistFits:
    :param key:
    :return: countData, fitData
    """
    if key is None:
        key = []
    if waistFits is None:
        waistFits = []
    countData = {}
    countData['xlabel'], countData['title'] = getLabels(scanType)
    if not xLabel == "":
        countData['xlabel'] = xLabel
    if not plotTitle == "":
        countData['title'] = plotTitle
    ax1 = {}
    if location == (-1, -1):
        data = []
        legendLabels = []
        if "raw" in plottedData:
            data.append(integrateData(rawData))
            legendLabels.append(['Int(Raw Data)'])
        if "-bg" in plottedData:
            data.append(integrateData(dataMinusBg))
            legendLabels.append('Int(Data - Background)')
        if "-avg" in plottedData:
            data.append(integrateData(dataMinusAverage))
            legendLabels.append('Int(Data - Average)')
        ax1['data'] = arr(data)
        ax1['legendLabels'] = arr(legendLabels)
        ax1['ylabel'] = 'Camera Counts (Integrated over Picture)'

        ax2 = {}
        data = []
        legendLabels = []
        if "raw" in plottedData:
            data.append([np.max(pic, axis=(1, 0)) for pic in rawData])
            legendLabels.append(['Max Value in Raw Data'])
        if "-bg" in plottedData:
            data.append([np.max(pic, axis=(1, 0)) for pic in dataMinusBg])
            legendLabels.append('Max Value in Data Without Background')
        if "-avg" in plottedData:
            data.append([np.max(pic, axis=(1, 0)) for pic in dataMinusAverage])
            legendLabels.append('Max Value in Data Without Average')
        ax2['data'] = arr(data)
        ax2['legendLabels'] = arr(legendLabels)
        ax2['ylabel'] = 'Maximum count in picture'
        countData['ax2'] = ax2
    else:
        ax1['data'] = arr([rawData[:, location[0], location[1]]])
        ax1['ylabel'] = 'Camera Counts (single pixel)'
    countData['ax1'] = ax1
    ax1['data'] = {'data': waists}
    ax1['ylabel'] = "Waist ($2\sigma$) (pixels)"
    if len(waistFits) == 0:
        ax1['legendLabels'] = ["fit $w_x$", "fit $w_y$"]
    else:
        print('...')
        ax1['legendLabels'] = ["fit $w_x$", "fit $w_y$", 'Fitted X: ' + str(waistFits[0]),
                               'Fitted Y: ' + str(waistFits[1])]
        fitYData = []
        xpts = np.linspace(min(key), max(key), 1000)
        for fitParams in waistFits:
            fitYData.append(beamWaistExpansion(xpts, fitParams[0], fitParams[1], 850e-9))
        ax1['fitYData'] = fitYData
        ax1['fitXData'] = [xpts, xpts]
    fitData = {'ax1': ax1}
    ax2 = {'data': positions, 'ylabel': "position (pixels)", 'legendLabels': ["Center x", "Center y"]}
    fitData['ax2'] = ax2
    fitData['title'] = "Fit Information"
    fitData['xlabel'] = countData['xlabel']

    return countData, fitData


def showPicComparisons(data, key, fitParameters=np.array([])):
    """
    formerly the "individualPics" option.
    expects structure:
    data[key value number][raw, -background, -average][2d pic]

    :param data:
    :param key:
    :param fitParameters:
    :return:
    """
    if data.ndim != 4:
        raise ValueError("Incorrect dimensions for data input to show pics if you want individual pics.")
    titles = ['Raw Picture', 'Background Subtracted', 'Average Subtracted']
    for inc in range(len(data)):
        figure()
        fig, plts = subplots(1, len(data[inc]), figsize=(15, 6))
        count = 0
        for pic in data[inc]:
            x = np.linspace(1, pic.shape[1], pic.shape[1])
            y = np.linspace(1, pic.shape[0], pic.shape[0])
            x, y = np.meshgrid(x, y)
            im = plts[count].pcolormesh(pic, extent=(x.min(), x.max(), y.min(), y.max()))
            fig.colorbar(im, ax=plts[count], fraction=0.046, pad=0.04)
            plts[count].set_title(titles[count])
            plts[count].axis('off')
            if fitParameters.size != 0:
                if (fitParameters[count] != np.zeros(len(fitParameters[count]))).all():
                    data_fitted = fitFunc.gaussian_2D((x, y), *fitParameters[count])
                    try:
                        # used to be "picture" which was unresolved, assuming should have been pic, as I've changed
                        # below.
                        plts[count].contour(x, y, data_fitted.reshape(pic.shape[0], pic.shape[1]), 2, colors='w',
                                            alpha=0.35, linestyles="dashed")
                    except ValueError:
                        pass
            count += 1
        fig.suptitle(str(key[inc]))


def showBigPics(data, key, fitParameters=np.array([]), individualColorBars=False, colorMax=-1):
    """
    formerly the "bigPictures" option.

    :param data:
    :param key:
    :param fitParameters:
    :param individualColorBars:
    :param colorMax:
    :return:
    """
    if data.ndim != 3:
        raise ValueError("Incorrect dimensions for data input showBigPics.")
    count = 0
    maximum = sorted(data.flatten())[colorMax]
    minimum = min(data.flatten())
    # get picture fits & plots
    for picture in data:
        fig = figure()
        grid(0)
        if individualColorBars:
            maximum = max(picture.flatten())
            minimum = min(picture.flatten())
        x = np.linspace(1, picture.shape[1], picture.shape[1])
        y = np.linspace(1, picture.shape[0], picture.shape[0])
        x, y = np.meshgrid(x, y)
        im = pcolormesh(picture, vmin=minimum, vmax=maximum)
        axis('off')
        title(str(round_sig(key[count], 4)), fontsize=8)
        if fitParameters.size != 0:
            if (fitParameters[count] != np.zeros(len(fitParameters[count]))).all():
                data_fitted = fitFunc.gaussian_2D((x, y), *fitParameters[count])
                try:
                    contour(x, y, data_fitted.reshape(picture.shape[0], picture.shape[1]), 2, colors='w', alpha=0.35,
                            linestyles="dashed")
                except ValueError:
                    pass
        count += 1
        # final touches
        cax = fig.add_axes([0.95, 0.1, 0.03, 0.8])
        fig.colorbar(im, cax=cax)


def showPics(data, key, fitParameters=np.array([]), individualColorBars=False, colorMax=-1):
    """
    formerly the default option.

    :param data:
    :param key:
    :param fitParameters:
    :param individualColorBars:
    :param colorMax:
    :return:
    """
    # if data.ndim != 3:
    #    raise ValueError("Incorrect dimensions for data input to show pics if you don't want individual pics.")
    num = len(data)
    gridsize1, gridsize2 = (0, 0)
    for i in range(100):
        if i*i >= num:
            gridsize1 = i
            if i*(i-1) >= num:
                gridsize2 = i-1
            else:
                gridsize2 = i
            break
    fig, plts = subplots(gridsize2, gridsize1, figsize=(15, 10))
    count = 0
    rowCount = 0
    picCount = 0
    maximum = sorted(data.flatten())[colorMax]
    minimum = min(data.flatten())
    # get picture fits & plots
    for row in plts:
        for _ in row:
            plts[rowCount, picCount].grid(0)
            if count >= len(data):
                count += 1
                picCount += 1
                continue
            picture = data[count]
            if individualColorBars:
                maximum = max(picture.flatten())
                minimum = min(picture.flatten())
            x = np.linspace(1, picture.shape[1], picture.shape[1])
            y = np.linspace(1, picture.shape[0], picture.shape[0])
            x, y = np.meshgrid(x, y)
            im = plts[rowCount, picCount].imshow(picture, origin='bottom', extent=(x.min(), x.max(), y.min(), y.max()),
                                                 vmin=minimum, vmax=maximum)
            plts[rowCount, picCount].axis('off')
            plts[rowCount, picCount].set_title(str(round_sig(key[count], 4)), fontsize=8)
            if fitParameters.size != 0:
                if (fitParameters[count] != np.zeros(len(fitParameters[count]))).all():
                    # data_fitted = gaussian_2D((x, y), *fitParameters[count])
                    try:
                        ellipse = Ellipse(xy=(fitParameters[count][1], fitParameters[count][2]),
                                          width=2*fitParameters[count][3], height=2*fitParameters[count][4],
                                          angle=fitParameters[count][5], edgecolor='r', fc='None', lw=2, alpha=0.2)
                        plts[rowCount, picCount].add_patch(ellipse)
                        # plts[rowCount, picCount].contour(x, y,
                        #                                 data_fitted.reshape(picture.shape[0], picture.shape[1]),
                        #                                 2, colors='w', alpha=0.35, linestyles="dashed")
                    except ValueError:
                        pass
            count += 1
            picCount += 1
        picCount = 0
        rowCount += 1
    # final touches
    cax = fig.add_axes([0.95, 0.1, 0.03, 0.8])
    fig.colorbar(im, cax=cax)


def beamIntensity(power, waist, radiusOfInterest=0):
    """
    computes the average beam intensity, in mW/cm^2, of a beam over some radius of interest.

    :param power: power of the laser beam, in mW
    :param waist: waist of the laser beam, in cm.
    :param radiusOfInterest: the radius of interest. In the case that this is << waist, the equation below
        reduces to a simpler, more commonly referenced form. The literal math gives 0/0 though, so I
        include the reduced form.
    """
    if radiusOfInterest == 0:
        return 2 * power / (consts.pi * waist ** 2)
    else:
        return power * (1 - np.exp(-2 * radiusOfInterest ** 2 / waist ** 2)) / (consts.pi * radiusOfInterest ** 2)


def computeBaslerGainDB(rawGain):
    """
    Gain (NOT currently used in fluorescence calc...)
    """
    G_c = 20 * np.log10((658 + rawGain)/(658 - rawGain))
    if 110 <= rawGain <= 511:
        gainDB = 20 * np.log10((658 + rawGain)/(658 - rawGain)) - G_c
    elif 511 <= rawGain <= 1023:
        gainDB = 0.0354 * rawGain - G_c
    else:
        gainDB = None
        warn('raw gain out of range! gainDB set to None/')
    return gainDB


def computeScatterRate(totalIntensity, D2Line_Detuning):
    """
    Computes the rate of photons scattering off of a single atom. From steck, equation 48.

    Assumes 2-Level approximation, good for near resonant light since the near-resonant transition
    will be dominant.

    Assumes D2 Transition.

    :param totalIntensity: the total intensity (from all beams) shining on the atoms.
    :param D2Line_Detuning: the detuning, in Hz, of the light shining on the atoms from the D2 transition.
    """
    isat = consts.Rb87_I_ResonantIsotropicSaturationIntensity
    rate = (consts.Rb87_D2Gamma / 2) * (totalIntensity / isat) / (1 + 4 * (D2Line_Detuning / consts.Rb87_D2Gamma) ** 2
                                                                  + totalIntensity / isat)
    return rate


def computeFlorescence(greyscaleReading, imagingLoss, imagingLensDiameter, imagingLensFocalLength, exposure ):
    """
    TODO: incorporate gain into the calculation, currently assumes gain = X1... need to check proper conversion
    from basler software. I'd expect a power conversion so a factor of 20,
    Fluorescence

    :param greyscaleReading:
    :param imagingLoss:
    :param imagingLensDiameter:
    :param imagingLensFocalLength:
    :param exposure:
    :return:
    """
    term1 = greyscaleReading * consts.cameraConversion / (consts.h * consts.c / consts.Rb87_D2LineWavelength)
    term2 = 1 * imagingLoss * (imagingLensDiameter**2 / (16 * imagingLensFocalLength**2)) * exposure
    fluorescence = term1 / term2
    return fluorescence


# mot radius is in cm
def computeMotNumber(sidemotPower, diagonalPower, motRadius, exposure, imagingLoss, greyscaleReading, detuning=10e6):
    """
    :param sidemotPower: power in the sidemot beam, in mW.  Code Assumes 3.3mm sidemot waist
    :param diagonalPower: power in an individual diagonal mot beam, in mW
    :param motRadius: the approximate radius of the MOT. Used as a higher order part of the calculation which takes into
        account the spread of the intensity of the beams over the finite size of the MOT. Less needed for
        big MOT beams.
    :param exposure: exposure time of the camera, in seconds.
    :param imagingLoss: Approximate amount of light lost in the imaging line due to mirrors efficiency, filter
        efficiency, etc.
    :param greyscaleReading: the integrated greyscale count reading from the camera.

    ===
    The mot number is determined via the following formula:

    MOT # = (Scattered Light Collected) / (Scattered light predicted per atom)

    Here with sideMOT power in mW assuming 3.3mm radius waist and a very rough estimation of main MOT diameter
    one inch, motRadius using the sigma of the MOT size but should not matter if it's small enough, and exposure
    in sec, typically 0.8 for the imaging loss accounting for the line filter, greyscaleReading is the integrated gray
    scale count with 4by4 binning on the Basler camera, and assuming gain set to 260 which is  unity gain for Basler
    """
    # in cm
    sidemotWaist = .33 / (2 * np.sqrt(2))
    # in cm
    diagonalWaist = 2.54 / 2
    # intensities
    sidemotIntensity = beamIntensity(sidemotPower, sidemotWaist, motRadius)
    diagonalIntensity = beamIntensity(diagonalPower, diagonalWaist, motRadius)
    totalIntensity = sidemotIntensity + 2 * diagonalIntensity
    rate = computeScatterRate(totalIntensity, detuning)
    imagingLensDiameter = 2.54
    imagingLensFocalLength = 10
    fluorescence = computeFlorescence(greyscaleReading, imagingLoss, imagingLensDiameter, imagingLensFocalLength,
                                      exposure)

    print('Light Scattered off of full MOT:', fluorescence * consts.h * consts.Rb87_D2LineFrequency * 1e9, "nW")
    motNumber = fluorescence / rate
    return motNumber


# TODO: for some reason this fitting is currently very finicky with respect to sigma_I. Don't understand why. fix this.
def calcMotTemperature(times, sigmas):
    print(sigmas[0])
    guess = [sigmas[0], 0.1]
    # guess = [0.001, 0.1]
    # in cm
    # sidemotWaist = .33 / (2*np.sqrt(2))
    sidemotWaist = 8 / (2*np.sqrt(2))
    # sidemotWaist^2/2 = 2 sigma_sidemot^2
    # different gaussian definitions
    sigma_I = sidemotWaist / 2
    # convert to m
    sigma_I /= 100
    # modify roughly for angle of beam
    # sigma_I /= np.cos(2*pi/3)
    sigma_I /= np.cos(consts.pi/4)
    sigma_I = 100
    fitVals, fitCovariances = fit(lambda x, a, b: ballisticMotExpansion(x, a, b, sigma_I), times, sigmas, p0=guess)
    simpleVals, simpleCovariances = fit(simpleMotExpansion, times, sigmas, p0=guess)
    temperature = consts.Rb87_M / consts.k_B * fitVals[1]**2
    tempFromSimple = consts.Rb87_M / consts.k_B * simpleVals[1]**2
    return temperature, tempFromSimple, fitVals, fitCovariances, simpleVals, simpleCovariances


def orderData(data, key, keyDim=None, otherDimValues=None):
    """

    :param data:
    :param key:
    :param keyDim:
    :param otherDimValues:
    :return: data, key, otherDimValues
    """
    zipObj = (zip(key, data, otherDimValues) if otherDimValues is not None else zip(key, data))
    if keyDim is not None:
        key, data, otherDimValues = list(zip(*sorted(zipObj, key=lambda x: x[0][keyDim])))
        # assuming 2D
        count = 0
        for val in key:
            if val[keyDim] == key[0][keyDim]:
                count += 1
        majorKeySize = int(len(key) / count)
        tmpKey = arr(key[:])
        tmpVals = arr(data[:])
        tmpKey.resize([majorKeySize, count, 2])
        tmpVals.resize([majorKeySize, count, arr(data).shape[1], arr(data).shape[2], arr(data).shape[3]])
        finKey = []
        finData = []
        for k, d in zip(tmpKey, tmpVals):
            k1, d1 = list(zip(*sorted(zip(k, d), key=lambda x: x[0][int(not keyDim)])))
            for k2, d2 in zip(k1, d1):
                finKey.append(arr(k2))
                finData.append(arr(d2))
        return arr(finData), arr(finKey), arr(otherDimValues)
    else:
        key, data = list(zip(*sorted(zipObj, key=lambda x: x[0])))
    return arr(data), arr(key), arr(otherDimValues)


def groupMultidimensionalData(key, varyingDim, atomLocations, survivalData, survivalErrs, loadingRate):
    """
    Normally my code takes all the variations and looks at different locations for all those variations.
    In the multi-dim case, this there are multiple variations for the same primary key value. I need to
    split up those multiple variations.
    """
    if len(key.shape) == 1:
        # no grouping needed
        return (key, atomLocations, survivalErrs, survivalData, loadingRate,
                [None for _ in range(len(key)*len(atomLocations))])
    # make list of unique indexes for each dimension
    uniqueSecondaryAxisValues = []
    newKey = []
    for i, secondaryValues in enumerate(transpose(key)):
        if i == varyingDim:
            for val in secondaryValues:
                if val not in newKey:
                    newKey.append(val)
            continue
        uniqueSecondaryAxisValues.append([])
        for val in secondaryValues:
            if val not in uniqueSecondaryAxisValues[-1]:
                uniqueSecondaryAxisValues[-1].append(val)
    extraDimValues = 1
    for i, dim in enumerate(uniqueSecondaryAxisValues):
        extraDimValues *= len(dim)
    newLoadingRate, newTransferData, newErrorData, locationsList, otherDimsList = [[] for _ in range(5)]
    allSecondaryDimVals = arr(uniqueSecondaryAxisValues).flatten()
    # iterate through all locations
    for loc, locData, locErrs, locLoad in zip(atomLocations, survivalData, survivalErrs, loadingRate):
        newData = locData[:]
        newErr = locErrs[:]
        newLoad = locLoad[:]
        newData.resize(int(len(locData)/extraDimValues), extraDimValues)
        newData = transpose(newData)
        newErr.resize(int(len(locData)/extraDimValues), extraDimValues)
        newErr = transpose(newErr)
        newLoad.resize(int(len(locData)/extraDimValues), extraDimValues)
        newLoad = transpose(newLoad)
        # iterate through all extra dimensions in the locations
        secondIndex = 0
        for val, err, load in zip(newData, newErr, newLoad):
            newTransferData.append(val)
            newErrorData.append(err)
            newLoadingRate.append(load)
            locationsList.append(loc)
            otherDimsList.append(allSecondaryDimVals[secondIndex])
            secondIndex += 1
    return (arr(newKey), arr(locationsList), arr(newErrorData), arr(newTransferData), arr(newLoadingRate),
            arr(otherDimsList))


def getLoadingData(picSeries, loc, whichPic, picsPerExperiment, manThreshold, binWidth):
    """

    :param picSeries:
    :param loc:
    :param whichPic:
    :param picsPerExperiment:
    :param manThreshold:
    :param binWidth:
    :return:
    """
    # grab the first picture of each repetition
    pic1Data = normalizeData(picSeries, loc, whichPic, picsPerExperiment)
    if manThreshold is not None:
        threshold = manThreshold
        thresholdFid = 0
        bins, binnedData, fitVals = [None]*3
    else:
        bins, binnedData = getBinData(binWidth, pic1Data)
        guess1, guess2 = guessGaussianPeaks(bins, binnedData)
        # ## Calculate Atom Threshold
        numberOfPictures = len(picSeries)
        guess = arr([numberOfPictures/100, guess1, 40,  numberOfPictures/100, 200, 40])
        fitVals = fitDoubleGaussian(bins, binnedData, guess)
        threshold, thresholdFid = calculateAtomThreshold(fitVals)
    atomCount = 0
    pic1Atom = []
    for point in pic1Data:
        if point > threshold:
            atomCount += 1
            pic1Atom.append(1)
        else:
            pic1Atom.append(0)
    return list(pic1Data), pic1Atom, threshold, thresholdFid, fitVals, bins, binnedData, atomCount


def calculateAtomThreshold(fitVals):
    """
    TODO: Figure out how this is supposed to work.
    :param fitVals = [Amplitude1, center1, sigma1, amp2, center2, sigma2]
    """
    # difference between centers divided by total sigma?
    TCalc = (fitVals[4] - fitVals[1])/(np.abs(fitVals[5]) + np.abs(fitVals[2]))
    threshold = abs(fitVals[1] + TCalc * fitVals[2])
    fidelity = 1/2 * (1 + special.erf(np.abs(TCalc)/np.sqrt(2)))
    return threshold, fidelity


<<<<<<< HEAD
def postSelectOnAssembly(pic1Atoms, pic2Atoms, postSelectionPic):
    ensembleHits = ((getEnsembleHits(pic1Atoms, postSelectionPic) if True
                    else getEnsembleHits(pic2Atoms, postSelectionPic)))
    # ps for post-selected
    psPic1Atoms, psPic2Atoms = [[[] for _ in pic1Atoms] for _ in range(2)]
    for i, hit in enumerate(ensembleHits):
        if hit:
            for atom1, orig1, atom2, orig2 in zip(psPic1Atoms, pic1Atoms, psPic2Atoms, pic2Atoms):
                atom1.append(orig1[i])
                atom2.append(orig2[i])
            # else nothing!
    print('Post-Selecting on assembly condition:', postSelectionPic, '. Number of hits:', len(psPic1Atoms[0]))
=======
def postSelectOnAssembly(pic1Atoms, pic2Atoms, hitCondition, connected=False):
    ensembleHits = getEnsembleHits(pic1Atoms, hitCondition=hitCondition, connected=connected)
    # ps for post-selected
    psPic1Atoms, psPic2Atoms = [[[] for _ in range(len(pic1Atoms))] for _ in range(2)]
    for i, ensembleHit in enumerate(ensembleHits):
        if ensembleHit:
            for atomInc in range(len(psPic1Atoms)):
                psPic1Atoms[atomInc].append(pic1Atoms[atomInc][i])
                psPic2Atoms[atomInc].append(pic2Atoms[atomInc][i])
        # else nothing!
    psPic1Atoms = arr(psPic1Atoms)
    psPic2Atoms = arr(psPic2Atoms)
    print('Post-Selecting on Condition:', hitCondition, '. Hits:', len(psPic2Atoms[0]))
>>>>>>> 2aa20b43f0a60d509d2c74e69d9e3d993d4c46e2
    return psPic1Atoms, psPic2Atoms


def normalizeData(data, atomLocation, picture, picturesPerExperiment, subtractBorders=True):
    """
    :param picturesPerExperiment:
    :param picture:
    :param subtractBorders:
    :param data: the array of pictures
    :param atomLocation: The location to analyze
    :return: The data at atomLocation with the background subtracted away (commented out at the moment).
    """
    allData = arr([])
    dimensions = data.shape
    if len(dimensions) == 4:
        rawData = data.reshape((data.shape[0] * data.shape[1], data.shape[2], data.shape[3]))
    else:
        rawData = data
    dimensions = rawData.shape
    for imageInc in range(0, dimensions[0]):
        if (imageInc + picturesPerExperiment - picture) % picturesPerExperiment == 0:
            pic = rawData[imageInc]
            border = 0
            if subtractBorders:
                normFactor = (2*len(pic[0][:])+2*len(pic[:][0]))
                border += (np.sum(pic[0][:]) + np.sum(pic[:][0]) + np.sum(pic[dimensions[1] - 1][:])
                           + np.sum([pic[i][dimensions[2] - 1] for i in range(len(pic))]))
                corners = (pic[0][0] + pic[dimensions[1]-1][dimensions[2] - 1] + pic[0][dimensions[2] - 1]
                           + pic[dimensions[1]-1][0])
                border -= corners
                # the number of pixels counted in the border
                border /= normFactor - 4
            allData = np.append(allData, pic[atomLocation[0]][atomLocation[1]] - border)
    return allData


def guessGaussianPeaks(binCenters, binnedData):
    """
    This function guesses where the gaussian peaks of the data are. It assumes one is near the maximum of the binned
    data. Then, from the binned data it subtracts an over-weighted (i.e. extra tall) poissonion distribution e^-k k^n/n!
    From the binned data. This should squelch the peak that it found. It then assumes that the second peak is near the
    maximum of the (data-poissonian) array.
    :param binCenters: The pixel-numbers corresponding to the binned data data points.
    :param binnedData: the binned data data points.
    :return: the two guesses.
    """
    randomOffset = 300
    binCenters += randomOffset
    # get index corresponding to global max
    guess1Index = np.argmax(binnedData)
    # get location of global max
    guess1Location = binCenters[guess1Index]
    binnedDataNoPoissonian = []
    for binInc in range(0, len(binCenters)):
        binnedDataNoPoissonian.append(binnedData[binInc]
                                      - fitFunc.poissonian(binCenters[binInc], guess1Location, 2 * max(binnedData) /
                                                           fitFunc.poissonian(guess1Location, guess1Location, 1)))
    guess2Index = np.argmax(binnedDataNoPoissonian)
    guess2Location = binCenters[guess2Index]
    binCenters -= randomOffset
    return guess1Location - randomOffset, guess2Location - randomOffset


def getBinData(binWidth, data):
    """

    :param binWidth:
    :param data:
    :return:
    """
    binBorderLocation = min(data)
    binsBorders = arr([])
    # get bin borders
    while binBorderLocation < max(data):
        binsBorders = np.append(binsBorders, binBorderLocation)
        binBorderLocation = binBorderLocation + binWidth
    # trash gets set but is unused.
    binnedData, trash = np.histogram(data, binsBorders)
    binCenters = binsBorders[0:binsBorders.size-1]
    return binCenters, binnedData


def getSurvivalEvents(pic1Atoms, pic2Atoms):
    """
    It returns a raw array that includes every survival data point, including points where the the atom doesn't get
    loaded at all.
    """
    # this will include entries for when there is no atom in the first picture.
    survivalData = np.array([])
    survivalData.astype(int)
    # flattens variations & locations?
    # if len(data.shape) == 4:
    #     data.reshape((data.shape[0] * data.shape[1], data.shape[2], data.shape[3]))

    # this doesn't take into account loss, since these experiments are feeding-back on loss.
    for atom1, atom2 in zip(pic1Atoms, pic2Atoms):
        if atom1 and atom2:
            # atom survived
            survivalData = np.append(survivalData, [1])
        elif atom1 and not atom2:
            # atom didn't survive
            survivalData = np.append(survivalData, [0])
        else:
            # no atom in the first place
            survivalData = np.append(survivalData, [-1])
    return survivalData


def getSurvivalData(survivalData, repetitionsPerVariation):
    # Take the previous data, which includes entries when there was no atom in the first picture, and convert it to
    # an array of just loaded and survived or loaded and died.
    survivalAverages = np.array([])
    loadingProbability = np.array([])
    survivalErrors = np.array([])
    if survivalData.size < repetitionsPerVariation:
        repetitionsPerVariation = survivalData.size
    for variationInc in range(0, int(survivalData.size / repetitionsPerVariation)):
        survivalList = np.array([])
        for repetitionInc in range(0, repetitionsPerVariation):
            if survivalData[variationInc * repetitionsPerVariation + repetitionInc] != -1:
                survivalList = np.append(survivalList,
                                         survivalData[variationInc * repetitionsPerVariation + repetitionInc])
        if survivalList.size == 0:
            # catch the case where there's no relevant data, typically if laser becomes unlocked.
            survivalErrors = np.append(survivalErrors, [0])
            loadingProbability = np.append(loadingProbability, [0])
            survivalAverages = np.append(survivalAverages, [0])
        else:
            # normal case
            survivalErrors = np.append(survivalErrors, np.std(survivalList)/np.sqrt(survivalList.size))
            loadingProbability = np.append(loadingProbability, survivalList.size / repetitionsPerVariation)
            survivalAverages = np.append(survivalAverages, np.average(survivalList))
    return survivalAverages, survivalErrors, loadingProbability


def getAvgPic(picSeries):
    if len(picSeries.shape) == 3:
        avgPic = np.zeros(picSeries[0].shape)
        for pic in picSeries:
            avgPic += pic
        avgPic = avgPic / len(picSeries)
        return avgPic
    elif len(picSeries.shape) == 4:
        avgPic = np.zeros(picSeries[0][0].shape)
        for variation in picSeries:
            for pic in variation:
                avgPic += pic
        avgPic = avgPic / (len(picSeries) * len(picSeries[0]))
        return avgPic


def processSingleImage(rawData, bg, window, xMin, xMax, yMin, yMax, accumulations, zeroCorners, smartWindow,
                       manuallyAccumulate=True):
    """
    Process the original data, giving back data that has been ordered and windowed as well as two other versions that
    have either the background or the average of the pictures subtracted out.

    This is a helper function that is expected to be embedded in a package. As such, many parameters are simply
    passed through some other function in order to reach this function, and all parameters are required.
    """
    # handle manual accumulations, where the code just sums pictures together.
    if manuallyAccumulate and not len(rawData.shape) == 3:
        print('ERROR: Requested manual accumulation but raw data doesn"t have the correct shape for that.')
    if manuallyAccumulate:
        avgPics = np.zeros((rawData.shape[1], rawData.shape[2]))
        count = 0
        for pic in rawData:
            avgPics += pic
            count += 1
        rawData = avgPics
    # handle windowing defaults
    allXPts = np.arange(1, rawData.shape[1])
    allYPts = np.arange(1, rawData.shape[0])

    if smartWindow:
        maxLocs = coordMax(rawData)
        xMin = maxLocs[1] - rawData.shape[1] / 5
        xMax = maxLocs[1] + rawData.shape[1] / 5
        yMin = maxLocs[0] - rawData.shape[0] / 5
        yMax = maxLocs[0] + rawData.shape[0] / 5
    elif window != (0, 0, 0, 0):
        xMin = window[0]
        xMax = window[1]
        yMin = window[2]
        yMax = window[3]
    else:
        if xMax == 0:
            xMax = len(rawData[0])
        if yMax == 0:
            yMax = len(rawData)
        if xMax < 0:
            xMax = 0
        if yMax < 0:
            yMax = 0

    xPts = allXPts[xMin:xMax]
    yPts = allYPts[yMin:yMax]

    # window images.
    rawData = np.copy(arr(rawData[yMin:yMax, xMin:xMax]))

    # final normalized data
    normData = rawData / accumulations

    # ### -Background Analysis
    # if user just entered a number, assume that it's a file number.
    if type(bg) == int and not bg == 0:
        print('loading background file ', bg)
        bg, _, _, _ = loadHDF5(bg)
        if manuallyAccumulate:
            avgPics = np.zeros((bg.shape[1], bg.shape[2]))
            count = 0
            for pic in bg:
                avgPics += pic
                count += 1
            bg = avgPics
        else:
            bg = bg[0]
        bg /= accumulations
    # window the background
    if not bg.size == 1:
        bg = np.copy(arr(bg[yMin:yMax, xMin:xMax]))
    dataMinusBg = np.copy(normData)
    dataMinusBg -= bg

    # it's important and consequential that the zeroing here is done after the background / corner is subtracted.
    if zeroCorners:
        cornerAvg = (pic[0, 0] + pic[0, -1] + pic[-1, 0] + pic[-1, -1]) / 4
        dataMinusBg -= cornerAvg
        cornerAvg = (pic[0, 0] + pic[0, -1] + pic[-1, 0] + pic[-1, -1]) / 4
        normData -= cornerAvg
    return normData, dataMinusBg, xPts, yPts


def processImageData(key, rawData, bg, window, xMin, xMax, yMin, yMax, accumulations, dataRange, zeroCorners,
                     smartWindow, manuallyAccumulate=False):
    """
    Process the orignal data, giving back data that has been ordered and windowed as well as two other versions that
    have either the background or the average of the pictures subtracted out.

    This is a helper function that is expected to be embedded in a package. As such, many parameters are simply
    passed through some other function in order to reach this function, and all parameters are required.
    """
    # handle windowing defaults
    print('beg', rawData.shape)
    if smartWindow:
        maxLocs = []
        for dat in rawData:
            maxLocs.append(coordMax(dat))
        maxLocs = arr(maxLocs)
        xMin = min(maxLocs[:, 0])
        xMax = max(maxLocs[:, 0])
        yMin = min(maxLocs[:, 1])
        yMax = max(maxLocs[:, 1])
        xRange = rawData.shape[2] / 2
        yRange = rawData.shape[1] / 2
        if xRange < xMax - xMin:
            xRange = xMax - xMin
        if yRange < yMax - yMin:
            yRange = yMax - yMin
        xMin -= 0.2 * xRange
        xMax += 0.2 * xRange
        yMin -= 0.2 * yRange
        yMax += 0.2 * yRange
    elif window != (0, 0, 0, 0):
        xMin = window[0]
        xMax = window[1]
        yMin = window[2]
        yMax = window[3]
    else:
        if xMax == 0:
            xMax = len(rawData[0][0])
        if yMax == 0:
            yMax = len(rawData[0])
        if xMax < 0:
            xMax = 0
        if yMax < 0:
            yMax = 0

    if manuallyAccumulate:
        # ignore shape[1], which is the number of pics in each variation. These are what are getting averaged.
        avgPics = np.zeros((int(rawData.shape[0] / accumulations), rawData.shape[1], rawData.shape[2]))
        print('avgpics shape:', avgPics.shape)
        varCount = 0
        for var in avgPics:
            for picNum in range(accumulations):
                var += rawData[varCount * accumulations + picNum]
            varCount += 1
        rawData = avgPics

    # combine and order data.
    rawData, key = combineData(rawData, key)
    rawData, key, _ = orderData(rawData, key)

    # window images.
    rawData = np.copy(arr(rawData[:, yMin:yMax, xMin:xMax]))
    # pull out the images to be used for analysis.
    if not dataRange == (0, 0):
        rawData = rawData[dataRange[0]:dataRange[-1]]
        key = key[dataRange[0]:dataRange[-1]]
        # final normalized data
    normData = rawData / accumulations

    # ### -Background Analysis
    # if user just entered a number, assume that it's a file number.
    if type(bg) == int and not bg == 0:
        bg = loadFits(bg)
        if manuallyAccumulate:
            avgPics = np.zeros((bg.shape[1], bg.shape[2]))
            count = 0
            for pic in bg:
                avgPics += pic
                count += 1
            bg = avgPics

        bg /= accumulations
    # window the background
    if not bg.size == 1:
        bg = np.copy(arr(bg[yMin:yMax, xMin:xMax]))

    dataMinusBg = np.copy(normData)
    for pic in dataMinusBg:
        pic -= bg
    # ### -Average Analysis
    # make a picture which is an average of all pictures over the run.
    avgPic = 0
    for pic in normData:
        avgPic += pic
    avgPic /= len(normData)
    print('avgpic shape:', avgPic.shape)
    dataMinusAvg = np.copy(normData)
    for pic in dataMinusAvg:
        pic -= avgPic

    # it's important and consequential that the zeroing here is done after the background / corner is subtracted.
    if zeroCorners:
        for pic in dataMinusBg:
            cornerAvg = (pic[0, 0] + pic[0, -1] + pic[-1, 0] + pic[-1, -1]) / 4
            pic -= cornerAvg
        for pic in dataMinusAvg:
            cornerAvg = (pic[0, 0] + pic[0, -1] + pic[-1, 0] + pic[-1, -1]) / 4
            pic -= cornerAvg
        for pic in normData:
            cornerAvg = (pic[0, 0] + pic[0, -1] + pic[-1, 0] + pic[-1, -1]) / 4
            pic -= cornerAvg

    return key, normData, dataMinusBg, dataMinusAvg, avgPic


def handleFitting(fitType, key, data):
    """

    :param fitType:
    :param key:
    :param data:
    :return: returns fitInfo, fitType, where fitInfo = {'x': xFit, 'nom': fitNom, 'std': fitStd, 'center': centerIndex,
            'vals': fitValues, 'errs': fitErrs, 'cov': fitCovs}
    """
    xFit = (np.linspace(min(key), max(key), 1000) if len(key.shape) == 1 else np.linspace(min(transpose(key)[0]),
                                                                                          max(transpose(key)[0]), 1000))
    fitNom = fitStd = centerIndex = fitValues = fitErrs = fitCovs = None
    if fitType == 'Quadratic-Bump':
        try:
            # 3 parameters
            if len(key) < len(signature(fitFunc.quadraticBump).parameters) - 1:
                print('Not enough data points to constrain a fit!')
                raise RuntimeError()
            widthGuess = np.std(key) / 2
            centerGuess = key[list(data).index(max(data))]
            fitValues, fitCovs = fit(fitFunc.quadraticBump, key, data, p0=[max(data), -1/widthGuess, centerGuess])
            fitErrs = np.sqrt(np.diag(fitCovs))
            a, b, c = unc.correlated_values(fitValues, fitCovs)
            fitObject = fitFunc.quadraticBump(xFit, a, b, c)
            fitNom = unp.nominal_values(fitObject)
            fitStd = unp.std_devs(fitObject)
            centerIndex = 2
        except RuntimeError:
            warn('Data Fit Failed!')
            fitType = None
    elif fitType == 'Gaussian-Bump':
        try:
            if len(key) < len(signature(fitFunc.gaussian).parameters) - 1:
                print('Not enough data points to constrain a fit!')
                raise RuntimeError()
            widthGuess = np.std(key) / 2
            # Get all the atoms
            centerGuess = key[list(data).index(max(data))]
            fitValues, fitCovs = fit(fitFunc.gaussian, key, data, p0=[-0.95, centerGuess, widthGuess, 0.95])
            fitErrs = np.sqrt(np.diag(fitCovs))
            a, b, c, d = unc.correlated_values(fitValues, fitCovs)
            fitObject = fitFunc.uncGaussian(xFit, a, b, c, d)
            fitNom = unp.nominal_values(fitObject)
            fitStd = unp.std_devs(fitObject)
            centerIndex = 1
        except RuntimeError:
            warn('Data Fit Failed!')
            fitType=None
    elif fitType == 'Gaussian-Dip':
        try:
            if len(key) < len(signature(fitFunc.gaussian).parameters) - 1:
                print('Not enough data points to constrain a fit!')
                raise RuntimeError()

            widthGuess = np.std(key) / 2
            centerGuess = key[list(data).index(min(data))]
            fitValues, fitCovs = fit(fitFunc.gaussian, key, data, p0=[-0.95, centerGuess, widthGuess, 0.95])
            fitErrs = np.sqrt(np.diag(fitCovs))
            a, b, c, d = unc.correlated_values(fitValues, fitCovs)
            fitObject = fitFunc.uncGaussian(xFit, a, b, c, d)
            fitNom = unp.nominal_values(fitObject)
            fitStd = unp.std_devs(fitObject)
        except RuntimeError:
            warn('Data Fit Failed!')
            fitType = None
    elif fitType == 'Exponential-Decay':
        try:
            if len(key) < len(signature(fitFunc.exponentialDecay).parameters) - 1:
                print('Not enough data points to constrain a fit!')
                raise RuntimeError()

            decayConstantGuess = np.std(key)
            ampGuess = data[0]
            fitValues, fitCovs = fit(fitFunc.exponentialDecay, key, data, p0=[ampGuess, decayConstantGuess])
            fitErrs = np.sqrt(np.diag(fitCovs))
            a, b = unc.correlated_values(fitValues, fitCovs)
            fitObject = fitFunc.uncExponentialDecay(xFit, a,b)
            fitNom = unp.nominal_values(fitObject)
            fitStd = unp.std_devs(fitObject)
        except RuntimeError:
            warn('Data Fit Failed!')
            fitType = None
    elif fitType == 'Exponential-Saturation':
        try:
            if len(key) < len(signature(fitFunc.exponentialSaturation).parameters) - 1:
                print('Not enough data points to constrain a fit!')
                fitType=None
                raise RuntimeError()

            decayConstantGuess = (np.max(key)+np.min(key))/4
            ampGuess = data[-1]
            fitValues, fitCovs = fit(fitFunc.exponentialSaturation, key, data,
                                     p0=[-ampGuess, decayConstantGuess, ampGuess])
            fitErrs = np.sqrt(np.diag(fitCovs))
            a, b, c = unc.correlated_values(fitValues, fitCovs)
            fitObject = fitFunc.uncExponentialSaturation(xFit, a, b, c)
            fitNom = unp.nominal_values(fitObject)
            fitStd = unp.std_devs(fitObject)
        except RuntimeError:
            warn('Data Fit Failed!')
            fitType = None
    elif fitType == 'DecayingSine':
        try:
            if len(key) < len(signature(fitFunc.RabiFlop).parameters) - 1:
                print('Not enough data points to constrain a fit!')
                raise RuntimeError()
            ampGuess = 1
            phiGuess = 0
            OmegaGuess = np.pi / np.max(key)
            # Omega is the Rabi rate
            fitValues, fitCovs = fit(fitFunc.RabiFlop, key, data, p0=[ampGuess, OmegaGuess, phiGuess])
            fitErrs = np.sqrt(np.diag(fitCovs))
            a, b, c = unc.correlated_values(fitValues, fitCovs)
            fitObject = fitFunc.uncRabiFlop(xFit, a, b, c)
            fitNom = unp.nominal_values(fitObject)
            fitStd = unp.std_devs(fitObject)
        except RuntimeError:
            warn('Data Fit Failed!')
            fitType = None
    elif fitType is not None:
        warn("fitType Not Understood! Not Trying to Fit!")
        fitType = None
    # these are None if fit fail
    fitInfo = {'x': xFit, 'nom': fitNom, 'std': fitStd, 'center': centerIndex, 'vals': fitValues, 'errs': fitErrs,
               'cov': fitCovs}
    return fitInfo, fitType


def outputDataToMmaNotebook(fileNumber, survivalData, survivalErrs, captureArray, key):
    """

    :param fileNumber:
    :param survivalData:
    :param survivalErrs:
    :param captureArray:
    :param key:
    :return:
    """
    runNum = fileNumber
    try:
        f = open(dataAddress + '\\run' + str(runNum)+'_key.txt', "w")
        for item in key:
            f.write("%s\n" % item)
        f.close()
        f = open(dataAddress + '\\run' + str(runNum)+'_survival.txt', "w")
        for i in survivalData:
            tmp = ''
            for j in i:
                tmp += "%s," % round_sig(j, 9)
            tmp2 = (tmp.replace("[", "").replace("]", ""))[0:-1]+"\n"
            f.write(tmp2)
        f.close()
        f = open(dataAddress + '\\run' + str(runNum)+'_err.txt', "w")
        for i in survivalErrs:
            tmp = ''
            for j in i:
                tmp += "%s," % round_sig(j, 9)
            tmp2 = (tmp.replace("[", "").replace("]", ""))[0:-1]+"\n"
            f.write(tmp2)
        f.close()
        f = open(dataAddress + '\\run' + str(runNum) + '_loading.txt', "w")
        for i in captureArray:
            tmp = ''
            for j in i:
                tmp += "%s," % round_sig(j, 9)
            tmp2 = (tmp.replace("[", "").replace("]", ""))[0:-1]+"\n"
            f.write(tmp2)
        f.close()
    except:
        print("Error while outputting data to mathematica file.")


def unpackAtomLocations(locs):
    """

    :param locs:
    :return:
    """
    if not (type(locs[0]) == int):
        # already unpacked
        return locs
    # assume atom grid format.
    bottomLeftRow, bottomLeftColumn, spacing, width, height = locs
    locArray = []
    for widthInc in range(width):
        for heightInc in range(height):
            locArray.append([bottomLeftRow + spacing * heightInc, bottomLeftColumn + spacing * widthInc])
    return locArray


def sliceMultidimensionalData(dimSlice, origKey, rawData, varyingDim=None):
    """

    :param dimSlice: e.g. [80, None]
    :param origKey:
    :param rawData:
    :param varyingDim:
    :return:
    """
    key = origKey[:]
    if dimSlice is not None:
        runningKey = key[:]
        runningData = rawData[:]
        for i, dimSpec in enumerate(dimSlice):
            if dimSpec is None:
                varyingDim = i
                continue
            tempKey = []
            tempData = []
            for j, elem in enumerate(transpose(runningKey)[i]):
                if abs(elem - dimSpec) < 1e-6:
                    tempKey.append(runningKey[j])
                    tempData.append(runningData[j])
            runningKey = tempKey[:]
            runningData = tempData[:]
        key = runningKey[:]
        rawData = runningData[:]
    otherDimValues = None
    if varyingDim is not None:
        otherDimValues = []
        for keyVal in key:
            otherDimValues.append('')
            for i, dimVal in enumerate(keyVal):
                if not i == varyingDim:
                    otherDimValues[-1] += str(dimVal) + ","
    if dimSlice is not None:
        key = arr(transpose(key)[varyingDim])
    if varyingDim is None and len(arr(key).shape) > 1:
        key = arr(transpose(key)[0])
    return arr(key), arr(rawData), otherDimValues, varyingDim


def applyDataRange(dataRange, groupedDataRaw, key):
    if dataRange is not None:
        groupedData, newKey = [[] for _ in range(2)]
        for count, variation in enumerate(groupedDataRaw):
            if count in dataRange:
                groupedData.append(variation)
                newKey.append(key[count])
        groupedData = arr(groupedData)
        key = arr(newKey)
    else:
        groupedData = groupedDataRaw
    return key, groupedData


def getNetLossStats(netLoss, reps):
    lossAverages = np.array([])
    lossErrors = np.array([])
    for variationInc in range(0, int(len(netLoss) / reps)):
        lossList = np.array([])
        # pull together the data for only this variation
        for repetitionInc in range(0, reps):
            if netLoss[variationInc * reps + repetitionInc] != -1:
                lossList = np.append(lossList, netLoss[variationInc * reps + repetitionInc])
        if lossList.size == 0:
            # catch the case where there's no relevant data, typically if laser becomes unlocked.
            lossErrors = np.append(lossErrors, [0])
            lossAverages = np.append(lossAverages, [0])
        else:
            # normal case, compute statistics
            lossErrors = np.append(lossErrors, np.std(lossList)/np.sqrt(lossList.size))
            lossAverages = np.append(lossAverages, np.average(lossList))
    return lossAverages, lossErrors


def getNetLoss(pic1Atoms, pic2Atoms):
    """
    Calculates the net loss fraction for every experiment. Assumes 2 pics per experiment.
    Useful for experiments where atoms move around, e.g. rearranging.
    """
    netLoss = []
    for inc, (atoms1, atoms2) in enumerate(zip(transpose(pic1Atoms), transpose(pic2Atoms))):
        loadNum, finNum = [0.0 for _ in range(2)]

        for atom1, atom2 in zip(atoms1, atoms2):
            if atom1:
                loadNum += 1.0
            if atom2:
                finNum += 1.0
        if loadNum == 0:
            netLoss.append(0)
        else:
            netLoss.append(1 - float(finNum) / loadNum)
    return netLoss


def getAtomInPictureStatistics(atomsInPicData, reps):
    """
    assumes atomsInPicData is a 2D array. atomsInPicData[0,:] refers to all of the atom events for a single location,
    atomsInPicData[1,:] refers to all the events for the second, etc.
    """
    stats = []
    for singleLocData in atomsInPicData:
        singleLocData = arr(singleLocData)
        variationData = singleLocData.reshape([int(len(singleLocData)/reps), reps])
        avgs = [np.average(singleVarData) for singleVarData in variationData]
        errs = [np.std(singleVarData)/np.sqrt(len(singleVarData)) for singleVarData in variationData]
        stats.append({'avg': avgs, 'err': errs})
    return stats


<<<<<<< HEAD
def getEnsembleHits(atomsList, hitCondition=None, requireConsecutive=False):
=======
def getEnsembleHits(atomsList, hitCondition=None, connected=False):
>>>>>>> 2aa20b43f0a60d509d2c74e69d9e3d993d4c46e2
    """
    This function determines whether an ensemble of atoms was hit in a given picture. Give it whichever
    picture data you need.

    NEW: this function is now involved in post-selection work

    atomsList should be a 2 dimensional array. 1 Dim for each atom location, one for each picture.
    """
    if hitCondition is None:
        hitCondition = np.ones(atomsList.shape[0])
    ensembleHits = []
<<<<<<< HEAD
    if type(hitCondition) is int:
        # condition is, e.g, 5 out of 6 of the ref pic.
        for inc, atoms in enumerate(transpose(atomsList)):
            matches = 0
            consecutive = True
            for atom in atoms:
                if atom:
                    matches += 1
                # else there's no atom. 3 possibilities: before string of atoms, after string, or in middle.
                # if in middle, consecutive requirement is not met.
                elif 0 < matches < hitCondition:
                    consecutive = False
            if requireConsecutive:
                ensembleHits.append((matches == hitCondition) and consecutive)
            else:
                ensembleHits.append(matches == hitCondition)
    else:
        for inc, atoms in enumerate(transpose(atomsList)):
            ensembleHits.append(True)
            for atom, needAtom in zip(atoms, hitCondition):
                if not atom and needAtom:
                    ensembleHits[inc] = False
                if atom and not needAtom:
                    ensembleHits[inc] = False
=======
    for inc, atoms in enumerate(transpose(atomsList)):
        if hitCondition is None:
            ensembleHits.append(True)
            for atom in atoms:
                if not atom:
                    ensembleHits[inc] = False
        else:
            continuous = True
            atomNum = 0
            for atom in atoms:
                if atom:
                    atomNum += 1
                elif connected and (0 < atomNum < hitCondition):
                    continuous = False
            if connected:
                ensembleHits.append(atomNum == hitCondition and continuous)
            else:
                ensembleHits.append(atomNum == hitCondition)
>>>>>>> 2aa20b43f0a60d509d2c74e69d9e3d993d4c46e2
    return ensembleHits


def getEnsembleStatistics(ensembleData, reps):
    """
    EnsembleData is a list of "hits" of the deisgnated ensemble of atoms in a given picture, for different variations.
    This function calculates some statistics on that list.
    """
    ensembleAverages = np.array([])
    ensembleErrors = np.array([])
    for variationInc in range(0, int(len(ensembleData) / reps)):
        ensembleList = np.array([])
        # pull together the data for only this variation
        for repetitionInc in range(0, reps):
            if ensembleData[variationInc * reps + repetitionInc] != -1:
                ensembleList = np.append(ensembleList, ensembleData[variationInc * reps + repetitionInc])
        if ensembleList.size == 0:
            # catch the case where there's no relevant data, typically if laser becomes unlocked.
            ensembleErrors = np.append(ensembleErrors, [0])
            ensembleAverages = np.append(ensembleAverages, [0])
        else:
            # normal case, compute statistics
            ensembleErrors = np.append(ensembleErrors, np.std(ensembleList)/np.sqrt(ensembleList.size))
            ensembleAverages = np.append(ensembleAverages, np.average(ensembleList))
    ensembleStats = {'avg': ensembleAverages, 'err': ensembleErrors}
    return ensembleStats
