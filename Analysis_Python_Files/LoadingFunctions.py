# ##############
# ### Data-Loading Functions


from os import linesep
from pandas import read_csv
import csv
import numpy as np
from numpy import array as arr
from . import ExpFile as exp

def read_Tektronics_DPO_3034(fn):
    # for our nicer Oscilloscope
    with open(fn) as f:
        ls = f.readlines()
        times = []
        volts = []
        for lineNum, l in enumerate(ls[:-1]):
            if lineNum < 15:
                continue
            l_sp = l.split(',')
            times.append(float(l_sp[0]))
            volts.append(float(l_sp[1]))
    return times, volts


def load_Siglet_SSA_3021X(fn):
    """
    for our lab's siglent spectrum analyzer. 
    Returns (freqs, powers)
    """
    with open(fn) as f:
        cf = csv.reader(f)
        freqs = []
        pows = []
        flag = False
        for i, row in enumerate(cf):
            # important: first 30 lines include information about scan settings, data is afterwards.
            if row[0] == 'Trace Data':
                flag = True
                continue
            if not flag:
                continue
            freqs.append(float(row[0]))
            pows.append(float(row[1]))
    return freqs, pows


def loadDataRay(fileID):
    """
    :param num: either the filename or a number, in which case assumes file name is [dataAddress + "dataRay_" + str(fileID) + ".wct"]
    :return: image matrix
    """
    if type(fileID) == int:
        fileName = dataAddress + "dataRay_" + str(fileID) + ".wct"
    else:
        fileName = fileID
    file = read_csv(fileName, header=None, skiprows=[0, 1, 2, 3, 4])
    data = file.as_matrix()
    for i, row in enumerate(data):
        data[i][-1] = float(row[-1][:-2])
        for j, elem in enumerate(data[i]):
            data[i][j] = float(elem)
    return data.astype(float)


def loadCompoundBasler(fid, cameraName='ace', loud=False):
    if type(fid) == type('string'):
        path = fid
    else:
        if cameraName == 'ace':
            path = exp.dataAddress + "AceData_" + str(fid) + ".txt"
        elif cameraName == 'scout':
            path = exp.dataAddress + "ScoutData" + str(fid) + ".txt"
        else:
            raise ValueError('cameraName has a bad value for a Basler camera.')
    with open(path) as file:
        original = file.read()
        pics = original.split(";")
        if loud:
            print('Number of Pics:', len(pics))
        dummy = linesep.join([s for s in pics[0].splitlines() if s])
        dummy2 = dummy.split('\n')
        dummy2[0] = dummy2[0].replace(' \r', '')
        data = np.zeros((len(pics), len(dummy2), len(arr(dummy2[0].split(' ')))))
        picInc = 0
        for pic in pics:
            if loud:
                if picInc % 100 == 0:
                    print('')
                if picInc% 1000 == 0:
                    print('')
                print('.',end='')
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

def load_Anritsu_MS2721B(file):
    with open(file) as fid:
        lines = fid.readlines()
    powers = []
    freqs = []
    traceData = lines[315:]
    for line in traceData:
        if line == '\n':
            break
        powStr, freqStr = line.split(',')
        _, power = powStr.split('=')
        _,freq, _ = freqStr.split(' ')
        powers.append(float(power))
        freqs.append(float(freq))
    return freqs, powers