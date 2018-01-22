__version__ = "1.0"

import numpy as np
from numpy import array as arr
from matplotlib.cm import get_cmap
from pandas import DataFrame


def prefix(num):
    """
    Convert number to nearest numbers with SI prefixes.
    :param num: the number to convert
    """
    # determine which range it lies in, r1/r2 means reduction 1 or reduction 2
    divisors = [1e-24 * pow(10, 3 * x) for x in range(17)]
    prefixes = list(reversed(['Yotta (Y)', 'Zetta (Z)', 'Exa (E)', 'Peta (P)', 'Tera (T)', 'Giga (G)', 'Mega (M)',
                              'Kilo (K)', '', 'Milli (m)', 'Micro ($\mu$)', 'Nano (n)', 'Pico (p)', 'Femto (f)',
                              'Atto (a)', 'Zepto (z)', 'Yocto (y)']))
    exp = np.floor(np.log10(np.abs(num)))
    if exp < 0:
        exp -= 3
    expIndex = int(exp / 3) + 8
    expIndex = 0 if expIndex < 0 else expIndex
    expIndex = len(prefixes)-1 if expIndex >= len(prefixes) else expIndex
    r1 = prefixes[expIndex]
    num1 = num / divisors[expIndex]
    if expIndex != len(prefixes):
        r2 = prefixes[expIndex + 1]
        num2 = num / divisors[expIndex + 1]
    else:
        num2 = None
    retStr = str(num1) + ' ' + r1
    if num2 is not None:
        retStr += '\nor\n' + str(num2) + ' ' + r2
    return retStr


def what(obj, callingLocals=locals()):
    """
    quick function to print name of input and value.
    If not for the default-Valued callingLocals, the function would always
    get the name as "obj", which is not what I want.

    :param obj: the object to print info for
    :param callingLocals: don't use, always should be locals().
    """
    name = "name not found"
    for k, v in list(callingLocals.items()):
        if v is obj:
            name = k
    if type(obj) == float:
        print(name, "=", "{:,}".format(obj))
    else:
        print(name, "=", obj)


def transpose(l):
    """
    Transpose a list.
    :param l: the list to be transposed
    :return: the tranposed list
    """
    return list(map(list, zip(*l)))


def getStats(data, printStats=False):
    """
    get some basic statistics about the input data, in the form of a pandas dataframe.
    :param data: the data to analyze
    :param printStats: an option to print the results
    :return: the dataframe containing the statistics
    """
    data = list(data)
    d = DataFrame()
    d['Avg'] = [np.mean(data)]
    d['len'] = [len(data)]
    d['min'] = [min(data)]
    d['max'] = [max(data)]
    d['std'] = [np.std(data)]
    d = d.transpose()
    d.columns = ['Stats']
    d = d.transpose()
    if printStats:
        print(d)
    return d


def getColors(num, rgb=False):
    """
    Get an array of colors, typically to use for plotting.

    :param rgb: an option to return the colors as an rgb tuple instead of a hex.
    :param num: number of colors to get
    :return: the array of colors, hex or rgb (see above)
    """
    cmapRGB = get_cmap('nipy_spectral', num)
    c = [cmapRGB(i)[:-1] for i in range(num)][1:]
    if rgb:
        return c
    # the negative of the first color
    c2 = [tuple(arr((1, 1, 1)) - arr(color)) for color in c]
    c = ['#%02x%02x%02x' % tuple(int(255 * color[i]) for i in range(len(color))) for color in c]
    c2 = ['#%02x%02x%02x' % tuple(int(255 * color[i]) for i in range(len(color))) for color in c2]
    return c, c2


def round_sig(x, sig=3):
    """
    round a float to some number of significant digits
    :param x: the numebr to round
    :param sig: the number of significant digits to use in the rounding
    :return the rounded number, as a float.
    """
    if np.isnan(x):
        x = 0
    try:
        return round(x, sig-int(np.floor(np.log10(abs(x)+2*np.finfo(float).eps)))-1)
    except ValueError:
        print(abs(x))
