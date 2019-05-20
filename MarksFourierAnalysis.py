__version__ = "1.0"

import scipy.fftpack as FT
import numpy as np


def fft(field, xpts, normalize=True):
    """
    :param field: the field amplitudes of the field to be transformed
    :param xpts: the positions that the field array above have been sampled at. Note that this function assumes
        evenly spaced points.
    :param normalize: This normalizes the fft to the continuous version, multiplying the spectrum by the spacing.
        If using my Ifft function, this should match the de-normalize option there.

    :return dic with 'Freq' and 'Field' Arguments.

    This is a wrapper for the fourier-transform function from Scipy.
    Takes the fourier transform of a 1-dimensional field, i.e. g{x} -> g_f{f_x}
x    returns the transform amplitudes and the frequencies in two objects. By default, it normalizes the FFT,
    taking into account the spacing of the points your field is defined over. This makes the FFT match the
    result you'd calculate from a continuous fourier transform, up to a factor of (2pi)^(?) depending on your
    FT convention.
    """

    assert (len(xpts) > 1)
    assert (len(xpts) == len(field))

    # assumes evenly spaced.
    spacing = (max(xpts) - min(xpts)) / (len(xpts) - 1)
    freqs = FT.fftshift(FT.fftfreq(len(xpts), spacing))
    fieldFFT = FT.fftshift(FT.fft(FT.ifftshift(field)))
    if normalize:
        fieldFFT *= spacing
    return {'Freq': freqs, 'Field': fieldFFT}


def ifft(fieldFFT, xpts, denormalize=True):
    """
    :param fieldFFT: the field frequency amplitudes
    :param xpts:
    :param denormalize:

    A wrapper for the inverse Fourier-transform function from Scipy
    Takes the inverse fourier transform of a 1-dimensional field, i.e. g_f{f_x} -> g{x}
    returns the amplitudes (presumably you already have the x points for those amplitudes.)

    By default, it re-compensates for the normalization done by default in my FFT wrapper,
    discussed above.
    """
    assert (len(fieldFFT) > 1)

    spacing = (max(xpts) - min(xpts)) / (len(xpts) - 1)
    if denormalize:
        fieldFFT /= spacing
    field = FT.fftshift(FT.ifft(FT.ifftshift(fieldFFT)))
    return {'Field': field, 'xpts': xpts}


def propagate(field, fieldPos, z_fin, wavelength, n=1):
    """
    @param field: a field in real space to be propagated.
    @param fieldPos: the positions that the field array above have been sampled at. Note that this function assumes
        evenly spaced points.
    @param z_fin: the distance to be propagated.
    @param wavelength: the wavelength of the light being propagated.
    @param n=1: The index of refraction of the medium being propagated through. The default is 1 for vacuum.
        Note that air is something like 1.002 =/= 1
    Propagate a field a given distance.
    """
    k = 2 * np.pi * n / wavelength
    # Important assumption (even spacing) in the following line.
    spacing = (max(fieldPos) - min(fieldPos))/(len(fieldPos) - 1)
    if not (spacing < wavelength / 2):
        print('WARNING: spacing is not sufficient to see evanescant frequencies. Spacing is ' + str(spacing))
    fftData = fft(field, fieldPos)
    transferFunc = np.exp(1j * k * z_fin
                          * np.sqrt((1 - ((wavelength / n) * fftData['Freq'])**2).astype(complex)))
    propFieldFFT = fftData['Field'] * transferFunc
    # need to make sure the result of ifft is interpreted as complex.
    field = ifft(propFieldFFT, fieldPos)
    return field


def diagnosticProp(field, fieldPos, z_fin, wavelength, n=1):
    """
    ### ########
    FOR TESTING.
    ### ########
    @param field: a field in real space to be propagated.
    @param fieldPos: the positions that the field array above have been sampled at. Note that this function assumes
        evenly spaced points.
    @param z_fin: the distance to be propagated.
    @param wavelength: the wavelength of the light being propagated.
    @param n=1: The index of refraction of the medium being propagated through. The default is 1 for vacuum.
        Note that air is something like 1.002 =/= 1
    Propagate a field a given distance. This function gives some extra info about the propagation in order to help
    debug what's going on during the propagation
    """
    k = 2 * np.pi * n / wavelength
    # Important assumption in the following line.
    spacing = (max(fieldPos) - min(fieldPos))/(len(fieldPos) - 1)
    if not (spacing < wavelength / 2):
        print('WARNING: spacing is not sufficient to see evanescant frequencies. Spacing is ' + str(spacing))
    fftData = fft(field, fieldPos)
    transferFunc = np.exp(1j * k * z_fin
                          * np.sqrt((1 - ((wavelength / n) * fftData['Freq'])**2).astype(complex)))
    propFieldFFT = fftData['Field'] * transferFunc
    field = ifft(propFieldFFT, fieldPos)
    return field, fftData, transferFunc, propFieldFFT


def fft2D(field, xpts, ypts, normalize=True):
    """
    :@param field: the field amplitudes of the field to be transformed
    :@param xpts: the positions that the field array above have been sampled at. Note that this function assumes
        evenly spaced points.
    :@param ypts: the positions that the field array above have been sampled at. Note that this function assumes
        evenly spaced points.
    Takes the fourier transform of a 2-dimensional field, i.e. g{x} -> g_f{f_x}
    returns the transform amplitudes and the frequencies in two objects.
    """
    assert(len(xpts) > 1)
    assert(len(ypts) > 1)
    assert(len(xpts) == len(field))
    assert(len(ypts) == len(field[0]))
    xspacing = (max(xpts) - min(xpts))/(len(xpts) - 1)    
    yspacing = (max(ypts) - min(ypts))/(len(ypts) - 1)    
    xFreqs = FT.fftshift(FT.fftfreq(len(xpts), xspacing))
    yFreqs = FT.fftshift(FT.fftfreq(len(ypts), yspacing))
    fieldFFT = FT.fftshift(FT.fft2(FT.ifftshift(field)))
    if normalize:
        fieldFFT *= (xspacing * yspacing)
    return {'Field': fieldFFT, 'xFreq': xFreqs, 'yFreq': yFreqs}


def ifft2D(field2DFFT, xpts, ypts, denormalize=True):
    """
    @param field2DFFT: the 2D field frequency amplitudes 
    
    Takes the inverse fourier transform of a 2-dimensional field;
    returns the amplitudes (presumably you already have the x points for those amplitudes.)
    """
    assert(len(field2DFFT) > 1)
    field = FT.fftshift(FT.ifft2(FT.ifftshift(field2DFFT)))
    if denormalize:
        xspacing = (max(xpts) - min(xpts))/(len(xpts) - 1)    
        yspacing = (max(ypts) - min(ypts))/(len(ypts) - 1)    
        field /= xspacing * yspacing
    return {'Field': field, 'xpts': xpts, 'ypts': ypts }



def propagate2D(field, fieldPosX, fieldPosY, z_fin, wavelength, n=1):
    """
    Propagate a field a given distance.
    """
    k = 2 * np.pi * n / wavelength
    # Important assumption in the following line.
    xSpacing = (max(fieldPosX) - min(fieldPosX))/(len(fieldPosX) - 1)
    if not (xSpacing <= wavelength / 2):
        print('WARNING: spacing is not sufficient to see evanescant frequencies in X direction.'
              ' Spacing is ' + str(xSpacing))
    ySpacing = (max(fieldPosY) - min(fieldPosY))/(len(fieldPosY) - 1)
    if not (ySpacing <= wavelength / 2):
        print('WARNING: spacing is not sufficient to see evanescant frequencies in Y direction.'
              ' Spacing is ' + str(ySpacing))
    theFft = fft2D(field, fieldPosX, fieldPosY)
    #fieldFFT, xFreqs, yFreqs = 
    transferFunc = np.zeros((len(theFft['xFreq']), len(theFft['yFreq']))).astype(complex)
    for x in range(len(theFft['xFreq'])):
        for y in range(len(theFft['yFreq'])):
            transferFunc[x][y] = np.exp(1j * k * z_fin * 
                                        np.sqrt((1 - ((wavelength / n) * theFft['xFreq'][x])**2 
                                                 - ((wavelength / n) * theFft['yFreq'][y])**2).astype(complex)))
    propFieldFFT = theFft['Field'] * transferFunc
    field = np.array([], dtype=complex)
    propField = ifft2D(propFieldFFT, fieldPosX, fieldPosY)
    return propField

def splitStepPropagate(field, xpts, n, z, wavelength, returnFinal=False, informProgress=True):
    """
    this function propagates the input field through a 3D material with varying index n.
    It alternates between propagating through the material (in fourier space) and
    refracting through the material (in real space.)
    :param field: is the input field to be propagated. It should be the field directly before
            material in question.
    :param xpts: the x-coordinates for the field values.
    :param n: the index of the 3D material, discretized to the desired precision of this method.
            I.e. if n contains 100 different values, this method will complete the above
            calculation 100 times. This is also expected to be defined for every x coordinate,
            and so should be of size (z_iterations, len(xpts)).
    :param z: the length of the material being propagated through, a constant is expected.
            This function could be easily modified in order to accomodate variable Delta z
            increments, which might be useful for numerically challenging calculations.
    :param wavelength: the (vacuum) wavelength of the light being propagated.
    :param returnFinal: if this is true then only the field at the end of the material is returned.
            the method does not attempt to keep track of all fields throughout the propagation,
            so this might be more efficient.
    :param informProgress: if this and returnFinal are true then while incrementing through the
            material, the function will inform the caller (via print) of the progress.
    :return this function by default returns all the fields calculated (in real space)
             throughout this function's operation. If n and or x is dense, then this will be a
             large data object.
    """
    assert type(n) == np.ndarray
    assert type(n[0]) == np.ndarray
    dz = z / len(n)
    k = 2 * np.pi / wavelength
    # (working field)
    wField = field
    if not returnFinal:
        # this includes the initial field! The returned data does not include the
        # initial point though.
        allData = np.zeros((len(n) + 1, len(xpts))).astype(complex)
        allData[0] = field
        posNumber = 0
        for n_z in n:
            posNumber += 1
            if informProgress:
                print('Working on #' + str(posNumber) + '...')
            # see math above for neat form of what this is doing.
            newField = allData[posNumber - 1] * np.exp(1j * k * n_z * dz)
            allData[posNumber] = propagate(newField, xpts, dz, wavelength, n_z)['Field']
        # cut off the initial field, which the user should have since it was an input to this function.
        allData = allData[1:]
    else:
        for n_z in n:
            allData = field
            for n_z in n:
                allData = propagate(allData * np.exp(1j * k * n_z * dz),
                                    xpts, dz, wavelength, n_z)['Field']
    return allData


