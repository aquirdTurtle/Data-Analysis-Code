from matplotlib.colors import ListedColormap
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import IPython.display as disp

from .fitters.Gaussian import bump3, bump
from .fitters import LargeBeamMotExpansion 

from . import Miscellaneous as misc
from . import MatplotlibPlotters as mp
from . import ExpFile as exp
from . import AnalysisHelpers as ah
from . import TransferAnalysis as ta
from . import PictureWindow as pw

import warnings

viridis = cm.get_cmap('viridis', 256)
dark_viridis = []
bl = 0.15
for i in range(256):
    dark_viridis.append(list(viridis(i)))
    dark_viridis[-1][0] = dark_viridis[-1][0] *(bl+(1-bl)*i/255)
    dark_viridis[-1][1] = dark_viridis[-1][1] *(bl+(1-bl)*i/255)
    dark_viridis[-1][2] = dark_viridis[-1][2] *(bl+(1-bl)*i/255)
dark_viridis_cmap = ListedColormap(dark_viridis)

def rmHighCountPics(pics, threshold):
    deleteList = []
    discardThreshold = 7000
    for i, p in enumerate(pics):
        if max(p.flatten()) > discardThreshold:
            deleteList.append(i)
    if len(deleteList) != 0:
        print('Not using suspicious data:', deleteList)
    for index in reversed(deleteList):
        pics = np.delete(pics, index, 0)
    return pics


def photonCounting(pics, threshold):
    pcPics = np.copy(pics)
    # digitize
    pcPics[pcPics<=threshold] = 0
    pcPics[pcPics>threshold] = 1
    # sum
    pcPicTotal = np.sum(pcPics,axis=0)
    return np.array(pcPicTotal), pcPics

def getBgImgs(fid, incr=2,startPic=1):
    if type(fid) == int:
        with exp.ExpFile(fid) as file:
            pics = file.get_pics()
    else:
        pics = fid
    pics2 = pics[startPic::incr]
    pics2 = rmHighCountPics(pics2, 7000)
    avgBg = np.mean(pics2,0)
    avgPcBg = photonCounting(pics2, 120)[0] / len(pics2)
    return avgBg, avgPcBg


def freespaceImageAnalysis( fids, guesses = None, fit=True, bgInput=None, bgPcInput=None, shapes=[None], zeroCorrection=0, zeroCorrectionPC=0,
                            keys=None, fitModule=bump, extraPicDictionaries=None, newAnnotation=False, onlyThisPic=None, pltVSize=5,              
                            plotSigmas=False, plotCounts=False, manualColorRange=None, calcTemperature=False, clearOutput=True, 
                            dataRange=None, guessTemp=10e-6, trackFitCenter=False, picsPerRep=1, startPic=0, binningParams=None, 
                            win=pw.PictureWindow(), transferAnalysisOpts=None, tferBinningParams=None, tferWin= pw.PictureWindow(),
                            extraTferAnalysisArgs={}, emGainSetting=300, lastConditionIsBackGround=True):
    if type(fids) == int:
        fids = [fids]
    if keys is None:
        keys = [None for _ in fids]
    sortedStackedPics = {}
    initThresholds = [None]
    picsForBg = []
    for filenum, fid in enumerate(fids):
        if transferAnalysisOpts is not None:
            res = ta.stage1TransferAnalysis( fid, transferAnalysisOpts, **extraTferAnalysisArgs )
            (initAtoms, tferAtoms, initAtomsPs, tferAtomsPs, key, keyName, initPicCounts, tferPicCounts, repetitions, initThresholds,
             avgPics, tferThresholds, initAtomImages, tferAtomImages, basicInfoStr, ensembleHits, groupedPostSelectedPics) = res
            # assumes that you only want to look at the first condition. 
            for varPics in groupedPostSelectedPics:
                picsForBg += varPics[-1 if lastConditionIsBackGround else 0]
            allFSIPics = [ varpics[0][startPic::picsPerRep] for varpics in groupedPostSelectedPics]
            fig, axs = plt.subplots(1,2)
            mp.makeAvgPlts(axs[0],axs[1], avgPics, transferAnalysisOpts, ['r','g','b']) 
            allFSIPics = [win.window( np.array(pics) ) for pics in allFSIPics]
            allFSIPics = ah.softwareBinning( binningParams, allFSIPics )
        elif type(fid) == int:
            ### For looking at either PGC imgs or FSI imgs 
            with exp.ExpFile(fid) as file:
                picsForBg += list(file.get_pics())
                allFSIPics = file.get_pics()[startPic::picsPerRep]
                _, key = file.get_key()
                if len(np.array(key).shape) == 2:
                    key = key[:,0]
                file.get_basic_info()
            allFSIPics = win.window( allFSIPics )
            allFSIPics = ah.softwareBinning( binningParams, allFSIPics )
            allFSIPics = np.reshape( allFSIPics, (len(key), int(allFSIPics.shape[0]/len(key)), allFSIPics.shape[1], allFSIPics.shape[2]) )
        else:
            ### Assumes given pics have the same start pic and increment (picsPerRep).
            picsForBg += fid
            allFSIPics = fid[startPic::picsPerRep]
            print("Assuming input is list of all pics, then splices to get FSI pics. Old code assumed the given were FSI pics.")
            allFSIPics = win.window( allFSIPics )
            allFSIPics = ah.softwareBinning( binningParams, allFSIPics )
            allFSIPics = np.reshape( allFSIPics, (len(key), int(allFSIPics.shape[0]/len(key)), allFSIPics.shape[1], allFSIPics.shape[2]) )
        if keys[filenum] is not None:
            key = keys[filenum]
        for i, keyV in enumerate(key):
            keyV = misc.round_sig_str(keyV)
            sortedStackedPics[keyV] = np.append(sortedStackedPics[keyV], allFSIPics[i],axis=0) if (keyV in sortedStackedPics) else allFSIPics[i]         

    if lastConditionIsBackGround:
        bgInput, pcBgInput = getBgImgs(picsForBg, startPic = startPic, picsPerRep = picsPerRep)
    elif bgInput == 'lastPic':
        bgInput, pcBgInput = getBgImgs(picsForBg, startPic = picsPerRep-1, picsPerRep = picsPerRep)
    if bgInput is not None: # was broken and not working if not given bg
        bgInput = win.window(bgInput)
        bgInput = ah.softwareBinning(binningParams, bgInput)
    if bgPcInput is not None:
        bgPcInput = win.window(bgPcInput)
        bgPcInput = ah.softwareBinning(binningParams, bgPcInput)   
    
    if extraPicDictionaries is not None:
        if type(extraPicDictionaries) == dict:
            extraPicDictionaries = [extraPicDictionaries]
        for dictionary in extraPicDictionaries:
            for keyV, pics in dictionary.items():
                sortedStackedPics[keyV] = (np.append(sortedStackedPics[keyV], pics,axis=0) if keyV in sortedStackedPics else pics)    
    sortedStackedKeyFl = [float(keyStr) for keyStr in sortedStackedPics.keys()]
    sortedKey = list(sorted(sortedStackedKeyFl))
    sortedKey, sortedStackedPics = ah.applyDataRange(dataRange, sortedStackedPics, sortedKey)
    numVars = len(sortedStackedPics.items())
    if len(np.array(shapes).shape) == 1:
        shapes = [shapes for _ in range(numVars)]       
    if guesses is None:
        guesses = [[None for _ in range(4)] for _ in range(numVars)]
    if len(np.array(bgInput).shape) == 2 or bgInput == None:
        bgInput = [bgInput for _ in range(numVars)]
    if len(np.array(bgPcInput).shape) == 2 or bgPcInput == None:
        bgPcInput = [bgPcInput for _ in range(numVars)]
    
    datalen, avgFitSigmas, images, hFitParams, hFitCovs, vFitParams, vFitCovs = [{} for _ in range(7)]
    titles = ['Bare', 'Photon-Count', 'Bare-mbg', 'Photon-Count-mbg']
    assert(len(sortedKey)>0)
    for vari, keyV in enumerate(sortedKey):
        keyV=misc.round_sig_str(keyV)
        varPics = sortedStackedPics[keyV]
        # 0 is init atom pics for post-selection on atom number... if we wanted to.
        expansionPics = rmHighCountPics(varPics,7000)
        datalen[keyV] = len(expansionPics)
        expPhotonCountImage = photonCounting(expansionPics, 120)[0] / len(expansionPics)
        bgPhotonCountImage = np.zeros(expansionPics[0].shape) if bgPcInput[vari] is None else bgPcInput[vari]
        expAvg = np.mean(expansionPics, 0)
        bgAvg = np.zeros(expansionPics[0].shape) if (bgInput[vari] is None or len(bgInput[vari]) == 1) else bgInput[vari]
        
        if bgPhotonCountImage is None:
            print('no bg photon', expAvg.shape)
            bgPhotonCount = np.zeros(photonCountImage.shape)
        avg_mbg = expAvg - bgAvg
        avg_mbgpc = expPhotonCountImage - bgPhotonCountImage
        images[keyV] = [expAvg, expPhotonCountImage, avg_mbg, avg_mbgpc]
        hFitParams[keyV], hFitCovs[keyV], vFitParams[keyV], vFitCovs[keyV] = [[] for _ in range(4)]
        for im, guess in zip(images[keyV], guesses[vari]):
            hAvg, vAvg = ah.collapseImage(im)
            if fit:
                guess = fitModule.guess(np.arange(len(hAvg)), hAvg) if guess is None else guess
                try:
                    hparams, hcov = opt.curve_fit(fitModule.f, np.arange(len(hAvg)), hAvg, p0=guess)
                    vparams, vcov = opt.curve_fit(fitModule.f, np.arange(len(vAvg)), vAvg, p0=guess)
                except RuntimeError:
                    warnings.warn('1D Fit Failed!')
                    vparams = hparams = guess
                    vcov = hcov = np.zeros((len(guess), len(guess)))
                hFitParams[keyV].append(hparams)
                hFitCovs[keyV].append(hcov)
                vFitParams[keyV].append(vparams)
                vFitCovs[keyV].append(vcov)
    # conversion from the num of pixels on the camera to microns at the focus of the tweezers
    cf = 16e-6/64
    mins, maxes = [[], []]
    imgs_ = np.array(list(images.values()))
    for imgInc in range(4):
        if manualColorRange is None:
            mins.append(min(imgs_[:,imgInc].flatten()))
            maxes.append(max(imgs_[:,imgInc].flatten()))
        else:
            mins.append(manualColorRange[0])
            maxes.append(manualColorRange[1])
    numVariations = len(images)
    if onlyThisPic is None:
        fig, axs = plt.subplots(numVariations, 4, figsize=(20, pltVSize*numVariations))
        if numVariations == 1:
            axs = np.array([axs])
        bgFig, bgAxs = plt.subplots(1, 2, figsize=(20, pltVSize))
    else:
        numRows = int(np.ceil((numVariations+2)/4))
        fig, axs = plt.subplots(numRows, 4 if numVariations>1 else 3, figsize=(20, pltVSize*numRows))
        bgAxs = [axs.flatten()[-1], axs.flatten()[-2]]
        bgFig = fig
    
    keyPlt = np.zeros(len(images))
    (totalSignal, hfitCenter, hFitCenterErrs, hSigmas, hSigmaErrs, h_amp, hAmpErrs, vfitCenter, vFitCenterErrs, vSigmas, vSigmaErrs, v_amp, 
     vAmpErrs) = [np.zeros((len(images), 4)) for _ in range(13)]
    for vari, ((keyV,ims), hParamSet, hCov_set, vParamSet, vCov_set) in enumerate(zip(images.items(), hFitParams.values(), hFitCovs.values(),
                                                                                      vFitParams.values(), vFitCovs.values())):
        for which in range(4):
            if onlyThisPic is None:
                (im, ax, title, min_, max_, hparams, hcovs, vparams, 
                 vcovs) = [obj[which] for obj in (ims, axs[vari], titles, mins, maxes, hParamSet, hCov_set, vParamSet, vCov_set)] 
            else:
                which = onlyThisPic
                ax = axs.flatten()[vari]
                (im, title, min_, max_, hparams, hcovs, vparams, 
                 vcovs) = [obj[which] for obj in (ims, titles, mins, maxes, hParamSet, hCov_set, vParamSet, vCov_set)] 
            v_errs, h_errs = np.sqrt(np.diag(vcovs)), np.sqrt(np.diag(hcovs))
            h_amp[vari][which], hfitCenter[vari][which], hSigmas[vari][which] = hparams[0], hparams[1], hparams[2]*cf*1e6
            hAmpErrs[vari][which], hFitCenterErrs[vari][which], hSigmaErrs[vari][which] = h_errs[0], h_errs[1], h_errs[2]*cf*1e6
            v_amp[vari][which], vfitCenter[vari][which], vSigmas[vari][which] = vparams[0], vparams[1], vparams[2]*cf*1e6
            vAmpErrs[vari][which], vFitCenterErrs[vari][which], vSigmaErrs[vari][which] = v_errs[0], v_errs[1], v_errs[2]*cf*1e6
            totalSignal[vari][which] = np.sum(im.flatten())
            keyPlt[vari] = keyV
            res = mp.fancyImshow(fig, ax, im, imageArgs={'cmap':dark_viridis_cmap, 'vmin':min_, 'vmax':max_}, 
                                 hFitParams=hparams, vFitParams=vparams, fitModule=fitModule, flipVAx = True)
            ax.set_title(keyV + ': ' + str(datalen[keyV]) + ';\n' + title + ': ' + misc.errString(hSigmas[vari][which],hSigmaErrs[vari][which]) 
                + r'$\mu m$ sigma, ' + misc.round_sig_str(totalSignal[vari][which],5), fontsize=12)            
            if onlyThisPic is not None:
                break
    ### Plotting background and photon counted background
    mp.fancyImshow(bgFig, bgAxs[0], bgAvg, imageArgs={'cmap':dark_viridis_cmap},flipVAx = True) 
    bgAxs[0].set_title('Background image (' + str(len(picsForBg)/picsPerRep) + ')')
    mp.fancyImshow(bgFig, bgAxs[1], bgPhotonCountImage, imageArgs={'cmap':dark_viridis_cmap},flipVAx = True) 
    bgAxs[1].set_title('Photon counted background image (' + str(len(picsForBg)/picsPerRep) + ')')
    fig.subplots_adjust(left=0,right=1,bottom=0.1, hspace=0.2, **({'top': 0.7, 'wspace': 0.4} if (onlyThisPic is None) else {'top': 0.9, 'wspace': 0.3}))
    
    disp.display(fig)
    if calcTemperature: 
        mbgSigmas = np.array([elt[2] for elt in hSigmas])
        mbgSigmaErrs = np.array([elt[2] for elt in hSigmaErrs])
        myGuess = [0.0, min((mbgSigmas)*1e-6), guessTemp]
        temp, fitV, cov = ah.calcBallisticTemperature(keyPlt*1e-3, (mbgSigmas)*1e-6, guess = myGuess, sizeErrors = mbgSigmaErrs)
        error = np.sqrt(np.diag(cov))
    numAxisCol = int(plotSigmas) + int(plotCounts) + int(trackFitCenter)
    if numAxisCol != 0:
        fig2, axs = plt.subplots(1, numAxisCol, figsize = (15, 5)) 
        fig2.subplots_adjust(top=0.75, wspace = 0.4)
    if plotSigmas:
        ax = (axs if numAxisCol == 1 else axs[0])        
        stdStyle = dict(marker='o',linestyle='',capsize=3)
        if onlyThisPic is not None:
            ax.errorbar(keyPlt, hSigmas[:,onlyThisPic], hSigmaErrs[:,onlyThisPic], color='b', label='h '+titles[onlyThisPic], **stdStyle);
            ax.errorbar(keyPlt, vSigmas[:,onlyThisPic], vSigmaErrs[:,onlyThisPic], color='c', label='v '+titles[onlyThisPic], **stdStyle);
        else:
            for whichPic in range(4):
                ax.errorbar(keyPlt, hSigmas[:,whichPic], hSigmaErrs[:,whichPic], color='b', label='h '+titles[whichPic], **stdStyle);
                ax.errorbar(keyPlt, vSigmas[:,whichPic], vSigmaErrs[:,whichPic], color='c', label='v '+titles[whichPic], **stdStyle);
        ax.legend()
        ax.set_ylabel(r'Fit Sigma ($\mu m$)')
            
        if calcTemperature:
            # converting time to s, hSigmas in um 
            xPoints = np.linspace(min(keyPlt), max(keyPlt))*1e-3
            yPoints = LargeBeamMotExpansion.f(xPoints, *fitV)*1e6
            yGuess = LargeBeamMotExpansion.f(xPoints, *myGuess)*1e6
            ax.plot(xPoints*1e3, yGuess, label = 'guess')
            ax.plot(xPoints*1e3, yPoints, label = 'fit')
            ax.legend()
            
        ampAx = ax.twinx()
        
        if onlyThisPic is not None:
            ampAx.errorbar(keyPlt, h_amp[:,onlyThisPic], hAmpErrs[:,onlyThisPic], label='h '+titles[onlyThisPic], color = 'orange', **stdStyle);
            ampAx.errorbar(keyPlt, v_amp[:,onlyThisPic], vAmpErrs[:,onlyThisPic], label='v '+titles[onlyThisPic], color = 'r', **stdStyle);
        else:
            for whichPic in range(4):
                ampAx.errorbar(keyPlt, h_amp[:,whichPic], hAmpErrs[:,whichPic], label='h '+titles[whichPic], color = 'orange', **stdStyle);
                ampAx.errorbar(keyPlt, v_amp[:,whichPic], vAmpErrs[:,whichPic], label='v '+titles[whichPic], color = 'r', **stdStyle);
        [tick.set_color('red') for tick in ampAx.yaxis.get_ticklines()]
        [tick.set_color('red') for tick in ampAx.yaxis.get_ticklabels()]
        ampAx.set_ylabel(r'Fit h_amps', color = 'r')
        
    hTotalPhotons, vTotalPhotons = None, None
    if plotCounts:
        # numAxCol = 1: ax = axs
        # numAxCol = 2: plotSigmas + plotCounts -- ax = axs[1]
        # numAxCol = 2: plotCounts + trackFitCenter -- ax = axs[0]
        # numAxCol = 3: ax = axs[1]
        if numAxisCol == 1:
            ax = axs
        elif numAxisCol == 2:
            ax = axs[1 if plotSigmas else 0]
        else:
            ax = axs[1]
        # Create axis to plot photon counts
        ax.set_ylabel(r'Integrated signal')
        photon_axis = ax.twinx()
        colors = ['red', 'orange', 'yellow', 'pink']
        # This is not currently doing any correct for e.g. the loading rate.
        countToCameraPhotonEM = 0.018577 / (emGainSetting/200) # the float is is EM200. 
        countToScatteredPhotonEM = 0.018577/0.07 / (emGainSetting/200)

        if onlyThisPic is not None:
            # calculate number of photons
            hamp = h_amp[:,onlyThisPic]*len(expansionPics[0][0]) # Horizontal "un"normalization for number of columns begin averaged.
            vamp = v_amp[:,onlyThisPic]*len(expansionPics[0]) 
            hsigpx = hSigmas[:,onlyThisPic]/(16/64) # Convert from um back to to pixels.
            vsigpx = vSigmas[:,onlyThisPic]/(16/64)
            htotalCountsPerPic = bump.area_under(hamp, hsigpx)
            vtotalCountsPerPic = bump.area_under(vamp, vsigpx)
            hTotalPhotons = countToScatteredPhotonEM*htotalCountsPerPic
            vTotalPhotons = countToScatteredPhotonEM*vtotalCountsPerPic
            ax.plot(keyPlt, totalSignal[:,onlyThisPic], marker='o', linestyle='', label=titles[onlyThisPic]);
            photon_axis.plot(keyPlt, hTotalPhotons, marker='o', linestyle='', color = 'r', label='Horizontal')
            photon_axis.plot(keyPlt, vTotalPhotons, marker='o', linestyle='', color = 'orange', label='Vertical')
        else:
            for whichPic in range(4):
                # See above comments
                amp = h_amp[:,whichPic]*len(expansionPics[0][0]) 
                sig = hSigmas[:,whichPic]/(16/64) 
                totalCountsPerPic = bump.area_under(amp, sig)
                hTotalPhotons = countToScatteredPhotonEM*totalCountsPerPic
                ax.plot(keyPlt, totalSignal[:,whichPic], marker='o', linestyle='', label=titles[whichPic]);
                photon_axis.plot(keyPlt, hTotalPhotons, marker='o', linestyle='', color = colors[whichPic])               
        ax.legend()
        photon_axis.legend()
        [tick.set_color('red') for tick in photon_axis.yaxis.get_ticklines()]
        [tick.set_color('red') for tick in photon_axis.yaxis.get_ticklabels()]
        photon_axis.set_ylabel(r'Fit-Based Avg Scattered Photon/Img', color = 'r')
    if trackFitCenter:
        #numaxcol = 1: ax = axs
        #numaxcol = 2: trackfitcenter + plothSigmas: ax = axs[1]
        #numaxcol = 2: trackfitcenter + plotCounts: ax = axs[1]
        #numaxcol = 3: ax = axs[2]
        if numAxisCol == 1:
            ax = axs
        else:
            ax = axs[-1]     
        if onlyThisPic is not None:
            ax.errorbar(keyPlt, hfitCenter[:,onlyThisPic], hFitCenterErrs[:,onlyThisPic], marker='o', linestyle='', capsize=3, label=titles[onlyThisPic]);
            def accel(t, x0, a):
                return x0 + 0.5*a*t**2
            accelFit, AccelCov = opt.curve_fit(accel, keyPlt*1e-3, hfitCenter[:,onlyThisPic], sigma = hFitCenterErrs[:,onlyThisPic])
            fitx = np.linspace(keyPlt[0], keyPlt[-1])*1e-3
            fity = accel(fitx, *accelFit)
            ax.plot(fitx*1e3, fity)
        else:
            for whichPic in range(4):
                ax.errorbar(keyPlt, hfitCenter[:,whichPic], hFitCenterErrs[:,whichPic], marker='o', linestyle='', capsize=3, label=titles[whichPic]);
        accelErr = np.sqrt(np.diag(AccelCov))
        fig2.legend()
        ax.set_ylabel(b'Fit Centers (pix)')
        ax.set_xlabel('time (ms)')
       
    if numAxisCol != 0:
        disp.display(fig2) 
    
    for fid in fids:
        if type(fid) == int:
            if newAnnotation or not exp.checkAnnotation(fid, force=False, quiet=True):
                exp.annotate(fid)
    if clearOutput:
        disp.clear_output()
    
    if calcTemperature: 
        tempCalc = temp*1e6
        tempCalcErr = error[2]*1e6
        print('temperature = ' + misc.errString(tempCalc, tempCalcErr) + 'uk')
    else:
        tempCalc = None
        tempCalcErr = None

    for fid in fids:
        if type(fid) == int:
            expTitle, _, lev = exp.getAnnotation(fid)
            expTitle = ''.join('#' for _ in range(lev)) + ' File ' + str(fid) + ': ' + expTitle
            disp.display(disp.Markdown(expTitle))
            with exp.ExpFile(fid) as file:
                file.get_basic_info()
    if trackFitCenter:
        print('Acceleration in Mpix/s^2 = ' + misc.errString(accelFit[1], accelErr[1]))
    if transferAnalysisOpts is not None:
        colors, colors2 = misc.getColors(len(transferAnalysisOpts.initLocs()) + 2)#, cmStr=dataColor)
        pltShape = (transferAnalysisOpts.initLocsIn[-1], transferAnalysisOpts.initLocsIn[-2])
        #mp.plotThresholdHists([initThresholds[0][0],initThresholds[1][0]], colors, shape=pltShape)    
    return {'images':images, 'fits':hFitParams, 'cov':hFitCovs, 'pics':sortedStackedPics, 'hSigmas':hSigmas, 'sigmaErrors':hSigmaErrs, 'dataKey':keyPlt, 
            'hTotalPhotons':hTotalPhotons, 'tempCalc':tempCalc, 'tempCalcErr':tempCalcErr, 'initThresholds':initThresholds[0]}

def getBgImgs(bgSource, startPic=1, picsPerRep=2):
    """ Expects either a file ID number or a list (or an array) of images as input.
    """
    if type(bgSource) == int:
        with exp.ExpFile(bgSource) as file:
            pics = file.get_pics()
    if type(bgSource) == type(np.array([])) or type(bgSource) == type([]):
        pics = bgSource
    pics2 = pics[startPic::picsPerRep]
    pics2 = rmHighCountPics(pics2, 7000)
    avgBg = np.mean(pics2,0)
    avgPcBg = photonCounting(pics2, 120)[0] / len(pics2)
    return avgBg, avgPcBg


