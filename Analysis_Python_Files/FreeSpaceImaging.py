from matplotlib.colors import ListedColormap
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import IPython.display as disp

from . import Miscellaneous as misc
from . import MatplotlibPlotters as mp
from . import ExpFile as exp
from . import AnalysisHelpers as ah
from .fitters.Gaussian import bump3, bump
from .fitters import LargeBeamMotExpansion 

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
    pcPics = np.sum(pcPics,axis=0)
    return np.array(pcPics)

def freespaceImageAnalysis( fids, guesses = None, fit=True, bgInput=[None], bgPcInput=None, shapes=[None], zeroCorrection=0, zeroCorrectionPC=0,
                            keys = None, fitModule=bump, extraPicDictionaries=None, newAnnotation=False, onlyThisPic=None, pltVSize=5,              
                            plotSigmas=False, plotCounts=False, manualColorRange=None, picsPerRep=1, calcTemperature = False, clearOutput = True, 
                           dataRange = None):
    if keys is None:
        keys = [None for _ in fids]
    sortedStackedPics = {}
    for filenum, fid in enumerate(fids):
        if type(fid) == int:
            with exp.ExpFile(fid) as f:
                allpics = f.get_pics()
                kn, key = f.get_key()
                f.get_basic_info()
        else:
            allpics = fid
            print("Assuming input is list of all pics.")
        if keys[filenum] is not None:
            key = keys[filenum]
        allpics = np.reshape(allpics, (len(key), int(allpics.shape[0]/len(key)), allpics.shape[1], allpics.shape[2]))
        for i, keyV in enumerate(key):
            keyV = misc.round_sig_str(keyV)
            sortedStackedPics[keyV] = np.append(sortedStackedPics[keyV], allpics[i],axis=0) if (keyV in sortedStackedPics) else allpics[i]
    if extraPicDictionaries is not None:
        if type(extraPicDictionaries) == dict:
            extraPicDictionaries = [extraPicDictionaries]
        for dictionary in extraPicDictionaries:
            for keyV, pics in dictionary.items():
                sortedStackedPics[keyV] = (np.append(sortedStackedPics[keyV], pics,axis=0) if keyV in sortedStackedPics else pics)
    sortedKey = list(sorted(key))
    sortedKey, sortedStackedPics = ah.applyDataRange(dataRange, sortedStackedPics, sortedKey)
    numVars = len(sortedStackedPics.items())
    if len(np.array(shapes).shape) == 1:
        shapes = [shapes for _ in range(numVars)]       
    if guesses is None:
        guesses = [[None for _ in range(4)] for _ in range(numVars)]
    if len(np.array(bgInput).shape) == 2:
        bgInput = [bgInput for _ in range(numVars)]
    if bgPcInput is None:
        bgPcInput = [None for _ in range(numVars)]
    datalen, avgFitSigmas, images, fitParams, fitCovs = [{} for _ in range(5)]
    titles = ['Bare', 'Photon-Count', 'Bare-mbg', 'Photon-Count-mbg']
    for vari, keyV in enumerate(sortedKey):
        keyV=misc.round_sig_str(keyV)
        varPics = sortedStackedPics[keyV]
        # 0 is init atom pics for post-selection on atom number... if we wanted to.
        expansionPics = rmHighCountPics(varPics[picsPerRep-1::picsPerRep],7000)
        datalen[keyV] = len(expansionPics)
        expPhotonCountImage = photonCounting(expansionPics, 120) / len(expansionPics)
        if bgPcInput[vari] is None:
            bgPhotonCountImage = np.zeros(expansionPics[0].shape)
        else:
            bgPhotonCountImage = bgPcInput[vari]
        expAvg = np.mean(expansionPics, 0)
        if len(bgInput[vari]) == 1:
            bgAvg = np.zeros(expansionPics[0].shape)
        else:
            bgAvg = bgInput[vari]
        if bgPhotonCountImage is None:
            print('no bg photon', expAvg.shape)
            bgPhotonCount = np.zeros(photonCountImage.shape)
        s_ = [0,-1,0,-1] if shapes[vari] == [None] else shapes[vari]
        avg_mbg = ah.windowImage(expAvg, s_) - ah.windowImage(bgAvg, s_)
        avg_mbgpc = ah.windowImage(expPhotonCountImage, s_) - ah.windowImage(bgPhotonCountImage, s_)
        images[keyV] = [ah.windowImage(expAvg, s_), ah.windowImage(expPhotonCountImage, s_), avg_mbg, avg_mbgpc]
        fitParams[keyV], fitCovs[keyV] = [[] for _ in range(2)]
        for im, guess in zip(images[keyV], guesses[vari]):
            hAvg, vAvg = ah.collapseImage(im)
            if fit:
                x = np.arange(len(hAvg))
                if guess is None:
                    guess = fitModule.guess(x, hAvg)
                try:
                    params, cov = opt.curve_fit(fitModule.f, x, hAvg, p0=guess)
                except RuntimeError:
                    print('fit failed!')
                fitParams[keyV].append(params)
                fitCovs[keyV].append(cov)

    cf = 16e-6/40
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
    else:
        numRows = int(np.ceil(numVariations/4))

        if numVariations == 1:
            fig, axs = plt.subplots(figsize=(20, pltVSize))
            axs = np.array(axs)
        else:
            fig, axs = plt.subplots(numRows, 4, figsize=(20, pltVSize*numRows))
    #if numVariations == 1:
     #   axs = [axs]
 
    sigmas = np.zeros((len(images), 4))
    totalSignal = np.zeros((len(images), 4))
    sigmaErrs = np.zeros((len(images), 4))
    keyPlt = np.zeros(len(images))
    amplitude = np.zeros((len(images), 4))
    ampErrs = np.zeros((len(images), 4))
    for vari, ((keyV,ims), (_,param_set), (_, cov_set)) in enumerate(zip(images.items(), fitParams.items(), fitCovs.items())):
        if onlyThisPic is None:
            for which, (im, ax, params, title, min_, max_, covs) in enumerate(zip(ims, axs[vari], param_set, titles, mins, maxes, cov_set)):
                sigmas[vari][which] = params[2]*cf*1e6
                sigmaErrs[vari][which] = np.sqrt(np.diag(covs))[2]*cf*1e6
                amplitude[vari][which] = params[0]
                ampErrs[vari][which] = np.sqrt(np.diag(covs))[0]
                totalSignal[vari][which] = np.sum(im.flatten())
                keyPlt[vari] = keyV
                res = mp.fancyImshow(fig, ax, im, imageArgs={'cmap':dark_viridis_cmap, 'vmin':min_, 'vmax':max_}, hFitParams=params, fitModule=fitModule)
                ax, hAvg, vAvg = [res[0], res[4], res[5]]
                ax.set_title(keyV + ': ' + str(datalen[keyV]) + ';\n' + title + ': ' + misc.errString(sigmas[vari][which],sigmaErrs[vari][which]) 
                    + r'$\mu m$ sigma, ' + misc.round_sig_str(totalSignal[vari][which],5), fontsize=12)
            fig.subplots_adjust(left=0,right=1,bottom=0.1,top=0.9, wspace=0.4, hspace=0.5)
        else:
            # ax = (axs.flatten()[vari] if type(axs)==type(np.array([])) else axs)
            ax = axs.flatten()[vari]
            sigmas[vari][onlyThisPic] = param_set[onlyThisPic][2]*cf*1e6
            sigmaErrs[vari][onlyThisPic] = np.sqrt(np.diag(cov_set[onlyThisPic]))[2]*cf*1e6
            amplitude[vari][onlyThisPic] = param_set[onlyThisPic][0]
            ampErrs[vari][onlyThisPic] = np.sqrt(np.diag(cov_set[onlyThisPic]))[0]
            totalSignal[vari][onlyThisPic] = np.sum(ims[onlyThisPic].flatten())
            keyPlt[vari] = keyV
            res = mp.fancyImshow(fig, ax, ims[onlyThisPic], imageArgs={'cmap':dark_viridis_cmap, 'vmin':mins[onlyThisPic], 
                                                                       'vmax':maxes[onlyThisPic]}, 
                                 hFitParams=param_set[onlyThisPic], fitModule=fitModule)
            ax, hAvg, vAvg = [res[0], res[4], res[5]]
            ax.set_title(keyV + ': ' + str(datalen[keyV]) + ';\n' 
                         + titles[onlyThisPic] + ': ' + misc.errString(param_set[onlyThisPic][2]*cf*1e6, np.sqrt(np.diag(cov_set[onlyThisPic]))[2]*cf*1e6) 
                         + r'$\mu m$ sigma, ' + misc.round_sig_str((totalSignal[vari][onlyThisPic]),5), 
                         fontsize=12)
            fig.subplots_adjust(left=0,right=1,bottom=0.1,top=0.9, wspace=0.3, hspace=0.5)
        
    disp.display(fig)
    
    if calcTemperature: 
        mbgSigmas = np.array([elt[2] for elt in sigmas])
        mbgSigmaErrs = np.array([elt[2] for elt in sigmaErrs])
        myGuess = [0.0, min((mbgSigmas)*1e-6), 10e-6]
        temp, fitV, cov = ah.calcBallisticTemperature(keyPlt*1e-3, (mbgSigmas)*1e-6, guess = myGuess, sizeErrors=mbgSigmaErrs)
        error = np.sqrt(np.diag(cov))
        
    numAxisCol = int(plotSigmas) + int(plotCounts)
               
    if numAxisCol != 0:
        fig2, axs = plt.subplots(1, numAxisCol, figsize = (15, 5)) 
        fig2.subplots_adjust(wspace = 0.3)
    if plotSigmas:
        ax = (axs[0] if numAxisCol == 2 else axs)
            
        if onlyThisPic is not None:
            ax.errorbar(keyPlt, sigmas[:,onlyThisPic], sigmaErrs[:,onlyThisPic], marker='o', linestyle='', capsize=3, label=titles[onlyThisPic]);
        else:
            for whichPic in range(4):
                ax.errorbar(keyPlt, sigmas[:,whichPic], sigmaErrs[:,whichPic], marker='o', linestyle='', capsize=3, label=titles[whichPic]);
        fig2.legend()
        ax.set_ylabel(r'Fit Sigma ($\mu m$)')
        
    
        if calcTemperature:
            # converting time to s, sigmas in um 
            xPoints = np.linspace(min(keyPlt), max(keyPlt))*1e-3
            yPoints = LargeBeamMotExpansion.f(xPoints, *fitV)*1e6
            yGuess = LargeBeamMotExpansion.f(xPoints, *myGuess)*1e6
            ax.plot(xPoints*1e3, yGuess, label = 'guess')
            ax.plot(xPoints*1e3, yPoints, label = 'fit')
            ax.legend()
            
        ampAx = ax.twinx()
        
        if onlyThisPic is not None:
            ampAx.errorbar(keyPlt, amplitude[:,onlyThisPic], ampErrs[:,onlyThisPic], marker='o', linestyle='', capsize=3, label=titles[onlyThisPic], color = 'r');
        else:
            for whichPic in range(4):
                ampAx.errorbar(keyPlt, amplitude[:,whichPic], ampErrs[:,whichPic], marker='o', linestyle='', capsize=3, label=titles[whichPic], color = 'r');
        ampAx.set_ylabel(r'Fit amplitudes', color = 'r')
        
        
    if plotCounts:
        ax = (axs[1] if numAxisCol == 2 else axs)
        if onlyThisPic is not None:
            ax.plot(keyPlt, totalSignal[:,onlyThisPic], marker='o', linestyle='', label=titles[onlyThisPic]);
        else:
            for whichPic in range(4):
                ax.plot(keyPlt, totalSignal[:,whichPic], marker='o', linestyle='', label=titles[whichPic]);
        fig.legend()
        ax.set_ylabel(r'Integrated signal')
        
        
        ##########################
        photon_axis = ax.twinx()
        
        amp = amplitude[:,onlyThisPic]*len(expansionPics[0][0]) #Horizontal "un"normalization for number of columns begin averaged.
        sig = sigmas[:,onlyThisPic]/(16/40) #Convert from um to pixels.
        countToCameraPhotonEM200 = 0.018577 * 2 #(EM100)
        countToScatteredPhotonEM200 = 0.018577/0.07 * 2
        totalCountsPerPic = bump.area_under(amp, sig)
        totalPhotons = countToScatteredPhotonEM200*totalCountsPerPic
        photon_axis.plot(keyPlt, totalPhotons, marker='o', linestyle='', color = 'r')
        photon_axis.set_ylabel(r'Photon count', color = 'r')
        
       
    if numAxisCol != 0:
        disp.display(fig2) 
    
    for fid in fids:
        if type(fid) == int:
            if newAnnotation or not exp.checkAnnotation(fid, force=False, quiet=True):
                exp.annotate(fid)
    if clearOutput:
        disp.clear_output()
    
    if calcTemperature: 
        temperatureCalculated = temp*1e6
        print('temperature = ' + misc.errString(temp*1e6, error[2]*1e6) + 'uk')

    for fid in fids:
        if type(fid) == int:
            expTitle, _, lev = exp.getAnnotation(fid)
            expTitle = ''.join('#' for _ in range(lev)) + ' File ' + str(fid) + ': ' + expTitle
            disp.display(disp.Markdown(expTitle))
            with exp.ExpFile(fid) as file:
                file.get_basic_info()
    
    return {'images':images, 'fits':fitParams, 'cov':fitCovs, 'pics':sortedStackedPics, 'sigmas':sigmas, 'sigmaErrors':sigmaErrs, 'dataKey':keyPlt, 'totalPhotons':totalPhotons, 'temperatureCalculated':temperatureCalculated}

def getBgImgs(fid):
    with exp.ExpFile(fid) as file:
        pics = file.get_pics()
    pics2 = pics[1::2]
    pics2 = rmHighCountPics(pics2, 7000)
    avgBg = np.mean(pics2,0)
    avgPcBg = photonCounting(pics2, 120) / len(pics2)
    return avgBg, avgPcBg