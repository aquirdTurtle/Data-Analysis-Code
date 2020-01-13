from matplotlib.colors import ListedColormap
from matplotlib import cm
import Miscellaneous as misc
import numpy as np
import matplotlib.pyplot as plt
import MatplotlibPlotters as mp
import ExpFile as exp
import AnalysisHelpers as ah
import scipy.optimize as opt
from fitters.Gaussian import bump3, bump

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
    if len(deleteList) is not 0:
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

def freespaceImageAnalysis( fids, guesses = None, fit=True, bgInput=None, bgPcInput=None, shapes=None, zeroCorrection=0, zeroCorrectionPC=0,
                            keys = None, fitModule=bump, extraPicDictionaries=None):
    if keys is None:
        keys = [None for _ in fids]
    sortedStackedPics = {}
    for filenum, fid in enumerate(fids):
        with exp.ExpFile(fid) as f:
            allpics = f.get_pics()
            kn, key = f.get_key()
            f.get_basic_info()
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
            for key, pics in dictionary.items():
                print(sortedStackedPics[key].shape, pics.shape)
                sortedStackedPics[key] = (np.append(sortedStackedPics[key], pics,axis=0) if key in sortedStackedPics else pics)
    numVars = len(sortedStackedPics.items())
    if shapes is None:
        shapes = [None for _ in range(numVars)]
    if guesses is None:
        guesses = [[None for _ in range(4)] for _ in range(numVars)]
    if bgInput is None:
        bgInput = [None for _ in range(numVars)]
    if bgPcInput is None:
        bgPcInput = [None for _ in range(numVars)]
    datalen, avgFitWaists, images, fitParams, fitCovs = [{} for _ in range(5)]
    titles = ['Bare', 'Photon-Count', 'Bare-mbg', 'Photon-Count-mbg']
    for vari, (keyV, varPics) in enumerate(sorted(sortedStackedPics.items())):
        # 0 is init atom pics for post-selection on atom number... if we wanted to.
        expansionPics = rmHighCountPics(varPics[1::2],7000)
        datalen[keyV] = len(expansionPics)
        expPhotonCountImage = photonCounting(expansionPics, 120) / len(expansionPics)
        if bgPcInput[vari] is None:
            bgPhotonCountImage = np.zeros(expansionPics[0].shape)
        else:
            bgPhotonCountImage = bgPcInput[vari]
        expAvg = np.mean(expansionPics, 0)
        if bgInput[vari] is None:
            bgAvg = np.zeros(expansionPics[0].shape)
        else:
            bgAvg = bgInput[vari]
        if bgAvg is None:
            bgAvg = np.zeros(expAvg.shape)
        if bgPhotonCountImage is None:
            print('no bg photon', expAvg.shape)
            bgPhotonCount = np.zeros(photonCountImage.shape)
        s_ = [0,-1,0,-1] if shapes[vari] is None else shapes[vari]
        avg_mbg = expAvg[s_[0]:s_[1], s_[2]:s_[3]] - bgAvg[s_[0]:s_[1], s_[2]:s_[3]]# - zeroCorrection
        avg_mbgpc = expPhotonCountImage[s_[0]:s_[1], s_[2]:s_[3]] - bgPhotonCountImage[s_[0]:s_[1], s_[2]:s_[3]]# - zeroCorrectionPC        
        images[keyV] = [expAvg[s_[0]:s_[1], s_[2]:s_[3]], expPhotonCountImage[s_[0]:s_[1], s_[2]:s_[3]], avg_mbg, avg_mbgpc]
        avgFitWaists[keyV], fitParams[keyV], fitCovs[keyV] = [[] for _ in range(3)]
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
                avgFitWaists[keyV].append(fitModule.fitCharacter(params))
            else:
                avgFitWaists[keyV].append(None)
    cf = 16e-6/50
    mins, maxes = [[], []]
    imgs_ = np.array(list(images.values()))
    for imgInc in range(4):
        mins.append(min(imgs_[:,imgInc].flatten()))
        maxes.append(max(imgs_[:,imgInc].flatten()))
    for vari, ((keyV,ims), (_,param_set)) in enumerate(zip(images.items(), fitParams.items())):
        fig, axs = plt.subplots(1,4, figsize=(20,3))
        for which, (im, ax, params, title, min_, max_) in enumerate(zip(ims, axs, param_set, titles, mins, maxes)):
            ax, _, _, _, hAvg, vAvg = mp.fancyImshow(fig, ax, im, imageArgs={'cmap':dark_viridis_cmap, 'vmin':min_, 'vmax':max_}, hFitParams=params, fitModule=fitModule)
            ax.set_title(title + ': ' + misc.round_sig_str(avgFitWaists[keyV][which]*cf*2e6) + r'$\mu m$ waist, ' + misc.round_sig_str(np.sum(im.flatten()),5))
        fig.subplots_adjust(left=0,right=1,bottom=0,top=0.95, wspace=0.3, hspace=0)
        fig.suptitle(keyV + ': ' + str(datalen[keyV]), fontsize=26)
    return avgFitWaists, images, fitParams, fitCovs, sortedStackedPics

def getBgImgs(fid):
    with exp.ExpFile(fid) as file:
        pics = file.get_pics()
    pics2 = pics[1::2]
    pics2 = rmHighCountPics(pics2, 7000)
    avgBg = np.mean(pics2,0)
    avgPcBg = photonCounting(pics2, 120) / len(pics2)
    return avgBg, avgPcBg