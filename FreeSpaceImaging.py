from matplotlib.colors import ListedColormap
from matplotlib import cm
import Miscellaneous as misc
import numpy as np
import matplotlib.pyplot as plt
import MatplotlibPlotters as mp
import ExpFile as exp
import AnalysisHelpers as ah
import scipy.optimize as opt
from fitters.Gaussian import bump3

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
    for i, p in enumerate(pics):
        if max(p.flatten()) > threshold:
            deleteList.append(i)
    if len(deleteList) is not 0:
        print('Not using suspicious bg data:', deleteList)
    for index in reversed(deleteList):
        pics = np.delete(pics, index, 0)
    return pics

def freespaceImageAnalysis(fids, guesses = [[64, 8, 40, 2, 8, 60, 2, 8, 78, 2],
                                            [64, 8, 40, 2, 8, 60, 2, 8, 78, 2],
                                            [64, 8, 40, 2, 8, 60, 2, 8, 79, 2],
                                            [64, 8, 40, 2, 8, 60, 2, 8, 78, 2]], 
                           nrPics = 3, fit=True, bgInput=None, bgPcInput=None, shape=None, 
                           zeroCorrection=0, zeroCorrectionPC=0, plotAll=False, fitModule=bump3):
    s = [0,-1,0,-1] if shape is None else shape
    sortedStackedPics = {}
    for fid in fids:
        with exp.ExpFile(fid) as f:
            allpics = f.get_pics()
            kn, key = f.get_key()
            f.get_basic_info()
        allpics = np.reshape(allpics, (len(key), int(allpics.shape[0]/len(key)), allpics.shape[1], allpics.shape[2]))
        for i, keyV in enumerate(key):
            if keyV in sortedStackedPics:
                sortedStackedPics[keyV] = np.append(sortedStackedPics[keyV], allpics[i],axis=0)
            else:
                sortedStackedPics[keyV] = allpics[i]
    fits = {}
    fitErrs = {}
    images = {}
    for keyV, varPics in sorted(sortedStackedPics.items()):
        figtitle=str(keyV) + ': ' + str(len(varPics)/nrPics)
        # 0 is init atom pics for post-selection on atom number... if we wanted to.
        expansionPics = varPics[1::nrPics]
        if nrPics == 3:
            inseqbackgroundpic=varPics[2::nrPics]
        discardThreshold = 7000
        expansionPics = rmHighCountPics(expansionPics,7000)
        inseqbackgroundpic = rmHighCountPics(inseqbackgroundpic,7000)

        threshold = 110
        
        expPhotonCountImage = np.zeros(expansionPics[0].shape)
        for p in expansionPics:
            for i, r in enumerate(p):
                for j,c in enumerate(r):
                    if c > threshold:
                        expPhotonCountImage[i,j] += 1
        
        if bgPcInput is None:
            bgPhotonCountImage = np.zeros(expansionPics[0].shape)
            if nrPics==3:
                for p in inseqbackgroundpic:
                    for i, r in enumerate(p):
                        for j,c in enumerate(r):
                            if c > threshold:
                                bgPhotonCountImage[i,j] += 1
        else:
            bgPhotonCountImage = bgPcInput
        
                        
        expPhotonCountImage /= len(varPics)/nrPics
        bgPhotonCountImage /= len(varPics)/nrPics
            
            
        expAvg = np.zeros(expansionPics[0].shape)
        for p in expansionPics:
            expAvg += p
        
        expAvg /= len(expansionPics)
        if bgInput is None:
            bgAvg = np.zeros(expansionPics[0].shape)
            if nrPics == 3:
                bgAvg = np.mean(inseqbackgroundpic,0)
        else:
            bgAvg = bgInput

        fig, axs = plt.subplots(1, 4 if plotAll else 1, figsize=(20,5))
        if bgAvg is None:
            print('no bg')
            bgAvg = np.zeros(expAvg.shape)
        if bgPhotonCountImage is None:
            print('no bg photon', expAvg.shape)
            bgPhotonCount = np.zeros(photonCountImage.shape)
            print(bgPhotonCount.shape)

        
        avg_mbg = expAvg[s[0]:s[1], s[2]:s[3]] - bgAvg[s[0]:s[1], s[2]:s[3]]# - zeroCorrection
        #print(np.sum(expAvg[s[0]:s[1], s[2]:s[3]]), np.sum(bgAvg[s[0]:s[1], s[2]:s[3]]),np.sum(avg_mbg))
        avg_mbgpc = expPhotonCountImage[s[0]:s[1], s[2]:s[3]] - bgPhotonCountImage[s[0]:s[1], s[2]:s[3]]# - zeroCorrectionPC
        ims = [expAvg[s[0]:s[1], s[2]:s[3]], expPhotonCountImage[s[0]:s[1], s[2]:s[3]], avg_mbg, avg_mbgpc] if plotAll else [avg_mbg]
        titles = ['Bare', 'Photon-Count', 'Bare-mbg', 'Photon-Count-mbg'] if plotAll else ['Bare-mbg']
        fits[keyV] = []
        fitErrs[keyV] = []
        images[keyV] = []
        for ax, im, title, guess in zip(axs.flatten() if plotAll else [axs], ims, titles, guesses):
            images[keyV].append(im)
            hAvg, vAvg = ah.collapseImage(im)
            if fit:
                x = np.arange(len(hAvg))
                if guess is None:
                    #guess = bump3.guess(x, hAvg)
                    guess = fitModule.guess(x,hAvg)
                try:
                    #params, cov = opt.curve_fit(bump3.f, x, hAvg, p0=guess)
                    params, cov = opt.curve_fit(fitModule.f, x, hAvg, p0=guess)
                except RuntimeError:
                    print('fit failed')
                fits[keyV].append(params)
                fitErrs[keyV].append(np.sqrt(np.diag(cov)))
            else:
                fits[keyV].append(None)
                fitErrs[keyV].append(None)
            cf = 16/50
            ax, _, _, _, hAvg, vAvg = mp.fancyImshow(fig, ax, im, imageArgs={'cmap':dark_viridis_cmap}, hFitParams=params, fitModule=fitModule)

            ax.set_title(title + ': ' + misc.round_sig_str(fits[keyV][-1][2]*cf*2) + r'$\mu m$ waist, ' + misc.round_sig_str(np.sum(im.flatten()),5))
        fig.subplots_adjust(left=0,right=1,bottom=0,top=0.95, wspace=0.3, hspace=0)
        if plotAll:
            fig.suptitle(figtitle, fontsize=26)
    return ims, fits, fitErrs, images