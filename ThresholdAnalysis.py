import MainAnalysis as ma
import matplotlib.pyplot as plt
import Miscellaneous as misc
import numpy as np
import ExpFile as exp


def updateThresholds(fid, atomLocations, picsPerRep):
    exp.setPath('17', 'May', '2019')
    res = standardPopulationAnalysis(fid, atomLocations, 0, picsPerRep)
    (locCounts, thresholds, avgPic, key, allPopsErr, allPops, avgPop, avgPopErr, fits,
     fitModules, keyName, atomData, rawData, atomLocations, avgFits, atomImages,
     totalAvg, totalErr) = res
    colors, _ = misc.getColors(len(atomLocations) + 1)
    f, ax = plt.subplots()
    for i, atomLoc in enumerate(atomLocations):
        ax.hist(locCounts[i], 50, color=colors[i], orientation='vertical', alpha=0.3, histtype='stepfilled')
        ax.axvline(thresholds[i].t, color=colors[i], alpha=0.3)    
    plt.show()
    # output thresholds
    threshVals = [t.t for t in thresholds]
    threshVals = np.flip(np.reshape(threshVals, (5,5)),1)
    with open('C:/Users/Mark-Brown/Code/Chimera-Control/T_File.txt','w') as f:
        for row in threshVals:
            for thresh in row:
                f.write(str(thresh) + ' ') 
    plt.show(block=False)

updateThresholds(38, [2,2,2,5,5], 2)

#plt.plot([0,1,2],[2,1,2],'o:')
#plt.show()
