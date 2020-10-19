import MainAnalysis as ma
import matplotlib.pyplot as plt
import Miscellaneous as misc
import numpy as np
import ExpFile as exp

def updateThresholds():
    with open('ThresholdAnalysisInfo.txt') as f:
        lines = f.readlines()
        dataLines = [l.rstrip() for l in lines[5:]]
    dateTuple = dataLines[0].split(',')
    fid = int(dataLines[1])
    atomLocations = dataLines[2].split(';')
    atomLocations = [[int(x) for x in loc.replace('[','').replace(']','').split(',')] for loc in atomLocations]
    picsPerRep = int(dataLines[3])    
    exp.setPath(*dateTuple)
    res = ma.standardPopulationAnalysis(fid, atomLocations, 0, picsPerRep)
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
    with open('C:/Users/Mark-Brown/Code/Chimera-Control/T_File.txt','w') as f:
        for val in threshVals:
            f.write(str(val) + ' ') 
    plt.show(block=False)

updateThresholds()
