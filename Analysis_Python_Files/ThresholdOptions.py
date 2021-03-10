import dataclasses as dc

@dc.dataclass
class ThresholdOptions:
    histBinSize: int = 5
    indvVariationThresholds:bool = False
    subtractEdgeCounts:bool = True
    tferThresholdSame:bool = True
    rigorousThresholdFinding:bool = True
    manualThreshold:bool = False
    manualThresholdValue: float = 0
    autoHardThreshold:bool=False
    autoThresholdFittingGuess:bool=False
    
    def __str__(self):
        return ('[ThresholdOptions with subtractEdgeCouunts: '+str(self.subtractEdgeCounts) 
                + ', tferThresholdSame:' + str(self.tferThresholdSame)
                + ', rigorousThresholdFinding:' + str(self.rigorousThresholdFinding)
                + ', manualThreshold:' + str(self.manualThreshold)
                + ', manualThresholdValue:' + str(self.manualThresholdValue)
                + ', indvVariationThresholds:' + str(self.indvVariationThresholds) + ']')
    def __repr__(self):
        return self.__str__()
    
    