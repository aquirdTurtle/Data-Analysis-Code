import dataclasses as dc
import numpy as np

@dc.dataclass
class AtomThreshold:
    """
    A structure that holds all of the info relevant for determining thresholds.
    """
    # the actual threshold
    th:float = 0
    fidelity:float = 0
    binCenters:tuple = ()
    binHeights:tuple = ()
    fitVals:tuple = ()
    rmsResidual:float = 0
    rawData:tuple = ()
    
    def binWidth(self):
        return self.binCenters[1] - self.binCenters[0]
    def binEdges(self):
        # for step style plotting
        return np.array([self.binCenters[0] - self.binWidth()] + list(self.binCenters) 
                        + [self.binCenters[-1] + self.binWidth()]) + self.binWidth()/2
    def binEdgeHeights(self):
        return np.array([0] + list(self.binHeights) + [0])
    def __copy__(self):
        return type(self)(th=self.th, fidelity=self.fidelity, binCenters=self.binCenters, binHeights=self.binHeights, 
                          fitVals=self.fitVals, rmsResidual=self.rmsResidual, rawData=self.rawData)
    def __str__(self):
        return ('[AtomThrehsold with t='+str(self.th) + ', fid='
                + str(self.fidelity) + ', fitVals=' + str(self.fitVals) + ']')
    def __repr__(self):
        return self.__str__()

