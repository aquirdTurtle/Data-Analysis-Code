import dataclasses as dc
import typing
import AnalysisHelpers as ah

@dc.dataclass
class AnalysisOptions:
    initLocsIn: tuple = ()
    tferLocsIn: tuple = ()
    initPic: int = 0
    tferPic: int = 1
    postSelectionConditions: typing.Any = None
    postSelectionConnected: bool = False
    def numAtoms(self):
        return len(self.initLocs())
    def initLocs(self):
        return ah.unpackAtomLocations(self.initLocsIn)
    def tferLocs(self):
        return ah.unpackAtomLocations(self.tferLocsIn)
    def __str__(self):
        return ('[AnalysisOptions with:\ninitLocations:' + str(self.initLocsIn)
                + ',\ntferLocations:' + str(self.tferLocsIn)
                + ",\ninitPic:" + str(self.initPic)
                + ',\ntferPic:' + str(self.tferPic)
                + ',\npostSelectionCondition:' + str(self.postSelectionConditions)
                + ',\npostSelectionConnected:' + str(self.postSelectionConnected) + ']')
    def __repr__(self):
        return self.__str__()
