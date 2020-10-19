import dataclasses as dc
import typing
from . import AnalysisHelpers as ah

@dc.dataclass
class condition:
    # right now the code only supports conditions that are isolated to analysis of a single picture.
    whichPic: tuple = ()
    # list of atom indexes.
    whichAtoms: tuple = ()
    # list of conditions for each atom. 
    conditions: tuple = ()
    # -1 means all are required. typically -1 or 1 (i.e. all conditions met or only one.)
    numRequired: int = -1
    # for some introspection.    
    name: str = ""
    # optional markers which tell plotters where to mark the average images for this data set. 
    markerWhichPicList: tuple = ()
    markerLocList: tuple = ()

        
@dc.dataclass
class TransferAnalysisOptions:
    initLocsIn: tuple = ()
    tferLocsIn: tuple = ()
    initPic: int = 0
    tferPic: int = 1

    postSelectionConditions: typing.Any = None
    positiveResultConditions: typing.Any = None

        
    def numDataSets(self):
        return len(self.positiveResultConditions)
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
