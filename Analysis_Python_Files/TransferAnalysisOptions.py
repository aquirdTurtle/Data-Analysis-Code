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
                + ',\npostSelectionCondition:' + str(self.postSelectionConditions) + ']')
    def __repr__(self):
        return self.__str__()

def getStandardLoadingOptions(atomLocs):
    prConditions = [None for _ in range(len(ah.unpackAtomLocations(atomLocs)))]
    for atomNum in range(len(ah.unpackAtomLocations(atomLocs))):
        singleLoadCondition = condition(whichPic=[0],whichAtoms=[atomNum],conditions=[True],numRequired=-1, 
                                        markerWhichPicList=(0,1), markerLocList=(atomNum,atomNum))
        prConditions[atomNum] = singleLoadCondition
    return TransferAnalysisOptions( initLocsIn=atomLocs, tferLocsIn=atomLocs,
                                   postSelectionConditions=[[] for _ in range(len(ah.unpackAtomLocations(atomLocs)))], 
                                   positiveResultConditions=prConditions )

    
def getStandardSurvivalOptions(atomLocs):
    psConditions = [[] for _ in range(len(ah.unpackAtomLocations(atomLocs)))]
    for atomNum in range(len(ah.unpackAtomLocations(atomLocs))):
        singleLoadCondition = condition(whichPic=[0],whichAtoms=[atomNum],conditions=[True],
                                        numRequired=-1, markerWhichPicList=(0,1), markerLocList=(atomNum,atomNum))
        psConditions[atomNum].append(singleLoadCondition)
    return TransferAnalysisOptions( initLocsIn=atomLocs, tferLocsIn=atomLocs, postSelectionConditions=psConditions, 
                                    positiveResultConditions=[None for _ in ah.unpackAtomLocations(atomLocs)] )

def getStandard3AtomTransferConditions():
    loadLeft = condition( name="LoadLeft", whichPic=[0,0,0], whichAtoms=[0,1,2],
                             conditions=[True, False, False],numRequired=-1, markerWhichPicList=(0,), markerLocList=(0,))
    loadCenter = condition( name="LoadCenter", whichPic=[0,0,0],whichAtoms=[0,1,2],
                               conditions=[False, True, False], numRequired=-1, markerWhichPicList=(0,), markerLocList=(1,))
    loadRight = condition( name="LoadRight", whichPic=[0,0,0],whichAtoms=[0,1,2],
                              conditions=[False, False, True], numRequired=-1, markerWhichPicList=(0,), markerLocList=(2,))      
    loadEdges = condition( name="LoadEdges", whichPic=[0,0,0],whichAtoms=[0,1,2],
                              conditions=[True, False, True], numRequired=-1, markerWhichPicList=(0,0), markerLocList=(0,2))      
    
    finLeft = condition( name="finLeft", whichPic=[1,1,1], whichAtoms=[0,1,2],
                             conditions=[True, False, False],numRequired=-1, markerWhichPicList=(1,), markerLocList=(0,))
    finCenter = condition( name="finCenter", whichPic=[1,1,1],whichAtoms=[0,1,2],
                              conditions=[False, True, False], numRequired=-1, markerWhichPicList=(1,), markerLocList=(1,))
    finRight = condition( name="finRight", whichPic=[1,1,1],whichAtoms=[0,1,2],
                              conditions=[False, False, True], numRequired=-1, markerWhichPicList=(1,), markerLocList=(2,))      
    finNone = condition( name="finNone", whichPic=[1,1,1],whichAtoms=[0,1,2],
                              conditions=[False, False, False], numRequired=-1, markerWhichPicList=(), markerLocList=())      
    
    sv = condition( name="Survive", whichPic=[1,1,1],whichAtoms=[0,1,2],
                              conditions=[True, True, True], numRequired=1)
    return loadLeft, loadCenter, loadRight, loadEdges, finLeft, finCenter, finRight, finNone, sv

def getStandard2AtomTransferConditions():
    loadLeft = condition( name="LoadLeft", whichPic=[0,0], whichAtoms=[0,1],
                             conditions=[True, False],numRequired=-1, markerWhichPicList=(0,), markerLocList=(0,))
    loadRight = condition( name="LoadRight", whichPic=[0,0],whichAtoms=[0,1],
                              conditions=[False, True], numRequired=-1, markerWhichPicList=(0,), markerLocList=(1,))

    finLeft = condition( name="finLeft", whichPic=[1,1], whichAtoms=[0,1],
                             conditions=[True, False],numRequired=-1, markerWhichPicList=(1,), markerLocList=(0,))
    finRight = condition( name="finRight", whichPic=[1,1],whichAtoms=[0,1],
                              conditions=[False, True], numRequired=-1, markerWhichPicList=(1,), markerLocList=(1,))
    
    sv = condition( name="Survive", whichPic=[1,1],whichAtoms=[0,1], conditions=[True, True], numRequired=1)
    return loadLeft, loadRight, finLeft, finRight, sv
