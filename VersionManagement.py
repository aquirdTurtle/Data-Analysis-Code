__version__ = "1.0"
from MainAnalysis import __version__ as mainAnalysisVersion
from AnalysisHelpers import __version__ as AnalysisHelpersVersion
from BokehPlotters import __version__ as bokehPlottersVersion
from FittingFunctions import __version__ as fittingFunctionsVersion
from MarksConstants import __version__ as MarksConstantsVersion
from MarksFourierAnalysis import __version__ as MarksFourierAnalysisVersion
from MatplotlibPlotters import __version__ as matplotlibPlottersVersion
from Miscellaneous import __version__ as miscellaneousVersion
from PlotlyPlotters import __version__ as plotlyPlotterVersion
from pandas import DataFrame


def getVersions():
    versions = DataFrame()
    versions['PrintVersions;'] = [__version__]
    versions['mainAnalysis:'] = [mainAnalysisVersion]
    versions['AnalysisHelpers:'] = [AnalysisHelpersVersion]
    versions['bokehPlotters:'] = [bokehPlottersVersion]
    versions['fittingFunctions:'] = [fittingFunctionsVersion]
    versions['MarksConstants:'] = [MarksConstantsVersion]
    versions['MarksFourierAnalysis:'] = [MarksFourierAnalysisVersion]
    versions['matplotlibPlotters:'] = [matplotlibPlottersVersion]
    versions['miscellaneous:'] = [miscellaneousVersion]
    versions['plotlyPlotters:'] = [plotlyPlotterVersion]
    versions = versions.transpose()
    versions.columns = ['Version']
    return versions
