#import matplotlib
#matplotlib.use('gtkagg')
import wifisIO
#import wifisSlices as slices
#import matplotlib.pyplot as plt
import numpy as np
#from matplotlib.backends.backend_pdf import PdfPages
import os
#import warnings
import wifisQuickReduction as quickReduction

#REQUIRED INPUT
varFile = 'wifisConfig.inp'

#initialize variables using configuration file
varInp = wifisIO.readInputVariables(varFile)
for var in varInp:
    globals()[var[0]]=var[1]            

waveLst = wifisIO.readAsciiList(waveLstFile)
if waveLst.ndim==0:
    waveLst = np.asarray([waveLst])

flatLst = wifisIO.readAsciiList(flatLstFile)    
if flatLst.ndim == 0:
    flatLst = np.asarray([flatLst])

for i in range(len(waveLst)):
    if len(waveLst)>1:
        quickReduction.procArcData(waveLst[i], flatLst[i], colorbarLims=None, hband=hband, varFile=varFile, noPlot=True)
    else:
        quickReduction.procArcData(waveLst[i], flatLst[i], colorbarLims=None, hband=hband, varFile=varFile, noPlot=False)

