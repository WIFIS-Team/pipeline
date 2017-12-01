import matplotlib
matplotlib.use('gtkagg')
import wifisIO
import wifisSlices as slices
import wifisSpatialCor as spatialCor
import matplotlib.pyplot as plt
import numpy as np
import wifisBadPixels as badPixels
import wifisCreateCube as createCube
from matplotlib.backends.backend_pdf import PdfPages
import os
import wifisProcessRamp as processRamp
import warnings
import wifisQuickReduction as quickReduction

#****************************************************************************************
#REQUIRED INPUT FILES
varFile = 'wifisConfig.inp'

varInp = wifisIO.readInputVariables(varFile)
for var in varInp:
    globals()[var[0]]=var[1]    

#over-ride variables here
noFlat = True
#****************************************************************************************

ronchiLst = wifisIO.readAsciiList(ronchiFile)
if ronchiLst.ndim==0:
    ronchiLst = np.asarray([ronchiLst])

flatLst = wifisIO.readAsciiList(flatLstFile)    
if flatLst.ndim == 0:
    flatLst = np.asarray([flatLst])

    
for i in range(len(ronchiLst)):
    if len(ronchiLst) > 1:
        quickReduction.procRonchiData(ronchiLst[i], flatLst[i], colorbarLims=None, noPlot=True, varFile=varFile, noFlat=noFlat)
    else:
        quickReduction.procRonchiData(ronchiLst[i], flatLst[i], colorbarLims=None, noPlot=False, varFile=varFile, noFlat=noFlat)

        
