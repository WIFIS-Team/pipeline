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

os.environ['PYOPENCL_CTX'] = '1' # Used to specify which OpenCL device to target. Should be uncommented and pointed to correct device to avoid future interactive requests

#****************************************************************************************
#REQUIRED INPUT FILES
ronchiLstFile = 'ronchi.lst'
flatLstFile = 'flat.lst'

hband = False

#****************************************************************************************

ronchiLst = wifisIO.readAsciiList(ronchiLstFile)
if ronchiLst.ndim==0:
    ronchiLst = np.asarray([ronchiLst])

flatLst = wifisIO.readAsciiList(flatLstFile)    
if flatLst.ndim == 0:
    flatLst = np.asarray([flatLst])

quickReduction.initPaths()
for fle in range(len(ronchiLst)):
    quickReduction.procRonchiData(ronchiLst[fle], flatLst[fle], colorbarLims=None)
    
