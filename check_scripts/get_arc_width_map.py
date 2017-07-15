import matplotlib
matplotlib.use('gtkagg')
import wifisIO
import wifisSlices as slices
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import os
import warnings
import wifisQuickReduction as quickReduction

os.environ['PYOPENCL_CTX'] = '1' # Used to specify which OpenCL device to target. Should be uncommented and pointed to correct device to avoid future interactive requests

#****************************************************************************************
#REQUIRED INPUT FILES
waveLstFile = 'wave.lst'
flatLstFile = 'flat.lst'

hband = False
colorbarLims = [0,10]

#****************************************************************************************

waveLst = wifisIO.readAsciiList(waveLstFile)
if waveLst.ndim==0:
    waveLst = np.asarray([waveLst])

flatLst = wifisIO.readAsciiList(flatLstFile)    
if flatLst.ndim == 0:
    flatLst = np.asarray([flatLst])

quickReduction.initPaths()
for fle in range(len(waveLst)):
    quickReduction.procArcData(waveLst[fle], flatLst[fle], colorbarLims=colorbarLims)
    
