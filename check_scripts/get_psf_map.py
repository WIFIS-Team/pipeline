import matplotlib
matplotlib.use('gtkagg')
import wifisQuickReduction as quickRed
import wifisIO
import os
import numpy as np

#initialize variables
varFile = 'wifisConfig.inp'

varInp = wifisIO.readInputVariables(varFile)

for var in varInp:
    globals()[var[0]]=var[1]    

#execute pyOpenCL section here
os.environ['PYOPENCL_COMPILER_OUTPUT'] = pyCLCompOut

if len(pyCLCTX)>0:
    os.environ['PYOPENCL_CTX'] = pyCLCTX 


obsLst = wifisIO.readAsciiList(obsLstFile)
if os.path.exists(skyLstFile):
    skyLst = wifisIO.readAsciiList(skyLstFile)
else:
    skyLst = None
    
if obsLst.ndim == 0:
    obsLst = np.asarray([obsLst])

if len(obsLst)>1:
    noPlot=True
else:
    noPlot=False

for i in range(len(obsLst)):
    if skyLst is not None:
        quickRed.getPSFMap(rampFolder=obsLst[i], skyFolder=skyLst[i], varFile=varFile, noPlot=noPlot)
    else:
        quickRed.getPSFMap(rampFolder=obsLst[i], varFile=varFile, noPlot=noPlot)

