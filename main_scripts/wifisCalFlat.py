"""

Calibrate flat field images

Produces:
- master flat image
- slitlet traces
- ??

"""
import matplotlib
matplotlib.use('gtkagg')
import matplotlib.pyplot as plt
import os
import wifisIO 
import warnings
import wifisCalFlatFunc as calFlat
import numpy as np
import time
import colorama

colorama.init()

#******************************* Required input ******************************
#INPUT VARIABLE FILE NAME
varFile = 'wifisConfig.inp'
#*****************************************************************************

logfile = open('wifis_reduction_log.txt','a')
logfile.write('********************\n')
logfile.write(time.strftime("%c")+'\n')
logfile.write('Processing flatfield files with WIFIS pyPline\n')

print('Reading input variables from file ' + varFile)
logfile.write('Reading input variables from file ' + varFile)

varInp = wifisIO.readInputVariables(varFile)
for var in varInp:
    locals()[var[0]]=var[1]
  
#execute pyOpenCL section here
os.environ['PYOPENCL_COMPILER_OUTPUT'] = pyCLCompOut 
os.environ['PYOPENCL_CTX'] = pyCLCTX 

logfile.write('Root folder containing raw data: ' + str(rootFolder)+'\n')

#first check if required input exists
print('Reading in calibration files')

#open calibration files
if os.path.exists(nlFile):
    nlCoef = wifisIO.readImgsFromFile(nlFile)[0]
    logfile.write('Using non-linearity corrections from file:\n')
    logfile.write(nlFile+'\n')
else:
    nlCoef =None
    print(colorama.Fore.RED+'*** WARNING: No non-linearity coefficient array provided, corrections will be skipped ***' + colorama.Style.RESET_ALL)

    logfile.write('*** WARNING: No non-linearity corrections file provided or file ' + str(nlFile) +' does not exist ***\n')

if os.path.exists(satFile):
    satCounts = wifisIO.readImgsFromFile(satFile)[0]
    logfile.write('Using saturation limits from file:\n')
    logfile.write(satFile+'\n')
 
else:
    satCounts = None
    print(colorama.Fore.RED+'*** WARNING: No saturation counts array provided and will not be taken into account ***'+colorama.Style.RESET_ALL)

    logfile.write('*** WARNING: No saturation counts file provided or file ' + str(satFile) +' does not exist ***\n')
            

if (os.path.exists(bpmFile)):
    BPM = wifisIO.readImgsFromFile(bpmFile)[0]
    logfile.write('Using bad pixel mask from file:\n')
    logfile.write(bpmFile+'\n')
else:
    BPM = None
    logfile.write('*** WARNING: No bad pixel mask provided or file ' + str(bpmFile) +' does not exist ***\n')
    

if (darkFile is not None) and os.path.exists(darkFile):
    darkLst = wifisIO.readAsciiList(darkFile)
    
    if darkLst.ndim == 0:
        darkLst = [darkLst]
    else:
        darkLst = None
else:
    darkLst = None
    logfile.write('*** WARNING: No darks provided or dark list ' + str(darkFile) +' does not exist ***\n')

#read file list
if os.path.exists(flatLstFile):
    flatLst= wifisIO.readAsciiList(flatLstFile)
else:
    logfile.write('*** FAILURE: Flat file list ' + str(flatLstFile) +' does not exist ***\n')
    raise Warning('*** Flat file list ' + flatLstFile + ' does not exist ***')

if flatLst.ndim == 0:
    flatLst = np.asarray([flatLst])

logfile.write('\n')
calFlat.runCalFlat(flatLst, hband=hband, darkLst=darkLst, rootFolder=rootFolder, nlCoef=nlCoef, satCounts=satCounts, BPM=BPM, distMapLimitsFile=distMapLimitsFile, plot=True, nChannel=nChannel, nRowsAvg=nRowsAvg, rowSplit=nRowSplitFlat, nlSplit=nlSplit, combSplit=nCombSplit, bpmCorRng=flatbpmCorRng, crReject=crReject, skipObsinfo=skipObsinfo, imgSmth=flatImgSmth, polyFitDegree=flatPolyFitDegree,nlFile=nlFile, satFile=satFile, bpmFile=bpmFile, flatCutOff=flatCutOff, logfile=logfile, winRng=flatWinRng, dispAxis=dispAxis, limSmth=flatLimSmth, obsCoords=obsCoords, darkFile=darkFile)

logfile.write('********************\n')
logfile.write('\n')

logfile.close()
