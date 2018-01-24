"""

Main script used to process raw flat field data

Input:
- Uses input as defined in configuration file denoted by variable varFile

Produces:
- for each observation listed:
- flat field ramp image (XXX_flat.fits)
- limits corresponding to the edges of all identified slices (XXX_flat_limits.fits)
- multi-extension image including all extracted slices
- multi-extension image containing the normalized response function of each slice


"""

#change the next two lines as needed
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
if len(pyCLCTX)>0:
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
    darkLst = wifisIO.readImgsFromFile(darkFile)[0]

    #if len returns more than 3, assume it is a single image
    if len(darkLst) > 3:
        darkLst = [darkLst]
else:
    darkLst = None
    logfile.write('*** WARNING: No dark provided or dark file ' + str(darkFile) +' does not exist ***\n')

#prioritize RON file over RON from associated dark?
if os.path.exists(ronFile):
    RON = wifisIO.readImgsFromFile(ronFile)[0]
    logfile.write('Using RON file:\n')
    logfile.write(ronFile+'\n')
elif darkFile is not None and os.path.exists(darkFile):
    RON = wifisIO.readImgsFromFile(darkFile.strip('.fits')+'_RON.fits')[0]
    logfile.write('Using RON file:\n')
    logfile.write(darkFile.strip('.fits')+'_RON.fits\n')

else:
    RON = None
    logfile.write('*** WARNING: No RON file provided, or ' + str(ronFile) +' does not exist ***\n')

    
#read file list
if os.path.exists(flatLstFile):
    flatLst= wifisIO.readAsciiList(flatLstFile)
else:
    logfile.write('*** FAILURE: Flat file list ' + str(flatLstFile) +' does not exist ***\n')
    raise Warning('*** Flat file list ' + flatLstFile + ' does not exist ***')

if flatLst.ndim == 0:
    flatLst = np.asarray([flatLst])

logfile.write('\n')
calFlat.runCalFlat(flatLst, hband=hband, darkLst=darkLst, rootFolder=rootFolder, nlCoef=nlCoef, satCounts=satCounts, BPM=BPM, distMapLimitsFile=distMapLimitsFile, plot=True, nChannel=nChannel, nRowsAvg=nRowsAvg, rowSplit=nRowSplitFlat, nlSplit=nlSplit, combSplit=nCombSplit, bpmCorRng=flatbpmCorRng, crReject=crReject, skipObsinfo=skipObsinfo, imgSmth=flatImgSmth, polyFitDegree=limitsPolyFitDegree,nlFile=nlFile, satFile=satFile, bpmFile=bpmFile, flatCutOff=flatCutOff, logfile=logfile, winRng=flatWinRng, dispAxis=dispAxis, limSmth=flatLimSmth, obsCoords=obsCoords, darkFile=darkFile, ron=RON,centGuess=centGuess, flatCor=flatCor, flatCorFile=flatCorFile)

logfile.write('********************\n')
logfile.write('\n')

logfile.close()
