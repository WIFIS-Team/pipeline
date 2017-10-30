"""

Calibrates arc lamp images

Requires:
- 

Produces:
- per pixel wavelength solution


"""

import matplotlib
matplotlib.use('gtkagg')
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import wifisIO 
import wifisWaveSol as waveSol
import wifisUncertainties
import wifisSlices as slices
import wifisHeaders as headers
import wifisProcessRamp as processRamp
import wifisCreateCube as createCube
from matplotlib.backends.backend_pdf import PdfPages
import warnings
import wifisCalWaveFunc as calWave
import colorama

colorama.init()

#************************** user input ***********************************
#INPUT VARIABLE FILE NAME
varFile = 'wifisConfig.inp'

#*****************************************************************************

logfile = open('wifis_reduction_log.txt','a')
logfile.write('********************\n')
logfile.write(time.strftime("%c")+'\n')
logfile.write('Processing flatfield files with WIFIS pyPline\n')

print('Reading input variables from file ' + varFile)
logfile.write('Reading input variables from file ' + varFile+'\n')
varInp = wifisIO.readInputVariables(varFile)
for var in varInp:
    locals()[var[0]]=var[1]

#execute pyOpenCL section here
os.environ['PYOPENCL_COMPILER_OUTPUT'] = pyCLCompOut
if pyCLCTX:
    os.environ['PYOPENCL_CTX'] = pyCLCTX 

logfile.write('Root folder containing raw data: ' + str(rootFolder)+'\n')

print('Reading in calibration files')

#first check if required input exists

#read file list
if os.path.exists(waveLstFile):
    waveLst= wifisIO.readAsciiList(waveLstFile)
    if waveLst.ndim ==0:
        waveLst = np.asarray([waveLst])
else:
    logfile.write('*** FAILURE: Wave file list ' + waveLstFile + ' does not exist ***')
    raise Warning('*** Wave file list ' + waveLstFile + ' does not exist ***')

if os.path.exists(flatLstFile):
    flatLst= wifisIO.readAsciiList(flatLstFile)
    if flatLst.ndim ==0:
        flatLst = np.asarray([flatLst])
else:
    raise Warning('*** Flat file list ' + flatLstFile + ' does not exist ***')

if len(waveLst) != len(flatLst):
    raise Warning('*** Length of wave file list is different from flat file list ***')

#open calibration files
if os.path.exists(nlFile):
    nlCoef = wifisIO.readImgsFromFile(nlFile)[0]
    logfile.write('Using non-linearity corrections from file:\n')
    logfile.write(nlFile+'\n')
else:
    nlCoef =None
    print(colorama.Fore.RED+'*** WARNING: No non-linearity coefficient array provided, corrections will be skipped *** + colorama.Style.RESET_ALL')

    logfile.write('*** WARNING: No non-linearity corrections file provided or file ' + str(nlFile) +' does not exist ***\n')
    
if os.path.exists(satFile):
    satCounts = wifisIO.readImgsFromFile(satFile)[0]
    logfile.write('Using saturation limitts from file:\n')
    logfile.write(satFile+'\n')

else:
    satCounts = None
    print(colorama.Fore.RED+'*** WARNING: No saturation counts array provided and will not be taken into account ***' + colorama.Style.RESET_ALL)

    logfile.write('*** WARNING: No saturation counts file provided or file ' + str(satFile) +' does not exist ***\n')

if (os.path.exists(bpmFile)):
    BPM = wifisIO.readImgsFromFile(bpmFile)[0]
else:
    BPM = None

if (darkFile is not None) and os.path.exists(darkFile):
    darkLst = wifisIO.readImgsFromFile(darkFile)[0]
    if len(darkLst) >3:
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

if not (os.path.exists(atlasFile)):
    raise Warning('*** Cannot continue, line atlas file ' + atlasFile + ' does not exist. Please provide the necessary atlas file***')

if not (os.path.exists(distMapFile)):
    raise Warning('*** Cannot continue, distorion map file ' + distMapFile + ' does not exist. Please process a Ronchi calibration sequence or provide the necessary file ***')

if not (os.path.exists(spatGridPropsFile)):
    raise Warning ('*** Cannot continue, spatial propertites grid file ' + spatGridPropsFile + ' does not exist. Please process a Ronchi calibration sequence or provide the necessary file ***')

#create processed directory, in case it doesn't exist
wifisIO.createDir('processed')
wifisIO.createDir('quality_control')

calWave.runCalWave(waveLst, flatLst, hband=hband, darkLst=darkLst, rootFolder=rootFolder, nlCoef=nlCoef, satCounts=satCounts, BPM=BPM, distMapLimitsFile=distMapLimitsFile, plot=True, nChannel=nChannel, nRowsAvg=nRowsAvg, rowSplit=nRowSplitFlat, nlSplit=nlSplit, combSplit=nCombSplit, bpmCorRng=bpmCorRng, crReject=crReject, skipObsinfo=skipObsinfo, flatWinRng=flatWinRng,flatImgSmth=flatImgSmth, limitsPolyFitDegree=limitsPolyFitDegree, distMapFile=distMapFile, spatGridPropsFile=spatGridPropsFile, atlasFile=atlasFile, templateFile=waveTempFile, prevResultsFile=waveTempResultsFile,  sigmaClip=sigmaClip, sigmaClipRounds=sigmaClipRounds, sigmaLimit=sigmaLimit, cleanDispSol=cleanDispSol,cleanDispThresh = cleanDispThresh, waveTrimThresh=waveTrimThresh,nlFile=nlFile,satFile=satFile,bpmFile=bpmFile, obsCoords=obsCoords, dispAxis=dispAxis, darkFile=darkFile, logfile=logfile, mxOrder=mxOrder, waveSmooth=waveSmooth, waveSolMP=waveSolMP,waveSolPlot=waveSolPlot, winRng=waveWinRng, nRowSplitFlat=nRowSplitFlat, gain=gain, ron=RON,flatbpmCorRng=flatbpmCorRng, mxCcor=waveMxCcor, adjustFitWin=waveAdjustFitWin)

logfile.write('********************\n')
logfile.write('\n')
logfile.close()
