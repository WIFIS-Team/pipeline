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
#INPUT VARIABLE FILE NAME
varFile = 'wifisConfig.inp'

#*****************************************************************************
#************************** user input ***********************************
#initialize all variables.
#DO NOT CHANGE VALUES HERE, EDIT THE 'variables.inp' FILE, WHICH OVERWRITES VALUES HERE

waveLstFile = 'wave.lst' 
flatLstFile = 'flat.lst'
hband = False
rootFolder = '/data/WIFIS/H2RG-G17084-ASIC-08-319'
pipelineFolder = '/data/pipeline/'
templateFile = pipelineFolder+'external_data/waveTemplate.fits'
prevResultsFile = pipelineFolder+'external_data/waveTemplateFittingResults.pkl'
distMapFile = pipelineFolder+'external_data/distMap.fits'
spatGridPropsFile = pipelineFolder+'external_data/distMap_spatGridProps.dat'
distMapLimitsFile = pipelineFolder+'external_data/distMap_limits.fits'
darkFile = None
mxOrder=3
nlFile = pipelineFolder + 'external_data/master_detLin_NLCoeff.fits' # the non-linearity correction coefficients file        
satFile = pipelineFolder+'external_data/master_detLin_satCounts.fits' # the saturation limits file
bpmFile = pipelineFolder+'external_data/bpm.fits' # the bad pixel mask   
atlasFile = pipelineFolder+'external_data/best_lines2.dat'
#optional behaviour
plot = True
crReject = False
cleanDispSol = True
skipObsinfo = False
#flat field specific options
flatWinRng = 51
flatImgSmth = 5
flatPolyFitDegree=2
cleanDispThresh = 1.5
waveTrimThresh = 0
sigmaClipRounds=2 #number of iterations when sigma-clipping of dispersion solution
sigmaClip = 2 #sigma-clip cutoff when sigma-clipping dispersion solution
sigmaLimit= 3 #relative noise limit (x * noise level) for which to reject lines

#parameters used for processing of ramps
nChannel=32 #specifies the number of channels used during readout of detector
flatbpmCorRng=20 #specifies the maximum separation of pixel search to use during bad pixel correction
nRowsAvg=4 # specifies the number of rows of reference pixels to use to correct for row bias (+/- nRowsAvg)
nRowSplit=1 # specifies how many processing steps to use during reference row correction. Must be integer multiple of number of frames. For very long ramps, use a higher number to avoid OpenCL issues and/or high memory consumption.
nlSplit=32 #specifies how many processing steps to use during non-linearity correction. Must be integer multiple of detector width. For very long ramps, use a higher number to avoid OpenCL issues and/or high memory consumption. 
nCombSplit=32 #specifies how many processing steps to use during creation of ramp image. Must be integer multiple of detector width. For very long ramps, use a higher number to avoid OpenCL issues and/or high memory consumption.
dispAxis=0
obsCoords = [-111.600444444,31.9629166667,2071]

#*****************************************************************************
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
    darkLst = wifisIO.readAsciiList(darkListFile)[0]
    darkLst = darkLst[:2]
else:
    darkLst = None

if not (os.path.exists(atlasFile)):
    raise Warning('*** Cannot continue, line atlas file ' + atlasFile + ' does not exist. Please provide the necessary atlas file***')

if not (os.path.exists(distMapFile)):
    raise Warning('*** Cannot continue, distorion map file ' + distMapFile + ' does not exist. Please process a Ronchi calibration sequence or provide the necessary file ***')

if not (os.path.exists(spatGridPropsFile)):
    raise Warning ('*** Cannot continue, spatial propertites grid file ' + spatGridPropsFile + ' does not exist. Please process a Ronchi calibration sequence or provide the necessary file ***')

#create processed directory, in case it doesn't exist
wifisIO.createDir('processed')
wifisIO.createDir('quality_control')

calWave.runCalWave(waveLst, flatLst, hband=hband, darkLst=darkLst, rootFolder=rootFolder, nlCoef=nlCoef, satCounts=satCounts, BPM=BPM, distMapLimitsFile=distMapLimitsFile, plot=plot, nChannel=nChannel, nRowsAvg=nRowsAvg, rowSplit=nRowSplit, nlSplit=nlSplit, combSplit=nCombSplit, bpmCorRng=flatbpmCorRng, crReject=crReject, skipObsinfo=skipObsinfo, flatWinRng=flatWinRng,flatImgSmth=flatImgSmth, flatPolyFitDegree=3, distMapFile=distMapFile, spatGridPropsFile=spatGridPropsFile, atlasFile=atlasFile, templateFile=templateFile, prevResultsFile=prevResultsFile,  sigmaClip=sigmaClip, sigmaClipRounds=sigmaClipRounds, sigmaLimit=sigmaLimit, cleanDispSol=cleanDispSol,cleanDispThresh = cleanDispThresh, waveTrimThresh=waveTrimThresh,nlFile=nlFile,satFile=satFile,bpmFile=bpmFile, obsCoords=obsCoords, dispAxis=dispAxis, darkFile=darkFile, logfile=logfile, mxOrder=mxOrder)
    
logfile.write('\n')
logfile.close()
