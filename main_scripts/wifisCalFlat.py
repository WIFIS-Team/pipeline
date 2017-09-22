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
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0' # Used to show compile errors for debugging, can be removed
os.environ['PYOPENCL_CTX'] = '2' # Used to specify which OpenCL device to target. Should be uncommented and pointed to correct device to avoid future interactive requests

#*****************************************************************************
#******************************* Required input ******************************

flatListFile = 'flat.lst' # a simple ascii file containing a list of the folder names that contain the ramp data
darkListFile = '' #None # list of processed dark ramps

hband = False

#pipeline location and RAW data location
rootFolder = '/data/WIFIS/H2RG-G17084-ASIC-08-319'
pipelineFolder = '/data/pipeline/'

#location of calibration files
nlFile = pipelineFolder + 'external_data/master_detLin_NLCoeff.fits' # the non-linearity correction coefficients file        
satFile = pipelineFolder+'external_data/master_detLin_satCounts.fits' # the saturation limits file
bpmFile = pipelineFolder+'external_data/bpm.fits' # the bad pixel mask

if hband:
    distMapLimitsFile = '/home/jason/wifis/data/ronchi_map_june/hband/processed/20170607221050_flat_limits.fits'
else:
    #distMapLimitsFile = pipelineFolder+'external_data/distMap_limits.fits'
    distMapLimitsFile = '/home/jason/wifis/data/ronchi_map_august/tb/processed/20170831210255_flat_limits.fits'
    
#distMapLimitsFile = ''
#optional behaviour of pipeline
plot = True #whether to plot the traces
crReject = False
skipObsinfo = False
winRng = 51
imgSmth = 5
limSmth = 20
polyFitDegree=2
dispAxis=0

flatSmooth=0
flatCutOff = 0.1
#parameters used for processing of ramps
nChannel=32 #specifies the number of channels used during readout of detector
bpmCorRng=20 #specifies the maximum separation of pixel search to use during bad pixel correction
nRowsAvg=0 # specifies the number of rows of reference pixels to use to correct for row bias (+/- nRowAvg)
rowSplit=1 # specifies how many processing steps to use during reference row correction. Must be integer multiple of number of frames. For very long ramps, use a higher number to avoid OpenCL issues and/or high memory consumption.
nlSplit=32 #specifies how many processing steps to use during non-linearity correction. Must be integer multiple of detector width. For very long ramps, use a higher number to avoid OpenCL issues and/or high memory consumption. 
combSplit=32 #specifies how many processing steps to use during creation of ramp image. Must be integer multiple of detector width. For very long ramps, use a higher number to avoid OpenCL issues and/or high memory consumption.
gain = 1.
ron = 1.

obsCoords = [-111.600444444,31.9629166667,2071]

#*****************************************************************************
#*****************************************************************************

logfile = open('wifis_reduction_log.txt','a')
logfile.write('********************\n')
logfile.write(time.strftime("%c")+'\n')
logfile.write('Processing flatfield files with WIFIS pyPline\n')
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
    

if (darkListFile is not None) and os.path.exists(darkListFile):
    darkLst = wifisIO.readAsciiList(darkListFile)
    
    if darkLst.ndim == 0:
        darkLst = [darkLst]
    else:
        darkLst = None
else:
    darkLst = None
    logfile.write('*** WARNING: No darks provided or dark list ' + str(darkListFile) +' does not exist ***\n')

#read file list
if os.path.exists(flatListFile):
    flatLst= wifisIO.readAsciiList(flatListFile)
else:
    logfile.write('*** FAILURE: Flat file list ' + str(flatListFile) +' does not exist ***\n')
    raise Warning('*** Flat file list ' + flatListFile + ' does not exist ***')

if flatLst.ndim == 0:
    flatLst = np.asarray([flatLst])

logfile.write('\n')
calFlat.runCalFlat(flatLst, hband=hband, darkLst=darkLst, rootFolder=rootFolder, nlCoef=nlCoef, satCounts=satCounts, BPM=BPM, distMapLimitsFile=distMapLimitsFile, plot=plot, nChannel=nChannel, nRowsAvg=nRowsAvg, rowSplit=rowSplit, nlSplit=nlSplit, combSplit=combSplit, bpmCorRng=bpmCorRng, crReject=crReject, skipObsinfo=skipObsinfo, imgSmth=imgSmth, polyFitDegree=polyFitDegree, avgRamps=True, nlFile=nlFile, satFile=satFile, bpmFile=bpmFile, flatCutOff=flatCutOff, logfile=logfile, winRng=winRng, dispAxis=dispAxis, limSmth=limSmth, obsCoords=obsCoords)

logfile.write('********************\n')
logfile.write('\n')

logfile.close()
