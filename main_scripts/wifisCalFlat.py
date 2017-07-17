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

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0' # Used to show compile errors for debugging, can be removed
os.environ['PYOPENCL_CTX'] = '1' # Used to specify which OpenCL device to target. Should be uncommented and pointed to correct device to avoid future interactive requests

#*****************************************************************************
#******************************* Required input ******************************

flatListFile = 'dome.lst' # a simple ascii file containing a list of the folder names that contain the ramp data
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
    distMapLimitsFile = pipelineFolder+'external_data/distMap_limits.fits'

distMapLimitsFile = ''
#optional behaviour of pipeline
plot = True #whether to plot the traces
crReject = False
skipObsinfo = False
winRng = 51
imgSmth = 5
polyFitDegree=2

#parameters used for processing of ramps
nChannel=32 #specifies the number of channels used during readout of detector
bpmCorRng=20 #specifies the maximum separation of pixel search to use during bad pixel correction
nRowAverage=4 # specifies the number of rows of reference pixels to use to correct for row bias (+/- nRowAverage)
rowSplit=1 # specifies how many processing steps to use during reference row correction. Must be integer multiple of number of frames. For very long ramps, use a higher number to avoid OpenCL issues and/or high memory consumption.
nlSplit=32 #specifies how many processing steps to use during non-linearity correction. Must be integer multiple of detector width. For very long ramps, use a higher number to avoid OpenCL issues and/or high memory consumption. 
combSplit=32 #specifies how many processing steps to use during creation of ramp image. Must be integer multiple of detector width. For very long ramps, use a higher number to avoid OpenCL issues and/or high memory consumption.

#*****************************************************************************
#*****************************************************************************

#first check if required input exists
print('Reading in calibration files')

#open calibration files
if os.path.exists(nlFile):
    nlCoef = wifisIO.readImgsFromFile(nlFile)[0]
else:
    nlCoef =None
    warnings.warn('*** No non-linearity coefficient array provided, corrections will be skipped ***')

if os.path.exists(satFile):
    satCounts = wifisIO.readImgsFromFile(satFile)[0]
else:
    satCounts = None
    warnings.warn('*** No saturation counts array provided and will not be taken into account ***')
        
if (os.path.exists(bpmFile)):
    BPM = wifisIO.readImgsFromFile(bpmFile)[0]
else:
    BPM = None

if (darkListFile is not None) and os.path.exists(darkListFile):
    darkLst = wifisIO.readAsciiList(darkListFile)
    
    if darkLst.ndim == 0:
        darkLst = [darkLst]
    else:
        darkLst = None
else:
    darkLst = None

#read file list
if os.path.exists(flatListFile):
    flatLst= wifisIO.readAsciiList(flatListFile)
else:
    raise Warning('*** Flat file list ' + flatListFile + ' does not exist ***')

if nlCoef is None:
    warnings.warn('*** No non-linearity coefficient array provided, corrections will be skipped ***')

if satCounts is None:
    warnings.warn('*** No saturation counts array provided and will not be taken into account ***')

if flatLst.ndim == 0:
    flatLst = np.asarray([lst])

calFlat.runCalFlat(flatLst, hband=hband, darkLst=darkLst, rootFolder=rootFolder, nlCoef=nlCoef, satCounts=satCounts, BPM=BPM, distMapLimitsFile=distMapLimitsFile, plot=plot, nChannel=nChannel, nRowAverage=nRowAverage, rowSplit=rowSplit, nlSplit=nlSplit, combSplit=combSplit, bpmCorRng=bpmCorRng, crReject=crReject, skipObsinfo=skipObsinfo, imgSmth=imgSmth, polyFitDegree=2, avgRamps=True)

    
