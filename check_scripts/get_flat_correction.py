"""

Script used to create a flat field correction file to correct the calibration flats to match the dome flats. Due to flexure issues, a new correction file needs to be associated with every Ronchi image.

"""

import matplotlib
matplotlib.use('gtkagg')
import matplotlib.pyplot as plt
import wifisIO
import colorama
import time
import os
from astropy.visualization import ZScaleInterval
import wifisProcessRamp as processRamp
import wifisCalFlatFunc as calFlat
import warnings
import numpy as np
import wifisSlices as slices
from matplotlib.backends.backend_pdf import PdfPages

t0 = time.time()

#*****************************************************************************
#************************** user input ******************************
varFile = 'wifisConfig.inp'

logfile = open('wifis_reduction_log.txt','a')
logfile.write('******************************\n')
logfile.write(time.strftime("%c")+'\n')
logfile.write('Processing science observations with WIFIS pyPline\n')
print('Reading input variables from file ' + varFile)
logfile.write('Reading input variables from file ' + varFile)

varInp = wifisIO.readInputVariables(varFile)
for var in varInp:
    locals()[var[0]]=var[1]

#*****************************************************************************

#execute pyOpenCL section here
os.environ['PYOPENCL_COMPILER_OUTPUT'] = pyCLCompOut
if len(pyCLCTX) >0:
    os.environ['PYOPENCL_CTX'] = pyCLCTX 

interval=ZScaleInterval() 

#read in input data

if os.path.exists(flatLstFile):
    print('Reading flat field input file ' + flatLstFile)
    logfile.write('Reading flat field input file ' + flatLstFile + '\n')
    flatFolder = wifisIO.readAsciiList(flatLstFile).tostring()
else:
    print(colorama.Fore.RED+'*** WARNING: flat field input file ' + flatLstFile + ' does not exist ***'+colorama.Style.RESET_ALL)
    logfile.write('*** WARNING: flat field input file ' + flatLstFile + ' does not exist\n')
    raise Warning('*** Cannot continue, flat field input file does not exist ***')

if os.path.exists(domeFile):
    print('Reading dome flat field input file ' + domeFile)
    logfile.write('Reading dome flat field input file ' + domeFile + '\n')
    domeFolder = wifisIO.readAsciiList(domeFile).tostring()
else:
    print(colorama.Fore.RED+'*** WARNING: dome flat field input file ' + domeFile + ' does not exist ***'+colorama.Style.RESET_ALL)
    logfile.write('*** WARNING: dome flat field input file ' + domeFile + ' does not exist\n')
    raise Warning('*** Cannot continue, dome flat field input file does not exist ***')        

if os.path.exists(domeBackFile):
    print('Reading dome flat field background emission input file ' + domeBackFile)
    logfile.write('Reading dome flat field background emission input file ' + domeBackFile + '\n')
    backFolder = wifisIO.readAsciiList(domeBackFile).tostring()
else:
    print(colorama.Fore.RED+'*** WARNING: dome flat field background emission input file ' + domeBackFile + ' does not exist. No background subtraction will occur ***'+colorama.Style.RESET_ALL)
    logfile.write('*** WARNING: dome flat field background emission input file ' + domeBackFile + ' does not exist. No background subtraction will occur\n')
  
if (darkFile is not None) and os.path.exists(darkFile):
    darkLst = wifisIO.readImgsFromFile(darkFile)[0]

    #if len returns more than 3, assume it is a single image
    if len(darkLst) > 3:
        darkLst = [darkLst]
else:
    darkLst = None
    logfile.write('*** WARNING: No dark provided or dark file ' + str(darkFile) +' does not exist. No dark will be subtracted from the calibration flats ***\n')


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

if not (os.path.exists(distMapLimitsFile)):
    logfile.write('*** FAILURE: Cannot continue, distorion map limits file ' + distMapLimitsFile + ' does not exist. Please process a Ronchi calibration sequence or provide the necessary file ***\n')
    raise Warning('*** Cannot continue, distorion map limits file ' + distMapLimitsFile + ' does not exist. Please process a Ronchi calibration sequence or provide the necessary file ***')
else:
    distMapLimits = wifisIO.readImgsFromFile(distMapLimitsFile)[0]
    logfile.write('Using distortion/spatial mapping limits file from file:\n')
    logfile.write(distMapLimitsFile+'\n')

logfile.write('\n')

#create processed directory, in case it doesn't exist
wifisIO.createDir('processed')
wifisIO.createDir('quality_control')

#open calibration files
if os.path.exists(nlFile):
    nlCoef = wifisIO.readImgsFromFile(nlFile)[0]
    logfile.write('Using non-linearity corrections from file:\n')
    logfile.write(nlFile+'\n')
else:
    nlCoef =None 
    print(colorama.Fore.RED+'*** WARNING: No non-linearity coefficient array provided, corrections will be skipped ***'+colorama.Style.RESET_ALL)
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
else:
    BPM = None
    
#now process data
print('Processing calibration flats')
logfile.write('Processing calibration flats\n')
calFlat.runCalFlat(np.asarray([flatFolder]), hband=hband, darkLst = darkLst, rootFolder=rootFolder, nlCoef=nlCoef, satCounts=satCounts, BPM = BPM, distMapLimitsFile = distMapLimitsFile, plot=True, nChannel = nChannel, nRowsAvg=nRowsAvg,rowSplit=nRowSplitFlat,nlSplit=nlSplit, combSplit=nCombSplit,bpmCorRng=flatbpmCorRng, crReject=False, skipObsinfo=False,nlFile=nlFile, bpmFile=bpmFile, satFile=satFile,darkFile=darkFile, logfile=logfile, ask=False, obsCoords=obsCoords, limSmth=flatLimSmth, flatCutOff=flatCutOff, gain=gain, ron=RON, polyFitDegree=limitsPolyFitDegree)

flatLst, flatHdr = wifisIO.readImgsFromFile('processed/'+flatFolder+'_flat.fits')
flat = flatLst[0]
flatSig = flatLst[1]

print('Reading slice limits')
logfile.write('Reading slice limits from file:\n')
logfile.write('processed/'+flatFolder+'_flat_limits.fits\n')

limits, limitsHdr = wifisIO.readImgsFromFile('processed/'+flatFolder+'_flat_limits.fits')
shft = limitsHdr['LIMSHIFT']

cont='n'
if os.path.exists('processed/'+domeFolder+'_dome_flat.fits'):
    cont = wifisIO.userInput('Processed files already exists for ' +domeFolder+', do you want to reprocess (y/n)?')
else:
    cont='y'

if cont=='y':
    print('Processing dome flats')
    logfile.write('Processing dome flats\n')

    dome, domeSig, domeSat, domeHdr =  processRamp.auto(domeFolder,rootFolder, 'processed/'+domeFolder+'_dome_flat.fits', satCounts, nlCoef, BPM, nChannel=nChannel, rowSplit=nRowSplit, nlSplit=nlSplit, combSplit=nCombSplit, crReject=False, bpmCorRng=flatbpmCorRng,avgAll=True, ron=RON)
else:
    print('Reading processed dome flats from file ')
    logfile.write('Reading processed dome flats from file\n')
    domeLst, domeHdr = wifisIO.readImgsFromFile('processed/'+domeFolder+'_dome_flat.fits')
    dome = domeLst[0]
    domeSig = domeLst[1]
    domeHdr = domeHdr[0]
    
if 'backFolder' in locals():
    if os.path.exists('processed/'+backFolder+'_back.fits'):
        cont = wifisIO.userInput('Processed files already exists for ' +backFolder+', do you want to reprocess (y/n)?')
    else:
        cont='y'

    if cont=='y':
        print('Processing background emission data')
        logfile.write('Processing background emission data\n')
    
        back, backSig, backSat, backHdr =  processRamp.auto(backFolder,rootFolder, 'processed/'+backFolder+'_back.fits', satCounts, nlCoef, BPM, nChannel=nChannel, rowSplit=nRowSplit, nlSplit=nlSplit, combSplit=nCombSplit, crReject=False, bpmCorRng=flatbpmCorRng,ron=RON,avgAll=True)
    else:
        print('Reading processed background emission data from file ')
        logfile.write('Reading processed background emission data from file\n')
        backLst, backHdr = wifisIO.readImgsFromFile('processed/'+backFolder+'_back.fits')
        back = backLst[0]
        backSig = backLst[1]
        backHdr = backHdr[0]
        
    print('Subtracting background emission from dome flat')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        dome -= back
        domeSig = np.sqrt(domeSig**2 + backSig**2)
        domeHdr.add_history('Image was background subtracted')
        
print('Extracting slices')
logfile.write('Extracting slices\n')
flatSlices = slices.extSlices(flat[4:-4,4:-4],limits,shft=shft)
domeSlices = slices.extSlices(dome[4:-4,4:-4], limits, shft=shft)

print('Getting response functions from flats')
logfile.write('Getting response functions from flats\n')
domeNorm = slices.getResponseAll(domeSlices, 1., 0.1)
calNorm = slices.getResponseAll(flatSlices, 0, 0.1)

print('Getting correction function')
logfile.write('Getting correction function')

calResp = slices.ffCorrectAll(flatSlices, domeNorm)
medSpec = slices.getMedLevelAll(calResp)

respSNorm = []
x = np.arange(medSpec[0].shape[0])
for i in range(len(medSpec)):
    respSNorm.append(calResp[i]/medSpec[i])

calCor = slices.ffCorrectAll(calNorm, respSNorm)

print('Plotting quality control figures')

with PdfPages('quality_control/flat_field_correction.pdf') as pdf:
    for r in calCor:
        fig = plt.figure()
        clim = interval.get_limits(r)
        plt.imshow(r, aspect='auto',clim=clim)
        plt.colorbar()
        plt.tight_layout()
        pdf.savefig()
        plt.close()


print('Saving correction function')
logfile.write('Saving correction function\n')

domeHdr.add_history('Correction function determined from the following files:')
domeHdr.add_history('Dome flat: ' + domeFolder)
domeHdr.add_history('Calibration flat: ' + flatFolder)
wifisIO.writeFits(respSNorm, 'processed/'+domeFolder+'_flatCor.fits')

