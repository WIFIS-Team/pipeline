"""

Main script used to process science data, sky data, and standard star data (NOT YET IMPLEMENTED)

Input:
- All input is read from the configuration file listed in variable varFile
- Processing requires a corresponding flat field data, arc lamp/wavelength calibration data associated with each observation
- A distortion map that is applicable to all data
- All observation to be processed are assumed to be part of a single observing sequence and hence the data can be combined

Produces:
For each target observation:
- The ramp image (XXX_obs.fits)
- A multi-extension fits file containing the extracted slices (XXX_obs.slices.fits)
- The distortion corrected/spatially rectified slices (XXX_obs_slices_distCor.fits)
- The fully gridded (spatially and spectrally rectified) slices (XXX_obs_slices_fullGrid.fits)
- The cube derived from the gridded data (XXX_obs_cube.fits)
- A collapsed image of the cube (XXX_obs_cubeImg.fits)

For each sky observation:
- The ramp image (XXX_obs.fits)
- A multi-extension fits file containing the extracted slices (XXX_sky.slices.fits)
- The distortion corrected/spatially rectified slices (XXX_sky_slices_distCor.fits)
- The fully gridded (spatially and spectrally rectified) slices (XXX_sky_slices_fullGrid.fits)
- The cube derived from the gridded data (XXX_sky_cube.fits)

- A combined cube of the final processed science observations ('target name'_combined_cube_#.fits)
- The collapsed image from the final combined cube ('target name'_combined_cubeImg_#.fits)

"""



import matplotlib
matplotlib.use('gtkagg')

import wifisIO
import wifisSlices as slices
import numpy as np
import time
import matplotlib.pyplot as plt
import wifisUncertainties
import wifisBadPixels as badPixels
import wifisCreateCube as createCube
import wifisHeaders as headers
import wifisProcessRamp as processRamp
import os
import copy
import wifisPostProcess as postProcess
import glob
import warnings
import wifisCalFlatFunc as calFlat
import wifisCalWaveFunc as calWave
from astropy.io import fits
from astropy import time as astrotime, coordinates as coord, units
import astropy
import colorama
from matplotlib.backends.backend_pdf import PdfPages

colorama.init()

#INPUT VARIABLE FILE NAME
varFile = 'wifisConfig.inp'

#*****************************************************************************

logfile = open('wifis_reduction_log.txt','a')
logfile.write('******************************\n')
logfile.write(time.strftime("%c")+'\n')
logfile.write('Processing science observations with WIFIS pyPline\n')
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

if not (os.path.exists(atlasFile)):
    logfile.write('*** FAILURE: Cannot continue, line atlas file ' + atlasFile + ' does not exist. Please provide the necessary atlas file ***\n')
    raise Warning('*** Cannot continue, line atlas file does not exist. Please provide the necessary atlas file ***')
    
if not (os.path.exists(distMapFile)):
    logfile.write('*** FAILURE: Cannot continue, distorion map file ' + distMapFile + ' does not exist. Please process a Ronchi calibration sequence or provide the necessary file ***\n')
    raise Warning('*** Cannot continue, distorion map file ' + distMapFile + ' does not exist. Please process a Ronchi calibration sequence or provide the necessary file ***')
else:
    distMap = wifisIO.readImgsFromFile(distMapFile)[0]
    logfile.write('Using distortion/spatial mapping from file:\n')
    logfile.write(distMapFile+'\n')

if not (os.path.exists(distMapLimitsFile)):
    logfile.write('*** FAILURE: Cannot continue, distorion map limits file ' + distMapLimitsFile + ' does not exist. Please process a Ronchi calibration sequence or provide the necessary file ***\n')
    raise Warning('*** Cannot continue, distorion map limits file ' + distMapLimitsFile + ' does not exist. Please process a Ronchi calibration sequence or provide the necessary file ***')
else:
    logfile.write('Using distortion/spatial mapping limits file from file:\n')
    logfile.write(distMapLimitsFile+'\n')
    distMapLimits = wifisIO.readImgsFromFile(distMapLimitsFile)[0]
    
if not (os.path.exists(spatGridPropsFile)):
    logfile.write('*** FAILURE: Cannot continue, spatial propertites grid file ' + spatGridPropsFile + ' does not exist. Please process a Ronchi calibration sequence or provide the necessary file ***\n')
    raise Warning ('*** Cannot continue, spatial propertites grid file does not exist. Please process a Ronchi calibration sequence or provide the necessary file ***')
else:
    spatGridProps = wifisIO.readTable(spatGridPropsFile)

#first check if BPM is provided
if os.path.exists(bpmFile):
    BPM = wifisIO.readImgsFromFile(bpmFile)[0]
    BPM = BPM.astype(bool)
else:
    BPM = None

#read input lists
if os.path.exists(obsLstFile):
    obsLst = wifisIO.readAsciiList(obsLstFile)

    if obsLst.ndim == 0:
        obsLst = np.asarray([obsLst])
else:
    logfile.write('*** FAILURE: Observation list file ' + obsLstFile + ' does not exist ***\n')
    raise Warning('Observation list file does not exist')

if skyLstFile is not None:
    if os.path.exists(skyLstFile):
        skyLst = wifisIO.readAsciiList(skyLstFile)
        if skyLst.ndim == 0:
            skyLst = np.asarray([skyLst])

        if len(skyLst) != len(obsLst):
            logfile.write('*** FAILURE: Observation list ' + obsLstFile + ' is not the same size as sky list ' + skyLstFile+' ***\n')
            raise Warning('Observation list is not the same size as ' + skyLstFile)

    else:
        skyLst = None
else:
    skyLst = None


if os.path.exists(flatLstFile):
    flatLst = wifisIO.readAsciiList(flatLstFile)

    if flatLst.ndim == 0:
        flatLst = np.asarray([flatLst])
else:
    logfile.write('*** FAILURE: Flat list file ' + flatLstFile + ' does not exist ***\n')
    raise Warning('*** Flat list file does not exist ***')

if os.path.exists(waveLstFile):
    waveLst = wifisIO.readAsciiList(waveLstFile)

    if waveLst.ndim == 0:
        waveLst = np.asarray([waveLst])
else:
    logfile.write('*** FAILURE: Wave/arc list file ' + waveLstFile + ' does not exist ***\n')
    raise Warning('*** Wave/arc list file does not exist ***')

if len(waveLst) != len(obsLst):
    logfile.write('FAILURE: Wave/arc list file ' + waveLstFile + ' is not the same size as observation list ' + obsLstFile + ' \n')
    raise Warning('*** Wave/arc list file is not the same size as observation list ***')


#deal with darks
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
    
logfile.write('\n')

#here begins the meat of the script
#**************************************************************************************************

#check to make sure that 
for i in range(len(obsLst)):

    print('\n*** Working on ' + obsLst[i] + ' ***')
    logfile.write('\n *** Working on ' + obsLst[i] + '***\n')
    #**************************************************************************************************

    #deal with flats
    flatFolder = flatLst[i]

    if not os.path.exists('processed/'+flatFolder+'_flat_limits.fits') or not os.path.exists('processed/'+flatFolder+'_flat_slices_norm.fits'):
        print('\n*** Processed flat field files do not exist for folder ' +flatFolder +', processing flat folder ***')
        logfile.write('Processed flat field files do not exist for folder ' +flatFolder +', processing flat folder\n')
        
        calFlat.runCalFlat(np.asarray([flatFolder]), hband=hband, darkLst = darkLst, rootFolder=rootFolder, nlCoef=nlCoef, satCounts=satCounts, BPM = BPM, distMapLimitsFile = distMapLimitsFile, plot=True, nChannel = nChannel, nRowsAvg=nRowsAvg,rowSplit=nRowSplitFlat,nlSplit=nlSplit, combSplit=nCombSplit,bpmCorRng=flatbpmCorRng, crReject=False, skipObsinfo=False,nlFile=nlFile, bpmFile=bpmFile, satFile=satFile,darkFile=darkFile, logfile=logfile, ask=False, obsCoords=obsCoords, limSmth=flatLimSmth, flatCutOff=flatCutOff, gain=gain, ron=RON, polyFitDegree=limitsPolyFitDegree,centGuess=centGuess)
        
    print('Reading slice limits')
    logfile.write('Reading slice limits from file:\n')
    logfile.write('processed/'+flatFolder+'_flat_limits.fits\n')
    
    limits, limitsHdr = wifisIO.readImgsFromFile('processed/'+flatFolder+'_flat_limits.fits')
    shft = limitsHdr['LIMSHIFT']

    print('Reading flat field response function')
    logfile.write('Reading flat field response function from file:\n')
    logfile.write('processed/'+flatFolder+'_flat_slices_norm.fits\n')

    flatNormLst = wifisIO.readImgsFromFile('processed/'+flatFolder+'_flat_slices_norm.fits')[0]
    flatNorm = flatNormLst[:18]
    flatSigma = flatNormLst[18:36]

    if flatCor:
        if os.path.exists(flatCorFile):
            print('Correcting flat field response function')
            logfile.write('Correcting flat field response function using file:\n')
            logfile.write(flatCorFile+'\n')
        
            flatCorSlices = wifisIO.readImgsFromFile(flatCorFile)[0]
            flatNorm = slices.ffCorrectAll(flatNorm, flatCorSlices)

            if len(flatCorSlices)>18:
                logfile.write('*** WARNING: Response correction does not include uncertainties ***\n')
                flatSigma = wifisUncertainties.multiplySlices(flatNorm,flatSigma,flatCorSlices[:18],flatCorSlices[18:36])
        else:
            print(colorama.Fore.RED+'*** WARNING: Flat field correction file does not exist, skipping ***'+colorama.Style.RESET_ALL)
    
            logfile.write('*** WARNING: Flat field correction file does not exist, skipping ***\n')
            
    #**************************************************************************************************
    #deal with wavelength calibration

    waveFolder = waveLst[i]
    if not os.path.exists('processed/'+waveFolder+'_wave_waveMap.fits'):
        print('\n*** Processed arc files do not exist for folder ' +waveFolder +', processing wave folder ***')
        logfile.write('\nProcessed arc files do not exist for folder ' +waveFolder +', processing wave folder\n')
        
        calWave.runCalWave(waveLst, flatLst, hband=hband, darkLst=darkLst, rootFolder=rootFolder, nlCoef=nlCoef, satCounts=satCounts, BPM=BPM, distMapLimitsFile=distMapLimitsFile, plot=True, nChannel=nChannel, nRowsAvg=nRowsAvg, rowSplit=nRowSplitFlat, nlSplit=nlSplit, combSplit=nCombSplit, bpmCorRng=flatbpmCorRng, crReject=crReject, skipObsinfo=skipObsinfo, flatWinRng=flatWinRng,flatImgSmth=flatImgSmth, limitsPolyFitDegree=limitsPolyFitDegree, distMapFile=distMapFile, spatGridPropsFile=spatGridPropsFile, atlasFile=atlasFile, templateFile=waveTempFile, prevResultsFile=waveTempResultsFile,  sigmaClip=sigmaClip, sigmaClipRounds=sigmaClipRounds, sigmaLimit=sigmaLimit, cleanDispSol=cleanDispSol,cleanDispThresh = cleanDispThresh, waveTrimThresh=waveTrimThresh,nlFile=nlFile,satFile=satFile,bpmFile=bpmFile,darkFile=darkFile,logfile=logfile,ask=False, obsCoords=obsCoords, waveSmooth=waveSmooth,winRng=waveWinRng,waveSolMP=waveSolMP, gain=gain, ron=RON, mxCcor=waveMxCcor,adjustFitWin=waveAdjustFitWin,centGuess=centGuess)

    print('Reading in wavelength mapping files')
    logfile.write('Using wavelength map from file:\n')
    logfile.write('processed/'+waveFolder+'_wave_waveMap.fits\n')
    logfile.write('Using wavelength grid properties from file:\n')
    logfile.write('processed/'+waveFolder+'_wave_waveGridProps.dat\n')
    
    #read in wavelength mapping files
    waveMap = wifisIO.readImgsFromFile('processed/'+waveFolder+'_wave_waveMap.fits')[0]
    waveGridProps=wifisIO.readTable('processed/'+waveFolder+'_wave_waveGridProps.dat')

    #**************************************************************************************************

    obsFolder = obsLst[i]
    obsSaveName = 'processed/'+obsFolder
    print('\n***Processing science folder '+ obsFolder + ' ***')
    logfile.write('\nProcessing science observation folder:\n')
    logfile.write(obsFolder+'\n')

    if os.path.exists(obsSaveName+'_obs.fits'):
        if not skipReprocPrompt:
            cont = wifisIO.userInput('Processed files already exists for ' +obsFolder+', do you want to reprocess (y/n)?')
        else:
            cont ='n'
            
        if (not cont.lower() == 'y'):
            contProc = False
        else:
            contProc = True
    else:
        contProc = True
    
    if (contProc):
        obs, sigmaImg, satFrame, hdr = processRamp.auto(obsFolder, rootFolder,'processed/'+obsFolder+'_obs.fits', satCounts, nlCoef, BPM, nChannel=nChannel, rowSplit=nRowSplit, nlSplit=nlSplit, combSplit=nCombSplit, crReject=crReject, bpmCorRng=bpmCorRng, nlFile=nlFile,satFile=satFile,bpmFile=bpmFile, gain=gain, ron=RON,logfile=logfile,nRows=nRowsAvg, obsCoords=obsCoords,avgAll=True, satSplit=nSatSplit)
            
    else:
        print('Reading science image')
        dataLst, hdr = wifisIO.readImgsFromFile('processed/'+obsFolder+'_obs.fits')
        
        obs = dataLst[0]
        sigmaImg=dataLst[1]
        satFrame=dataLst[2]

        if type(hdr) is list:
            hdr = hdr[0]

        logfile.write('Science data read from file:\n')
        logfile.write('processed/'+obsFolder+'_obs.fits\n')
    
    if skyLst is not None and skyLst[i].lower() != 'none':
        skyFolder = skyLst[i]
        
        if not os.path.exists('processed/'+skyFolder+'_sky.fits'):
            print('Processing sky folder '+skyFolder)
            logfile.write('\nProcessing sky folder ' + skyFolder+'\n')
            sky, skySigmaImg, skySatFrame, skyHdr = processRamp.auto(skyFolder, rootFolder,'processed/'+skyFolder+'_sky.fits', satCounts, nlCoef, BPM, nChannel=nChannel, rowSplit=nRowSplit, nlSplit=nlSplit, combSplit=nCombSplit, crReject=crReject, bpmCorRng=bpmCorRng,nlFile=nlFile,satFile=satFile,bpmFile=bpmFile, gain=gain, ron=RON,logfile=logfile,nRows=nRowsAvg, obsCoords=obsCoords,avgAll=True,satSplit=nSatSplit)
                
        #subtract sky from data at this stage
        print('Reading sky data from ' + skyFolder)
        logfile.write('Reading sky data from observation ' + skyFolder +'\n')
        skyDataLst,skyHdr = wifisIO.readImgsFromFile('processed/'+skyFolder+'_sky.fits')
        sky = skyDataLst[0]
        skySigmaImg = skyDataLst[1]
        skySatFrame = skyDataLst[2]
        skyHdr = skyHdr[0]
        del skyDataLst
        
        if skyShiftCor:
            pixDiff = postProcess.crossCorImage(obs[4:-4,4:-4],sky[4:-4,4:-4],maxFluxLevel = skyShitMaxLevel, oversample=skyShiftOverSample, regions=skyShiftRegions,position=np.arange(obs[4:-4,4:-4].shape[1]),contFit1=skyShiftContFit1, contFit2=skyShiftContFit2)

            #remove bad outliers, then compute the mean average
            pixShift = np.nanmedian(pixDiff)
            pixDiff[pixDiff > pixShift+1] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                pixDiff[pixDiff < pixShift-1] = np.nan
            
            pixShift = np.float32(np.round(np.nanmean(pixDiff),decimals=4))
                
            print('Found mean pixel shift of ' + str(pixShift))
            logfile.write('Found mean pixel shift of '+str(pixShift)+'\n')
            
            # save pixDiff slices for quality control monitoring
            print('Plotting quality control results')
            with PdfPages('quality_control/'+obsLst[i]+'_obs_'+skyLst[i]+'_sky_PixDiff.pdf') as pdf:
                fig=plt.figure()
                plt.plot(pixDiff, 'o')
                plt.xlabel('Pixel number along slice direction')
                plt.ylabel('Measured pixel shift')
                plt.title('Determined pixel shift of ' + str(pixShift))
                plt.tight_layout()
                pdf.savefig(dpi=300)
                plt.close()

            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                if pixShift != 0:
                    #now correct for shift
                    print('Correcting sky image for pixel shift')
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore',RuntimeWarning)
                        skyShift = np.empty(sky.shape,dtype=sky.dtype)
                        skyShift[:] = sky[:]
                        skyShift[4:-4,4:-4] = postProcess.shiftImage(sky[4:-4,4:-4],pixShift)
                        sky = skyShift
                    logfile.write('Sky image corrected by pixel offset\n')
                    hdr.add_history('Sky image corrected by offset')
        else:
            pixShift = 0

        reProcSky = False
        #check if flexure was measured/corrected before, and if so make sure to recompute all sky files to 
        if os.path.exists('processed/'+skyFolder+'_sky_slices.fits'):
            skyTmpHdr = fits.open('processed/'+skyFolder+'_sky_slices.fits')[0].header
            if 'FLEXSHFT' in skyTmpHdr:
                pixShiftOld = np.float32(np.round(skyTmpHdr['FLEXSHFT'],decimals=4))
            else:
                pixShiftOld = 0

            if pixShift != pixShiftOld:
                print('Previous processed sky files had a different pixel shift correction, reprocessing necessary')
                logfile.write('Previous processed sky files had a different pixel shift correction, reprocessing necessary\n')
                reProcSky = True
                    
        if not skyScaleCor:
            print('Subtracting sky from obs')
            logfile.write('Subtracting sky flux from science image flux\n')

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                obs -= sky
                sigmaImg  = np.sqrt(sigmaImg**2 + skySigmaImg**2)
                
            hdr.add_history('Subtracted sky flux image using:')
            hdr.add_history(skyFolder)

        #slices stage
        if not os.path.exists('processed/'+skyFolder+'_sky_slices.fits') or reProcSky:
            if not 'sky' in locals():
                print('Reading sky data from ' + skyFolder)
                logfile.write('Reading sky data from observation ' + skyFolder +'\n')
                skyDataLst,skyHdr = wifisIO.readImgsFromFile('processed/'+skyFolder+'_sky.fits')
                sky = skyDataLst[0]
                skySigmaImg = skyDataLst[1]
                skySatFrame = skyDataLst[2]
                skyHdr = skyHdr[0]
                del skyDataLst
                
            sky = sky[4:-4,4:-4]
            skySigmaImg = skySigmaImg[4:-4,4:-4]
            skySatFrame = skySatFrame[4:-4,4:-4]

            if skyShiftCor:
                #skyHdr.add_history('Determined sky image was off by ' + str(pixShift) + ' pixels along the dispersion direction')
                skyHdr.set('FLEXSHFT',np.float32(np.round(pixShift,decimals=4)),'Measured flexure shift in pixels')
            else:
                if 'FLEXSHFT' in skyHdr:
                    skyHdr.remove('FLEXSHFT')

            print('Extracting sky slices')
            logfile.write('\nExtracting sky slices using:\n')
            logfile.write('flatFolder')
                
            skySlices = slices.extSlices(sky, distMapLimits, shft=shft)
            skySigmaSlices = slices.extSlices(skySigmaImg, distMapLimits, shft=shft)
            skySatSlices = slices.extSlices(skySatFrame,distMapLimits, shft=shft)
            
            skyHdr.add_history('Used following flat field file for slice limits:')
            skyHdr.add_history(flatFolder)

            #apply flat-field
            if not noFlat:
                skyFlat = slices.ffCorrectAll(skySlices, flatNorm)
                skySigmaSlices = wifisUncertainties.multiplySlices(skySlices,skySigmaSlices,flatNorm, flatSigma)

                logfile.write('Sky slices were flat fielded using:\n')
                logfile.write(flatFolder+'\n')
                
                skyHdr.add_history('Slices were flat fielded using')
                skyHdr.add_history(flatFolder)
            else:
                skyFlat = skySlices

            #remove previous description of file
            hdrTmp = skyHdr[::-1]
            hdrTmp.remove('COMMENT')
            skyHdr = hdrTmp[::-1]

            skyHdr.add_comment('File contains flux, sigma, sat info for each slice as multi-extensions')
            wifisIO.writeFits(skyFlat+skySigmaSlices+skySatSlices, 'processed/'+skyFolder+'_sky_slices.fits', ask=False, hdr=skyHdr)
            hdrTmp = skyHdr[::-1]
            hdrTmp.remove('COMMENT')
            skyHdr = hdrTmp[::-1]

        #distortion correction stage
        if not os.path.exists('processed/'+skyFolder+'_sky_slices_distCor.fits') or reProcSky:
            print('Distortion correcting sky slices and placing on uniform spatial grid')
            logfile.write('Distortion correcting sky slices and placing on uniform spatial grid\n')

            if not 'skyFlat' in locals():
                print('Reading sky slices for ' + skyFolder)
                logfile.write('Reading sky slices from:\n')
                logfile.write('processed/'+skyFolder+'_sky_slices.fits\n')
                
                skySlicesLst,skyHdr = wifisIO.readImgsFromFile('processed/'+skyFolder+'_sky_slices.fits')
                skyFlat = skySlicesLst[0:18]
                skySigmaSlices = skySlicesLst[18:36]
                skySatSlices = skySlicesLst[36:]
                skyHdr =skyHdr[0]
                del skySlicesLst

            skyCor = createCube.distCorAll(skyFlat, distMap, spatGridProps=spatGridProps)
            skyHdr.add_history('Used following file for distortion map:')
            skyHdr.add_history(distMapFile)

            tmpHdr = skyHdr[::-1]
            tmpHdr.remove('COMMENT')
            skyHdr = tmpHdr[::-1]
            skyHdr.add_comment('File contains the distortion-corrected flux slices as multi-extensions')

            wifisIO.writeFits(skyCor, 'processed/'+skyFolder+'_sky_slices_distCor.fits', ask=False, hdr=skyHdr)
                        
        #wavelength gridding stage
        if not os.path.exists('processed/'+skyFolder+'_sky_slices_fullGrid.fits') or reProcSky:
            print('Placing sky slices on uniform wavelength grid')
            logfile.write('Placing sky slices on uniform wavelength grid using wavelength map:\n')
            logfile.write('processed/'+waveFolder+'_wave_waveMap.fits\n')

            if not 'skyCor' in locals():
                print('Reading distortion corrected sky slices for ' + skyFolder)
                logfile.write('Reading sky slices from:\n')
                logfile.write('processed/'+skyFolder+'_sky_slices_distCor.fits\n')
                skyCor,skyHdr = wifisIO.readImgsFromFile('processed/'+skyFolder+'_sky_slices_distCor.fits')
                skyHdr = skyHdr[0]
                
            skyGrid = createCube.waveCorAll(skyCor, waveMap, waveGridProps=waveGridProps)
            
            tmpHdr = skyHdr[::-1]
            tmpHdr.remove('COMMENT')
            skyHdr = tmpHdr[::-1]
            skyHdr.add_comment('File contains the spatial and wavelength gridded flux slices as multi-extensions')
            skyHdr.add_history('Used following observation to map wavelength:')
            skyHdr.add_history(waveFolder)
            wifisIO.writeFits(skyGrid, 'processed/'+skyFolder+'_sky_slices_fullGrid.fits', ask=False, hdr=skyHdr)

        #check if previous processing of sky exists and if sky RV measurement already carried out
        if os.path.exists('processed/'+skyFolder+'_sky_cube.fits') and not reProcSky and getSkyCorRV:
            print('Reading RVSKYCOR from FITS header')
            logfile.write('Readings RVSKYCOR from ' + skyFolder + ' header instead\n')
            skyTmpHdr = fits.open('processed/'+skyFolder+'_sky_cube.fits')[0].header
            if 'RVSKYCOR' in skyTmpHdr:
                rvSkyCor = skyTmpHdr['RVSKYCOR']
                hdr.set('RVSKYCOR',rvSkyCor,'RV offset between sky and template')
            else:
                print(colorama.Fore.RED+'*** WARNING: RVSKYCOR keyword not found in FITS header, reprocessing necessary ***'+colorama.Style.RESET_ALL)
                logfile.write('*** WARNING:  RVSKYCOR keyword not found in FITS header, reprocessing necessary ***')
                reProcSky = True        
                
        #create cube stage
        if not os.path.exists('processed/'+skyFolder+'_sky_cube.fits') or reProcSky:
            print('Creating sky cube')
            logfile.write('Creating sky cube from gridded slices\n')

            print('Reading fully gridded sky slices for ' + skyFolder)
            logfile.write('Reading fully gridded corrected sky slices from:\n')
            logfile.write('processed/'+skyFolder+'_sky_slices_fullGrid.fits\n')

            if not 'skyGrid' in locals():
                skyGrid,skyHdr = wifisIO.readImgsFromFile('processed/'+skyFolder+'_sky_slices_fullGrid.fits')
                skyHdr = skyHdr[0]

            tmpHdr = skyHdr[::-1]
            tmpHdr.remove('COMMENT')
            tmpHdr.remove('COMMENT')
            skyHdr = tmpHdr[::-1]
            skyHdr.add_comment('File contains the flux data cube')
            skyCube = createCube.mkCube(skyGrid, ndiv=ndiv).astype('float32')
            headers.getWCSCube(skyCube, skyHdr, xScale, yScale, waveGridProps)

            if getSkyCorRV:
                #compute RV difference between sky slices and some sky template spectrum
                
                print('Measuring RV difference between median sky spectrum and template')
                logfile.write('Measuring RV difference between median sky spectrum and template\n')
                logfile.write('Using the following parameters:\n')
                
                if not 'skyEmTemp' in locals():
                    skyEmTemp = wifisIO.readTable(skyEmTempFile)
          
                #compute median-averaged sky spectrum
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    avgSkySpec = np.nanmedian(skyCube,axis=[0,1])

                waveArray = 1e9*(np.arange(skyCube.shape[2])*skyHdr['CDELT3'] +skyHdr['CRVAL3'])
            
                #interpolate sky template onto observed wavelength grid
                tempInterp = np.interp(waveArray, skyEmTemp[:,0],skyEmTemp[:,1])

                rvSkyCor = postProcess.crossCorSpec(waveArray, avgSkySpec, tempInterp, plot=False, oversample=50, absorption=False, mode='idl', contFit1=True, contFit2=False,nContFit=50,contFitOrder=1, mxVel=1000, reject=0, velocity=True)

                skyHdr.set('RVSKYCOR',rvSkyCor,'RV offset between sky and template')
                hdr.set('RVSKYCOR',rvSkyCor,'RV offset between sky and template')
                print('Found RV offset between sky spectrum and template of ' + str(rvSkyCor))
                logfile.write('Found RV offset between sky spectrum and template of ' + str(rvSkyCor)+'\n')

            #write WCS to header
            wifisIO.writeFits(skyCube, 'processed/'+skyFolder+'_sky_cube.fits', hdr=skyHdr, ask=False)

    else:
        if darkLst is not None and darkLst[0] is not None:
            print('Subtracting dark')
            logfile.write('Subtracting dark image from science image\n')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                obs -= dark
                sigmaImg = np.sqrt(darkSig**2 + sigmaImg**2)

            hdr.add_history('Subtracted dark image from file:')
            hdr.add_history(darkFile)

    #extracting slices
    print('Extracting science data slices')
    logfile.write('Extracting science slices using:\n')
    logfile.write(flatFolder+'\n')
    
    dataSlices = slices.extSlices(obs[4:-4,4:-4], distMapLimits, shft=shft)
    sigmaSlices = slices.extSlices(sigmaImg[4:-4,4:-4], distMapLimits,shft=shft)
    satSlices = slices.extSlices(satFrame[4:-4,4:-4], distMapLimits, shft=shft)

    hdr.add_history('Used following flat field file for slice limits:')
    hdr.add_history(flatFolder)
    
    #apply flat-field correction
    #print('Applying flat field corrections')
    if not noFlat:
        dataFlat = slices.ffCorrectAll(dataSlices, flatNorm)
        sigmaSlices = wifisUncertainties.multiplySlices(dataSlices, sigmaSlices, flatNorm, flatSigma)
        logfile.write('Science slices were flat fielded using:\n')
        logfile.write(flatFolder+'\n')

        hdr.add_history('Slices were flat fielded using:')
        hdr.add_history(flatFolder)
    else:
        dataFlat = dataSlices

    hdrTmp = hdr[::-1]
    hdrTmp.remove('COMMENT')
    hdr = hdrTmp[::-1]
        
    hdr.add_comment('File contains flux, sigma, sat info for each slice as multi-extensions')
    wifisIO.writeFits(dataFlat+sigmaSlices+satSlices, obsSaveName+'_obs_slices.fits', ask=False, hdr=hdr)

    print('Distortion correcting science data and placing on uniform spatial grid')
    logfile.write('Distortion correcting sky slices and placing on uniform spatial grid\n')
                
    #distortion correct data
    dataCor = createCube.distCorAll(dataFlat, distMap, spatGridProps=spatGridProps)
    
    hdr.add_history('Used following file for distortion map:')
    hdr.add_history(distMapFile)

    tmpHdr = hdr[::-1]
    tmpHdr.remove('COMMENT')
    hdr = tmpHdr[::-1]
    hdr.add_comment('File contains the distortion-corrected flux slices as multi-extensions')

    #carry out additional 

    if skyLst is not None:
        if skyScaleCor:
            if not 'skyCor' in locals():
                print('Reading distortion corrected sky slices for ' + skyFolder)
                logfile.write('Reading sky slices from:\n')
                logfile.write('processed/'+skyFolder+'_sky_slices_distCor.fits\n')
                skyCor,skyHdr = wifisIO.readImgsFromFile('processed/'+skyFolder+'_sky_slices_distCor.fits')
                skyHdr = skyHdr[0]
                
            if skyMxScale == 0:
                subSlices = []
                for oSlc,sSlc in zip(dataCor,skyCor):
                    subSlices.append(oSlc-sSlc)
                
                hdr.add_history('Subtracted sky slices using sky obs:')
                hdr.add_history(skyFolder)

            else:

                print('Finding and correcting strength of sky emission lines to best match science observation')
                logfile.write('Finding and correcting strength of sky emission lines to best match science observation\n')
                logfile.write('Using the following parameters:\n')
                logfile.write('Wavlength regions:\n')
                
                if skyLineRegions is not None:
                    for reg in skyLineRegions:
                        logfile.write(str(reg[0]) + ' - ' + str(reg[1])+'\n')
                logfile.write('mxScale: ' + str(skyMxScale)+'\n')
                logfile.write('Fitting individual lines: '+str(skyFitIndLines)+'\n')

                
                #check if RV correction exists, if so update wave mapping to take correction into effect
                if 'rvSkyCor' in locals():
                    waveShift = []
                    for j in range(len(waveMap)):
                        waveShift.append(waveMap[j]*(1. + rvSkyCor/2.99792458e5))

                    waveGridShift = np.zeros(3,dtype='float32')
                    waveGridShift[:] = waveGridProps
                    waveGridShift[:1] = waveGridProps[:1]*(1. + rvSkyCor/2.99792458e5)
 
                else:
                    waveShift = waveMap
                    waveGridShift = waveGridProps

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    subSlices= postProcess.subScaledSkySlices2(waveShift, dataCor, skyCor, waveGridShift,skyHdr,regions=skyLineRegions, mxScale=skyMxScale, fluxThresh = skyFluxThresh, fitInd=skyFitIndLines, saveFile = 'quality_control/'+obsLst[i]+'_obs_'+skyLst[i]+'_sky')

                hdr.add_history('Subtracted scaled sky slices using sky obs:')
                hdr.add_history(skyFolder)
                if skyFitIndLines:
                    hdr.add_history('Scaled sky lines individually')

            dataCor = subSlices


    #add additional header info
    #barycentric corrections
    print('Determining barycentric corrections')
    logfile.write('Determining barycentric corrections\n')
    #input code here to compute barycentric time and RV corrections
    objCoords = coord.SkyCoord(hdr['RA'],hdr['DEC'],unit=(units.hourangle,units.deg),frame='icrs')
        
    #get barycentric time correction
    times = astrotime.Time(float(hdr['JD']), format='jd',scale='utc', location=obsCoords)
    ltt_bary = times.light_travel_time(objCoords, kind='barycentric')
    time_bary = times.tdb+ltt_bary

    hdr.set('BARY_COR', ltt_bary.value,'Barycentric date correction')
    hdr.set('BARY_JD', time_bary.value,'Barycentric Julian date')
    
    #get barycentric velocity correction
    if astropy.version.major>1:
        baryCorr = objCoords.radial_velocity_correction('barycentric',obstime=times)
        baryCorr = baryCorr.to(units.km/units.s)
        hdr.set('BARY_RV', baryCorr.value, 'Barycentric velocity correction in km/s')
    else:
        print(colorama.Fore.RED+'*** WARNING: Only astropy version > 2 supports radial velocity corrections, no corrections computed ***'+colorama.Style.RESET_ALL)
        
        logfile.write('*** WARNING: Only astropy version > 2 supports radial velocity correctsion, no correctsion computed ***')
        
    #compute galactic coordinates
    print('Determining galactic coordiantes')
    logfile.write('Determining galactic coordinates\n')

    icrs_coords = coord.ICRS(ra=hdr['RA_DEG']*units.deg,dec=hdr['DEC_DEG']*units.deg)
    gal_coords = icrs_coords.transform_to(coord.Galactic)
    hdr.set('GAL_l',gal_coords.l.value,'Galactic longitude in degrees')
    hdr.set('GAL_b',gal_coords.b.value,'Galactic latitude in degrees')
    
    wifisIO.writeFits(dataCor, obsSaveName+'_obs_slices_distCor.fits', ask=False, hdr=hdr)

    #continue processing 
    #place on uniform wavelength grid
    print('Placing science data on uniform wavelength grid')

    logfile.write('Placing science slices on uniform wavelength grid using wavelength map:\n')
    logfile.write('processed/'+waveFolder+'_wave_waveMap.fits\n')

    tmpHdr = hdr[::-1]
    tmpHdr.remove('COMMENT')
    hdr = tmpHdr[::-1]
    hdr.add_comment('File contains the spatial and wavelength gridded flux slices as multi-extensions')

    dataGrid = createCube.waveCorAll(dataCor, waveMap, waveGridProps=waveGridProps)

    hdr.add_history('Used following observation to map wavelength:')
    hdr.add_history(waveFolder)

    wifisIO.writeFits(dataGrid, obsSaveName+'_obs_slices_fullGrid.fits', ask=False, hdr=hdr)

    #create cube
    print('Creating science data cube')
    logfile.write('Creating science cube from gridded slices\n')

    tmpHdr = hdr[::-1]
    tmpHdr.remove('COMMENT')
    tmpHdr.remove('COMMENT')
    hdr = tmpHdr[::-1]
    
    dataCube = createCube.mkCube(dataGrid, ndiv=ndiv).astype('float32')

    if tellCor:
        #input code here to deal with telluric corrections
        pass
   
    hdrCube = copy.copy(hdr[:])
    hdrImg = copy.copy(hdr[:])
    
    headers.getWCSCube(dataCube, hdrCube, xScale, yScale, waveGridProps, useSesameCoords=useSesameCoords)

    hdrCube.add_comment('File contains the flux data cube')
    wifisIO.writeFits(dataCube, obsSaveName+'_obs_cube.fits', hdr=hdrCube, ask=False)

    dataImg = np.nansum(dataCube, axis=2).astype('float32')

    hdrImg.add_comment('File contains the summed flux along each spaxel')
    headers.getWCSImg(dataImg, hdrImg, xScale, yScale, useSesameCoords=useSesameCoords)
    wifisIO.writeFits(dataImg, obsSaveName+'_obs_cubeImg.fits',hdr=hdrImg, ask=False)

if len(obsLst) > 1:
    print('\n*** Averaging all cubes ***')
    logfile.write('Median averaging all cubes\n')

    combHdr = fits.Header()
    combHdr.set('SIMPLE',True,'conforms to FITS standard')
    combHdr.set('BITPIX',-32, 'array data type')
    combHdr.set('NAXIS',2,'number of array dimensions')
    combHdr.set('NAXIS1', hdr['NAXIS1'])
    combHdr.set('NAXIS2',hdr['NAXIS2'])
    
    #get obsinfo from last observation to populate new header 
    path = wifisIO.getPath(obsLst[-1],rootFolder=rootFolder)
    fileLst =glob.glob(path+'H2R*fits')
    fileLst = wifisIO.sorted_nicely(fileLst)
    headers.addTelInfo(combHdr, path+'/obsinfo.dat')

    #define wave grid properties using first file
    waveGridProps = wifisIO.readTable('processed/'+waveLst[0]+'_wave_waveGridProps.dat')
    
    #create output cube
    combGrid = [] # np.zeros(cube.shape, dtype=cube.dtype)
    dataCombLst = []
    iTimeLst = []
    utTimeLst = []
    for i in range(0,len(obsLst)):
        
        #go back to original map/dispersion solution and interpolate all obs onto same grid
        dataCor, hdr = wifisIO.readImgsFromFile('processed/'+obsLst[i]+'_obs_slices_distCor.fits')
        hdr = hdr[0]
        iTimeLst.append(hdr['INTTIME'])
        utTimeLst.append(hdr['UT'])
        
        waveMap = wifisIO.readImgsFromFile('processed/'+waveLst[i]+'_wave_waveMap.fits')[0]

        #check if rv corrections exist 
        if 'RVSKYCOR' in hdr:
            if hdr['RVSKYCOR']!=0:
                rvSkyCor = hdr['RVSKYCOR']
                print('Correcting observation ' + obsLst[i] + ' for RV offset of '+str(rvSkyCor))
                logfile.write('Correcting observation ' + obsLst[i] + ' for RV offset of ' + str(rvSkyCor)+'\n')

                waveShift = []
                for j in range(len(waveMap)):
                    waveShift.append(waveMap[j]*(1. + rvSkyCor/2.99792458e5))
            else:
                waveShift = waveMap
        else:
            waveShift = waveMap
            
        print('Interpolating slices of ' + obsLst[i] + ' onto uniform grid')
        logfile.write('Interpolating slices of ' + obsLst[i] + ' onto uniform grid\n')
        dataGrid = createCube.waveCorAll(dataCor, waveShift, waveGridProps=waveGridProps)
        dataCombLst.append(dataGrid)
        
        #combine data by summing all of the same slices together
        #if len(combGrid)>0:
        #    for j in range(len(dataGrid)):
        #        combGrid[j] += dataGrid[j]
        #else:
        #    combGrid = dataGrid
                
    #now divide each slice by the total number of observations
    #for i in range(len(combGrid)):
    #    combGrid[i] /= float(len(obsLst))

    #now combine data
    print('Creating median slices')
    logfile.write('Creating median slices\n')

    #for i in range(obsLst.shape[0]):
    #    plt.plot(np.nanmedian(dataCombLst[i][0],axis=0))
    #plt.show()
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        
        combGrid = []
        for i in range(len(dataGrid)):
            tmpLst = []
            for j in range(obsLst.shape[0]):
                tmpLst.append(dataCombLst[j][i])
            combGrid.append(np.nanmedian(tmpLst,axis=0))
    del tmpLst
    del dataCombLst

    #now create cube from gridded data
    print('Creating median data cube')
    logfile.write('Creating median cube from gridded slices\n')
    combCube = createCube.mkCube(combGrid, ndiv=ndiv)

    #add additional header info here
    #compute total iTime
    combHdr.set('INTTIME',np.float32(np.sum(iTimeLst)),'Total integration time on target in seconds')

    #compute total observation time, from start of first exposure to end of last exposure
    #convert UT into decimal representation
    t1Split = utTimeLst[0].split(':')
    time1 = np.float(t1Split[0])*3600.+np.float(t1Split[1])*60.+np.float(t1Split[2])
    #subtract first integration time
    time1 -= iTimeLst[0]
    t2Split = utTimeLst[-1].split(':')
    time2 = np.float(t2Split[0])*3600.+np.float(t2Split[1])*60.+np.float(t2Split[2])
    deltaTime = time2-time1
    
    if deltaTime < 0:
        #account for possible changeover in day
        deltaTime+=24.*3600.
        
    combHdr.set('OBJTIME',np.float32(deltaTime),'Total time spent on target in seconds')
    
    if getSkyCorRV:
        combHdr.add_history('RV offset between sky and template was determined for each observation')
    if skyShiftCor:
        combHdr.add_history('Corrected each observation for determined pixel shift between sky and target ramp image')
    if skyScaleCor:
        if skyFitIndLines:
            combHdr.add_history('Subtracted sky image with individually scaled sky lines from each observation')
        else:
            combHdr.add_history('Subtracted sky image with per-region scaled sky lines from each observation')

            
else:
    combCube, combHdr = wifisIO.readImgsFromFile('processed/'+obsLst[0]+'_obs_cube.fits')

#get object name
objectName = combHdr['Object']   

#check if cube with this name already exists
existsLst = glob.glob('processed/'+objectName+'_combined_cube_*fits')
obsNum = len(existsLst)+1

#add JD/BJD and additional corrections from last observation
combHdr.set('JD',hdr['JD'],'Julian date at end of observation')
combHdr.set('BARY_COR',hdr['BARY_COR'],'Barycentric date correction')
combHdr.set('BARY_JD', hdr['BARY_JD'],'Barycentric Julian date')

    
#now save combined cube
combHdr.add_history('Data is mean average of ' +str(len(obsLst)) + ' observations:')
for name in obsLst:
    combHdr.add_history(name)

hdrCube = copy.copy(combHdr[:])
hdrImg = copy.copy(combHdr[:])

hdrCube.add_comment('File contains the flux data cube')
hdrImg.add_comment('File contains the summed flux along each spaxel')

headers.getWCSCube(combCube, hdrCube, xScale, yScale, waveGridProps, useSesameCoords=useSesameCoords)
wifisIO.writeFits(combCube.astype('float32'), 'processed/'+objectName+'_combined_cube_'+str(obsNum)+'.fits', hdr=hdrCube, ask=False)

combImg = np.nansum(combCube, axis=2)

headers.getWCSImg(combImg, hdrImg, xScale, yScale, useSesameCoords=useSesameCoords)
wifisIO.writeFits(combImg.astype('float32'), 'processed/'+objectName+'_combined_cubeImg_'+str(obsNum)+'.fits', hdr=hdrImg, ask=False)
