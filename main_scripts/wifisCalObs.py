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
import wifisWaveSol as waveSol
import wifisProcessRamp as processRamp
import os
import copy
import wifisPostProcess as postProcess
from scipy.ndimage.interpolation import shift
import glob
import warnings
import wifisCalFlatFunc as calFlat
import wifisCalWaveFunc as calWave
from astropy.io import fits
from scipy.interpolate import interp1d
from astropy import time as astrotime, coordinates as coord, units
import astropy

import colorama
from matplotlib.backends.backend_pdf import PdfPages

colorama.init()

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0' # Used to show compile errors for debugging, can be removed
os.environ['PYOPENCL_CTX'] = '2' # Used to specify which OpenCL device to target. Should be uncommented and pointed to correct device to avoid future interactive requests

#*****************************************************************************
#REQUIRED INPUT

#specifies whether data is hband or not
hband = False

#likely static
rootFolder = '/data/WIFIS/H2RG-G17084-ASIC-08-319'
pipelineFolder = '/data/pipeline/'

#input list/file names
flatLstFile = 'flat.lst'
waveLstFile = 'wave.lst'
obsLstFile = 'obs.lst'
skyLstFile = 'sky.lst'
darkFile = ''#'dark.lst'
noFlat = False

#sky subtraction options
skySubFirst = True

if skySubFirst:
    skyCorRegions = [[0,350]]
else:
    skyCorRegions = [[1025,1045],[1080,1105],[1140,1175],[1195,1245],[1265,1330]]

#options for flexure/pixel shift between sky/obs cubes
skyShiftCor = True
skyShiftOverSample = 20
skyShiftContFit1 = True
skyShiftContFit2 = True
skyShiftnPix = 50
skyShiftContFitOrd = 1
skyShiftMxShift = 4
skyShiftReject = 1
skyShitMaxLevel = 0. #only used for determining which columns to use if subtracting sky at the image level

#options to determine line strength scaling difference between sky and obs cubes
skyScaleCor = True
skyScaleMx = 0.9
skyScaleSigClip = 3
skyScaleSigClipRnds = 1
skyScaleUseMxOnly = 0.2 # only uses pixels with >0.2 of maximum signal for fitting/subtracting

#determine additional RV corrections relative to sky emission template
getSkyCorRV = True
skyEmTempFile = pipelineFolder+'/external_data/sky_emission_template.dat'

#telluric correction
tellCor = False

#optional flat field correction to better match cal flats to dome flats
flatCorFile = 'processed/flat_correction_slices.fits'
flatCor = False

#specify calibration files
if hband:
    waveTempFile = '/data/pipeline/external_data/waveTemplate_hband.fits'
    waveTempResultsFile = '/data/pipeline/external_data/waveTemplate_hband_fitResults.pkl'
    distMapFile = '/home/jason/wifis/data/ronchi_map_june/hband/processed/20170607222050_ronchi_distMap.fits'
    spatGridPropsFile = '/home/jason/wifis/data/ronchi_map_june/hband/processed/20170607222050_ronchi_spatGridProps.dat'
else:
    waveTempFile = '/data/pipeline/external_data/waveTemplate.fits'
    waveTempResultsFile = '/data/pipeline/external_data/waveTemplateFittingResults.pkl'

    #may
    #distMapFile = '/home/jason/wifis/data/ronchi_map_may/testing/processed/20170511222022_ronchi_distMap.fits'
    #distMapLimitsFile = '/home/jason/wifis/data/ronchi_map_may/testing/processed/20170511223518_flat_limits.fits'
    #spatGridPropsFile = '/home/jason/wifis/data/ronchi_map_may/testing/processed/20170511222022_ronchi_spatGridProps.dat'

    #june
    #distMapFile = '/home/jason/wifis/data/ronchi_map_june/testing/processed/20170611221759_ronchi_distMap.fits'
    #distMapLimitsFile = '/home/jason/wifis/data/ronchi_map_june/testing/processed/20170611222844_flat_limits.fits'
    #spatGridPropsFile = '/home/jason/wifis/data/ronchi_map_june/testing/processed/20170611221759_ronchi_spatGridProps.dat'

    #july
    #distMapFile = '/home/jason/wifis/data/ronchi_map_july/tb/processed/20170707175840_ronchi_distMap.fits'
    #distMapLimitsFile = '/home/jason/wifis/data/ronchi_map_july/tb/processed/20170707180443_flat_limits.fits'
    #spatGridPropsFile = '/home/jason/wifis/data/ronchi_map_july/tb/processed/20170707175840_ronchi_spatGridProps.dat'

    #august
    distMapFile = '/home/jason/wifis/data/ronchi_map_august/tb/testing/processed/20170831211259_ronchi_distMap.fits'
    distMapLimitsFile = '/home/jason/wifis/data/ronchi_map_august/tb/testing/processed/20170831210255_flat_limits.fits'
    spatGridPropsFile = '/home/jason/wifis/data/ronchi_map_august/tb/testing/processed/20170831211259_ronchi_spatGridProps.dat'


nlFile = '/home/jason/wifis/data/june_cals/processed/master_detLin_NLCoeff.fits'        
satFile = '/home/jason/wifis/data/june_cals/processed/master_detLin_satCounts.fits'
bpmFile = '/home/jason/wifis/data/june_cals/processed/master_dark_BPM.fits'
atlasFile = pipelineFolder+'external_data/best_lines2.dat'

#bad pixel mask correction range
bpmCorRng = 1

#pixel scale
#may old
xScale = 0.532021532706
yScale = -0.545667026386

#may new
#xScale = 0.529835976681
#yScale = 0.576507533367

#june
#xScale = 0.549419840181
#yScale = -0.581389824133

#wavelength fitting
mxOrder = 3
cleanDispSol = True
if hband:
    cleanDispThresh = 1.5
    waveTrimThresh=0.25
else:
    cleanDispThresh = 1.5
    waveTrimThresh = 0
    
sigmaClipRounds=2 #number of iterations when sigma-clipping of dispersion solution
sigmaClip = 2 #sigma-clip cutoff when sigma-clipping dispersion solution
sigmaLimit= 3 #relative noise limit (x * noise level) for which to reject lines

#determine final cube parameters
ndiv = 1
if ndiv == 0:
    yScale = yScale*35/18.

crReject = False

#optional behaviour
skipObsinfo = False

#flat field specific options
flatWinRng = 51
flatImgSmth = 5
flatPolyFitDegree=2

#parameters used for processing of ramps
dispAxis=0
nChannel=32 #specifies the number of channels used during readout of detector
flatbpmCorRng=20 #specifies the maximum separation of pixel search to use during bad pixel correction
nRowsAvg=4 # specifies the number of rows of reference pixels to use to correct for row bias (+/- nRowsAvg)
rowSplit=1 # specifies how many processing steps to use during reference row correction. Must be integer multiple of number of frames. For very long ramps, use a higher number to avoid OpenCL issues and/or high memory consumption.
nlSplit=32 #specifies how many processing steps to use during non-linearity correction. Must be integer multiple of detector width. For very long ramps, use a higher number to avoid OpenCL issues and/or high memory consumption.
satSplit=32 #specifies how many processing steps to use during identification of first saturated frame. Must be integer multiple of detector width. For very long ramps, use a higher number to avoid OpenCL issues and/or high memory consumption.
combSplit=32 #specifies how many processing steps to use during creation of ramp image. Must be integer multiple of detector width. For very long ramps, use a higher number to avoid OpenCL issues and/or high memory consumption.
gain = 1.
ron = 1.

#coordinates
obsCoords = [-111.600444444,31.9629166667,2071]
useSesameCoords=True
#*****************************************************************************
#*****************************************************************************

logfile = open('wifis_reduction_log.txt','a')
logfile.write('******************************\n')
logfile.write(time.strftime("%c")+'\n')
logfile.write('Processing science observations with WIFIS pyPline\n')
logfile.write('Root folder containing raw data: ' + str(rootFolder)+'\n')

#create processed directory, in case it doesn't exist
wifisIO.createDir('processed')

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
    distMapLimits = wifisIO.readImgsFromFile(distMapLimitsFile)[0]
    logfile.write('Using distortion/spatial mapping limits file from file:\n')
    logfile.write(distMapLimitsFile+'\n')

if not (os.path.exists(spatGridPropsFile)):
    logfile.write('*** FAILURE: Cannot continue, spatial propertites grid file ' + spatGridPropsFile + ' does not exist. Please process a Ronchi calibration sequence or provide the necessary file ***\n')
    raise Warning ('*** Cannot continue, spatial propertites grid file does not exist. Please process a Ronchi calibration sequence or provide the necessary file ***')

satCounts = wifisIO.readImgsFromFile(satFile)[0]
nlCoeff = wifisIO.readImgsFromFile(nlFile)[0]

#first check if BPM is provided
if os.path.exists(bpmFile):
    BPM = wifisIO.readImgsFromFile(bpmFile)[0]
    BPM = BPM.astype(bool)
else:
    BPM = None

#read in ronchi mask
distMap = wifisIO.readImgsFromFile(distMapFile)[0]
spatGridProps = wifisIO.readTable(spatGridPropsFile)

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

#here begins the meat of the script
#**************************************************************************************************
#deal with darks
if darkFile is not None and os.path.exists(darkFile):
    dark, darkSig, darkSat = wifisIO.readImgsFromFile(darkFile)[0]
    darkLst = [dark, darkSig]
else:
    print(colorama.Fore.RED+'*** WARNING: No dark image provided, or file does not exist ***'+colorama.Style.RESET_ALL)
    
    if logfile is not None:
        logfile.write('*** WARNING: No dark image provide, or file ' + str(darkFile)+' does not exist ***\n')
    darkLst = [None,None]

logfile.write('\n')
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
        
        calFlat.runCalFlat(np.asarray([flatFolder]), hband=hband, darkLst = darkLst, rootFolder=rootFolder, nlCoef=nlCoef, satCounts=satCounts, BPM = BPM, distMapLimitsFile = distMapLimitsFile, plot=True, nChannel = nChannel, nRowsAvg=nRowsAvg,rowSplit=rowSplit,nlSplit=nlSplit, combSplit=combSplit,bpmCorRng=flatbpmCorRng, crReject=False, skipObsinfo=False,nlFile=nlFile, bpmFile=bpmFile, satFile=satFile,darkFile=darkFile, logfile=logfile, ask=False, obsCoords=obsCoords)
        
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
        
        calWave.runCalWave(waveLst, flatLst, hband=hband, darkLst=darkLst, rootFolder=rootFolder, nlCoef=nlCoef, satCounts=satCounts, BPM=BPM, distMapLimitsFile=distMapLimitsFile, plot=True, nChannel=nChannel, nRowsAvg=nRowsAvg, rowSplit=rowSplit, nlSplit=nlSplit, combSplit=combSplit, bpmCorRng=flatbpmCorRng, crReject=crReject, skipObsinfo=skipObsinfo, flatWinRng=flatWinRng,flatImgSmth=flatImgSmth, flatPolyFitDegree=mxOrder, distMapFile=distMapFile, spatGridPropsFile=spatGridPropsFile, atlasFile=atlasFile, templateFile=waveTempFile, prevResultsFile=waveTempResultsFile,  sigmaClip=sigmaClip, sigmaClipRounds=sigmaClipRounds, sigmaLimit=sigmaLimit, cleanDispSol=cleanDispSol,cleanDispThresh = cleanDispThresh, waveTrimThresh=waveTrimThresh,nlFile=nlFile,satFile=satFile,bpmFile=bpmFile,darkFile=darkFile,logfile=logfile,ask=False, obsCoords=obsCoords)

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
        cont = 'n'
        cont = wifisIO.userInput('Processed files already exists for ' +obsFolder+', do you want to continue processing (y/n)?')
        if (not cont.lower() == 'y'):
            contProc = False
        else:
            contProc = True
    else:
        contProc = True
    
    if (contProc):
        obs, sigmaImg, satFrame, hdr = processRamp.auto(obsFolder, rootFolder,'processed/'+obsFolder+'_obs.fits', satCounts, nlCoeff, BPM, nChannel=nChannel, rowSplit=rowSplit, nlSplit=nlSplit, combSplit=combSplit, crReject=crReject, bpmCorRng=bpmCorRng, nlFile=nlFile,satFile=satFile,bpmFile=bpmFile, gain=gain, ron=ron,logfile=logfile,nRows=nRowsAvg, obsCoords=obsCoords,avgAll=True, satSplit=satSplit)
            
    else:
        print('Reading science data from file')
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
            sky, skySigmaImg, skySatFrame, skyHdr = processRamp.auto(skyFolder, rootFolder,'processed/'+skyFolder+'_sky.fits', satCounts, nlCoeff, BPM, nChannel=nChannel, rowSplit=rowSplit, nlSplit=nlSplit, combSplit=combSplit, crReject=crReject, bpmCorRng=bpmCorRng,nlFile=nlFile,satFile=satFile,bpmFile=bpmFile, gain=gain, ron=ron,logfile=logfile,nRows=nRowsAvg, obsCoords=obsCoords,avgAll=True,satSplit=satSplit)
                
        if skySubFirst:
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
                pixDiff = postProcess.crossCorImage(obs[4:-4,4:-4],sky[4:-4,4:-4],maxFluxLevel = skyShitMaxLevel, oversample=20, regions=skyCorRegions,position=np.arange(obs[4:-4,4:-4].shape[1]))

                pixShift = np.nanmedian(pixDiff)
                if pixShift !=0:
                    print('Found median pixel shift of ' + str(pixShift))
                    logfile.write('Found median pixel shift of '+str(pixShift)+'\n')
                    hdr.add_history('Determined sky image was off by ' + str(pixShift) + ' pixels along the dispersion direction')
                else:
                    pixShift = np.nanmean(pixDiff)
                    print('Found mean pixel shift of ' + str(pixShift))
                    logfile.write('Found mean pixel shift of '+str(pixShift)+'\n')
                    hdr.add_history('Determined sky image was off by ' + str(pixShift) + ' pixels along the dispersion direction')

                # save pixDiff slices for quality control monitoring
                print('Plotting quality control results')
                with PdfPages('quality_control/'+obsLst[i]+'_sky_PixDiff.pdf') as pdf:
                    fig=plt.figure()
                    plt.plot(pixDiff, 'o')
                    plt.xlabel('Pixel number along slice direction')
                    plt.ylabel('Measured pixel shift')
                    plt.title('Determined pixel shift of ' + str(pixShift))
                    plt.tight_layout()
                    pdf.savefig(dpi=300)
                    plt.close()      

                if pixShift != 0:
                    #now correct for shift
                    print('Correcting sky image for pixel shift')
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore',RuntimeWarning)
                        skyShift = sky
                        skyShift[4:-4,4:-4] = postProcess.shiftImage(sky[4:-4,4:-4],pixShift)
                        sky = skyShift
                    
                    logfile.write('Sky image corrected by pixel offset\n')
                    hdr.add_history('Sky image corrected by offset')

                #skySmooth = postProcess.getSmoothedImage(sky[4:-4,4:-4],kernSize=4)
                                
            print('Subtracting sky from obs')
            logfile.write('Subtracting sky flux from science image flux\n')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                obs -= sky
                sigmaImg  = np.sqrt(sigmaImg**2 + skySigmaImg**2)
                
            hdr.add_history('Subtracted sky flux image using:')
            hdr.add_history(skyFolder)
        else:
            #slices stage
            if not os.path.exists('processed/'+skyFolder+'_sky_slices.fits'):
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
            if not os.path.exists('processed/'+skyFolder+'_sky_slices_distCor.fits'):
                print('Distortion correcting sky slices and placing on uniform spatial grid')
                logfile.write('Distortion correcting sky slices and placing on uniform spatial grid\n')

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
            else:
                print('Reading distortion corrected sky slices for ' + skyFolder)
                logfile.write('Reading distortion corrected sky slices from:\n')
                logfile.write('processed/'+skyFolder+'_sky_slices_distCor.fits\n')
                skyCor,skyHdr = wifisIO.readImgsFromFile('processed/'+skyFolder+'_sky_slices_distCor.fits')
                skyHdr = skyHdr[0]
                
            #wavelength gridding stage
            if not os.path.exists('processed/'+skyFolder+'_sky_slices_fullGrid.fits'):
                print('Placing sky slices on uniform wavelength grid')
                logfile.write('Placing sky slices on uniform wavelength grid using wavelength map:\n')
                logfile.write('processed/'+waveFolder+'_wave_waveMap.fits\n')

                skyGrid = createCube.waveCorAll(skyCor, waveMap, waveGridProps=waveGridProps)

                tmpHdr = skyHdr[::-1]
                tmpHdr.remove('COMMENT')
                skyHdr = tmpHdr[::-1]
                skyHdr.add_comment('File contains the spatial and wavelength gridded flux slices as multi-extensions')
                skyHdr.add_history('Used following observation to map wavelength:')
                skyHdr.add_history(waveFolder)
                wifisIO.writeFits(skyGrid, 'processed/'+skyFolder+'_sky_slices_fullGrid.fits', ask=False, hdr=skyHdr)

            #create cube stage
            if not os.path.exists('processed/'+skyFolder+'_sky_cube.fits'):
                print('Creating sky cube')
                logfile.write('Creating sky cube from gridded slices\n')

                print('Reading fully gridded sky slices for ' + skyFolder)
                logfile.write('Reading fully gridded corrected sky slices from:\n')
                logfile.write('processed/'+skyFolder+'_sky_slices_fullGrid.fits\n')
                
                skyGrid,skyHdr = wifisIO.readImgsFromFile('processed/'+skyFolder+'_sky_slices_fullGrid.fits')
                skyHdr = skyHdr[0]

                tmpHdr = skyHdr[::-1]
                tmpHdr.remove('COMMENT')
                tmpHdr.remove('COMMENT')
                skyHdr = tmpHdr[::-1]
                skyHdr.add_comment('File contains the flux data cube')
                skyCube = createCube.mkCube(skyGrid, ndiv=ndiv).astype('float32')
                
                #write WCS to header
                headers.getWCSCube(skyCube, skyHdr, xScale, yScale, waveGridProps)
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
    
    #subtract sky at this stage

    if skyLst is not None and not skySubFirst:
        if skyShiftCor:
            print('Determining amount of pixel shift between sky and science slices along dispersion axis')
            logfile.write('Determining amount of pixel shift between sky and science slices along dispersion axis\n')

            #waveArray = 1e9*(np.arange(skyCube.shape[2])*skyHdr['CDELT3'] +skyHdr['CRVAL3'])
            logfile.write('Using the following parameters:\n')
            logfile.write('Wavlength regions:\n')
            for reg in skyCorRegions:
                logfile.write(str(reg[0]) + ' - ' + str(reg[1])+'\n')
                
            logfile.write('oversample: ' + str(skyShiftOverSample)+'\n')
            logfile.write('contFit1: ' + str(skyShiftContFit1)+'\n')
            logfile.write('contFit2: ' + str(skyShiftContFit2)+'\n')
            logfile.write('nContFit: ' + str(skyShiftnPix)+'\n')
            logfile.write('contFitOrder: ' + str(skyShiftContFitOrd)+'\n')
            logfile.write('mxShift: ' + str(skyShiftMxShift)+'\n')
            logfile.write('reject: ' + str(skyShiftReject)+'\n')
            
            #input code here to check and correct for potential pixel shift between sky/science slices
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                pixDiff = postProcess.crossCorSlices(waveMap, dataCor, skyCor,regions=skyCorRegions,oversample=skyShiftOverSample,contFit1=skyShiftContFit1,contFit2=skyShiftContFit2 ,nContFit=skyShiftnPix, contFitOrder=skyShiftContFitOrd, mxShift=skyShiftMxShift, reject=skyShiftReject,velocity=False)

            #compute median from image, assuming that there is a constant shift
            pixShift = np.nanmedian(pixDiff)
            print('Found median pixel shift of ' + str(pixShift))
            logfile.write('Found median pixel shift of '+str(pixShift)+'\n')
            hdr.add_history('Determined sky cube was off by ' + str(pixShift) + ' pixels along the dispersion direction')

            if pixShift != 0:
                #now correct for shift
                print('Correcting sky slices for pixel shift')
                skyShift = postProcess.shiftSlicesAll(skyCor,pixShift)
                skyCor = skyShift
                    
                logfile.write('Sky slices corrected by pixel offset\n')
                hdr.add_history('Sky slices corrected by offset')

            # save pixDiff slices for quality control monitoring
            print('Plotting quality control results')
            with PdfPages('quality_control/'+obsLst[i]+'_sky_PixDiff.pdf') as pdf:
                for i_slc in range(len(pixDiff)):
                    slc = pixDiff[i_slc]
                    fig=plt.figure()

                    plt.title('Slice '+str(i_slc))
                    plt.imshow(slc, aspect='auto', origin='lower')

                    plt.colorbar()
                    plt.tight_layout()
                    pdf.savefig(dpi=300)
                    plt.close()                    
             
        if skyScaleCor:
            print('Finding and correcting strength of sky emission lines to best match science observation')
            logfile.write('Finding and correcting strength of sky emission lines to best match science observation\n')
            logfile.write('Using the following parameters:\n')
            logfile.write('Wavlength regions:\n')
            for reg in skyCorRegions:
                logfile.write(str(reg[0]) + ' - ' + str(reg[1])+'\n')
            logfile.write('mxScale: ' + str(skyScaleMx)+'\n')
            logfile.write('sigmaClip: ' + str(skyScaleSigClip)+'\n')
            logfile.write('sigmaClipRounds: '+str(skyScaleSigClipRnds)+'\n')

            #waveArray = 1e9*(np.arange(skyCube.shape[2])*skyHdr['CDELT3'] +skyHdr['CRVAL3'])

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                subSlices, fSlices = postProcess.subScaledSkySlices(waveMap, dataCor, skyCor, regions=skyCorRegions, mxScale=skyScaleMx, sigmaClip=skyScaleSigClip, sigmaClipRounds=skyScaleSigClipRnds,useMaxOnly=skyScaleUseMxOnly)

            #create quality control figures from fSlices
            fMap = postProcess.buildfSlicesMap(fSlices)

            print('Plotting quality control results')
            with PdfPages('quality_control/'+obsLst[i]+'_sky_scalings.pdf') as pdf:
                for i_slc in range(len(fMap)):
                    slc = fMap[i_slc]
                    fig=plt.figure()
                    fig.add_subplot(111)
                    xTickLabel = []
                    for reg in skyCorRegions:
                        xTickLabel.append(str(reg))

                    plt.xlabel('Wavelength regions')
                    plt.title('Slice '+str(i_slc))
                    #ax.set_xticklabels(xTickLabel)
                    xTickNums = np.asarray(np.arange(len(skyCorRegions)))
                    plt.xticks(xTickNums,xTickLabel,rotation=45)
                    plt.imshow(slc, aspect='auto', origin='lower')

                    plt.colorbar()
                    plt.tight_layout()
                    pdf.savefig(dpi=300)
                    plt.close()                    
            
            dataCor = subSlices
            hdr.add_history('Subtracted scaled sky slices using sky obs:')
            hdr.add_history(skyFolder)

        else:
            print('Subtracting sky slices from science slices')
            logfile.write('Subtracting sky slices from science slices:\n')
            logfile.write(skyFolder+'\n')
            for i_slc in range(len(dataCor)):
                dataCor[i_slc] = dataCor[i_slc] - skyCor[i_slc]

            hdr.add_history('Subtracted sky slices using:')
            hdr.add_history(skyFolder)

        if getSkyCorRV:
            #compute RV difference between sky slices and some sky template spectrum

            print('Measuring RV difference between median sky spectrum and template')
            logfile.write('Measuring RV difference between median sky spectrum and template\n')
            logfile.write('Using the following parameters:\n')
                
            if not 'skyEmTemp' in locals():
                skyEmTemp = wifisIO.readTable(skyEmTempFile)

            print('Placing sky slices on uniform wavelength grid')
            skyGrid = createCube.waveCorAll(skyCor, waveMap, waveGridProps=waveGridProps)
            
            #compute median-averaged sky spectrum
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                avgSkySpec = np.nanmedian(np.nanmedian(skyGrid,axis=0),axis=0)

            tmpHdr = fits.open('processed/'+skyFolder+'_sky_cube.fits')[0].header
            waveArray = 1e9*(np.arange(skyGrid[0].shape[1])*tmpHdr['CDELT3'] +tmpHdr['CRVAL3'])
            del tmpHdr
            
            #interpolate sky template onto observed wavelength grid
            tempInterp = np.interp(waveArray, skyEmTemp[:,0],skyEmTemp[:,1])
            
            rvSkyCor = postProcess.crossCorSpec(waveArray, avgSkySpec, tempInterp, plot=False, oversample=50, absorption=False, mode='idl', contFit1=True, contFit2=False,nContFit=50,contFitOrder=1, mxVel=1000, reject=0, velocity=True)

            hdr.set('RVSKYCOR',rvSkyCor,'RV offset between sky and template')
            print('Found RV offset between sky spectrum and template of ' + str(rvSkyCor))
            logfile.write('Found RV offset between sky spectrum and template of ' + str(rvSkyCor)+'\n')

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

    for i in range(0,len(obsLst)):
        
        #go back to original map/dispersion solution and interpolate all obs onto same grid
        dataCor = wifisIO.readImgsFromFile('processed/'+obsLst[i]+'_obs_slices_distCor.fits')[0]
        hdr = fits.open('processed/'+obsLst[i]+'_obs_slices_fullGrid.fits')[0].header
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

        #combine data by summing all of the same slices together
        if len(combGrid)>0:
            for j in range(len(dataGrid)):
                combGrid[j] += dataGrid[j]
        else:
            combGrid = dataGrid
                
    #now divide each slice by the total number of observations
    for i in range(len(combGrid)):
        combGrid[i] /= float(len(obsLst))
        
    #now create cube from gridded data
    print('Creating mean data cube')
    logfile.write('Creating mean cube from gridded slices\n')
    
    combCube = createCube.mkCube(combGrid, ndiv=ndiv)
    
else:
    combCube, combHdr = wifisIO.readImgsFromFile('processed/'+obsLst[0]+'_obs_cube.fits')

#get object name
objectName = combHdr['Object']   

#check if cube with this name already exists
existsLst = glob.glob('processed/'+objectName+'_combined_cube_*fits')
obsNum = len(existsLst)+1

#need to include JD/BJD and additional corrections
    
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
