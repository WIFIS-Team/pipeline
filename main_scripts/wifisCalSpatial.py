"""

Calibrates Ronchi mask and zero-point file

Requires:
- 

Produces:
-


"""
import matplotlib
matplotlib.use('gtkagg')
import wifisIO
import wifisSlices as slices
import wifisSpatialCor as spatialCor
import matplotlib.pyplot as plt
import numpy as np
import wifisBadPixels as badPixels
import wifisCreateCube as createCube
from matplotlib.backends.backend_pdf import PdfPages
import os
import wifisProcessRamp as processRamp
import colorama
import wifisUncertainties
import wifisHeaders as headers
import wifisCalFlatFunc as calFlat
import time
import warnings
from astropy.visualization import ZScaleInterval
import wifisBadPixels as badPixels

#*****************************************************************************
#support function

def fixSatPixelsAll(allSlices):
    for j in range(len(allSlices)):
        sliceImg = allSlices[j]
        x = np.arange(sliceImg.shape[0])
        sliceOut = sliceImg[:]
    
        for i in range(sliceImg.shape[1]):
            xGood = np.where(np.isfinite(sliceImg[:,i]))[0]
            xBad = np.where(~np.isfinite(sliceImg[:,i]))[0]
            yInt = sliceImg[:,i]
            ytmp = yInt[xGood]
            yInt[xBad] = np.interp(xBad,xGood, ytmp, left=np.nan, right=np.nan)
            sliceOut[:,i] = yInt
        allSlices[j] = sliceOut
        
    return allSlices
#*****************************************************************************
#main script starts here

colorama.init()

#INPUT VARIABLE FILE NAME
varFile = 'wifisConfig.inp'

#*****************************************************************************
#************************** user input ***********************************
#initialize all variables.
#DO NOT CHANGE VALUES HERE, EDIT THE 'variables.inp' FILE, WHICH OVERWRITES VALUES HERE

ronchiFile = 'ronchi.lst' #expected to be a single entry
ronchiFlatFile = 'ronchiFlat.lst' #expected to be a single entry
zpntFlatFile = 'zpntFlat.lst' #expected to be a single entry
zpntLstFile = 'zpntObs.lst'
zpntSkyFile = 'zpntSky.lst'
darkFile = ''
rootFolder = '/data/WIFIS/H2RG-G17084-ASIC-08-319'
satFile = '/home/jason/wifis/data/non-linearity/may/processed/master_detLin_satCounts.fits'
nlFile = '/home/jason/wifis/data/non-linearity/may/processed/master_detLin_NLCoeff.fits' 
bpmFile = '/data/pipeline/external_data/bpm.fits'
hband = False

#flat field specific options
flatWinRng = 51
flatImgSmth = 5
flatPolyFitDegree=2
flatbpmCorRng=20 
flatCorFile = 'processed/flat_correction_slices.fits'
flatCor = True
zpntNbin=1
zpntBright=True
zpntSmooth=10
zpntWinRng=61
zpntMxChange=5
obsBpmCorRng = 1
zpntFlatCutOff = 0.2

#ronchi tracing parameters
ronchiNbin=1
ronchiWinRng=7
ronchiMxWidth=3
ronchiSmth=20
ronchiBright=False
ronchiPolyOrder=2
ronchiSigmaClipRounds=2

#spatial grid parameters
spatTrim = 0.5

#parameters used for processing of ramps
nChannel=32 
nRowsAvg=4 
nRowSplit=1 
nlSplit=32 
nSatSplit=32 
nCombSplit=32 
gain =1.
ron = 1.
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

#first calibrate ronchi flat
#second calibrate zpnt flat, if exists
#then calibrate zero-point, if exists
#lastly, calibrate ronchi

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

satCounts = wifisIO.readImgsFromFile(satFile)[0]
nlCoeff = wifisIO.readImgsFromFile(nlFile)[0]

#deal with darks
if darkFile is not None and os.path.exists(darkFile):
    dark, darkSig, darkSat = wifisIO.readImgsFromFile(darkFile)[0]
    darkLst = [dark, darkSig]
else:
    print(colorama.Fore.RED+'*** WARNING: No dark image provided, or file does not exist ***'+colorama.Style.RESET_ALL)
    
    if logfile is not None:
        logfile.write('*** WARNING: No dark image provide, or file ' + str(darkFile)+' does not exist ***\n')
    darkLst = [None,None]
    
#first check if BPM is provided
if os.path.exists(bpmFile):
    BPM = wifisIO.readImgsFromFile(bpmFile)[0]
    BPM = BPM.astype(bool)
else:
    BPM = None
    
if os.path.exists(ronchiFlatFile):
    ronchiFlatFolder = wifisIO.readAsciiList(ronchiFlatFile).tostring() 
else:
    logfile.write('*** FAILURE: No provided flat field field input list associated with Ronchi data\n ***')
    raise Warning('No provided flat field field input list associated with Ronchi data  ***')

if os.path.exists(ronchiFile):
    ronchiFolder = wifisIO.readAsciiList(ronchiFile).tostring()
else:
    print(colorama.Fore.RED+'*** WARNING: No Ronchi input list provided. Skipping processing of Ronchi data ***'+colorama.Style.RESET_ALL)

    logfile.write('*** WARNING: No Ronci input list provided. Skipping processing of Ronchi data\n ***')
    ronchiFolder = None

if os.path.exists(zpntFlatFile):
    zpntFlatFolder = wifisIO.readAsciiList(zpntFlatFile).tostring()
else:
    print(colorama.Fore.RED+'*** WARNING: No provided flat field field input list associated with zero-point offset data. Skipping processing of zero-point offset data ***'+colorama.Style.RESET_ALL)
    logfile.write('*** WARNING: No provided flat field field input list associated with zero-point offset data. Skipping processing of zero-point offset data\n ***')
    zpntFlatFolder = None

if os.path.exists(zpntLstFile):
    zpntLst = wifisIO.readAsciiList(zpntLstFile)
    if zpntLst.ndim==0:
        zpntLst = np.asarray([zpntLst])
else:
    print(colorama.Fore.RED+'*** WARNING: No zero-point input list provided. Skipping processing of zero-point data ***'+colorama.Style.RESET_ALL)

    logfile.write('*** WARNING: No zero-point input list provided. Skipping processing of zero-point data\n ***')
    zpntLst = None

if zpntSkyFile is not None and os.path.exists(zpntSkyFile):
    zpntSkyLst = wifisIO.readAsciiList(zpntSkyFile)
    if zpntSkyLst.ndim==0:
        zpntSkyLst = np.asarray([zpntSkyLst])
else:
    print(colorama.Fore.RED+'*** WARNING: No sky frames associated with zero-point observations provided. Skipping sky subtraction ***'+colorama.Style.RESET_ALL)

    logfile.write('*** WARNING: No sky frames associated with zero-point observations provided. Skipping sky subtraction ***\n')
    zpntSkyLst = None

#******************************************************************************************************
#******************************************************************************************************
#check if processed flat field already exists for Ronchi, if not process the flat

if not os.path.exists('processed/'+ronchiFlatFolder+'_flat_limits.fits') or not os.path.exists('processed/'+ronchiFlatFolder+'_flat_slices_norm.fits'):
    print('Processed flat field data does not exist for folder ' +ronchiFlatFolder +', processing flat folder')
    calFlat.runCalFlat(np.asarray([ronchiFlatFolder]), hband=hband, darkLst = darkLst, rootFolder=rootFolder, nlCoef=nlCoef, satCounts=satCounts, BPM = BPM, plot=True, nChannel = nChannel, nRowsAvg=nRowsAvg,rowSplit=nRowSplit,nlSplit=nlSplit, combSplit=nCombSplit,bpmCorRng=flatbpmCorRng, crReject=False, skipObsinfo=False,nlFile=nlFile, bpmFile=bpmFile, satFile=satFile, darkFile=darkFile,satSplit=nSatSplit)

#******************************************************************************************************
#******************************************************************************************************
#check if processed flat field already exists for zero-point data, if not process the flat folder

if zpntFlatFolder is not None:
    if not os.path.exists('processed/'+zpntFlatFolder+'_flat_limits.fits') and not os.path.exists('processed/'+zpntFlatFolder+'_flat_slices_norm.fits'):
        print('Flat limits do not exist for folder ' +zpntFlatFolder +', processing flat folder')

        calFlat.runCalFlat(np.asarray([zpntFlatFolder]), hband=hband, darkLst = darkLst, rootFolder=rootFolder, nlCoef=nlCoef, satCounts=satCounts, BPM = BPM, plot=True, nChannel = nChannel, nRowsAvg=nRowsAvg,rowSplit=nRowSplit,nlSplit=nlSplit, combSplit=nCombSplit,bpmCorRng=flatbpmCorRng, crReject=False, skipObsinfo=False,nlFile=nlFile, bpmFile=bpmFile, satFile=satFile, darkFile=darkFile, flatCutOff=zpntFlatCutOff,distMapLimitsFile='processed/'+ronchiFlatFolder+'_flat_limits.fits')

#******************************************************************************************************
#******************************************************************************************************
#now process the zero-point data. If it is a list, then co-add all ramp images
if zpntLst is not None:
    if os.path.exists('processed/'+zpntLst[0]+'_zpnt_obs_comb.fits'):
        cont = wifisIO.userInput('Combined zero-point offset observation already exists, do you want to reprocess (y/n)?')
        if not cont.lower() == 'y':
            print('Reading combined zero-point data from file '+'processed/'+zpntLst[0]+'_zpnt_obs_comb.fits')
            logfile.write('Reading combined zero-point data from file'+'processed/'+zpntLst[0]+'_zpnt_obs_comb.fits\n')
            zpntComb = wifisIO.readImgsFromFile('processed/'+zpntLst[0]+'_zpnt_obs_comb.fits')[0]
    else:
        cont='y'

    if cont.lower()=='y':
        obsAll = []

        print('Processing and combining all zero-point observations')
        logfile.write('Processing and combinging all zero-point observations\n')

        for i in range(len(zpntLst)):
            zpntFolder = zpntLst[i]
                
            if not os.path.exists('processed/'+zpntFolder+'_zpnt_obs.fits'):
                print('Processing ' + zpntFolder)
                logfile.write('Processing '+ zpntFolder+'\n')

                zpntObs, zpntSigma, zptnSatFrame, zpntHdr = processRamp.auto(zpntFolder, rootFolder,'processed/'+zpntFolder+'_zpnt_obs.fits', satCounts, nlCoeff, BPM, nChannel=nChannel, rowSplit=nRowSplit, nlSplit=nlSplit, combSplit=nCombSplit, crReject=False, bpmCorRng=obsBpmCorRng,nlFile=nlFile,satFile=satFile,bpmFile='', gain=gain, ron=ron,logfile=logfile,nRows=nRowsAvg, obsCoords=obsCoords,saveAll=True, rampNum=None, avgAll=True)
                
            else:
                print('Processed data already exists for ' + zpntFolder + '. Reading data instead')
                logfile.write('Processed data already exists for ' + zpntFolder + '. Reading data instead\n')

                zpntObsLst, zpntHdr = wifisIO.readImgsFromFile('processed/'+zpntFolder+'_zpnt_obs.fits')
                zpntObs = zpntObsLst[0]
                zpntHdr = zpntHdr[0]
                
            #carry out sky subtraction
            if zpntSkyLst is not None:
                skyFolder = zpntSkyLst[i]
        
                if not os.path.exists('processed/'+skyFolder+'_sky.fits'):
                    print('Processing sky folder '+skyFolder)
                    logfile.write('\nProcessing sky folder ' + skyFolder+'\n')

                    sky, skySigmaImg, skySatFrame, skyHdr = processRamp.auto(skyFolder, rootFolder,'processed/'+skyFolder+'_sky.fits', satCounts, nlCoeff, BPM, nChannel=nChannel, rowSplit=nRowSplit, nlSplit=nlSplit, combSplit=nCombSplit, crReject=False, bpmCorRng=obsBpmCorRng, rampNum=None,nlFile=nlFile,satFile=satFile,bpmFile=bpmFile, gain=gain, ron=ron,logfile=logfile,nRows=nRowsAvg, obsCoords=obsCoords,avgAll=True)
                else:
                    print('Reading sky data from ' + skyFolder)
                    logfile.write('Reading processed sky image from:\n')
                    logfile.write('processed/'+skyFolder+'_sky.fits\n')
                        
                    skyDataLst,skyHdr = wifisIO.readImgsFromFile('processed/'+skyFolder+'_sky.fits')
                    sky = skyDataLst[0]
                    skySigmaImg = skyDataLst[1]
                    skySatFrame = skyDataLst[2]
                    skyHdr = skyHdr[0]
                    del skyDataLst
            
                print('Subtracting sky from obs')
                logfile.write('Subtracting sky flux from zero-point image flux\n')
                zpntObs -= sky
                zpntHdr.add_history('Subtracted sky flux image using:')
                zpntHdr.add_history(skyFolder)
            
            obsAll.append(zpntObs)
                
        print('Co-adding all zero-point data into a single image')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            zpntComb = np.sum(np.asarray(obsAll),axis=0)

        wifisIO.writeFits(zpntComb,'processed/'+zpntLst[0]+'_zpnt_obs_comb.fits', ask=False)

    #now trace the data 
    if os.path.exists('processed/'+zpntLst[0]+'_zpnt_traces.fits'):
        cont = wifisIO.userInput('Zero-point traces already exist, do you want to re-trace (y/n)?')
        if not cont.lower() == 'y':
            zpntTraces = wifisIO.readImgExtFromFile('processed/'+zpntLst[0]+'_zpnt_traces.fits')[0]
    else:
        cont='y'

    if cont.lower()=='y':
        if zpntFlatFolder is not None:
                print('Reading slice limits')
                logfile.write('Reading slice limits from file:\n')
                logfile.write('processed/'+zpntFlatFolder+'_flat_limits.fits\n')
        else:
            print(colorama.Fore.RED+'*** WARNING: No flat field associated with zero-point observations. Using Ronchi flat instead ***'+colorama.Style.RESET_ALL)
            logfile.write('*** WARNING: No flat field associated with zero-point observations. Using Ronchi flat instead ***\n')
            logfile.write('Reading slice limits from file:\n')
            logfile.write('processed/'+ronchiFlatFolder+'_flat_limits.fits\n')
            zpntFlatFolder = ronchiFlatFolder
            
        limits, limitsHdr = wifisIO.readImgsFromFile('processed/'+zpntFlatFolder+'_flat_limits.fits')
        shft = limitsHdr['LIMSHIFT']

        print('Reading flat field response function')
        logfile.write('Reading flat field response function from file:\n')
        logfile.write('processed/'+zpntFlatFolder+'_flat_slices_norm.fits\n')
                
        flatNormLst = wifisIO.readImgsFromFile('processed/'+zpntFlatFolder+'_flat_slices_norm.fits')[0]
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
                        
        #extract slices
        print('Extracting slices')
        logfile.write('Extracting slices\n')
        zpntSlices = slices.extSlices(zpntComb[4:-4,4:-4], limits, shft=shft, dispAxis=dispAxis)
        wifisIO.writeFits(zpntSlices, 'processed/'+zpntLst[0]+'_zpnt_obs_comb_slices.fits',ask=False)

        #correct remaining bad pixels using linear interpolation across the spatial axis
        zpntSlices = fixSatPixelsAll(zpntSlices)
        
        #apply flat field
        print('Applying flat field corrections')
        logfile.write('Applying flat field corrections\n')
        zpntFlat = slices.ffCorrectAll(zpntSlices, flatNorm)
            
        #now find traces
        print('Tracing zero-point offset slices')
        logfile.write('Tracing zero-point offset slices\n')
        
        zpntTraces = spatialCor.traceWireFrameAll(zpntFlat, nbin=zpntNbin, bright=zpntBright, MP=True, plot=False, smooth=zpntSmooth, winRng=zpntWinRng,mxChange=zpntMxChange)

        #optional section to address problematic slices
        #print('Fixing problematic traces')
                
        #now carry out polynomial fitting to further smooth the fits
        polyFitLst = []

        if hband:
            #change the code here reflect limited range with hband data
            pass
        else:
            xfit = np.arange(2040)

        x = np.arange(zpntSlices[0].shape[1])
        print('plotting results')
        with PdfPages('quality_control/'+zpntLst[0]+'_zpnt_traces.pdf') as pdf:
            for i in range(len(zpntSlices)):
                
                #if i==17:
                #    xfit = np.arange(1100)
                #    pord=3
                #else:
                xfit = np.where(np.isfinite(zpntTraces[i]))[0]
                pord=3
                    
                fig=plt.figure()
                interval = ZScaleInterval()
                lims=interval.get_limits(zpntFlat[i])
                plt.imshow(zpntFlat[i], aspect='auto', interpolation='nearest',cmap='jet', clim=lims, origin='lower')
                plt.colorbar()
                plt.plot(x,zpntTraces[i], 'k', linewidth=2)

                #carry out a single iteration of sigma-clipping
                pcof = np.polyfit(xfit, zpntTraces[i][xfit],pord)
                poly = np.poly1d(pcof)
                res = (zpntTraces[i]-poly(x))
                med = np.nanmedian(res)
                std = np.nanstd(res)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    xfit = x[np.where(np.abs(res+med)<2.*std)]
                pcof = np.polyfit(xfit, zpntTraces[i][xfit],3)
                poly = np.poly1d(pcof)
         
                plt.plot(x,poly(x), 'r--')
                polyFitLst.append(np.clip(poly(x),0,zpntFlat[i].shape[0]-1))
                plt.title('Slice ' + str(i))
                plt.tight_layout()

                pdf.savefig(dpi=300)
                plt.close()

        wifisIO.writeFits(polyFitLst, 'processed/'+zpntLst[0]+'_zpnt_traces.fits',ask=False)
        zpntTraces = polyFitLst
    else:
        zpntTraces = wifisIO.readImgExtFromFile('processed/'+zpntLst[0]+'_zpnt_traces.fits')[0]
else:
    zpntTraces = None

#******************************************************************************************************
#******************************************************************************************************
#now deal with the Ronchi data

#process Ronchi data first

if ronchiFolder is not None:
    if not os.path.exists('processed/'+ronchiFolder+'_ronchi.fits'):

        ronchi, sigmaImg, satFrame, ronchiHdr = processRamp.auto(ronchiFolder, rootFolder,'processed/'+ronchiFolder+'_ronchi.fits', satCounts, nlCoeff, BPM,nChannel=nChannel, rowSplit=nRowSplit, satSplit=nSatSplit, nlSplit=nlSplit, combSplit=nCombSplit, crReject=False, bpmCorRng=flatbpmCorRng, saveAll=True)
    else:
        ronchiLst, ronchiHdr = wifisIO.readImgsFromFile('processed/'+ronchiFolder+'_ronchi.fits')
        ronchi = ronchiLst[0]
        ronchiHdr = ronchiHdr[0]

    if os.path.exists('processed/'+ronchiFolder+'_ronchi_traces.fits'):
        cont = wifisIO.userInput('Ronchi traces already exist, do you want to retrace (y/n)?')

        if not cont.lower() =='y':
            ronchiTraces = wifisIO.readImgsFromFile('processed/'+ronchiFolder+'_ronchi_traces.fits')[0]
            ronchiSlices = wifisIO.readImgsFromFile('processed/'+ronchiFolder+'_ronchi_slices.fits')[0]
    else:
        cont='y'
        
    if cont.lower() == 'y':
        print('Reading slice limits')
        logfile.write('Reading slice limits from file:\n')
        logfile.write('processed/'+ronchiFlatFolder+'_flat_limits.fits\n')

        limits, limitsHdr = wifisIO.readImgsFromFile('processed/'+ronchiFlatFolder+'_flat_limits.fits')
        shft = limitsHdr['LIMSHIFT']

        print('Reading flat field response function')
        logfile.write('Reading flat field response function from file:\n')
        logfile.write('processed/'+ronchiFlatFolder+'_flat_slices_norm.fits\n')

        flatSlicesLst = wifisIO.readImgsFromFile('processed/'+ronchiFlatFolder+'_flat_slices.fits')[0]
        flatNormLst = wifisIO.readImgsFromFile('processed/'+ronchiFlatFolder+'_flat_slices_norm.fits')[0]
        flatSlices = flatSlicesLst[:18]
        flatNorm = flatNormLst[:18]

        if flatCor:
            if os.path.exists(flatCorFile):
                print('Correcting flat field response function')
                logfile.write('Correcting flat field response function using file:\n')
                logfile.write(flatCorFile+'\n')
                
                flatCorSlices = wifisIO.readImgsFromFile(flatCorFile)[0]
                flatNorm = slices.ffCorrectAll(flatNorm, flatCorSlices)

        #extract ronchi slices
        ronchiSlices = slices.extSlices(ronchi[4:-4,4:-4], limits, dispAxis=dispAxis)
        wifisIO.writeFits(ronchiSlices, 'processed/'+ronchiFolder+'_ronchi_slices.fits',hdr=ronchiHdr, ask=False)

        #apply flat field correction
        ronchiFlat = slices.ffCorrectAll(ronchiSlices, flatNorm)

        print('Getting Ronchi traces')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            ronchiTraces, ronchiAmps = spatialCor.traceRonchiAll(ronchiFlat, nbin=ronchiNbin, winRng=ronchiWinRng, mxWidth=ronchiMxWidth,smth=ronchiSmth, bright=ronchiBright, flatSlices=flatSlices, MP=True)

            #if needed, address problematic fits here
            #better for May and june
            #ronchiTraces, ronchiAmps = spatialCor.traceRonchiAll(ronchiSlices, nbin=ronchiNbin, winRng=ronchiWinRng, mxWidth=ronchiMxWidth,smth=ronchiSmth, bright=ronchiBright, flatSlices=None, MP=True)

            #needed for june
            #ronchiTraces[4],ronchiAmps[4] = spatialCor.traceRonchiSlice([ronchiFlat[4],ronchiNbin,ronchiWinRng,2040, False,ronchiMxWidth,ronchiSmth,False,flatSlices[4],0.5])
            #ronchiTraces[17],ronchiAmps[17] = spatialCor.traceRonchiSlice([ronchiFlat[17],ronchiNbin,ronchiWinRng,2040, False,ronchiMxWidth,ronchiSmth,False,flatSlices[17],0.5])
            
        print('Plotting amplitude map')
        #build resolution map
        ampMapLst = spatialCor.buildAmpMap(ronchiTraces, ronchiAmps, ronchiSlices)

        #get median FWHM
        ampAll = []
        for f in ronchiAmps:
            for i in range(len(f)):
                for j in range(len(f[i])):
                    ampAll.append(f[i][j])
            
        ampMed = np.nanmedian(ampAll)
        
        ntot = 0
        for r in ronchiSlices:
            ntot += r.shape[0]
    
        ampMap = np.empty((r.shape[1],ntot),dtype='float32')
    
        strt=0
        for a in ampMapLst:
            ampMap[:,strt:strt+a.shape[0]] = a.T
            strt += a.shape[0]

        fig = plt.figure()
        m = np.nanmedian(ampMap)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            s = np.nanstd(ampMap[ampMap < 10*m])
            
        interval = ZScaleInterval()
        clim=interval.get_limits(ampMap)
        
        plt.imshow(ampMap, origin='lower', aspect='auto', clim=clim,cmap='jet')
        plt.title('Ronchi amplitude map - Med amp ' + '{:4.2f}'.format(ampMed))
        
        plt.colorbar()
        plt.tight_layout()
        plt.savefig('quality_control/'+ronchiFolder+'_ronchi_amp_map.pdf',dpi=300)
        plt.close()

        ronchiHdr.set('QC_AMP',ampMed,'Median Ronchi amplitude of all slices')
        wifisIO.writeFits(ronchiTraces,'processed/'+ronchiFolder+'_ronchi_traces.fits',hdr=ronchiHdr,ask=False)

    else:
        ronchiTraces = wifisIO.readImgsFromFile('processed/'+ronchiFolder+'_ronchi_traces.fits')[0]
        ronchiSlices = wifisIO.readImgsFromFile('processed/'+ronchiFolder+'_ronchi_slices.fits')[0]

    if os.path.exists('processed/'+ronchiFolder+'_ronchi_poly_traces.fits'):
        cont = wifisIO.userInput('Ronchi polynomial traces already exist, do you want to retrace (y/n)?')
        if not cont.lower() == 'y':
            print('Reading polynomial traces from file '+'processed/'+ronchiFolder+'_ronchi_poly_traces.fits')
            logfile.write('Reading polynomial traces from file '+'processed/'+ronchiFolder+'_ronchi_poly_traces.fits\n')
            ronchiPolyTraces = wifisIO.readImgsFromFile('processed/'+ronchiFolder+'_ronchi_poly_traces.fits')[0]
    else:
        cont = 'y'

    if cont.lower()=='y':
        print('Getting polynomial fits to traces')
        #get polynomial fits
        ronchiPolyTraces = []
        with PdfPages('quality_control/'+ronchiFolder+'_ronchi_traces.pdf') as pdf:
            interval = ZScaleInterval()
        
            for i in range(len(ronchiSlices)):
                trace = ronchiTraces[i]

                #add specific details to deal with bad traces
                #for june
               #if i==5:
               #    goodReg = []
               #    for j in range(10):
               #        goodReg.append([0,2040])
               #    goodReg.append([0,1000])
               #    goodReg.append([0,2040])
               #    goodReg.append([400,1500])
               #    goodReg.append([0,2040])
               #    goodReg.append([0,2040])
               #    
               #elif i==6:
               #    goodReg = []
               #    for j in range(10):
               #        goodReg.append([0,2040])
               #    goodReg.append([400,2040])
               #    goodReg.append([0,2040])
               #    goodReg.append([0,2040])
               #    goodReg.append([0,2040])
               #    goodReg.append([0,2040])
               #elif i==7:
               #    goodReg =[]
               #    for j in range(10):
               #        goodReg.append([0,2040])
               #    goodReg.append([0,1500])
               #    goodReg.append([0,2040])
               #    goodReg.append([0,2040])
               #    goodReg.append([0,2040])
               #    goodReg.append([0,2040])
               #elif i==15:
               #    goodReg = []
               #    for j in range(10):
               #        goodReg.append([0,2040])
               #    for j in range(5):
               #        goodReg.append([0,1600])
               #elif i==17:
               #    #goodReg = [[0,2040]]
               #    for j in range(11):
               #        goodReg.append([0,2040])
               #    goodReg.append([0,1000])
               #    goodReg.append([0,2040])
                
                    #goodReg=[]
                    #for j in range(9):
                    #    goodReg.append([0,2040])
                    #goodReg.append([0,500])
                    #goodReg.append([300,900])
                    #goodReg.append([300,1750])
              #  else:
              #      goodReg =[[0,2040]]

                #for may
                #if i==13:
                #    goodReg=[]
                #    for j in range(9):
                #        goodReg.append([0,2040])
                #    goodReg.append([0,1500])
                #    for j in range(4):
                #        goodReg.append([0,2040])
                #    goodReg.append([0,1500])
                #
                #elif i==15:
                #    goodReg = [[600,2040],[0,1250],[0,750],[0,500],[0,750],[0,1400],[0,1400],[0,1400],[0,1600],[0,1250],[0,900],[0,1250],[0,1250],[0,1500],[0,1250]]
                #elif i==16:
                #    goodReg=[]
                #    for j in range(4):
                #        goodReg.append([0,2040])
                #    goodReg.append([0,1500])
                #    for j in range(9):
                #        goodReg.append([0,2040])
                #elif i==17:
                #    goodReg=[[0,1100],[0,2040],[0,1100],[0,2040],[0,2040],[0,1500],[0,2040],[0,2040],[0,2040],[0,2040],[0,1100],[750,1750],[0,1100]]
                #else:
                #    goodReg = [[0,2040]]
                
                #for august
                #if i==17:
                #    goodReg = []
                #    for j in range(10):
                #        goodReg.append([0,2040])
                #    goodReg.append([0,1400])
                #else:
                #    goodReg = [[0,2040]]

                #for october
                if i==13:
                    goodReg = [[0,2040],[0,2040],[800,2040]]
                    for i in range(11):
                        goodReg.append([0,2040])
                    
                elif i==17:
                    goodReg = [[0,2040],[0,2040],[0,2040],[0,1400],[0,2040],[0,2040],[0,2040],[0,2040],[0,2040],[0,1550],[0,1500],[0,1100]]
                  
                else:
                    goodReg = [[0,2040]]
                
                #for july
                #if i==13:
                #    goodReg=[[0,1600]]
                #elif i==15:
                #    goodReg=[[0,2040],[0,2040],[0,2040],[0,2040],[0,750]]
                #    for j in range(0,11):
                #        goodReg.append([0,2040])
                #elif i ==17:
                #    goodReg= []
                #    for j in range(10):
                #        goodReg.append([0,2040])
                #    for j in range(3):
                #        goodReg.append([0,1500])
                #else:
                #    goodReg=[[0,2040]]

                polyTrace = spatialCor.polyFitRonchiTrace(trace, goodReg, order=ronchiPolyOrder, sigmaClipRounds=ronchiSigmaClipRounds)

                #more details to deal with bad/extra traces
                #for june
                #if i==5 or i==13:
                #    polyTrace = polyTrace[:-1,:]
                #if i==15:
                #    polyTrace = polyTrace[1:,:]
                #elif i==17:
                #    polyTrace[np.logical_or(polyTrace<0, polyTrace>=ronchiSlices[17].shape[0])] = np.nan
                #    #polyTrace[9,500:] = np.nan
                #    polyTrace = np.delete(polyTrace,10,axis=0)
                #    polyTrace[10,:]=np.nan
                #    polyTrace[11,:] = np.nan
                                
                #for may
                #if i==0 or i==14:
                #    polyTrace=polyTrace[1:,:]
                #elif i==7:
                #    polyTrace=polyTrace[:-1,:]
                #elif i==15:
                #    polyTrace = np.delete(polyTrace,4,axis=0)
                #    polyTrace[2,750:] = np.nan
                #    polyTrace[3,600:] = np.nan
                #    polyTrace[9,900:] = np.nan
                #elif i==17:
                #    polyTrace[np.logical_or(polyTrace<0, polyTrace>=ronchiSlices[17].shape[0])] = np.nan
                    
                
                #for july
                #if i==0:
                #     polyTrace = polyTrace[2:,:]
                #elif i==1:
                #    polyTrace=polyTrace[1:,:]
                #elif i==5:
                #    polyTrace=polyTrace[:-1,:]
                #elif i==8:
                #    polyTrace=polyTrace[:-1,:]
                #elif i ==14:
                #    polyTrace = polyTrace[1:-1,:]
                #elif i==15:
                #    polyTrace[4,:] = np.nan
                #    polyTrace = polyTrace[:-1,:]
                #elif i==16:
                #    polyTrace=polyTrace[:-2,:]
                #    
                #elif i==17:
                #    polyTrace[np.logical_or(polyTrace<0, polyTrace>=ronchiSlices[17].shape[0])] = np.nan
                #polyTrace = polyTrace[1:11]

                #for october
                if i==17:
                    polyTrace[np.logical_or(polyTrace<0, polyTrace>=ronchiSlices[17].shape[0])] = np.nan
                    polyTrace[3,1450:] = np.nan
                    
                ronchiPolyTraces.append(polyTrace)
                                    
                plt.ioff()
                fig = plt.figure()
                lims = interval.get_limits(ronchiSlices[i])
                plt.imshow(ronchiSlices[i], aspect='auto', origin='lower', cmap='jet', clim=lims)#clim=[0,np.nanmedian(ronchiSlices[i])*1.5])
                plt.colorbar()
                plt.title('slice number ' + str(i)+', # of dips: ' + str(len(trace)))
                for j in range(trace.shape[0]):
                    plt.plot(np.arange(trace.shape[1]),trace[j,:],'k', linewidth=2)
                for j in range(polyTrace.shape[0]):
                    plt.plot(np.arange(polyTrace.shape[1]), polyTrace[j,:],'r--', linewidth=1)
                plt.tight_layout()
                pdf.savefig(dpi=300)
                plt.close(fig)

        print('Saving Ronchi trace results')
        wifisIO.writeFits(ronchiPolyTraces, 'processed/'+ronchiFolder+'_ronchi_poly_traces.fits', ask=False)

if os.path.exists('processed/'+ronchiFolder+'_ronchi_distMap.fits') and os.path.exists('processed/'+ronchiFolder+'_ronchi_spatGridProps.dat'):
    cont = wifisIO.userInput('Distortion map files already exist, do you want to continue (y/n)')
else:
    cont ='y'

if cont.lower()=='y':
    if ronchiTraces is not None:
        print('Distortion correcting distortion map to get spatial limits')
        #get full distortion maps

        if zpntTraces is None:
            print(colorama.Fore.RED+'*** WARNING: No zero-point offset traces used to determine distortion map ***'+colorama.Style.RESET_ALL)
            logfile.write('*** WARNING: No zero-point offset traces used to determine distortion map ***\n')
    
        distMap = spatialCor.extendTraceAll2(ronchiPolyTraces, ronchiSlices, zpntTraces,order=3, MP=True, method='linear')

        #write maps
        wifisIO.writeFits(distMap, 'processed/'+ronchiFolder+'_ronchi_distMap.fits', ask=False)
        #distortion correct the flat field
        flatSlices = wifisIO.readImgsFromFile('processed/'+ronchiFlatFolder+'_flat_slices.fits')[0][:18]
        flatCor = createCube.distCorAll(flatSlices, distMap)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            trimLims = slices.getTrimLimsAll(flatCor, spatTrim)
        distCor = createCube.distCorAll(distMap, distMap)
        distTrim = slices.trimSliceAll(distCor, trimLims)
        spatGridProps = createCube.compSpatGrid(distTrim)
    
        print('saving results')
        #write results
        wifisIO.writeTable(spatGridProps, 'processed/'+ronchiFolder+'_ronchi_spatGridProps.dat')
        
        with PdfPages('quality_control/'+ronchiFolder+'_ronchi_distMap.pdf') as pdf:
            for i in range(len(distMap)):
                fig=plt.figure()
                interval = ZScaleInterval()
                plt.imshow(distMap[i], aspect='auto', interpolation='nearest',cmap='jet', origin='lower')
                plt.colorbar()
                plt.title('Slice ' + str(i))
                plt.tight_layout()

                pdf.savefig(dpi=300)
                plt.close()
