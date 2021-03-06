"""

Main script used to process raw Ronchi data and raw data corresponding that provides constraints about the zero-point offset of each slice

Input:
- All variables are read from the configuration file specified by varFile
- A flat field observation that specifies the limits of each slice that is applicable to the Ronchi data (and ideally the zero-point data, if needed)

Produces:
- The ramp image of the Ronchi data (XXX_ronchi.fits)
- A multi-extension file containing the extracted slices of the Ronchi data (XXX_ronchi_slices.fits)
- The traces of the Ronchi mask, for each slice (XXX_ronchi_traces.fits)
- The polynomial fits to the Ronchi traces, for each slice (XXX_ronchi_poly_traces.fits)
- The distortion map derived from the Ronchi tracing and (if provided) the zero-point offset (XXX_ronchi_distMap.fits)
- The grid parameters to be used for spatial rectification (XXX_ronchi_spatGridProps.dat)
- The combined (and individual) ramp image(s) associated with the zero-point offset observation(s) (XXX_zpnt_obs.fits and XXX_zpnt_obs_comb.fits)
- A multi-extension image containing the extracted slices from the combined zero-point offset data (XXX_zpnt_obs_comb_slices.fits)
- The traces (or the polynomial fits to the traces) of the zero-point offset, for each slice
"""

#change the next two lines as needed
#import matplotlib
#matplotlib.use('gtk3agg')

useRonchiMethod=True
useRonchiMethodAlso=False

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
import shutil

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

#deal with darks
if darkFile is not None and os.path.exists(darkFile):
    dark, darkSig, darkSat = wifisIO.readImgsFromFile(darkFile)[0]
    darkLst = [dark, darkSig]
else:
    print(colorama.Fore.RED+'*** WARNING: No dark image provided, or file does not exist ***'+colorama.Style.RESET_ALL)
    
    if logfile is not None:
        logfile.write('*** WARNING: No dark image provide, or file ' + str(darkFile)+' does not exist ***\n')
    darkLst = [None,None]

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

    
#first check if BPM is provided
if os.path.exists(bpmFile):
    BPM = wifisIO.readImgsFromFile(bpmFile)[0]
    BPM = BPM.astype(bool)
else:
    BPM = None
    
if os.path.exists(ronchiFile):
    ronchiFolder = wifisIO.readAsciiList(ronchiFile).tostring()

    if os.path.exists(ronchiFlatFile):
        ronchiFlatFolder = wifisIO.readAsciiList(ronchiFlatFile).tostring() 
    else:
        logfile.write('*** FAILURE: No provided flat field field input list associated with Ronchi data\n ***')
        raise Warning('No provided flat field field input list associated with Ronchi data  ***')
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
if 'ronchiFlatFolder' in locals():
    if not os.path.exists('processed/'+ronchiFlatFolder+'_flat_limits.fits') or not os.path.exists('processed/'+ronchiFlatFolder+'_flat_slices_norm.fits'):
        print('Processed flat field data does not exist for folder ' +ronchiFlatFolder +', processing flat folder')
        calFlat.runCalFlat(np.asarray([ronchiFlatFolder]), hband=hband, darkLst = darkLst, rootFolder=rootFolder, nlCoef=nlCoef, satCounts=satCounts, BPM = BPM, plot=True, nChannel = nChannel, nRowsAvg=nRowsAvg,rowSplit=nRowSplitFlat,nlSplit=nlSplit, combSplit=nCombSplit,bpmCorRng=flatbpmCorRng, crReject=False, skipObsinfo=False,nlFile=nlFile, bpmFile=bpmFile, satFile=satFile, darkFile=darkFile,satSplit=nSatSplit,logfile=logfile,centGuess=centGuess)


#******************************************************************************************************
#******************************************************************************************************
#start with the Ronchi data

#process Ronchi data first

if ronchiFolder is not None:
    if not os.path.exists('processed/'+ronchiFolder+'_ronchi.fits'):

        ronchi, sigmaImg, satFrame, ronchiHdr = processRamp.auto(ronchiFolder, rootFolder,'processed/'+ronchiFolder+'_ronchi.fits', satCounts, nlCoef, BPM,nChannel=nChannel, rowSplit=nRowSplitFlat, satSplit=nSatSplit, nlSplit=nlSplit, combSplit=nCombSplit, crReject=False, bpmCorRng=flatbpmCorRng, saveAll=True, ron=RON, gain=gain, avgAll=True)
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

        ronchiFlatImg = wifisIO.readImgsFromFile('processed/'+ronchiFlatFolder+'_flat.fits')[0]
        if len(ronchiFlatImg)<4:
            ronchiFlatImg = ronchiFlatImg[0]
            
        flatSlicesLst = wifisIO.readImgsFromFile('processed/'+ronchiFlatFolder+'_flat_slices.fits')[0]
        flatNormLst = wifisIO.readImgsFromFile('processed/'+ronchiFlatFolder+'_flat_slices_norm.fits')[0]

        nSlices = len(flatSlicesLst)/3
        flatSlices = flatSlicesLst[:nSlices]
        flatNorm = flatNormLst[:nSlices]

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
        if not noFlat:
            ronchiFlat = slices.ffCorrectAll(ronchiSlices, flatNorm)
        else:
            ronchiFlat = ronchiSlices
            
        print('Getting Ronchi traces')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            ronchiTraces, ronchiAmps = spatialCor.traceRonchiAll(ronchiFlat, nbin=ronchiNbin, winRng=ronchiWinRng, mxWidth=ronchiMxWidth,smth=ronchiSmth, bright=ronchiBright, flatSlices=flatSlices, MP=True)


        print('Plotting amplitude map')
        #build resolution map
        ampMapLst = spatialCor.buildAmpMap(ronchiTraces, ronchiAmps, ronchiSlices)
                        
        #get median amplitude/contrast measurement
        ampAll = []

        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)

            for f in ronchiAmps:

                #remove points from different traces that share same coordinates, to derive a more accurate average

                for j in range(f.shape[0]-1):
                    for k in range(j+1,f.shape[0]):
                        whr = np.where(np.abs(f[j,:]-f[k,:])<0.5)[0]
                        if len(whr)>0:
                            f[k,whr]=np.nan

                for i in range(f.shape[0]):
                    for j in range(f.shape[1]):
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
        if hband:
            print('Using suitable region of detector to determine Ronchi traces')
            if logfile is not None:
                logfile.write('Ronchi suitable region of detector to determine Ronchi traces:\n')
            if not 'ronchiFlatImg' in locals():
                ronchiFlatImg = wifisIO.readImgsFromFile('processed/'+ronchiFlatFolder+'_flat.fits')[0]
            if len(ronchiFlatImg)<4:
                ronchiFlatImg = ronchiFlatImg[0]
                                                                 
            #only use region with suitable flux
            if dispAxis == 0:
                flatImgMed = np.nanmedian(ronchiFlatImg[4:-4,4:-4], axis=1)
            else:
                flatImgMed = np.nanmedian(ronchiFlatImg[4:-4,4:-4], axis=0)
                            
            flatImgMedGrad = np.gradient(flatImgMed)
            medMax = np.nanargmax(flatImgMed)
            lim1 = np.nanargmax(flatImgMedGrad[:medMax])
            lim2 = np.nanargmin(flatImgMedGrad[medMax:])+medMax
                
            if logfile is not None:
                logfile.write('Using following detector limits for tracing:\n')
                logfile.write(str(lim1)+ ' ' + str(lim2)+'\n')
                
        #get polynomial fits
        ronchiPolyTraces = []
        with PdfPages('quality_control/'+ronchiFolder+'_ronchi_traces.pdf') as pdf:
            interval = ZScaleInterval()
        
            for i in range(len(ronchiSlices)):
                trace = ronchiTraces[i]

                #add specific details to deal with bad ronchi traces here
                #use this section to modify the ranges used for the polynomial traces
                
                #comment the line below and modify the specific slice 
                goodReg=[[0,2040]]

                if i==0:
                    pass
                elif i==1:
                    pass
                elif i==2:
                     pass
                elif i==3:
                    pass
                elif i==4:
                    pass
                elif i==5:
                    goodReg = []
                    for j in range(15):
                        goodReg.append([0,2040])
                    goodReg[-1] = [0,1000]
                elif i==6:
                    pass
                elif i==7:
                    pass
                elif i==8:
                    pass
                elif i==9:
                    pass
                elif i==10:
                     pass
                elif i==11:
                    pass
                elif i==12:
                    pass
                elif i==13:
                    pass
                elif i==14:
                    pass
                elif i==15:
                    goodReg = []
                    for j in range(15):
                        goodReg.append([0,2040])
                    goodReg[8] = [0,1500]
                    goodReg[10] = [0,1500]
    
                elif i==16:
                    pass
                elif i==17:
                    pass

                polyTrace = spatialCor.polyFitRonchiTrace(trace, goodReg, order=ronchiPolyOrder, sigmaClipRounds=ronchiSigmaClipRounds)

                #add specific details to deal with bad ronchi traces here
                #use this section to remove bad or extra traces that are deminishing the quality of the distortion map

                if i==0:
                    polyTrace=polyTrace[:-1,:]
                elif i==1:
                    polyTrace = polyTrace[:-1,:]
                elif i==2:
                     pass
                elif i==3:
                    pass
                elif i==4:
                    pass
                elif i==5:
                    polyTrace[-1, 1000:]=np.nan
                elif i==6:
                    pass
                elif i==7:
                    pass
                elif i==8:
                    pass
                elif i==9:
                    pass
                elif i==10:
                     pass
                elif i==11:
                    pass
                elif i==12:
                    pass
                elif i==13:
                    pass
                elif i==14:
                    polyTrace[1,:750]=np.nan
                elif i==15:
                    polyTrace[-1, :1000]=np.nan
                    #polyTrace[10, 1500:]=np.nan
                elif i==16:
                    polyTrace[-1,1000:]=np.nan
                elif i==17:
                    polyTrace[-1,:]=np.nan
                    
                polyTrace[np.logical_or(polyTrace<0, polyTrace>=ronchiSlices[i].shape[0])] = np.nan
                                        
                ronchiPolyTraces.append(polyTrace)
                                    
                plt.ioff()
                fig = plt.figure()
                lims = interval.get_limits(ronchiSlices[i])
                plt.imshow(ronchiSlices[i], aspect='auto', origin='lower', cmap='jet', clim=lims)
                plt.colorbar()
                plt.title('slice number ' + str(i)+', # of traces found: ' + str(len(trace))+', # of traces used: '+str(len(polyTrace)))

                for j in range(trace.shape[0]):
                    plt.plot(np.arange(trace.shape[1]),trace[j,:],'k', linewidth=2)
                for j in range(polyTrace.shape[0]):
                    plt.plot(np.arange(polyTrace.shape[1]), polyTrace[j,:],'r--', linewidth=1)
                plt.tight_layout()
                pdf.savefig(dpi=300)
                plt.close(fig)

        print('Saving Ronchi trace results')
        wifisIO.writeFits(ronchiPolyTraces, 'processed/'+ronchiFolder+'_ronchi_poly_traces.fits', ask=False)

#******************************************************************************************************
#******************************************************************************************************
#check if processed flat field already exists for zero-point data, if not process the flat folder

if zpntFlatFolder is not None:
    if not os.path.exists('processed/'+zpntFlatFolder+'_flat_limits.fits') and not os.path.exists('processed/'+zpntFlatFolder+'_flat_slices_norm.fits'):
        print('Flat limits do not exist for folder ' +zpntFlatFolder +', processing flat folder')

        if 'ronchiFlatFolder' in locals():
            calFlat.runCalFlat(np.asarray([zpntFlatFolder]), hband=hband, darkLst = darkLst, rootFolder=rootFolder, nlCoef=nlCoef, satCounts=satCounts, BPM = BPM, plot=True, nChannel = nChannel, nRowsAvg=nRowsAvg,rowSplit=nRowSplitFlat,nlSplit=nlSplit, combSplit=nCombSplit,bpmCorRng=flatbpmCorRng, crReject=False, skipObsinfo=False,nlFile=nlFile, bpmFile=bpmFile, satFile=satFile, darkFile=darkFile, flatCutOff=zpntFlatCutOff,distMapLimitsFile='processed/'+ronchiFlatFolder+'_flat_limits.fits',logfile=logfile,centGuess=centGuess)
        else:
            calFlat.runCalFlat(np.asarray([zpntFlatFolder]), hband=hband, darkLst = darkLst, rootFolder=rootFolder, nlCoef=nlCoef, satCounts=satCounts, BPM = BPM, plot=True, nChannel = nChannel, nRowsAvg=nRowsAvg,rowSplit=nRowSplitFlat,nlSplit=nlSplit, combSplit=nCombSplit,bpmCorRng=flatbpmCorRng, crReject=False, skipObsinfo=False,nlFile=nlFile, bpmFile=bpmFile, satFile=satFile, darkFile=darkFile, flatCutOff=zpntFlatCutOff,logfile=logfile,centGuess=centGuess)

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

                zpntObs, zpntSigma, zptnSatFrame, zpntHdr = processRamp.auto(zpntFolder, rootFolder,'processed/'+zpntFolder+'_zpnt_obs.fits', satCounts, nlCoef, BPM, nChannel=nChannel, rowSplit=nRowSplit, nlSplit=nlSplit, combSplit=nCombSplit, crReject=False, bpmCorRng=obsBpmCorRng,nlFile=nlFile,satFile=satFile,bpmFile='', gain=gain, ron=RON,logfile=logfile,nRows=nRowsAvg, obsCoords=obsCoords,saveAll=True, rampNum=None, avgAll=True)
                
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

                    sky, skySigmaImg, skySatFrame, skyHdr = processRamp.auto(skyFolder, rootFolder,'processed/'+skyFolder+'_sky.fits', satCounts, nlCoef, BPM, nChannel=nChannel, rowSplit=nRowSplit, nlSplit=nlSplit, combSplit=nCombSplit, crReject=False, bpmCorRng=obsBpmCorRng, rampNum=None,nlFile=nlFile,satFile=satFile,bpmFile=bpmFile, gain=gain, ron=RON,logfile=logfile,nRows=nRowsAvg, obsCoords=obsCoords,avgAll=True)
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

        zpntFlatImg = wifisIO.readImgsFromFile('processed/'+zpntFlatFolder+'_flat.fits')[0]
        if len(zpntFlatImg)<4:
            zpntFlatImg = zpntFlatImg[0]
            
        flatNormLst = wifisIO.readImgsFromFile('processed/'+zpntFlatFolder+'_flat_slices_norm.fits')[0]

        #extract the proper slices
        nSlices = len(flatNormLst)/3
        flatNorm = flatNormLst[:nSlices]
        flatSigma = flatNormLst[nSlices:2*nSlices]
                
        if flatCor:
            if os.path.exists(flatCorFile):
                print('Correcting flat field response function')
                logfile.write('Correcting flat field response function using file:\n')
                logfile.write(flatCorFile+'\n')
        
                flatCorSlices = wifisIO.readImgsFromFile(flatCorFile)[0]
                flatNorm = slices.ffCorrectAll(flatNorm, flatCorSlices)

                if len(flatCorSlices)>nSlices:
                    logfile.write('*** WARNING: Response correction does not include uncertainties ***\n')
                    flatSigma = wifisUncertainties.multiplySlices(flatNorm,flatSigma,flatCorSlices[:nSlices],flatCorSlices[nSlices:2*nSlices])
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

        if useRonchiMethod:

            print ('Interpolating from Ronch Slices')
            #iterate through each slice, find the position of the zero-point observation in the middle of the detector
            #then use the two closest ronchi traces to create a zero-point trace

            zpntTraces = []

            for zpntSlc, ronchiSlc in zip(zpntFlat, ronchiPolyTraces):

                #get average profile of middle 10 pixels
                mid = int(len(zpntSlc)/2.)
                rng = np.round(np.array([mid-5, mid+5])).astype(int)

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                
                    prof = np.nanmean(zpntSlc[:,rng],axis=1)

                    #now get gaussian fit to profile to find middle
                    finite_rng = np.where(np.isfinite(prof))[0]
                    x = np.arange(prof.shape[0])
                    fit = spatialCor.gaussFit(x[finite_rng],prof[finite_rng], plot=False)

                    #find nearest two traces
                    low = np.where(np.nanmean(ronchiSlc[:,rng],axis=1)<=fit[2])[0][-1]
                    high = np.where(np.nanmean(ronchiSlc[:,rng],axis=1)>=fit[2])[0][0]

                diff = ronchiSlc[high, mid] - ronchiSlc[low,mid]
                w1 = (fit[2]-ronchiSlc[low,mid])/diff
                w2 = (ronchiSlc[high,mid]-fit[2])/diff
                zpntTraces.append(w2*ronchiSlc[low,:] + w1*ronchiSlc[high,:])
                
            wifisIO.writeFits(zpntTraces, 'processed/'+zpntLst[0]+'_zpnt_traces.fits',ask=False)
                    
        else:
            print ('Tracing from full zero-point observation')

            if hband:
                #read in flat field file and use it to determine limits
                print('Using suitable region of detector to determine zero-point traces')
                if logfile is not None:
                    logfile.write('Using suitable region of detector to determine zero-point traces:\n')
                
                #only use region with suitable flux
                if dispAxis == 0:
                    flatImgMed = np.nanmedian(zpntFlatImg[4:-4,4:-4], axis=1)
                else:
                    flatImgMed = np.nanmedian(zpntFlatImg[4:-4,4:-4], axis=0)
                            
                flatImgMedGrad = np.gradient(flatImgMed)
                medMax = np.nanargmax(flatImgMed)
                lim1 = np.nanargmax(flatImgMedGrad[:medMax])
                lim2 = np.nanargmin(flatImgMedGrad[medMax:])+medMax
                
                if logfile is not None:
                    logfile.write('Using following detector limits for tracing:\n')
                    logfile.write(str(lim1)+ ' ' + str(lim2)+'\n')

                zpntTraces = spatialCor.traceWireFrameAll(zpntFlat, nbin=zpntNbin, bright=zpntBright, MP=True, plot=False, smooth=zpntSmooth, winRng=zpntWinRng,mxChange=zpntMxChange,constRegion=[lim1,lim2])
            else:
                zpntTraces = spatialCor.traceWireFrameAll(zpntFlat, nbin=zpntNbin, bright=zpntBright, MP=True, plot=False, smooth=zpntSmooth, winRng=zpntWinRng,mxChange=zpntMxChange)

            #optional section to address problematic slices
            print('Fixing problematic traces')
            #use a COG/COL algorithm to define the trace instead
            zpntTraces[0] = spatialCor.traceWireFrameSliceCOG([zpntFlat[0], zpntNbin, zpntWinRng,zpntSmooth,zpntBright,None,5])
            zpntTraces[1] = spatialCor.traceWireFrameSliceCOG([zpntFlat[1], zpntNbin, zpntWinRng,zpntSmooth,zpntBright,None,5])
        
            #now carry out polynomial fitting to further smooth the fits
            polyFitLst = []

            if hband:
                pord=2
                #xfit = np.arange(lim1,lim2)
                #x = np.arange(lim1,lim2)
                
            else:
                pord=2
                #xfit = np.arange(2040)
            x = np.arange(zpntSlices[0].shape[1])

            print('plotting results')
            with PdfPages('quality_control/'+zpntLst[0]+'_zpnt_traces.pdf') as pdf:
                for i in range(len(zpntSlices)):

                    #add specific details to deal with bad zpnt traces here
                    #comment out this line and replace the necessary fitting range in the corresponding slice below
                    xfit = np.where(np.isfinite(zpntTraces[i]))[0]

                    if i==0:
                        pass
                    elif i==1:
                        pass
                    elif i==2:
                        pass
                    elif i==3:
                        pass
                    elif i==4:
                        pass
                    elif i==5:
                        pass
                    elif i==6:
                        pass
                    elif i==7:
                        pass
                    elif i==8:
                        pass
                    elif i==9:
                        pass
                    elif i==10:
                        pass
                    elif i==11:
                        pass
                    elif i==12:
                        pass
                    elif i==13:
                        pass
                    elif i==14:
                        pass
                    elif i==15:
                        pass
                    elif i==16:
                        pass
                    elif i==17:
                        xfit = np.where(np.isfinite(zpntTraces[i]))[0]
                        xfit = xfit[xfit >=250]

                    fig=plt.figure()
                    interval = ZScaleInterval()
                    lims=interval.get_limits(zpntFlat[i])
                    plt.imshow(zpntFlat[i], aspect='auto', interpolation='nearest',cmap='jet', clim=lims, origin='lower')
                    plt.colorbar()
                    plt.plot(x,zpntTraces[i], 'k', linewidth=2)

                    #carry out 1 iterations of sigma-clipping
                    for j in range(2):
                        pcof = np.polyfit(xfit, zpntTraces[i][xfit],pord)
                        poly = np.poly1d(pcof)

                        res = (zpntTraces[i][xfit]-poly(xfit))
                        med = np.nanmedian(res)
                        std = np.nanstd(res)
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore',RuntimeWarning)
                            xfit = x[np.where(np.abs(res+med)<2.*std)[0]]
                            xfit =xfit[np.where(np.isfinite(zpntTraces[i][xfit]))]
                        
                    pcof = np.polyfit(xfit, zpntTraces[i][xfit],pord)
                    poly = np.poly1d(pcof)
         
                    plt.plot(x,poly(x), 'r--')
                    polyFitLst.append(np.clip(poly(x),0,zpntFlat[i].shape[0]-1))
                    plt.title('Slice ' + str(i))
                    plt.tight_layout()

                    pdf.savefig(dpi=300)
                    plt.close()

            
            if useRonchiMethodAlso:

                print ('Interpolating from Ronch Slices using zero-point traces as offset')
                #iterate through each slice, find the position of the zero-point observation in the middle of the detector
                #then use the two closest ronchi traces to create a zero-point trace
            
                zpntTraces = []
                
                for zpntSlc, ronchiSlc in zip(polyFitLst, ronchiPolyTraces):

                    #now get gaussian fit to profile to find middle
                    mid = int(len(zpntSlc)/2.)
                    fit = zpntSlc[mid]
                    
                    #find nearest two traces
                    low = np.where(ronchiSlc[:,mid]<=fit)[0][-1]
                    high = np.where(ronchiSlc[:,mid]>=fit)[0][0]

                    diff = ronchiSlc[high, mid] - ronchiSlc[low,mid]
                    w1 = (fit-ronchiSlc[low,mid])/diff
                    w2 = (ronchiSlc[high,mid]-fit)/diff
                    zpntTraces.append(w2*ronchiSlc[low,:] + w1*ronchiSlc[high,:])
                
            wifisIO.writeFits(polyFitLst, 'processed/'+zpntLst[0]+'_zpnt_traces.fits',ask=False)
            
    else:
        zpntTraces = wifisIO.readImgExtFromFile('processed/'+zpntLst[0]+'_zpnt_traces.fits')[0]
else:
    zpntTraces = None


#******************************************************************************************************
#******************************************************************************************************
#Now combine everything together to get a full distortion map

if 'ronchiFolder' in locals() and ronchiFolder is not None:
    if os.path.exists('processed/'+ronchiFolder+'_ronchi_distMap.fits') and os.path.exists('processed/'+ronchiFolder+'_ronchi_spatGridProps.dat'):
        cont = wifisIO.userInput('Distortion map files already exist, do you want to continue (y/n)')
    else:
        cont ='y'
else:
    cont='n'

if cont.lower()=='y':
    if ronchiTraces is not None:
        #get full distortion maps
        print('Creating distortion map')

        if zpntTraces is None:
            print(colorama.Fore.RED+'*** WARNING: No zero-point offset traces used to determine distortion map ***'+colorama.Style.RESET_ALL)
            logfile.write('*** WARNING: No zero-point offset traces used to determine distortion map ***\n')

        distMap = spatialCor.extendTraceAll2(ronchiPolyTraces, ronchiSlices, zpntTraces,order=1, MP=True, method='linear', mappingMethod='polyFit')

        #write maps
        wifisIO.writeFits(distMap, 'processed/'+ronchiFolder+'_ronchi_distMap.fits', ask=False)

        #copy flat field associated with distortion map to new file
        shutil.copyfile('processed/'+ronchiFlatFolder+'_flat_limits.fits','processed/'+ronchiFolder+'_ronchi_distMap_limits.fits')
                        
        #distortion correct the flat field
        flatSlices = wifisIO.readImgsFromFile('processed/'+ronchiFlatFolder+'_flat_slices.fits')[0]
        nSlices = len(flatSlices)/3
        flatSlices = flatSlices[:nSlices]

        print('Distortion correcting distortion map to get spatial limits')

        try:
            flatCor = createCube.distCorAll_CL(flatSlices, distMap)
        except:
            print(colorama.Fore.RED+'*** WARNING: AKIMA INTERPOLATION FAILED, FALLING BACK TO LINEAR INTERPOLATION ***'+colorama.Style.RESET_ALL)
            flatCor = createCube.distCorAll(flatSlices, distMap, method='linear')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            trimLims = slices.getTrimLimsAll(flatCor, spatTrim)

        #try:
        distCor = createCube.distCorAll_CL(distMap, distMap)
        #except:
        #    print(colorama.Fore.RED+'*** WARNING: AKIMA INTERPOLATION FAILED, FALLING BACK TO LINEAR INTERPOLATION ***'+colorama.Style.RESET_ALL)
        #    distCor = createCube.distCorAll(distMap, distMap, method='linear')

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
logfile.close()
