"""

Calibrates arc lamp images

Requires:
- 

Produces:
- per pixel wavelength solution


"""

import matplotlib
matplotlib.use('agg')
import numpy as np
import time
import matplotlib.pyplot as plt
import wifisGetSatInfo as satInfo
import wifisNLCor as NLCor
import wifisRefCor as refCor
import os
import wifisIO 
import wifisCombineData as combData
import wifisWaveSol as waveSol
import wifisUncertainties
import wifisBadPixels as badPixels
import wifisSlices as slices
import wifisHeaders as headers
import wifisProcessRamp as processRamp
import wifisCreateCube as createCube
from matplotlib.backends.backend_pdf import PdfPages

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0' # Used to show compile errors for debugging, can be removed
os.environ['PYOPENCL_CTX'] = '1' # Used to specify which OpenCL device to target
plt.ioff()

#*****************************************************************************
#************************** Required input ***********************************
fileList = 'wave.lst' # a table, with first column corresponding to arc folder and second to associated flat

#mostly static input
templateFile = '/data/pipeline/external_data/templateSlices.fits'
prevResultsFile = '/data/pipeline/external_data/prevSol.pkl'
nlFile = '/data/WIFIS/H2RG-G17084-ASIC-08-319/UpTheRamp/20170504201819/processed/master_detLin_NLCoeff.fits'        
satFile = '/data/WIFIS/H2RG-G17084-ASIC-08-319/UpTheRamp/20170504201819/processed/master_detLin_satCounts.fits'
bpmFile = '/data/pipeline/external_data/bpm.fits'
atlasFile = '/data/pipeline/external_data/best_lines2.dat'
distMapFile = '/data/pipeline/external_data/distortionMap.fits'
spatGridPropsFile = '/data/pipeline/external_data/spatGridProps.dat'

#optional behaviour
plot = True
crReject = False
rmsThresh = 0.3

#*****************************************************************************
#*****************************************************************************

t0 = time.time()

#first check if required input exists
if not (os.path.exists(nlFile) and os.path.exists(satFile) and os.path.exists(atlasFile) and os.path.exists(distMapFile) and os.path.exists(spatGridPropsFile)):
    if not (os.path.exists(satFile)):
        print ('*** ERROR: Cannot continue, saturation file ' + satFile + ' does not exist. Please process a detector linearity calibration sequence or provide the necessary file ***')
    if not (os.path.exists(nlFile)):
        print ('*** ERROR: Cannot continue, NL coefficients file ' + nlFile + ' does not exist. Please process a detector linearity calibration sequence or provide the necessary file ***')
    if not (os.path.exists(atlasFile)):
        print ('*** ERROR: Cannot continue, line atlas file ' + atlasFile + ' does not exist. Please provide the necessary atlas file***')
    if not (os.path.exists(distMapFile)):
        print ('*** ERROR: Cannot continue, distorion map file ' + distMapFile + ' does not exist. Please process a Ronchi calibration sequence or provide the necessary file ***')
    if not (os.path.exists(spatGridPropsFile)):
        print ('*** ERROR: Cannot continue, spatial propertites grid file ' + spatGridPropsFile + ' does not exist. Please process a Ronchi calibration sequence or provide the necessary file ***')
    raise SystemExit('*** Missing required calibration files, exiting ***')

#create processed directory, in case it doesn't exist
wifisIO.createDir('processed')
wifisIO.createDir('quality_control')

print('Reading in calibration files')
#open calibration files
nlCoeff = wifisIO.readImgsFromFile(nlFile)[0]
satCounts = wifisIO.readImgsFromFile(satFile)[0]

if (os.path.exists(bpmFile)):
    BPM = wifisIO.readImgsFromFile(bpmFile)[0]
else:
    BPM = None

#read file list
lst= wifisIO.readAsciiList(fileList)

if (lst.ndim == 1):
    lstLen = 1
else:
    lstLen = len(lst)

for lstNum in range(lstLen):
    if (lst.ndim>1):
        waveFolder = lst[lstNum,0]
        flatFolder = lst[lstNum,1]
    else:
        waveFolder = lst[0]
        flatFolder = lst[1]
        
    savename = 'processed/'+waveFolder
    flatName = 'processed/'+flatFolder
    
    if(os.path.exists(savename+'_wave.fits') and os.path.exists(savename+'_waveMap.fits') and os.path.exists(savename+'_waveFitResuls.pkl') and os.path.exists(savename+'_wave_distCor.fits')):
        cont = wifisIO.userInput('Processed wavelength calibration files already exists for ' +waveFolder+', do you want to continue processing (y/n)?')
        if (cont.lower() == 'y'):
            contProc = True
        else:
            contProc = False
    else:
        contProc = True
    
    if (contProc):
        print('*** Working on folder ' + waveFolder + ' ***')

        if (os.path.exists(savename+'_wave.fits')):
            cont = wifisIO.userInput('Processed arc lamp file already exists for ' + waveFolder+', do you want to continue processing (y/n)?')
            if (cont.lower() == 'n'):
                print('Reading image'+savename+'_wave.fits instead')
                waveCor, sigmaImg, satFrame= wifisIO.readImgsFromFile(savename+'_wave.fits')[0]
                contProc2 = False
            else:
                contProc2 = True
        else:
            contProc2 = True
        
        if (contProc2):
            if (os.path.exists(waveFolder+'/Result')):
                #process CDS/Fowler ramps
                wave, sigmaImg, satFrame,hdr = processRamp.fromFowler(waveFolder, savename+'_wave.fits', satCounts, nlCoeff, BPM, nChannel=32, rowSplit=1, nlSplit=1, combSplit=1, crReject=False, bpmCorRng=2)
            else:
                #process UTR
                wave, sigmaImg, satFrame,hdr = processRamp.fromUTR(waveFolder, savename+'_wave.fits', satCounts, nlCoeff, BPM, nChannel=32, rowSplit=1, nlSplit=32, combSplit=32, crReject=crReject, bpmCorRng=2)

        print('Extracting slices')
        #first get rid of reference pixels
        wave = wave[4:2044,4:2044]
        sigmaImg = sigmaImg[4:2044, 4:2044]
        satFrame = satFrame[4:2044, 4:2044]
        
        limits, limHdr = wifisIO.readImgsFromFile(flatName+'_flat_limits.fits')
        limShift = limHdr['LIMSHIFT']
        
        waveSlices = slices.extSlices(wave, limits, dispAxis=0, shft=limShift)
        sigmaSlices = slices.extSlices(sigmaImg, limits, dispAxis=0)
        satSlices = slices.extSlices(satFrame, limits, dispAxis=0)

        print('Flat fielding slices')
        flatNorm = wifisIO.readImgsFromFile(flatName+'_flat_slices_norm.fits')[0][0:18]
        waveFlat = slices.ffCorrectAll(waveSlices, flatNorm)
        sigmaFlat = slices.ffCorrectAll(sigmaSlices,flatNorm)
        
        print('Distortion correcting slices')
        distMap = wifisIO.readImgsFromFile(distMapFile)[0]
        spatGridProps = wifisIO.readTable(spatGridPropsFile)
        waveCor = createCube.distCorAll(waveFlat, distMap, spatGridProps=spatGridProps)

        #save distortion corrected arc image
        wifisIO.writeFits(waveCor, savename+'_wave_distCor.fits',hdr=hdr)
        
        #Determine dispersion solution
        print('Determining dispersion solution')
        #read in template
        template = wifisIO.readImgsFromFile(templateFile)[0]

        #read in template results to extract lambda -> wavelength solution
        prevSol = wifisIO.readPickle(prevResultsFile)
    
        #check if solution already exists.
        if(os.path.exists(savename+'_waveFitResults.pkl') and (os.path.exists(savename+'_waveMap.fits'))):
            cont = wifisIO.userInput('Dispersion solution and wavemap already exists for ' +waveFolder+', do you want to continue and replace (y/n)?')
            if (cont.lower() == ''):
                cont3 = True
            else:
                cont3 = False
        else:
            cont3 = True
                
        if cont3:
            results = waveSol.getWaveSol(waveSlices, template, atlasFile, 3, prevSol, winRng=13, mxCcor=150, weights=False, buildSol=False, sigmaClip=1, allowLower=True, lngthConstraint=True)
    
            #Save all results
            wifisIO.writePickle(results, savename+'_waveFitResults.pkl')

            dispSolLst = results[0]

            print('Getting smoothed dispersion solutions')

            #remove solution with rms > 0.3
            if (rmsThresh is not None):
                rms = results[4]
                for i in range(len(rms)):
                    for j in range(len(rms[i])):
                        if (rms[i][j] > rmsThresh):
                            dispSolLst[i][j] = np.asarray([np.nan])

            if (plot):
                polySol = waveSol.polyFitDispSolution(dispSolLst, degree=2, plotFile='quality_control/'+waveFolder+'_polyFit_dispSol.pdf')
                #gaussSol =waveSol.gaussSmoothDispSolution(result[0],nPix=3,plot='quality_control/'+waveFolder+'_polyFit_dispSol.pdf')
            else:
                polySol = waveSol.polyFitDispSolution(dispSolLst, degree=2, plotFile=None)
                #gaussSol =waveSol.gaussSmoothDispSolution(result[0],nPix=3,plotFile=None)  

            dispSolLst = polySol
            
            print('Creating wavelength map')
            #Create wavemap
            waveMapLst = waveSol.buildWaveMap(dispSolLst, waveSlices[0].shape[1])

            #save wavemap solution
            wifisIO.writeFits(waveMapLst, savename+'_waveMap.fits',hdr=hdr)

            if (plot):
                #save some output for quality control purposes
                plt.ioff()

                #first save RMS results
                rms = results[4]

                with PdfPages('quality_control/'+waveFolder+'_wave_RMS.pdf') as pdf:
                    for r in rms:
                        plt.plot(r)
                        plt.xlabel('Column #')
                        plt.ylabel('RMS in pixels')
                        pdf.savefig(dpi=300)
                        plt.close()

                #now build and save FWHM map
                fwhmMapLst = waveSol.buildFWHMMap(results[2], results[3], npts)
                #get max and min starting wavelength based on median of central slice (slice 8)
                waveMin = np.nanmedian(waveMapLst[:,0])
                waveMax = np.nanmedian(waveMapLst[:,-1])
                
                #determine length along spatial direction
                ntot = 0
                for j in range(len(rmsLst)):
                    ntot += len(rmsLst[j])
                    
                #get mean FWHM
                fwhmMean = 0.
                nFWHM = 0.
                for f in fwhmLst:
                    for i in range(len(f)):
                        for j in range(len(f[i])):
                            fwhmMean += f[i][j]
                            nFWHM += 1.
            
                fwhmMean /= nFWHM

                #build "detector" map images
                #wavelength solution
                waveMap = np.empty((npts,ntot),dtype='float32')
                strt=0
                for m in waveMapLst:
                    waveMap[:,strt:strt+m.shape[0]] = m.T
                    strt += m.shape[0]

                #fwhm map
                fwhmMap = np.empty((npts,ntot),dtype='float32')
                strt=0
                for f in fwhmMapLst:
                    fwhmMap[:,strt:strt+f.shape[0]] = f.T
                    strt += f.shape[0]

                #save results
                wifisIO.writeFits(waveMap, 'quality_control/'+waveFolder+'_waveMapImg.fits', ask=False,hdr=hdr)
                wifisIO.writeFits(fwhmMap, 'quality_control/'+waveFolder+'_fwhmMapImg.fits', ask=False,hdr=hdr)

                plt.imshow(fwhmMap, aspect='auto', cmap='jet')
                plt.colorbar()
                plt.title('Mean FWHM is '+str(fwhmMean) +', min wave is '+str(waveMin)+', max wave is '+str(waveMax))
                plt.savefig('quality_control/'+waveFolder+'fwhmMapImg.png', dpi=300)
                plt.close()

                           
print ("Total time to run entire script: ",time.time()-t0)

