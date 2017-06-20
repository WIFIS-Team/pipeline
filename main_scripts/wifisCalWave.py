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

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0' # Used to show compile errors for debugging, can be removed
os.environ['PYOPENCL_CTX'] = '1' # Used to specify which OpenCL device to target
plt.ioff()

#*****************************************************************************
#************************** Required input ***********************************
waveList = 'wave.lst' 
flatList = 'flat.lst'

hband = False

#mostly static input
rootFolder = '/data/WIFIS/H2RG-G17084-ASIC-08-319'

if hband:
    templateFile = '/data/pipeline/external_data/hband_template.fits'
    prevResultsFile = '/data/pipeline/external_data/hband_template.pkl'
else:
    templateFile = '/data/pipeline/external_data/waveTemplate.fits'
    prevResultsFile = '/data/pipeline/external_data/waveTemplateFittingResults.pkl'
    
nlFile = '/home/jason/wifis/data/non-linearity/may/processed/master_detLin_NLCoeff.fits'        
satFile = '/home/jason/wifis/data/non-linearity/may/processed/master_detLin_satCounts.fits'
bpmFile = '/data/pipeline/external_data/bpm.fits'
atlasFile = '/data/pipeline/external_data/best_lines2.dat'
distMapFile = '/home/jason/wifis/data/ronchi_map_may/distortionMap.fits'
spatGridPropsFile = '/home/jason/wifis/data/ronchi_map_may/spatGridProps.dat'

#optional behaviour
plot = True
crReject = False
cleanDispSol = True
waveTrimThresh=0.1

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
waveLst= wifisIO.readAsciiList(waveList)
flatLst = wifisIO.readAsciiList(flatList)

if (waveLst.ndim == 0):
    waveLst = np.asarray([waveLst])
    flatLst = np.asarray([flatLst])
    
for lstNum in range(len(waveLst)):
    waveFolder = waveLst[lstNum]
    flatFolder = flatLst[lstNum]
    
    savename = 'processed/'+waveFolder
    
    if(os.path.exists(savename+'_wave.fits') and os.path.exists(savename+'_wave_waveMap.fits') and os.path.exists(savename+'_wave_fitResults.pkl') and os.path.exists(savename+'_wave_distCor.fits') and os.path.exists(savename+'_wave_waveGridProps.dat')):
        cont = 'n'
        cont = wifisIO.userInput('Processed wavelength calibration files already exists for ' +waveFolder+', do you want to continue processing (y/n)?')
        if (not cont.lower() == 'y'):
            contProc = False
        else:
            contProc = True
    else:
        contProc = True
    
    if (contProc):
        print('*** Working on folder ' + waveFolder + ' ***')

        if (os.path.exists(savename+'_wave.fits')):
            cont = 'n'
            cont = wifisIO.userInput('Processed arc lamp file already exists for ' + waveFolder+', do you want to continue processing (y/n)?')
            if (not cont.lower() == 'y'):
                print('Reading image'+savename+'_wave.fits instead')
                [wave, sigmaImg, satFrame],hdr= wifisIO.readImgsFromFile(savename+'_wave.fits')
                hdr = hdr[0]
                contProc2 = False
            else:
                contProc2 = True
        else:
            contProc2 = True
        
        if (contProc2):
            wave, sigmaImg, satFrame,hdr = processRamp.auto(waveFolder, rootFolder, savename+'_wave.fits', satCounts, nlCoeff, BPM, nChannel=32, rowSplit=1, nlSplit=32, combSplit=32, crReject=crReject, bpmCorRng=2)

        if not os.path.exists('processed/'+flatFolder+'_flat_limits.fits'):
            raise SystemExit('*** CANNOT PROCEED WITHOUT PROCESSED FLAT FIELDS. PROCESS FLAT FIELDS FIRST ***')

        if os.path.exists(savename+'_wave_distCor.fits'):
            cont = 'n'
            cont = wifisIO.userInput('Distortion corrected wave slices file already exists for ' +waveFolder+', do you want to continue processing (y/n)?')
            if (not cont.lower() == 'y'):
                print('Reading in distortion corrected wave slices file '+savename+'_wave_distCor.fits instead')
                waveSlicesLst = wifisIO.readImgsFromFile(savename+'_wave_distCor.fits')[0]
                waveCor = waveSlicesLst[0:18]
                sigmaCor = waveSlicesLst[18:36]
                satSlices = waveSlicesLst[36:]
                
                contProc2 = False
            else:
                contProc2 = True
        else:
            contProc2= True

        if (contProc2):
            print('Extracting slices')
            #first get rid of reference pixels
            wave = wave[4:2044,4:2044]
            sigmaImg = sigmaImg[4:2044, 4:2044]
            satFrame = satFrame[4:2044, 4:2044]
        
            limits, limHdr = wifisIO.readImgsFromFile('processed/'+flatFolder+'_flat_limits.fits')
            limShift = limHdr['LIMSHIFT']
        
            waveSlices = slices.extSlices(wave, limits, dispAxis=0, shft=limShift)
            sigmaSlices = slices.extSlices(sigmaImg, limits, dispAxis=0)
            satSlices = slices.extSlices(satFrame, limits, dispAxis=0)

            wifisIO.writeFits(waveSlices + sigmaSlices + satSlices, savename+'_wave_slices.fits', ask=False)

            flatNormLst = wifisIO.readImgsFromFile('processed/'+flatFolder+'_flat_slices_norm.fits')[0]
            flatNorm = flatNormLst[0:18]
            flatSigmaNorm = flatNormLst[18:36]
        
            waveFlat = slices.ffCorrectAll(waveSlices, flatNorm)
            sigmaFlat = wifisUncertainties.multiplySlices(waveSlices, sigmaSlices, flatNorm, flatSigmaNorm)
            
            print('Distortion correcting slices')
            distMap = wifisIO.readImgsFromFile(distMapFile)[0]
            spatGridProps = wifisIO.readTable(spatGridPropsFile)
            waveCor = createCube.distCorAll(waveSlices, distMap, spatGridProps=spatGridProps)

            #save distortion corrected arc image
            wifisIO.writeFits(waveCor, savename+'_wave_distCor.fits',hdr=hdr, ask=False)

        if os.path.exists(savename+'_wave_fitResults.pkl'):
            cont = 'n'
            cont = wifisIO.userInput('Wavlength fitting results already exists for ' +waveFolder+', do you want to continue processing (y/n)?')
            if (not cont.lower() == 'y'):
                print('Reading in results '+savename+'_wave_fitResults.pkl instead')
                results = wifisIO.readPickle(savename+'_wave_fitResults.pkl')
                
                contProc2 = False
            else:
                contProc2 = True
        else:
            contProc2= True

        if (contProc2):
            #Determine dispersion solution
            print('Determining dispersion solution')
            #read in template
            template = wifisIO.readImgsFromFile(templateFile)[0]

            #read in template results to extract lambda -> wavelength solution
            prevResults = wifisIO.readPickle(prevResultsFile)
            prevSol = prevResults[5]
            
            results = waveSol.getWaveSol(waveCor, template, atlasFile, 3, prevSol, winRng=13, mxCcor=150, weights=False, buildSol=False, sigmaClip=1, allowLower=True, lngthConstraint=True)
    
            #Save all results
            wifisIO.writePickle(results, savename+'_wave_fitResults.pkl')

        if os.path.exists(savename+'_wave_waveMap.fits') and os.path.exists(savename+'_wave_waveGridProps.dat'):
            cont = 'n'
            cont = wifisIO.userInput('Wavlength map already exists for ' +waveFolder+', do you want to continue processing (y/n)?')
            if (not cont.lower() == 'y'):
                print('Reading in map '+savename+'_wave_waveMap.fits instead')
                waveMap = wifisIO.readImgsFromFile(savename+'_wave_waveMap.fits')
                waveGridProps = wifisIO.readTable(savename+'_wave_waveGridProps.dat')
                
                contProc2 = False
            else:
                contProc2 = True
        else:
            contProc2= True

        if (contProc2):
            print('Building wavelegth map')
            if cleanDispSol:
                print('Finding and replacing bad solutions')
                if plot:
                    rmsClean, dispSolClean, pixSolClean = waveSol.cleanDispSol(results, plotFile='quality_control/'+waveFolder+'_wave_waveFit_rms.pdf')
                else:
                    rmsClean, dispSolClean, pixSolClean = waveSol.cleanDispSol(results, plotFile=None)
            else:
                dispSolClean = result[0]

                if plot:
                    rms = results[4]
                    with PdfPages('quality_control/'+waveFolder+'_wave_RMS.pdf') as pdf:
                        for r in rms:
                            fig = plt.figure()
                            plt.plot(r)
                            plt.xlabel('Column #')
                            plt.ylabel('RMS in pixels')
                            pdf.savefig(dpi=300)
                            plt.close()
                
            dispSolLst = dispSolClean
            
            print('Creating wavelength map')
            #Create wavemap
            waveMap = waveSol.buildWaveMap(dispSolLst, waveCor[0].shape[1])

            if waveTrimThresh > 0:
                #now trim wavemap if needed
                #read in unnormalized flat field data
                flatSlices = wifisIO.readImgsFromFile('processed/'+flatFolder+'_flat_slices.fits')[0][0:18]
                waveMapTrim = waveSol.trimWaveSliceAll(waveMap, flatSlices, waveTrimThresh)
                waveMap = waveMapTrim

            #save wavemap solution
            wifisIO.writeFits(waveMap, savename+'_wave_waveMap.fits',hdr=hdr, ask=False)

            #get wave grid properties
            waveGridProps = createCube.compWaveGrid(waveMap)
            wifisIO.writeTable(waveGridProps, savename+'_wave_waveGridProps.dat')

            
            print('placing arc image on grid')
            waveGrid = createCube.waveCorAll(waveCor, waveMap, waveGridProps=waveGridProps)
            wifisIO.writeFits(waveGrid, savename+'_wave_fullGrid.fits', ask=False)
            
            if plot:
                print('Getting quality control checks')
                rms = results[4]
                pixCentLst = results[2]
                fwhmLst = results[1]
                npts = waveCor[0].shape[1]
                fwhmMapLst = waveSol.buildFWHMMap(pixCentLst, fwhmLst, npts)
                #get max and min starting wavelength based on median of central slice (slice 8)

                trimSlc = waveSol.trimWaveSlice([waveMap[8], flatSlices[8], 0.5])
                waveMax = np.nanmedian(trimSlc[:,0])
                waveMin = np.nanmedian(trimSlc[:,-1])
 
                #determine length along spatial direction
                ntot = 0
                for j in range(len(rms)):
                    ntot += len(rms[j])

                #get median FWHM
                fwhmAll = []
                for f in fwhmLst:
                    for i in range(len(f)):
                        for j in range(len(f[i])):
                            fwhmAll.append(f[i][j])
            
                fwhmMed = np.nanmedian(fwhmAll)

                #build "detector" map images
                #wavelength solution
                waveMapImg = np.empty((npts,ntot),dtype='float32')
                strt=0
                for m in waveMap:
                    waveMapImg[:,strt:strt+m.shape[0]] = m.T
                    strt += m.shape[0]

                #fwhm map
                fwhmMap = np.empty((npts,ntot),dtype='float32')
                strt=0
                for f in fwhmMapLst:
                    fwhmMap[:,strt:strt+f.shape[0]] = f.T
                    strt += f.shape[0]

                #save results
                wifisIO.writeFits(waveMapImg, 'quality_control/'+waveFolder+'_wave_wavelength_map.fits', ask=False)
                wifisIO.writeFits(fwhmMap, 'quality_control/'+waveFolder+'_wave_fwhm_map.fits', ask=False)

                #get improved clim for plotting
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    cMax = np.nanmax(fwhmMap[fwhmMap < 0.9*np.nanmax(fwhmMap)])
                    
                fig = plt.figure()
                plt.imshow(fwhmMap, aspect='auto', cmap='jet', clim=[0, cMax])
                plt.colorbar()
                plt.title('Median FWHM is '+'{:3.1f}'.format(fwhmMed) +', min wave is '+'{:6.1f}'.format(waveMin)+', max wave is '+'{:6.1f}'.format(waveMax))
                plt.savefig('quality_control/'+waveFolder+'_fwhm_map.png', dpi=300)
                plt.close()
                                
                           
print ("Total time to run entire script: ",time.time()-t0)

