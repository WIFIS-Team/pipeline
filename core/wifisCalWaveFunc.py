"""

Calibrates arc lamp images

Requires:
- 

Produces:
- per pixel wavelength solution


"""

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
import wifisCalFlatFunc as calFlat

def runCalWave(waveLst, flatLst, hband=False, nlCoef=None, satCounts=None, BPM=None, distMapLimitsFile='', plot=True, nChannel=32, nRowAverage=4,rowSplit=1,nlSplit=1, combSplit=1,bpmCorRng=2, crReject=False, skipObsinfo=False, darkLst=None, flatWinRng=51, flatImgSmth=5, flatPolyFitDegree=3, rootFolder='', distMapFile='', spatGridPropsFile='', atlasFile='', templateFile='', prevResultsFile='', sigmaClip=2, sigmaClipRounds=2, sigmaLimit=3,cleanDispSol=False,cleanDispThresh = 0, waveTrimThresh=0):
    """
    """
      
    t0 = time.time()
    
    plt.ioff()
    #create processed directory, in case it doesn't exist
    wifisIO.createDir('processed')
    wifisIO.createDir('quality_control')

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
                nRamps = wifisIO.getNumRamps(waveFolder,rootFolder=rootFolder)

                #if more than one ramp is present, average the results
                if nRamps > 1:
                    waveAll = []
                    sigmaImgAll = []
                    satFrameAll = []

                    for rampNum in range(1,nRamps+1):
                        wave, sigmaImg, satFrame,hdr = processRamp.auto(waveFolder, rootFolder, savename+'_wave.fits', satCounts, nlCoef, BPM, nChannel=nChannel, rowSplit=rowSplit, nlSplit=nlSplit, combSplit=combSplit, crReject=crReject, bpmCorRng=bpmCorRng, rampNum=rampNum)

                        waveAll.append(wave)
                        sigmaImgAll.append(sigmaImg)
                        satFrameAll.append(satFrame)

                    #now combine all images
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        wave = np.nanmean(np.asarray(waveAll),axis=0)
                        sigmaImg = np.zeros(sigmaImg.shape,dtype=sigmaImg.dtype)
                        for k in range(len(sigmaImgAll)):
                            sigmaImg += sigmaImgAll[k]**2
                        sigmaImg = np.sqrt(sigmaImg)/len(sigmaImgAll)
                        satFrame = np.nanmean(np.asarray(satFrameAll),axis=0).round().astype('int')
                    del flatImgAll
                    del sigmaImgAll
                    del satFrameAll
                else:
                    wave, sigmaImg, satFrame,hdr = processRamp.auto(waveFolder, rootFolder, savename+'_wave.fits', satCounts, nlCoef, BPM, nChannel=nChannel, rowSplit=rowSplit, nlSplit=nlSplit, combSplit=combSplit, crReject=crReject, bpmCorRng=bpmCorRng, rampNum=None)
            if os.path.exists(savename+'_wave_slices.fits'):
                cont = 'n'
                cont = wifisIO.userInput('Wave slices already exists for ' + waveFolder+', do you want to continue processing (y/n)?')
                if (not cont.lower() == 'y'):
                    print('Reading image '+savename+'_wave_slices.fits instead')
                    waveSlicesLst = wifisIO.readImgsFromFile(savename+'_wave.fits')[0]
                    waveSlices = waveSlicesLst[0:18]
                    sigmaSlices = waveSlicesLst[18:36]
                    satSlices = waveSlicesLst[36:]
                    contProc2 = False
                else:
                    contProc2 = True
            else:
                contProc2 = True

            if (contProc2):
                print('Extracting wave slices')
                #first get rid of reference pixels
                wave = wave[4:-4,4:-4]
                sigmaImg = sigmaImg[4:-4, 4:-4]
                satFrame = satFrame[4:-4, 4:-4]

                print(flatFolder)
                #check if processed flat already exists, if not process the flat folder
                if not os.path.exists('processed/'+flatFolder+'_flat_limits.fits'):
                    distMap = wifisIO.readImgsFromFile(distMapFile)[0]
                    print('Flat limits do not exist for folder ' +flatFolder +', processing flat folder')

                    calFlat.runCalFlat(np.asarray([flatFolder]), hband=hband, darkLst = darkLst, rootFolder=rootFolder, nlCoef=nlCoef, satCounts=satCounts, BPM = BPM, distMapLimitsFile = distMapLimitsFile, plot=True, nChannel = nChannel, nRowAverage=nRowAverage,rowSplit=rowSplit,nlSplit=nlSplit, combSplit=combSplit,bpmCorRng=2, crReject=False, skipObsinfo=False,avgRamps=True)

                limits, limHdr = wifisIO.readImgsFromFile('processed/'+flatFolder+'_flat_limits.fits')
                limShift = limHdr['LIMSHIFT']
        
                waveSlices = slices.extSlices(wave, limits, dispAxis=0, shft=limShift)
                sigmaSlices = slices.extSlices(sigmaImg, limits, dispAxis=0)
                satSlices = slices.extSlices(satFrame, limits, dispAxis=0)

                wifisIO.writeFits(waveSlices + sigmaSlices + satSlices, savename+'_wave_slices.fits', hdr=hdr,ask=False)

                #get flat fielded slices
                flatNormLst = wifisIO.readImgsFromFile('processed/'+flatFolder+'_flat_slices_norm.fits')[0]
                flatNorm = flatNormLst[0:18]
                flatSigmaNorm = flatNormLst[18:36]
               
                waveFlat = slices.ffCorrectAll(waveSlices, flatNorm)
                sigmaFlat = wifisUncertainties.multiplySlices(waveSlices, sigmaSlices, flatNorm, flatSigmaNorm)
 
                
            if os.path.exists(savename+'_wave_distCor.fits'):
                cont = 'n'
                cont = wifisIO.userInput('Distortion corrected wave slices file already exists for ' +waveFolder+', do you want to continue processing (y/n)?')
                if (not cont.lower() == 'y'):
                    print('Reading in distortion corrected wave slices file '+savename+'_wave_distCor.fits instead')
                    waveSlicesLst= wifisIO.readImgsFromFile(savename+'_wave_distCor.fits')[0]
                    waveCor =waveSlicesLst[:18]
                    sigmaCor = waveSlicesLst[18:36]
                    
                    contProc2 = False
                else:
                    contProc2 = True
            else:
                contProc2= True

            if contProc2:
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

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)

                    results = waveSol.getWaveSol(waveCor, template, atlasFile, 3, prevSol, winRng=9, mxCcor=150, weights=False, buildSol=False, sigmaClip=sigmaClip, allowLower=False, lngthConstraint=True, MP=True, adjustFitWin=True, sigmaLimit=sigmaLimit, allowSearch=False, sigmaClipRounds=sigmaClipRounds)

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
                            rmsClean, dispSolClean, pixSolClean = waveSol.cleanDispSol(results, plotFile='quality_control/'+waveFolder+'_wave_waveFit_rms.pdf', threshold=cleanDispThresh)
                        else:
                            rmsClean, dispSolClean, pixSolClean = waveSol.cleanDispSol(results, plotFile=None, threshold=cleanDispThresh)
                    else:
                        dispSolClean = result[0]

                        if plot:
                            rms = results[4]
                            with PdfPages('quality_control/'+waveFolder+'_wave_waveFit_RMS.pdf') as pdf:
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

                    #use linear interpolation and extrapolation to fill in missing solutions
                    #waveMapLst = waveSol.buildWaveMap(dispSolLst, waveCor[0].shape[1], extrapolate=False, fill_missing=False)
                    
                    #use linear interpolation and polynomial fitting to extrapolate for missing solutions
                    waveMapLst = waveSol.buildWaveMap2(dispSolLst, waveCor[0].shape[1], extrapolate=True, fill_missing=True)

                    #smooth waveMap solution to avoid pixel-to-pixel jumps
                    waveMap = waveSol.smoothWaveMapAll(waveMapLst,smth=1,MP=True )

                    #replace wavemap with polynomial fit
                    #waveMap = waveSol.polyFitWaveMapAll(waveMapLst, degree=1, MP=True)

                    #save wavemap solution
                    wifisIO.writeFits(waveMap, savename+'_wave_waveMap.fits',hdr=hdr, ask=False)

                    if waveTrimThresh > 0:
                        print('Trimming wavelength map to useful range')
                        #now trim wavemap if needed
                        #read in unnormalized flat field data
                        flatSlices = wifisIO.readImgsFromFile('processed/'+flatFolder+'_flat_slices.fits')[0][0:18]
                        waveMapTrim = waveSol.trimWaveSliceAll(waveMap, flatSlices, waveTrimThresh)
                
                        #get wave grid properties
                        waveGridProps = createCube.compWaveGrid(waveMapTrim)
                    else:
                        waveGridProps = createCube.compWaveGrid(waveMap) 
           
                    wifisIO.writeTable(waveGridProps, savename+'_wave_waveGridProps.dat')

                    print('placing arc image on grid')
                    waveGrid = createCube.waveCorAll(waveCor, waveMap, waveGridProps=waveGridProps)
                    wifisIO.writeFits(waveGrid, savename+'_wave_fullGrid.fits', hdr=hdr,ask=False)
            
                    if plot:
                        print('Getting quality control checks')
                        rms = results[4]
                        pixCentLst = results[2]
                        fwhmLst = results[1]
                        npts = waveCor[0].shape[1]
                        fwhmMapLst = waveSol.buildFWHMMap(pixCentLst, fwhmLst, npts)

                        #get max and min starting wavelength based on median of central slice (slice 8)
                        if waveTrimThresh > 0:
                            trimSlc = waveMapTrim[8]
                        else:
                            trimSlc = waveMap[8]

                        #find median wavelength of first and last non-NaN columns
                        wCol = np.nanmedian(trimSlc, axis=0)

                        #find non-NaN columns
                        whr = np.where(np.isfinite(wCol))[0]
        
                        waveMax = np.nanmedian(trimSlc[:,whr[0]])
                        waveMin = np.nanmedian(trimSlc[:,whr[-1]])
 
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
                        wifisIO.writeFits(waveMapImg, 'quality_control/'+waveFolder+'_wave_wavelength_map.fits', hdr=hdr,ask=False)
                        wifisIO.writeFits(fwhmMap, 'quality_control/'+waveFolder+'_wave_fwhm_map.fits',hdr=hdr, ask=False)

                        #get improved clim for plotting
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', RuntimeWarning)
                            cMax = np.nanmax(fwhmMap[fwhmMap < 0.9*np.nanmax(fwhmMap)])
                    
                        fig = plt.figure()
                        plt.imshow(fwhmMap, aspect='auto', cmap='jet', clim=[0, cMax], origin='lower')
                        plt.colorbar()
                        plt.title('Median FWHM is '+'{:3.1f}'.format(fwhmMed) +', min wave is '+'{:6.1f}'.format(waveMin)+', max wave is '+'{:6.1f}'.format(waveMax))
                        plt.savefig('quality_control/'+waveFolder+'_wave_fwhm_map.png', dpi=300)
                        plt.close()
                                
                        
    print ("Total time to run entire script: ",time.time()-t0)
    return