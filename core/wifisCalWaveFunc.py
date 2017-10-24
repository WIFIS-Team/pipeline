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
from astropy import time as astrotime, coordinates as coord, units
import colorama
from astropy.visualization import ZScaleInterval

def runCalWave(waveLst, flatLst, hband=False, nlCoef=None, satCounts=None, BPM=None, distMapLimitsFile='', plot=True, nChannel=32, nRowsAvg=0,rowSplit=1,nlSplit=1, combSplit=1,bpmCorRng=2, crReject=False, skipObsinfo=False, darkLst=None, flatWinRng=51, flatImgSmth=5, limitsPolyFitDegree=3, rootFolder='', distMapFile='', spatGridPropsFile='', atlasFile='', templateFile='', prevResultsFile='', sigmaClip=2, sigmaClipRounds=2, sigmaLimit=3,cleanDispSol=False,cleanDispThresh = 0, waveTrimThresh=0, waveSmooth=1, nlFile='', bpmFile='', satFile='',darkFile='',logfile=None, ask=True, obsCoords=None, dispAxis=0, mxOrder=3,winRng=9,waveSolMP=True,waveSolPlot=False, nRowSplitFlat=1,ron=None, gain=1.,flatbpmCorRng=20.):
    """
    """

    type(logfile)
    t0 = time.time()
    colorama.init()
    
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
            if ask:
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
                    print('Reading image '+savename+'_wave.fits instead')
                    [wave, sigmaImg, satFrame],hdr= wifisIO.readImgsFromFile(savename+'_wave.fits')
                    hdr = hdr[0]
                    contProc2 = False
                else:
                    contProc2 = True
            else:
                contProc2 = True
        
            if (contProc2):
                wave, sigmaImg, satFrame,hdr = processRamp.auto(waveFolder, rootFolder, savename+'_wave.fits', satCounts, nlCoef, BPM, nChannel=nChannel, rowSplit=rowSplit, nlSplit=nlSplit, combSplit=combSplit, crReject=crReject, bpmCorRng=bpmCorRng, rampNum=None, nlFile=nlFile, bpmFile=bpmFile, satFile=satFile, obsCoords=obsCoords, avgAll=True, logfile=logfile, ron=ron, gain=gain)

                #carry out dark subtraction
                if darkLst is not None and darkLst[0] is not None:
                    print('Subtracting dark ramp')
                    if len(darkLst)>1:
                        dark, darkSig = darkLst[:2]
                        sigmaImg = np.sqrt(sigmaImg**2 + darkSig**2)
                    else:
                        dark = darkLst[0]
                        logfile.write('*** Warning: No uncertainty associated with dark image ***\n')
                        print(colorama.Fore.RED+'*** WARNING: No uncertainty associated with dark image ***'+colorama.Style.RESET_ALL)
                    wave -= dark
                    hdr.add_history('Dark image subtracted using file:')
                    hdr.add_history(darkFile)
                    if logfile is not None:
                        logfile.write('Subtracted dark image using file:\n')
                        logfile.write(darkFile+'\n')
                    else:
                        print(colorama.Fore.RED+'*** WARNING: No dark image provided, or file does not exist, skipping ***'+colorama.Style.RESET_ALL)

                        if logfile is not None:
                            logfile.write('*** WARNING: No dark image provided, or file ' + str(darkFile)+' does not exist, skipping ***')
                   
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

                #check if processed flat already exists, if not process the flat folder
                if not os.path.exists('processed/'+flatFolder+'_flat_limits.fits'):
                    distMap = wifisIO.readImgsFromFile(distMapFile)[0]
                    print('Flat limits do not exist for folder ' +flatFolder +', processing flat folder')

                    calFlat.runCalFlat(np.asarray([flatFolder]), hband=hband, darkLst = darkLst, rootFolder=rootFolder, nlCoef=nlCoef, satCounts=satCounts, BPM = BPM, distMapLimitsFile = distMapLimitsFile, plot=True, nChannel = nChannel, nRowsAvg=nRowsAvg,rowSplit=nRowSplitFlat,nlSplit=nlSplit, combSplit=combSplit,bpmCorRng=flatbpmCorRng, crReject=False, skipObsinfo=False,nlFile=nlFile, bpmFile=bpmFile, satFile=satFile, darkFile=darkFile, logfile=logfile, polyFitDegree=limitsPolyFitDegree,gain=gain,ron=ron)

                print('Reading slice limits file')
                limits, limHdr = wifisIO.readImgsFromFile('processed/'+flatFolder+'_flat_limits.fits')
                limShift = limHdr['LIMSHIFT']
        
                waveSlices = slices.extSlices(wave, limits, dispAxis=dispAxis, shft=limShift)
                sigmaSlices = slices.extSlices(sigmaImg, limits, dispAxis=dispAxis)
                satSlices = slices.extSlices(satFrame, limits, dispAxis=dispAxis)

                hdr.add_history('Used following flat field file for slice limits:')
                hdr.add_history(flatFolder)

                #remove previous comment about file contents
                hdrTmp = hdr[::-1]
                hdrTmp.remove('COMMENT')
                hdr = hdrTmp[::-1]
                
                hdr.add_comment('File contains flux, sigma, sat info for each slice as multi-extensions')
                wifisIO.writeFits(waveSlices + sigmaSlices + satSlices, savename+'_wave_slices.fits', hdr=hdr,ask=False)

                hdrTmp = hdr[::-1]
                hdrTmp.remove('COMMENT')
                hdr = hdrTmp[::-1]
                
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

                hdr.add_history('Used following file for distortion map:')
                hdr.add_history(distMapFile)

                hdr.add_comment('File contains the distortion-corrected flux slices as multi-extensions')
                #save distortion corrected arc image
                wifisIO.writeFits(waveCor, savename+'_wave_distCor.fits',hdr=hdr, ask=False)

                hdrTmp = hdr[::-1]
                hdrTmp.remove('COMMENT')
                hdr = hdrTmp[::-1]
                
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

                    results = waveSol.getWaveSol(waveCor, template, atlasFile, mxOrder, prevSol, winRng=winRng, mxCcor=150, weights=False, buildSol=False, sigmaClip=sigmaClip, allowLower=False, lngthConstraint=True, MP=waveSolMP, adjustFitWin=True, sigmaLimit=sigmaLimit, allowSearch=False, sigmaClipRounds=sigmaClipRounds,plot=waveSolPlot)

                hdr.add_history('Used the following file as template for wavelength mapping:')
                hdr.add_history(templateFile)
                hdr.add_history('Used the following atlas file for the input line list:')
                hdr.add_history(atlasFile)
                
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
                        hdr.add_history('Removed badly fit solutions with threshold of ' + str(cleanDispThresh) + '-sigma')
                        
                        if plot:
                            rmsClean, dispSolClean, pixSolClean = waveSol.cleanDispSol(results, plotFile='quality_control/'+waveFolder+'_wave_waveFit_RMS.pdf', threshold=cleanDispThresh)
                        else:
                            rmsClean, dispSolClean, pixSolClean = waveSol.cleanDispSol(results, plotFile=None, threshold=cleanDispThresh)
                    else:
                        dispSolClean = results[0]

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
                    #hdr.add_history('Used linear interpolation to fill missing/badly fit wavelength regions')
                    
                    #use linear interpolation and polynomial fitting to extrapolate for missing solutions
                    waveMapLst = waveSol.buildWaveMap2(dispSolLst, waveCor[0].shape[1], extrapolate=True, fill_missing=True)
                    hdr.add_history('Used linear interpolation and a linear polynomial fitting to fill missing/badly fit wavelength regions')

                    #smooth waveMap solution to avoid pixel-to-pixel jumps
                    if waveSmooth>0:
                        print('Smoothing wavelength map')
                        if logfile is not None:
                            logfile.write('Gaussian smoothed wavelength map using 1-sigma width of ' + str(waveSmooth)+'\n')
                            
                        waveMap = waveSol.smoothWaveMapAll(waveMapLst,smth=waveSmooth,MP=True )
                        hdr.add_history('Gaussian smoothed wavelength map using 1-sigma width of ' + str(waveSmooth))
                    else:
                        waveMap = waveMapLst
                        
                    #replace wavemap with polynomial fit
                    #waveMap = waveSol.polyFitWaveMapAll(waveMapLst, degree=1, MP=True)

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

                    print('Placing arc image on grid')
                    waveGrid = createCube.waveCorAll(waveCor, waveMap, waveGridProps=waveGridProps)

                    hdr.add_comment('File contains the uniformly gridded spatial and wavelength mapped fluxes for each slice as multi-extensions')
                    wifisIO.writeFits(waveGrid, savename+'_wave_fullGrid.fits', hdr=hdr,ask=False)

                    hdrTmp = hdr[::-1]
                    hdrTmp.remove('COMMENT')
                    hdrTmp.remove('COMMENT')
                    hdr = hdrTmp[::-1]
                    
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
                        hdr.set('QC_WMIN',waveMin,'Minimum median wavelength for middle slice')
                        hdr.set('QC_WMAX',waveMax,'Maximum median wavelength for middle slice')
                        hdr.set('QC_WFWHM', fwhmMed, 'Median FWHM of all slices')

                        wifisIO.writeFits(waveMapImg, 'quality_control/'+waveFolder+'_wave_wavelength_map.fits', hdr=hdr,ask=False)
                        wifisIO.writeFits(fwhmMap, 'quality_control/'+waveFolder+'_wave_fwhm_map.fits',hdr=hdr, ask=False)

                        #get improved clim for plotting
                        #with warnings.catch_warnings():
                        #    warnings.simplefilter('ignore', RuntimeWarning)
                        #    cMax = np.nanmax(fwhmMap[fwhmMap < 0.9*np.nanmax(fwhmMap)])
                    
                        fig = plt.figure()
                        interval = ZScaleInterval()
                        lims = interval.get_limits(fwhmMap)
                        #plt.imshow(fwhmMap, aspect='auto', cmap='jet', clim=[0, cMax], origin='lower')
                        plt.imshow(fwhmMap, aspect='auto', cmap='jet', clim=lims, origin='lower')
                        plt.colorbar()
                        plt.title('Median FWHM is '+'{:3.1f}'.format(fwhmMed) +', min wave is '+'{:6.1f}'.format(waveMin)+', max wave is '+'{:6.1f}'.format(waveMax))
                        plt.tight_layout()
                        plt.savefig('quality_control/'+waveFolder+'_wave_fwhm_map.png', dpi=300)
                        plt.close()
                                            
                    hdr.add_comment('File contains the wavelength mapping slices as multi-extensions')
                    #save wavemap solution
                    wifisIO.writeFits(waveMap, savename+'_wave_waveMap.fits',hdr=hdr, ask=False)
            print('*** Finished processing ' + waveFolder + ' in ' + str(time.time()-t0) + ' seconds ***')

    return
