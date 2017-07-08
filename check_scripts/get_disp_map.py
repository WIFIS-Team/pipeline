import matplotlib
matplotlib.use('Agg')
import wifisWaveSol as waveSol
import wifisIO
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
import wifisSlices as slices
import os
import wifisGetSatInfo as satInfo
import wifisCombineData as combData
import warnings
import wifisCreateCube as createCube

os.environ['PYOPENCL_CTX'] = '1' # Used to specify which OpenCL device to target. Should be uncommented and pointed to correct device to avoid future interactive requests
warnings.simplefilter('ignore', RuntimeWarning)

#*******************************************************************************
#required input!

hband = False
pipelineFolder = '/data/pipeline'
rootFolder = '/data/WIFIS/H2RG-G17084-ASIC-08-319'
colorbarLims = [0,10]

waveFileLst = 'wave.lst'
flatFileLst = 'flat.lst'

if hband:
    #needs updating
    templateFile = '/data/pipeline/external_data/hband_template.fits'
    prevResultsFile = '/data/pipeline/external_data/hband_template.pkl'
    lngthConstraint = True
    mxOrder = 3
    distMapFile = pipelineFolder + '/external_data/distMap.fits'
    spatGridPropsFile = pipelineFolder + '/external_data/distMap_spatGridProps.dat'
    distMapLimitsFile = pipelineFolder+'/external_data/distMap_limits.fits'
else:
    #TB band
    lst = 'tb_arc.lst'
    lngthConstraint = True
    templateFile = pipelineFolder+'/external_data/waveTemplate.fits'
    prevResultsFile = pipelineFolder+'/external_data/waveTemplateFittingResults.pkl'
    distMapFile = pipelineFolder + '/external_data/distMap.fits'
    spatGridPropsFile = pipelineFolder + '/external_data/distMap_spatGridProps.dat'
    distMapLimitsFile = pipelineFolder+'/external_data/distMap_limits.fits'
    mxOrder = 3

#should be (mostly) static
atlasFile = pipelineFolder + '/external_data/best_lines2.dat'
satFile = pipelineFolder + '/external_data/master_detLin_satCounts.fits'

sigmaClipRounds=1 #number of iterations when sigma-clipping of dispersion solution
sigmaClip = 3 #sigma-clip cutoff when sigma-clipping dispersion solution
sigmaLimit= 3 #relative noise limit (x * noise level) for which to reject lines
#*******************************************************************************

print('getting ready')
wifisIO.createDir('quick_reduction')

#read in previous results and template
template = wifisIO.readImgsFromFile(templateFile)[0]
prevResults = wifisIO.readPickle(prevResultsFile)
prevSol = prevResults[5]
distMap = wifisIO.readImgsFromFile(distMapFile)[0]
spatGridProps = wifisIO.readTable(spatGridPropsFile)

print('Reading input data')
waveLst = wifisIO.readAsciiList(waveFileLst)
flatLst = wifisIO.readAsciiList(flatFileLst)

if (waveLst.ndim == 0):
    waveLst = np.asarray([waveLst])
if (flatLst.ndim == 0):
    flatLst = np.asarray([flatLst])
    
satCounts = wifisIO.readImgsFromFile(satFile)[0]

for fle in range(len(waveLst)):
    waveFolder = waveLst[fle]
    flatFolder = flatLst[fle]

    if (os.path.exists('quick_reduction/'+waveFolder+'_fwhm_map.png') and os.path.exists('quick_reduction/'+waveFolder+'_fwhm_map.fits') and os.path.exists('quick_reduction/'+waveFolder+'_wavelength_map.fits')):
        pass
    else:
        print('Processing arc file '+ waveFolder)
        #check the type of raw data, only assumes CDS or up-the-ramp
        if (os.path.exists(rootFolder + '/CDSReference/'+waveFolder+'/Result/CDSResult.fits')):
            #CDS image
            wave = wifisIO.readImgsFromFile(rootFolder + '/CDSReference/'+waveFolder+'/Result/CDSResult.fits')[0]
            wave = wave[4:2044, 4:2044] #trim off reference pixels
        else:
            #assume up-the-ramp
            data, inttime, hdr = wifisIO.readRampFromFolder(rootFolder + '/UpTheRamp/'+waveFolder)
            satFrame = satInfo.getSatFrameCL(data, satCounts,32)
    
            #get processed ramp
            wave = combData.upTheRampCL(inttime, data, satFrame, 32)[0]
            wave = wave[4:2044,4:2044]

        if (os.path.exists('quick_reduction/'+flatFolder+'_flat_limits.fits') and os.path.exists('quick_reduction/'+flatFolder+'_flat_slices.fits')):
            limits, limitsHdr = wifisIO.readImgsFromFile('quick_reduction/'+flatFolder+'_flat_limits.fits')
            flatSlices = wifisIO.readImgsFromFile('quick_reduction/'+flatFolder+'_flat_slices.fits')[0]
            shft = limitsHdr['LIMSHIFT']
        else:
            print('Processing flat file')
            #check the type of raw data, only assumes CDS or up-the-ramp
            if (os.path.exists(rootFolder + '/CDSReference/'+flatFolder+'/Result/CDSResult.fits')):
                #CDS image
                flat, flatHdr = wifisIO.readImgsFromFile(rootFolder + '/CDSReference/'+flatFolder+'/Result/CDSResult.fits')
            else:
                #assume up-the-ramp
                data, inttime, flatHdr = wifisIO.readRampFromFolder(rootFolder + '/UpTheRamp/'+flatFolder)
                satCounts = wifisIO.readImgsFromFile(satFile)[0]
                satFrame = satInfo.getSatFrameCL(data, satCounts,32)
                #get processed ramp
                flat = combData.upTheRampCL(inttime, data, satFrame, 32)[0]

            print('Finding flat limits')
            limits = slices.findLimits(flat, dispAxis=0, winRng=51, imgSmth=5, limSmth=20,rmRef=True)
            distMapLimits = wifisIO.readImgsFromFile(distMapLimitsFile)[0]
            shft = int(np.nanmedian(limits[1:-1,:] - distMapLimits[1:-1,:]))
            limits = distMapLimits
            flatSlices = slices.extSlices(flat[4:2044,4:2044], limits)

            flatHdr.set('LIMSHIFT',shft, 'Limits shift relative to Ronchi slices')
            wifisIO.writeFits(limits,'quick_reduction/'+flatFolder+'_flat_limits.fits',hdr=flatHdr)
            wifisIO.writeFits(flatSlices,'quick_reduction/'+flatFolder+'_flat_slices.fits')
                
        print('extracting wave slices')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", "RuntimeWarning")
            waveSlices = slices.extSlices(wave, limits, dispAxis=0, shft=shft)

        print('getting normalized wave slices')
        if hband:
            flatNorm = slices.getResponseAll(flatSlices, 0, 0.6)
        else:
            flatNorm = slices.getResponseAll(flatSlices, 0, 0.1)

        waveNorm = slices.ffCorrectAll(waveSlices, flatNorm)

        print ('getting distortion corrected slices')
        waveCor = createCube.distCorAll(waveSlices, distMap, spatGridProps=spatGridProps)

        #save data
        wifisIO.writeFits(waveCor, 'quick_reduction/'+waveFolder+'_wave_slices_distCor.fits')
        print('Getting dispersion solution')

        result = waveSol.getWaveSol(waveCor, template, atlasFile, 3, prevSol, winRng=9, mxCcor=150, weights=False, buildSol=False, sigmaClip=sigmaClip, allowLower=False, lngthConstraint=True, MP=True, adjustFitWin=True, sigmaLimit=sigmaLimit, allowSearch=False, sigmaClipRounds=sigmaClipRounds)        
       
        print('Extracting solution results')
        dispSolLst = result[0]
        fwhmLst = result[1]
        pixCentLst = result[2]
        waveCentLst = result[3]
        rmsLst = result[4]
        pixSolLst = result[5]

        print('Building maps of results')
        npts = waveSlices[0].shape[1]
        waveMapLst = waveSol.buildWaveMap(dispSolLst,npts)

        for fwhm in fwhmLst:
            for i in range(len(fwhm)):
                fwhm[i] = np.abs(fwhm[i])
        
        fwhmMapLst = waveSol.buildFWHMMap(pixCentLst, fwhmLst, npts)
        #get max and min starting wavelength based on median of central slice (slice 8)

        if hband:
            trimSlc = waveSol.trimWaveSlice([waveMapLst[8], flatSlices[8], 0.5])
            waveMin = np.nanmin(trimSlc)
            waveMax = np.nanmax(trimSlc)
        else:
            trimSlc = waveMapLst[8]
            waveMax = np.nanmedian(trimSlc[:,0])
            waveMin = np.nanmedian(trimSlc[:,-1])
 
        print('*******************************************************')
        print('*** Minimum median wavelength for slice 8 is ' + str(waveMin)+ ' ***')
        print('*** Maximum median wavelength for slice 8 is ' + str(waveMax)+ ' ***')
        print('*******************************************************')

        
        #determine length along spatial direction
        ntot = 0
        for j in range(len(rmsLst)):
            ntot += len(rmsLst[j])

        #get median FWHM
        fwhmAll = []
        for f in fwhmLst:
            for i in range(len(f)):
                for j in range(len(f[i])):
                    fwhmAll.append(f[i][j])
            
        fwhmMed = np.nanmedian(fwhmAll)
        print('**************************************')
        print('*** MEDIAN FWHM IS '+ str(fwhmMed) + ' ***')
        print('**************************************')

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
        wifisIO.writeFits(waveMap, 'quick_reduction/'+waveFolder+'_wave_wavelength_map.fits', ask=False)
        wifisIO.writeFits(fwhmMap, 'quick_reduction/'+waveFolder+'_wave_fwhm_map.fits', ask=False)

        print('plotting results')
        fig = plt.figure()

        plt.imshow(fwhmMap, aspect='auto', cmap='jet', clim=colorbarLims, origin='lower')

        plt.colorbar()
        plt.title('Median FWHM is '+'{:3.1f}'.format(fwhmMed) +', min wave is '+'{:6.1f}'.format(waveMin)+', max wave is '+'{:6.1f}'.format(waveMax))
        plt.savefig('quick_reduction/'+waveFolder+'_fwhm_map.png', dpi=300)
        plt.close()

