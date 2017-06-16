import matplotlib
matplotlib.use('pdf')

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

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0' # Used to show compile errors for debugging, can be removed
os.environ['PYOPENCL_CTX'] = '1' # Used to specify which OpenCL device to target. Should be uncommented and pointed to correct device to avoid future interactive requests

#*****************************************************************************
#REQUIRED INPUT
#likely changes for each obs
flatInfoFile = 'flat.lst'
waveInfoFile = 'wave.lst'
obsLstFile = 'obs.lst'
skyLstFile = None #'sky.lst'
darkInfoFile = 'dark.lst'
noFlat = True

#likely static
rootFolder = '/data/WIFIS/H2RG-G17084-ASIC-08-319'

#may
distMapFile = '/home/jason/wifis/data/ronchi_map_may/distortionMap.fits'
distMapLimitsFile = '/home/jason/wifis/data/ronchi_map_may/ronchiMap_limits.fits'
spatGridPropsFile = '/home/jason/wifis/data/ronchi_map_may/spatGridProps.dat'

#june
#distMapFile = '/home/jason/wifis/data/ronchi_map_june/20170607010313/processed/20170607001609_ronchi_distMap.fits'
#distMapLimitsFile = '/home/jason/wifis/data/ronchi_map_june/20170607010313/processed/20170607001828_flat_limits.fits'
#spatGridPropsFile = '/home/jason/wifis/data/ronchi_map_june/20170607010313/processed/20170607001609_ronchi_spatGridProps.dat'

satFile = '/data/WIFIS/H2RG-G17084-ASIC-08-319/UpTheRamp/20170504201819/processed/master_detLin_satCounts.fits'
bpmFile = '/data/pipeline/external_data/bpm.fits'
nlFile = '/data/WIFIS/H2RG-G17084-ASIC-08-319/UpTheRamp/20170504201819/processed/master_detLin_NLCoeff.fits'
waveTempFile = '/data/pipeline/external_data/waveTemplate.fits'
waveTempResultsFile = '/data/pipeline/external_data/waveTemplateFittingResults.pkl'
atlasFile = '/data/pipeline/external_data/best_lines2.dat'

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

mxOrder = 3

ndiv = 1
if ndiv == 0:
    yScale = yScale*35/18.

#*****************************************************************************

#create processed directory, in case it doesn't exist
wifisIO.createDir('processed')

#first read in all calibration data
distMap = wifisIO.readImgsFromFile(distMapFile)[0]
distMapLimits = wifisIO.readImgsFromFile(distMapLimitsFile)[0]
satCounts = wifisIO.readImgsFromFile(satFile)[0]
nlCoeff = wifisIO.readImgsFromFile(nlFile)[0]

#first check if BPM is provided
if os.path.exists(bpmFile):
    BPM = wifisIO.readImgsFromFile(bpmFile)[0]
    BPM = BPM.astype(bool)
else:
    BPM = None

#deal with darks
if darkInfoFile is not None:
    darkFolder = wifisIO.readAsciiList(darkInfoFile).tostring()
    
    if not os.path.exists('processed/'+darkFolder+'_dark.fits'):
        darkFolder = wifisIO.readAsciiList(darkInfoFile).tostring()
        print('processing dark')

        dark, sigmaImg, satFrame, hdr = processRamp.auto(darkFolder, rootFolder,'processed/'+darkFolder+'_dark.fits', satCounts, nlCoeff, None, nChannel=32, rowSplit=1, nlSplit=32, combSplit=32, crReject=False, bpmCorRng=0, saveAll=False, ignoreBPM=True)
    else:
        dark = wifisIO.readImgsFromFile('processed/'+darkFolder+'_dark.fits')[0]
else:
    darkFolder=None
    
#deal with flats first
flatFolder = wifisIO.readAsciiList(flatInfoFile).tostring()

if not os.path.exists('processed/'+flatFolder+'_flat.fits'):
    print('processing flat')
    flat, sigmaImg, satFrame, flatHdr = processRamp.auto(flatFolder, rootFolder,'processed/'+flatFolder+'_flat.fits', satCounts, nlCoeff, BPM, nChannel=32, rowSplit=1, nlSplit=32, combSplit=32, crReject=False, bpmCorRng=20, saveAll=False)
else:
    flat,flatHdr = wifisIO.readImgsFromFile('processed/'+flatFolder+'_flat.fits')

if not os.path.exists('processed/'+flatFolder+'_flat_limits.fits'):
    #get limits
    print('Getting limits relative to distortion map')
    limits = slices.findLimits(flat,limSmth=20, rmRef=True)
    shft = np.nanmedian(limits[1:-1,:] - distMapLimits[1:-1,:])
    flatHdr.set('LIMSHIFT',shft, 'Limits shift relative to Ronchi slices')
    wifisIO.writeFits(limits, 'processed/'+flatFolder+'_flat_limits.fits', hdr=flatHdr)
else:
    limits, flatHdr = wifisIO.readImgsFromFile('processed/'+flatFolder+'_flat_limits.fits')
    shft = flatHdr['LIMSHIFT']
    
if darkFolder is not None:
    print('subtracting dark from flat')
    flat -= dark
    
flat = flat[4:2044, 4:2044]

#get shift compared to ronchi mask
if not os.path.exists('processed/'+flatFolder+'_flat_slices.fits'):
    print('Extracting flat slices')
    flatSlices = slices.extSlices(flat, distMapLimits, shft=shft)
    wifisIO.writeFits(flatSlices,'processed/'+flatFolder+'_flat_slices.fits', ask=False, hdr=flatHdr)
else:
    flatSlices,flatHdr = wifisIO.readImgsFromFile('processed/'+flatFolder+'_flat_slices.fits')
    
if not os.path.exists('processed/'+flatFolder+'_flat_slcies_norm.fits'):
    print('Getting normalized slices')
    #get normalized flat field slices
    flatNorm = slices.getResponseAll(flatSlices, 0, 0.1)
    wifisIO.writeFits(flatNorm, 'processed/'+flatFolder+'_flat_slices_norm.fits',ask=False)
else:
    flatNorm = wifisIO.readImgsFromFile('processed/'+flatFolder+'_flat_slices_norm.fits')[0]
    
#read in ronchi mask
distMap = wifisIO.readImgsFromFile(distMapFile)[0]
spatGridProps = wifisIO.readTable(spatGridPropsFile)

waveFolder = wifisIO.readAsciiList(waveInfoFile).tostring()
if not os.path.exists('processed/'+waveFolder+'_wave.fits'):
    print('processing arc image')
    wave, sigmaImg, satFrame, hdr = processRamp.auto(waveFolder, rootFolder,'processed/'+waveFolder+'_wave.fits', satCounts, nlCoeff, BPM, nChannel=32, rowSplit=1, nlSplit=32, combSplit=32, crReject=False, bpmCorRng=2, saveAll=False)
else:
    wave = wifisIO.readImgsFromFile('processed/'+waveFolder+'_wave.fits')[0]
    
#remove reference pixels
wave = wave[4:2044,4:2044]

if not os.path.exists('processed/'+waveFolder+'wave_slices.fits'):
    print('extracting wave slices')
    waveSlices = slices.extSlices(wave,  distMapLimits, shft=shft)
    wifisIO.writeFits(waveSlices, 'processed/'+waveFolder+'_wave_slices.fits', ask=False)
else:
    waveSlices = wifisIO.readImgsFromFile('processed/'+waveFolder+'_wave_slices.fits')[0]

if not os.path.exists('processed/'+waveFolder+'wave_slices_distCor.fits'):
    print('Distortion correcting flat slices')
    waveCor = createCube.distCorAll(waveSlices, distMap, spatGridProps=spatGridProps)
    wifisIO.writeFits(waveCor,'processed/'+waveFolder+'_wave_slices_distCor.fits', ask=False)
else:
    waveCor = wifisIO.readImgsFromFile('processed/'+waveFolder+'_wave_slices_distCor.fits')[0]

if not os.path.exists('processed/'+waveFolder+'_waveFitResults.pkl'):
    print('getting dispersion solution')
    template = wifisIO.readImgsFromFile(waveTempFile)[0]
    templateResults = wifisIO.readPickle(waveTempResultsFile)
    prevSol = templateResults[5]

    result = waveSol.getWaveSol(waveCor, template, atlasFile,mxOrder, prevSol, winRng=9, mxCcor=150, weights=False, buildSol=False, allowLower=False, sigmaClip=3, lngthConstraint = True, MP=True, adjustFitWin=False, allowSearch=False, sigmaClipRounds=3)

    wifisIO.writePickle(result, 'processed/'+waveFolder+'_waveFitResults.pkl')
else:
    result = wifisIO.readPickle('processed/'+waveFolder+'_waveFitResults.pkl')
    
if not os.path.exists('processed/'+waveFolder+'_waveMap.fits'):
    print('Building wavelegth map')
    wifisIO.createDir('quality_control')
    rmsClean, dispSolClean, pixSolClean = waveSol.cleanDispSol(result, plotFile='quality_control/'+waveFolder+'_waveFit_rms.pdf')

    #resultClean = [dispSolClean, result[1],result[2],result[3], rmsClean, pixSolClean]
    #wifisIO.writePickle(resultClean, 'processed/'+waveFolder+'_waveFitResults_cleaned.pkl')

    waveMap = waveSol.buildWaveMap(dispSolClean, waveCor[0].shape[1], extrapolate=True)
    wifisIO.writeFits(waveMap, 'processed/'+waveFolder+'_waveMap.fits', ask=False)

    #now trim wavemap if needed

    waveGridProps = createCube.compWaveGrid(waveMap)
    wifisIO.writeTable(waveGridProps, 'processed/'+waveFolder+'_waveGridProps.dat')
else:
    waveMap = wifisIO.readImgsFromFile('processed/'+waveFolder+'_waveMap.fits')[0]
    waveGridProps = wifisIO.readTable('processed/'+waveFolder+'_waveGridProps.dat')

if not os.path.exists('processed/'+waveFolder+'_wave_slices_grid.fits'):
    print('placing arc image on grid')
    waveGrid = createCube.waveCorAll(waveCor, waveMap, waveGridProps=waveGridProps)
    wifisIO.writeFits(waveGrid, 'processed/'+waveFolder+'_wave_slices_grid.fits', ask=False)

print('processing observations')

obsLst = wifisIO.readAsciiList(obsLstFile)

if obsLst.ndim == 0:
    obsLst = np.asarray([obsLst])

if skyLstFile is not None:
    skyLst = wifisIO.readAsciiList(skyLstFile)
    if skyLst.ndim == 0:
        skyLst = np.asarray([skyLst])
else:
    skyLst = None
    
cubeLst = []
for i in range(len(obsLst)):

    dataFolder = obsLst[i]

    print('Working on data folder ' + dataFolder)

    print('Processing science data')
    data, sigmaImg, satFrame, hdr = processRamp.auto(dataFolder, rootFolder,'processed/'+dataFolder+'_obs.fits', satCounts, nlCoeff, BPM, nChannel=32, rowSplit=1, nlSplit=32, combSplit=32, crReject=False, bpmCorRng=2, saveAll=False)
    
    #remove reference pixels
    data = data[4:2044,4:2044]

    if skyLst is not None:
        skyFolder = skyLst[i]
        print('Processing sky data')
        sky, sigmaImg, satFrame, hdrSky = processRamp.auto(skyFolder, rootFolder,'processed/'+skyFolder+'_sky.fits', satCounts, nlCoeff, BPM, nChannel=32, rowSplit=1, nlSplit=32, combSplit=32, crReject=False, bpmCorRng=2,saveAll=False)

        #remove reference pixels
        sky = sky[4:2044,4:2044]

        #subtract sky from data at this stage
        print('Subtracting sky from obs')
        data -= sky
    else:
        if darkFolder is not None:
            data -= dark[4:2044,4:2044]
            
    print('Extracting slices')
    #extracting slices
    dataSlices = slices.extSlices(data, distMapLimits, shft=shft)
    wifisIO.writeFits(dataSlices, 'processed/'+dataFolder+'_obs_slices.fits', ask=False)

    #apply flat-field correction
    #print('Applying flat field corrections')
    if not noFlat:
        dataFlat = slices.ffCorrectAll(dataSlices, flatNorm)
    else:
        dataFlat = dataSlices
        
    print('distortion correcting')
    #distortion correct data
    dataCor = createCube.distCorAll(dataFlat, distMap, spatGridProps=spatGridProps)
    if noFlat:
        wifisIO.writeFits(dataCor, 'processed/'+dataFolder+'_obs_slices_distCor_noFlat.fits', ask=False)
    else:
        wifisIO.writeFits(dataCor, 'processed/'+dataFolder+'_obs_slices_distCor.fits', ask=False)

    print('placing on uniform wave grid')
    #place on uniform wavelength grid
    dataGrid = createCube.waveCorAll(dataCor, waveMap, waveGridProps=waveGridProps)
    if noFlat:
        wifisIO.writeFits(dataGrid, 'processed/'+dataFolder+'_obs_slices_grid_noFlat.fits', ask=False)
    else:
        wifisIO.writeFits(dataGrid, 'processed/'+dataFolder+'_obs_slices_grid.fits', ask=False)
                
    print('creating cube')
    #create cube
    dataCube = createCube.mkCube(dataGrid, ndiv=ndiv)

    dataImg = np.nansum(dataCube, axis=2)

    #write WCS to header
    hdrImg = hdr[:]
    hdrCube = hdr
    
    headers.getWCSImg(dataImg, hdrImg, xScale, yScale)
    if noFlat:
        wifisIO.writeFits(dataImg, 'processed/'+dataFolder+'_obs_cubeImg_noFlat.fits',hdr=hdrImg, ask=False)
    else:
        wifisIO.writeFits(dataImg, 'processed/'+dataFolder+'_obs_cubeImg.fits',hdr=hdrImg, ask=False)

    headers.getWCSCube(dataCube, hdrCube, xScale, yScale, waveGridProps)

    if noFlat:
        wifisIO.writeFits(dataCube, 'processed/'+dataFolder+'_obs_cube_noFlat.fits', hdr=hdrCube, ask=False)
    else:
        wifisIO.writeFits(dataCube, 'processed/'+dataFolder+'_obs_cube.fits', hdr=hdrCube, ask=False)

    cubeLst.append(dataCube)

#get object name
objectName = hdrCube['Object']

combCube = np.zeros(dataCube.shape, dtype=dataCube.dtype)
for cube in cubeLst:
    combCube += cube

combCube /= float(len(cubeLst))
#now save combined cube

if noFlat:
    wifisIO.writeFits(combCube, 'processed/'+objectName+'_combined_cube_noFlat.fits', hdr=hdrCube, ask=False)
else:
    wifisIO.writeFits(combCube, 'processed/'+objectName+'_combined_cube.fits', hdr=hdrCube, ask=False)

combImg = np.nansum(combCube, axis=2)

if noFlat:
    wifisIO.writeFits(combImg, 'processed/'+objectName+'_combined_cubeImg_noFlat.fits', hdr=hdrImg, ask=False)
else:
    wifisIO.writeFits(combImg, 'processed/'+objectName+'_combined_cubeImg.fits', hdr=hdrImg, ask=False)

    
    


