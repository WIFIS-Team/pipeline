import wifisIO
import wifisSlices as slices
import numpy as np
import time
import matplotlib.pyplot as plt
import wifisUncertainties
import wifisBadPixels as badPixels
import wifisCreateCube as createCube
import wifisWCS
import wifisWaveSol as waveSol
import wifisProcessRamp as processRamp
import os

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0' # Used to show compile errors for debugging, can be removed
os.environ['PYOPENCL_CTX'] = '1' # Used to specify which OpenCL device to target. Should be uncommented and pointed to correct device to avoid future interactive requests

#*****************************************************************************
#REQUIRED INPUT
#likely changes for each obs
flatFolder = '20170512004126'
waveFolder = '20170512004311'
obsLstFile = 'obs.lst'
skyLstFile = 'sky.lst'

#likely static
distMapFile = 'ronchi_map_polyfit.fits'
distMapLimitsFile = 'master_flat_limits.fits'
satFile = '/data/WIFIS/H2RG-G17084-ASIC-08-319/UpTheRamp/20170502141549/processed/master_detLin_satCounts.fits'
bpmFile = 'bpm.fits'
nlFile = '/data/WIFIS/H2RG-G17084-ASIC-08-319/UpTheRamp/20170502141549/processed/master_detLin_NLCoeff.fits'
spatGridPropsFile = 'spatGridProps.dat'
waveTempFile = 'waveFlat_distCor.fits'
waveTempResultsFile = 'waveFlat_fitting_results.pkl'
atlasFile = '/data/pipeline/external_data/best_lines2.dat'

#pixel scale
xScale = 0.532021532706
yScale = -0.545667026386
 
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

#deal with flats first
print('processing flat')

#check image type
if (os.path.exists(flatFolder+'/Result')):
    flat, sigmaImg, satFrame, hdr = processRamp.fromFowler(flatFolder, 'processed/'+flatFolder+'_flat.fits', satCounts, nlCoeff, BPM, nChannel=32, rowSplit=1, nlSplit=1, combSplit=1, crReject=False, bpmCorRng=20)
else:
    #if dealing with a very big ramp, increase rowSplit value below
    flat, sigmaImg, satFrame, hdr = processRamp.fromUTR(flatFolder, 'processed/'+flatFolder+'_flat.fits', satCounts, nlCoeff, BPM, nChannel=32, rowSplit=1, nlSplit=32, combSplit=32, crReject=False, bpmCorRng=20)

#get limits
print('Getting limits relative to distortion map')
limits = slices.findLimits(flat,limSmth=20, rmRef=True)
flat = flat[4:2044, 4:2044]

#get shift compared to ronchi mask
print('Extracting flat slices')
shft = np.nanmedian(limits[1:-1,:] - distMapLimits[1:-1,:])
flatSlices = slices.extSlices(flat, distMapLimits, shft=shft)
wifisIO.writeFits(flatSlices,'processed/'+flatFolder+'_flat_slices.fits', ask=False)

print('Getting normalized slices')
#get normalized flat field slices
flatNorm = slices.getResponseAll(flatSlices, 0, 0.1)
wifisIO.writeFits(flatNorm, 'processed/'+flatFolder+'_flat_slices_norm.fits',ask=False)

#read in ronchi mask
distMap = wifisIO.readImgsFromFile(distMapFile)[0]
spatGridProps = wifisIO.readTable(spatGridPropsFile)

print('processing arc image')
if (os.path.exists(waveFolder+'/Result')):
    wave, sigmaImg, satFrame, hdr = processRamp.fromFowler(waveFolder, 'processed/'+waveFolder+'_wave.fits', satCounts, nlCoeff, BPM, nChannel=32, rowSplit=1, nlSplit=1, combSplit=1, crReject=False, bpmCorRng=2)
else:
    wave, sigmaImg, satFrame, hdr = processRamp.fromUTR(waveFolder, 'processed/'+waveFolder+'_wave.fits', satCounts, nlCoeff, BPM, nChannel=32, rowSplit=1, nlSplit=32, combSplit=32, crReject=False, bpmCorRng=2)

#remove reference pixels
wave = wave[4:2044,4:2044]

print('extracting wave slices')
waveSlices = slices.extSlices(wave,  distMapLimits, shft=shft)
wifisIO.writeFits(waveSlices, 'processed/'+waveFolder+'_wave_slices.fits', ask=False)

print('Flat fielding wave slices')
waveFlat = slices.ffCorrectAll(waveSlices, flatNorm)

print('Distortion correcting flat slices')
waveCor = createCube.distCorAll(waveFlat, distMap, spatGridProps=spatGridProps)
wifisIO.writeFits(waveCor,'wave_slices_distCor.fits', ask=False)

print('getting dispersion solution')
template = wifisIO.readImgsFromFile(waveTempFile)[0]
templateResults = wifisIO.readPickle(waveTempResultsFile)
sol = templateResults[5]

result =  waveSol.getWaveSol(waveCor, template, atlasFile, 3, sol, winRng=9, mxCcor=150, weights=False, buildSol=False, sigmaClip=1, allowLower=True, lngthConstraint=True)
wifisIO.writePickle(result, 'processed/'+waveFolder+'_waveFitResults.pkl')

polySol = waveSol.polyFitDispSolution(result[0], plot=False,degree=2)

print('Building wavelegth map')
waveMap = waveSol.buildWaveMap(polySol, waveCor[0].shape[1])

#now trim wavemap if needed

waveGridProps = createCube.compWaveGrid(waveMap)
wifisIO.writeTable(waveGridProps, 'processed/'+waveFolder+'_waveGridProps.dat')

print('processing observations')

obsLst = wifisIO.readAsciiList(obsLstFile)

if obslst.ndim == 0:
    obsLst = np.asarray([obsLst])

if skyLst is not None:
    skyLst = wifisIO.readAsciiList(skyLstFile)
    if skyLst.ndim == 0:
        skyLst = np.asarray([skyLst])

cubeLst = []
for i in range(len(obsLst)):

    dataFolder = obsLst[i]

    print('Working on data folder ' + dataFolder)

    print('Processing science data')
    if (os.path.exists(dataFolder+'/Result')):
        data, sigmaImg, satFrame, hdr = processRamp.fromFowler(dataFolder, 'processed/'+dataFolder+'_obs.fits', satCounts, nlCoeff, BPM, nChannel=32, rowSplit=1, nlSplit=1, combSplit=1, crReject=False, bpmCorRng=2)
    else:
        data, sigmaImg, satFrame, hdr = processRamp.fromUTR(dataFolder, 'processed/'+dataFolder+'_obs.fits', satCounts, nlCoeff, BPM, nChannel=32, rowSplit=1, nlSplit=32, combSplit=32, crReject=False, bpmCorRng=2)

    #remove reference pixels
    data = data[4:2044,4:2044]

    if skyLst is not None:
        skyFolder = skyLst[i]
        print('Processing sky data')
        if (os.path.exists(skyFolder+'/Result')):
            sky, sigmaImg, satFrame, hdrSky = processRamp.fromFowler(skyFolder, 'processed/'+skyFolder+'_sky.fits', satCounts, nlCoeff, BPM, nChannel=32, rowSplit=1, nlSplit=1, combSplit=1, crReject=False, bpmCorRng=2)
        else:
            sky, sigmaImg, satFrame, hdrSky = processRamp.fromUTR(skyFolder, 'processed/'+skyFolder+'_sky.fits', satCounts, nlCoeff, BPM, nChannel=32, rowSplit=1, nlSplit=32, combSplit=32, crReject=False, bpmCorRng=2)
        #remove reference pixels
        sky = sky[4:2044,4:2044]

        #subtract sky from data at this stage
        print('Subtracting sky from obs')
        data -= sky
        
    print('Extracting slices')
    #extracting slices
    dataSlices = slices.extSlices(data, distMapLimits, shft=shft)
    wifisIO.writeFits(dataslices, 'processed/'+dataFolder+'_obs_slices.fits', ask=False)

    #apply flat-field correction
    print('Applying flat field corrections')
    dataFlat = slices.ffCorrectAll(dataSlices, flatNorm)
      
    print('distortion correcting')
    #distortion correct data
    dataCor = createCube.distCorAll(dataFlat, distMap, spatGridProps=spatGridProps)
    wifisIO.writeFits(dataCor, 'processed/'+dataFolder+'_obs_slices_distCor.fits', ask=False)

    print('placing on uniform wave grid')
    #place on uniform wavelength grid
    dataGrid = createCube.waveCorAll(dataCor, waveMap, waveGridProps=waveGridProps)
    wifisIO.writeFits(dataGrid, 'processed/'+dataFolder+'_obs_slices_grid.fits', ask=False)

    print('creating cube')
    #create cube
    dataCube = createCube.mkCube(dataGrid, ndiv=1)

    dataImg = np.nansum(dataCube, axis=2)
    
    #write WCS to header
    hdrCube = hdr
    hdrImg = hdr

    headers.getWCSCube(dataCube, hdrCube, xScale, yScale, waveGridProps)
    wifisIO.writeFits(dataCube, 'processed/'+dataFolder+'_obs_cube.fits', hdr=hdrCube, ask=False)
    headers.getWCSImg(dataImg, hdrImg, xScale, yScale)
    wifisIO.writeFits(dataImg, 'processed/'+dataFolder+'_obs_cubeImg.fits',hdr=hdrImg, ask=False)

    cubeLst.append(dataCube)

#get object name
objectName = hdrCube['Object']

combCube = np.zeros(dataCube.shape, dtype=dataCube.dtype)
for cube in cubeLst:
    combCube += cube

cubeAll /= float(len(cubeLst))
#now save combined cube

wifisIO.writeFits(combCube, 'processed/'+objectName+'_combined_cube.fits', hdr=hdrCube, ask=False)
combImg = np.nansum(combCube, axis=2)

wifisIO.writeFits(combImg, 'processed/'+objectName+'_combined_cubeImg.fits', hdr=hdrImg, ask=False)
    
    
    


