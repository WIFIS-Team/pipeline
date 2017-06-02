import wifisWaveSol as waveSol
import wifisIO
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
import wifisSlices as slices
import os
import wifisGetSatInfo as satInfo
import wifisCombineData as combData

os.environ['PYOPENCL_CTX'] = '1' # Used to specify which OpenCL device to target. Should be uncommented and pointed to correct device to avoid future interactive requests

#required input!
atlasFile = '/data/pipeline/external_data/best_lines2.dat'
satFile = '/data/WIFIS/H2RG-G17084-ASIC-08-319/UpTheRamp/20170504201819/processed/master_detLin_satCounts.fits'
flatFolder = '20170510233851'
waveFolder = '20170510234217'

print('getting ready')

#read in previous results and template
template = wifisIO.readImgsFromFile('/data/pipeline/external_data/waveTemplate.fits')[0]
prevResults = wifisIO.readPickle('/data/pipeline/external_data/waveTemplateFittingResults.pkl')
prevSol = prevResults[5]

print('Reading input data')

print('Processing arc file')
#check the type of raw data, only assumes CDS or up-the-ramp
if (os.path.exists(waveFolder+'/Results')):
    #CDS image
    wave = wifisIO.readImgsFromFile(waveFolder+'/Results/CDSResult.fits')[0]
    wave = wave[4:2044, 4:2044] #trim off reference pixels
else:
    #assume up-the-ramp
    data, inttime, hdr = wifisIO.readRampFromFolder(waveFolder)
    satCounts = wifisIO.readImgsFromFile(satFile)[0]
    satFrame = satInfo.getSatFrameCL(data, satCounts,32)
    
    #get processed ramp
    wave = combData.upTheRampCL(inttime, data, satFrame, 32)[0]

print('Processing flat file')
#check the type of raw data, only assumes CDS or up-the-ramp
if (os.path.exists(flatFolder+'/Results')):
    #CDS image
    flat = wifisIO.readImgsFromFile(flatFolder+'/Results/CDSResult.fits')[0]
else:
    #assume up-the-ramp
    data, inttime, hdr = wifisIO.readRampFromFolder(flatFolder)
    satCounts = wifisIO.readImgsFromFile(satFile)[0]
    satFrame = satInfo.getSatFrameCL(data, satCounts,32)
    
    #get processed ramp
    flat = combData.upTheRampCL(inttime, data, satFrame, 32)[0]

print('Finding flat limits')
limits = slices.findLimits(flat, dispAxis=0, winRng=51, imgSmth=5, limSmth=20)

print('extracting wave slices')
waveSlices = slices.extSlices(wave, limits, dispAxis=0)

print('Getting dispersion solution')
result = waveSol.getWaveSol(waveSlices, template, atlasFile,3, prevSol, winRng=9, mxCcor=150, weights=False, buildSol=False, allowLower=True, sigmaClip=2., lngthConstraint = True, MP=True)

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
fwhmMapLst = waveSol.buildFWHMMap(pixCentLst, fwhmLst, npts)

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
print('*****************************************')
print('****** MEAN FWHM IS '+ str(fwhmMean) + ' *******')
print('*****************************************')

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
wifisIO.createDir('quick_reduction')
wifisIO.writeFits(waveMap, 'quick_reduction/'+waveFolder+'_wavelength_map.fits', ask=False)
wifisIO.writeFits(fwhmMap, 'quick_reduction/'+waveFolder+'_fwhm_map.fits', ask=False)

plt.imshow(fwhmMap, aspect='auto', cmap='jet')
plt.colorbar()
plt.savefig('quick_reduction/'+waveFolder+'fwhm_map.png', dpi=300)
plt.close()

