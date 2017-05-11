import wifisWaveSol as waveSol
import wifisIO
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
import wifisSlices as slices

limitsFile = 'limits.fits' #must point to location of limits

print('Reading input data')
limits = wifisIO.readImgsFromFile(limitsFile)[0]

wave = wifisIO.readImgsFromFile('CDSResult.fits')[0][4:2044,:]#trim, if limits file doesn't include reference pixels, which is likely the case
    
print('extracting wave slices')
waveSlices = slices.extSlices(wave, limits, dispAxis=0)

print('getting ready to find dispersion solution')
atlas = 'external_data/best_lines2.dat'

#read in previous results and template
template = wifisIO.readPickle('template_input.pkl')
tmpLst = template[0]
prevSol = template[1]

print('Getting dispersion solution')
result = waveSol.getWaveSol(waveSlices, tmpLst, atlas,3, prevSol, winRng=9, mxCcor=30, weights=False, buildSol=False, allowLower=True, sigmaClip=2., lngthConstraint = True)

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

#determine length along dispersion direction
ntot = 0
for r in rmsLst:
    ntot += len(r)

#build "detector" map images
#wavelength solution
waveMap = np.empty((npts,ntot),dtype='float32')
strt=0
for m in waveMapLst:
    waveMap[:,strt:strt+m.shape[0]] = m.T
    strt += m.shape[0]

wifisIO.writeFits(waveMap, 'wavelength_map.fits')

#fwhm map
fwhmMap = np.empty((npts,ntot),dtype='float32')
strt=0
for f in fwhmMapLst:
    fwhmMap[:,strt:strt+f.shape[0]] = f.T
    strt += f.shape[0]

wifisIO.writeFits(fwhmMap, 'fwhm_map.fits')

plt.imshow(fwhmMap, aspect='auto')
plt.colorbar()
plt.savefig('fwhm_map.png', dpi=300)
plt.show()

