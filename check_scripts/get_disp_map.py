import wifisWaveSol as waveSol
import wifisIO
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import interp1d


print('Reading input data')
limits = wifisIO.readImgsFromFile('limits.fits')[0] #must point to location of limits
wave = wifisIO.readImgsFromFile('CDSResult.fits')[0][4:2044,:] (#trim, if limits file doesn't include reference pixels, which is likely the case
    
print('extracting wave slices')
waveSlices = slices.extSlices(wave, limits, dispAxis=0)

tmpLst = []
dispSol = []

print('getting mean templates')
for w in waveSlices:
    mid = int(w.shape[0]/2)
    y = np.nanmean(w[mid-4:mid+5,:],axis=0)
    tmpLst.append(y)

atlas = 'external_data/best_lines2.dat'

#dispersion solutions for template slices
dispTemp = [[6.94432076e+03,  -6.64823436e+00,   1.76614405e-03,-5.34567472e-07],
            [6.92071493e+03,  -6.58643402e+00,   1.71915780e-03,-5.21821679e-07],
            [ 6.84753705e+03,  -6.38834210e+00,   1.53920478e-03,-4.66558945e-07],
            [6.81231873e+03,  -6.29558259e+00,   1.46437240e-03,-4.45861833e-07],
            [6.68429143e+03,  -5.95333715e+00,   1.15927424e-03,-3.55071242e-07],
            [6.67497223e+03,  -5.91997038e+00,   1.12949634e-03,-3.45789655e-07],
            [6.60583640e+03,  -5.73866976e+00,   9.65454922e-04,-2.96461930e-07],
            [6.60191806e+03,  -5.72682246e+00,   9.57496137e-04,-2.94615309e-07],
            [6.58640181e+03,  -5.68493758e+00,   9.15028716e-04,-2.80684206e-07],
            [6.58022938e+03,  -5.67318200e+00,   9.08690459e-04,-2.79809200e-07],
            [  6.58344796e+03,  -5.68753082e+00,   9.19169946e-04,-2.82559581e-07],
            [  6.61836504e+03,  -5.79039083e+00,   1.01497156e-03,-3.11983041e-07],
            [  6.91494545e+03,  -6.60140133e+00,   1.74015653e-03,-5.26725524e-07],
            [  6.68846942e+03,  -5.99041161e+00,   1.19365263e-03,-3.65636458e-07],
            [6.72641404e+03,  -6.10327307e+00,   1.29371104e-03,-3.95806105e-07],
            [  6.75219240e+03,  -6.18322495e+00,   1.36783471e-03,-4.19092686e-07],
            [6.81670485e+03,  -6.37178475e+00,   1.53544599e-03,-4.69603182e-07],
            [  6.79383500e+03,  -6.31775216e+00,   1.49144616e-03,-4.58893942e-07]]

print('Getting dispersion solution')
result = waveSol.getWaveSol(waveSlices, tmpLst, atlas,3, dispTemp, winRng=9, mxCcor=30, weights=False, buildSol=False, allowLower=True, sigmaClip=2., lngthConstraint = True)

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

plt.imshow(fwhmMap_interp, aspect='auto')
plt.colorbar()
plt.savefig('fwhm_map_interp.png', dpi=300)
plt.show()

