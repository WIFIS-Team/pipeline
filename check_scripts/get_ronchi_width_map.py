import wifisIO
import wifisSpatialCor as spatialCor
import wifisBadPixels as badPixels
import wifisSlices as slices
import matplotlib.pyplot as plt
import numpy as np

print('Reading in data')

limits = wifisIO.readImgsFromFile('limits.fits')[0]
BPM = wifisIO.readImgsFromFile('bad_pixel_mask.fits')[0]
ronchi = wifisIO.readImgsFromFile('CDSResult.fits')[0]

#trim ronchi mask to match that of BPM
ronchi = ronchi[4:2044,4:2044]

#use bad pixel mask to correct for bad pixels
ronchiCor = badPixels.corBadPixelsAll(ronchi, BPM, mxRng=15,dispAxis=0)

#extract individual slices
ronchiSlices = slices.extSlices(ronchiCor, limits, dispAxis=0)

print('Getting ronchi trace')
ronchiTraces, ronchiWidths = spatialCor.traceRonchiAll(ronchiSlices, nbin=4, winRng=5, mxWidth=2,smth=10, bright=False)

print('creating width map')
widthMapLst = spatialCor.buildWidthMap(ronchiTraces, ronchiWidths, ronchiSlices)

ntot = 0
for r in ronchiSlices:
    ntot += r.shape[0]
    
fwhmMap = np.empty((r.shape[1],ntot),dtype='float32')

strt=0
for w in widthMapLst:
    fwhmMap[:,strt:strt+w.shape[0]] = 2.*np.sqrt(2*np.log(2.))*w.T
    strt += w.shape[0]

plt.imshow(fwhmMap)
plt.colorbar()
plt.savefig('ronchi_fwhm_map.png', dpi=300)
plt.show

wifisIO.writeFits(fwhmMap, 'ronchi_fwhm_map.fits')
