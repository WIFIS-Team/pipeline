"""
"""

import wifisIO
import numpy as np
import wifisSpatialCor as spatialCor

flatSliceFile = 'processed/master_flat_slices.fits'

print('reading flat')
#read in flat-field slices file
flatSlices = wifisIO.readImgsFromFile(flatSliceFile)[0]

print('tracing centre of flat')
#trace centre, assuming all slices are aligned
trace = spatialCor.traceCentreFlatAll(flatSlices,cutoff=0.8, limSmth=20,MP=False, plot=False)

print('Creating artificial distortion map')
#create fake ronchi maps
ronchiMaps = []
#create ronchi maps
for i in range(len(flatSlices)):
    ronchiMap = np.empty(flatSlices[i].shape)
    
    for j in range(ronchiMap.shape[0]):
        ronchiMap[j,:] = j - trace[i]

    ronchiMaps.append(ronchiMap)

#save to file

wifisIO.writeFits(ronchiMaps, 'processed/ronchi_map.fits')

