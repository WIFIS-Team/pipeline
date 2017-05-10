"""
"""

import wifisIO
import numpy as np

flatSliceFile = ''

#read in flat-field slices file
flatSlices = wifisIO.readImgsFromFile(flatSliceFile)[0][0]

#trace centre, assuming all slices are aligned
trace = spatialCor.traceCentreFlatAll(flatSlices,cutoff=0.5, limSmth=20,MP=True, plot=False)

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

