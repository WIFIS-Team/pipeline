"""

Calibrates arc lamp images

Requires:
- list of inpute files to 

Produces:
- per pixel wavelength solution

Suggestions:
- use OpenCL to carry out polynomial fitting for each row

"""

import numpy as np
import time
import matplotlib.pyplot as plt
import wifisGetSatInfo as satInfo
import wifisNLCor as NLCor
import wifisRefCor as refCor
import os
import wifisIO 
import wifisCombineData as combData
import wifisWaveSol as waveSol
import multiprocessing as mp

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1' # Used to show compile errors for debugging, can be removed
os.environ['PYOPENCL_CTX'] = ':1' # Used to specify which OpenCL device to target    
t0 = time.time()

#Read in data
t1 = time.time()
out = wifisIO.readFromFolder('test1')
print("time to read all files took", time.time()-t1, " seconds")

data = out[1]
inttime = np.array(out[0])
nFrames = inttime.shape[0]
nx = data.shape[1]
ny = data.shape[0]
#******************************************************************************

#Correct data for reference pixels
ta = time.time()
print("Subtracting reference pixel channel bias")
#refCor.channelCL(data, nFrames, 32)
print("Subtracting reference pixel row bias")
#refCor.row(data, nFrames, 4)
print("time to apply reference pixel corrections ", time.time()-ta, " seconds")
#******************************************************************************

#find if any pixels are saturated to avoid use in future calculations
satCounts = wifisIO.readFromFile('wifisSatCounts.fits')
satFrame = satInfo.getSatFrameCL(data, satCounts,1)
#******************************************************************************

#apply non-linearity correction
ta = time.time()
print("Correcting for non-linearity")
nlCoeff = wifisIO.readFromFile('wifisNLCoeff.fits')
dorg = np.array(data)
NLCor.applyNLCorCL(data, nlCoeff, 32)
print("time to apply non-linearity corrections ", time.time()-ta, " seconds")
#******************************************************************************

#Combine data into single image
fluxImg = combData.upTheRampCL(inttime, data, satFrame, 1)
#write image to a file
#wifisIO.writeFits(fluxImg, 'wifisUpTheRampImg.fits')

