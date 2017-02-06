"""

Fully calibrates set of images used to measure the non-linearity behaviour of the detector

Requires:
- specifying the folder where these files can be found

Produces:
- map of saturation level
- map of per-pixel non-linearity corrections
- bad pixel mask for pixels with very bad non-linearity (*** STILL TO DO ***)

"""

import numpy as np
import astropy.io.fits as fits
import astropy.io.ascii as ascii
import time
import matplotlib.pyplot as plt
import wifisGetSatInfo as satInfo
import wifisNLCor as NLCor
import wifisRefCor as refCor
import os
import wifisIO 

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1' # Used to show compile errors for debugging, can be removed
os.environ['PYOPENCL_CTX'] = ':1' # Used to specify which OpenCL device to target, should uncomment and set to preferred device to avoid interactively selecting each time

t0 = time.time()

#*****************************************************************************
#*************************** Required input **********************************

#set folder name
foldername = 'test1'

#*****************************************************************************

savename = 'processed/'+foldername

#create processed directory
wifisIO.createDir('processed')

#check if processing needs to be done
if(os.path.exists(savename+'_NLCoeff.fits')):
    cont = wifisIO.userInput('Processed file already exists for ' +foldername+', do you want to continue processing (y/n)?')
    if (cont.lower() == 'n'):
        print('Exiting')
        exit()

#******************************************************************************
#Read in data
t1 = time.time()
data = 0 #initilize to avoid spike in memory usage
data, inttime = wifisIO.readImgsFromFolder(foldername)
print("time to read all files took", time.time()-t1, " seconds")

nFrames = inttime.shape[0]
nx = data.shape[1]
ny = data.shape[0]

#******************************************************************************

#Correct data for reference pixels
ta = time.time()
print("Subtracting reference pixel channel bias")
refCor.channelCL(data, nFrames, 32)
print("Subtracting reference pixel row bias")
#refCor.row(data, nFrames, 4) #*** NOT CURRENTLY CARRIED OUT ***
print("time to apply reference pixel corrections ", time.time()-ta, " seconds")
#******************************************************************************

#******************************************************************************
#get saturation information for each pixel
ta = time.time()
satCounts = satInfo.getSatCountsCL(data,0.95, 32)
satFrame = satInfo.getSatFrameCL(data,satCounts,32)
print ("saturation code took ", time.time()-ta, " seconds")

#write saturation info to array
#check if file should be saved

if(os.path.exists(savename+'_satCounts.fits')):
    cont = wifisIO.userInput('satCounts file already exists for ' +foldername+', do you want to replace (y/n)?')
    if (cont.lower() == 'y'):
        wifisIO.writeFits(satCounts, savename+'_satCounts.fits') #ADD HEADER INFO?
else:
    wifisIO.writeFits(satCounts, savename+'_satCounts.fits') #ADD HEADER INFO?
     
#******************************************************************************

#******************************************************************************
# Get the non-linearity correction coefficients
ta = time.time()
nlCoeff = NLCor.getNLCorCL(data,satFrame,32)
print ("non-linearity code took", time.time()-ta, " seconds")

#write NL Coefficients to a FITS image
if(os.path.exists(savename+'_NLCoeff.fits')):
    cont = wifisIO.userInput('NLCoeff file already exists for ' +foldername+', do you want to replace (y/n)?')
    if (cont.lower() == 'y'):
        wifisIO.writeFits(nlCoeff, savename+'_NLCoeff.fits')
else:
    wifisIO.writeFits(nlCoeff, savename+'_NLCoeff.fits')

#******************************************************************************

t1 = time.time()
print ("Total time to run entire script: ",t1-t0)
