"""

Calibrate flat field images

Produces:
- master flat image
- slitlet traces

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
import glob
import wifisSlices as slices

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1' # Used to show compile errors for debugging, can be removed
os.environ['PYOPENCL_CTX'] = ':1' # Used to specify which OpenCL device to target. Should be uncommented and pointed to correct device to avoid future interactive requests

t0 = time.time()

#*****************************************************************************
#************************** Required input ***********************************
fileList = 'list' 
#*****************************************************************************

#read file list
lst= wifisIO.readAsciiList(fileList)

if lst.ndim == 0:
    lst = [lst]

procFiles = []
#go through list and process each file individually

for folder in lst:

    folder = folder.tostring()
    savename = 'processed/'+folder

    if(os.path.exists(savename+'_flat.fits')):
        cont = wifisIO.userInput('Processed flatfield file already exists for ' +folder+', do you want to continue processing (y/n)?')
        if (cont.lower() == 'n'):
            print('Reading image'+savename+'_flat.fits instead')
            fluxImg = wifisIO.readImgFromFile(savename+'_flat.fits')
            contProc = False
        else:
            contProc = True
    else:
        contProc = True
        
    if (contProc):
        #Read in data
        ta = time.time()
        data, inttime, hdr = wifisIO.readImgsFromFolder(folder)
        print("time to read all files took", time.time()-ta, " seconds")
        
        nFrames = inttime.shape[0]
        nx = data.shape[1]
        ny = data.shape[0]

        #******************************************************************************
        #Correct data for reference pixels
        ta = time.time()
        print("Subtracting reference pixel channel bias")
        refCor.channelCL(data, nFrames, 32)
        print("Subtracting reference pixel row bias")
        #refCor.row(data, nFrames, 4) # *** NOT CURRENTLY BEING USED ***
        print("time to apply reference pixel corrections ", time.time()-ta, " seconds")
        
        #******************************************************************************
        #find if any pixels are saturated to avoid use in future calculations
        #*** ADD CODE TO DEAL WITH POTENTIAL OF MULTIPLE NL FILES, BUT FOR NOW JUST TAKE THE FIRST
        satFile = glob.glob('processed/*satCounts.fits')[0]
        
        satCounts = wifisIO.readImgFromFile(satFile)[0]
        satFrame = satInfo.getSatFrameCL(data, satCounts,32)
        
        #******************************************************************************
        #apply non-linearity correction
        ta = time.time()
        print("Correcting for non-linearity")
        
        #find NL coefficient file
        #*** ADD CODE TO DEAL WITH POTENTIAL OF MULTIPLE NL FILES, BUT FOR NOW JUST TAKE THE FIRST
        nlFile = glob.glob('processed/*NLCoeff.fits')[0]
        
        nlCoeff = wifisIO.readImgFromFile(nlFile)[0]
        NLCor.applyNLCorCL(data, nlCoeff, 32)
        print("time to apply non-linearity corrections ", time.time()-ta, " seconds")
        
        #******************************************************************************
        #Combine data cube into single image
        fluxImg = combData.upTheRampCL(inttime, data, satFrame, 32)
        data = 0
        
        #write image to a file
        # Save saturation info as additional extension
        out = np.zeros((ny,nx,2))
        out[:,:,0] = fluxImg
        out[:,:,1] = satFrame        
        
        #add additional header information here
        
        wifisIO.writeFits(out, savename+'_flat.fits', hdr=hdr)

        out = 0
        
    procFiles.append(fluxImg)

#now combine all flatfield images into master flat

procFiles = np.array(procFiles)
master = np.median(procFiles, axis=0)

#******************************************************************************
#STILL TO DO CORRECT BAD PIXELS
# CODE GOES HERE
#******************************************************************************

#find limits of each slice
limits = slices.findLimits(master, dispAxis=1, winRng=51, imgSmth=5, limSmth=10)

#write limits to file
if(os.path.exists('processed/'+'master_flat_limits.fits')):
    cont = wifisIO.userInput('Master flat limits file already exists, do you want to replace (y/n)?')
    if (cont.lower() == 'y'):
        contWrite = True
    else:
        contWrite = False
else:
    contWrite = True
if (contWrite):
    #add/modify header information here
    wifisIO.writeImgSlices(fluxImg, response,'processed/'+'master_flat.fits')

#now extract the individual slices
extSlices = slices.extSlices(data, limits, dispAxis=0)

#now get smoothed and normalized response function
response = slices.getResponseAll(extSlices, 5, 0.01)

#write master image to file
if(os.path.exists('processed/'+'master_flat.fits')):
    cont = wifisIO.userInput('Master flat file already exists, do you want to replace (y/n)?')
    if (cont.lower() == 'y'):
        contWrite = True
    else:
        contWrite = False
else:
    contWrite = True
if (contWrite):
    #add/modify header information here
    wifisIO.writeImgSlices(fluxImg, response,'processed/'+'master_flat.fits')

print ("Total time to run entire script: ",time.time()-t0)
