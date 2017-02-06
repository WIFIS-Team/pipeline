"""

Calibrate dark images

Input: 
- BPM from linearity measurents (optional, will be merged) *** NOT IMPLEMENTED YET ***

Requires:
- specifying the ascii list containing folder names to process and from which to create a master dark frame

Produces: 
- master dark image
- bad pixel map (merged) *** STILL TO DO ***

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
    
    #Read in data
    t1 = time.time()
    data, inttime = wifisIO.readImgsFromFolder(folder)
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
    #refCor.row(data, nFrames, 4) # *** NOT CURRENTLY BEING USED ***
    print("time to apply reference pixel corrections ", time.time()-ta, " seconds")

    #******************************************************************************
    #find if any pixels are saturated to avoid use in future calculations
    #*** ADD CODE TO DEAL WITH POTENTIAL OF MULTIPLE NL FILES, BUT FOR NOW JUST TAKE THE FIRST
    satFile = glob.glob('processed/*satCounts.fits')[0]
    satCounts = wifisIO.readImgFromFile(satFile)
    satFrame = satInfo.getSatFrameCL(data, satCounts,32)

    #******************************************************************************
    #apply non-linearity correction
    ta = time.time()
    print("Correcting for non-linearity")

    #find NL coefficient file
    #*** ADD CODE TO DEAL WITH POTENTIAL OF MULTIPLE NL FILES, BUT FOR NOW JUST TAKE THE FIRST
    nlFile = glob.glob('processed/*NLCoeff.fits')[0]
    
    nlCoeff = wifisIO.readImgFromFile(nlFile)
    NLCor.applyNLCorCL(data, nlCoeff, 32)
    print("time to apply non-linearity corrections ", time.time()-ta, " seconds")

    #******************************************************************************
    #Combine data cube into single image
    fluxImg = combData.upTheRampCL(inttime, data, satFrame, 32)
    data = 0
    
    #write image to a file
    if(os.path.exists(savename+'_dark.fits')):
        cont = wifisIO.userInput('Processed dark file already exists for ' +folder+', do you want to replace (y/n)?')
        if (cont.lower() == 'y'):
            wifisIO.writeFits(fluxImg, savename+'_dark.fits', hdr=['INTTIME',inttime[-1],'Total integration time'])
    else:
        wifisIO.writeFits(fluxImg, savename+'_dark.fits', hdr=['INTTIME',inttime[-1],'Total integration time'])


    procFiles.append(fluxImg)

#now combine all dark images into master dark

procFiles = np.array(procFiles)
master = np.median(procFiles, axis=0)


#******************************************************************************
#STILL TO DO CORRECT BAD PIXELS
# CODE GOES HERE
#******************************************************************************

#write master image to file
if(os.path.exists('processed/'+'master_dark_Itime'+str(inttime[-1])+'.fits')):
    cont = wifisIO.userInput('Master dark file already exists for ' + str(inttime[-1])+', do you want to replace (y/n)?')
    if (cont.lower() == 'y'):
        wifisIO.writeFits(fluxImg, 'processed/'+'master_dark_Itime'+str(inttime[-1])+'.fits', hdr=['INTTIME',inttime[-1],'Total integration time'])
else:
    wifisIO.writeFits(fluxImg, 'processed/'+'master_dark_Itime'+str(inttime[-1])+'.fits', hdr=['INTTIME',inttime[-1],'Total integration time'])
            
