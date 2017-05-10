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
import astropy.io.fits as fits

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1' # Used to show compile errors for debugging, can be removed
os.environ['PYOPENCL_CTX'] = '1' # Used to specify which OpenCL device to target. Should be uncommented and pointed to correct device to avoid future interactive requests

t0 = time.time()

#*****************************************************************************
#************************** Required user input ******************************
fileList = 'list'
nlFile = 'processed/master_detLin_NLCoeff.fits'        
satFile = 'processed/master_detLin_satCounts.fits'
bpmFile = 'processed/bad_pixel_mask.fits'
#*****************************************************************************

#first check if required input exists
if not (os.path.exists(nlFile) and os.path.exists(satFile)):
    if not (os.path.exists(satFile)):
        print ('*** ERROR: Cannot continue, file ' + satFile + ' does not exist. Please process the a detector linearity calibration sequence or provide the necessary file ***')
    if not (os.path.exists(nlFile)):
        print ('*** ERROR: Cannot continue, file ' + nlFile + ' does not exist. Please process the a detector linearity calibration sequence or provide the necessary file ***')
    raise SystemExit('*** Missing required calibration files, exiting ***')

#create processed directory, in case it doesn't exist
wifisIO.createDir('processed')

#read file list
lst= wifisIO.readAsciiList(fileList)

if lst.ndim == 0:
    lst = [lst]

#*******************************************************************************************
#read last file to get total integration time (taken as time difference between end of last frame - end of first frame)
folder = lst[0]
fleLst = glob.glob(folder+'/H2*fits')
srtLst = wifisIO.sorted_nicely(fleLst)
hdr = fits.getheader(list[0])
iTime0 = hdr['INTTIME']
hdr = fits.getheader(list[-1])
iTime1 = hdr['INTTIME']
iTime = iTime1 - iTime0
#*******************************************************************************************

masterSave = 'processed/master_dark_I'+str(iTime)+'.fits'

#first check if master dark exists
if(os.path.exists(masterSave)):
    cont = wifisIO.userInput('Master dark file already exists for integration time (s)' + str(iTime)+', do you want to replace (y/n)?')
    if (cont.lower() == 'y'):
        contProc = True
    else:
        contProc = False
else:
    contProc = True
    
if (contProc):
    
    procDark = []
    procSigma = []
    procSatFrame = []

    #go through list and process each file individually

    for folder in lst:

        folder = folder.tostring()
        savename = 'processed/'+folder

        if(os.path.exists(savename+'_dark.fits')):
            cont = wifisIO.userInput('Processed dark file already exists for ' +folder+', do you want to continue processing (y/n)?')
            if (cont.lower() == 'n'):
                print('Reading image'+savename+'_dark.fits instead')
                fluxImg = wifisIO.readImgsFromFile(savename+'_dark.fits')
                contProc2 = False
            else:
                contProc2 = True
        else:
            contProc2 = True
        
        if (contProc2):
            #Read in data
            ta = time.time()
            data, inttime, hdr = wifisIO.readRampFromFolder(folder)
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
            refCor.rowCL(data, nFrames, 4,5) 
            print("time to apply reference pixel corrections ", time.time()-ta, " seconds")
        
            #******************************************************************************
            #find if any pixels are saturated to avoid use in future calculations
        
            satCounts = wifisIO.readImgsFromFile(satFile)[0]
            satFrame = satInfo.getSatFrameCL(data, satCounts,32)
        
            #******************************************************************************
            #apply non-linearity correction
            ta = time.time()
            print("Correcting for non-linearity")
        
            nlCoeff = wifisIO.readImgsFromFile(nlFile)[0]
            NLCor.applyNLCorCL(data, nlCoeff, 32)
            print("time to apply non-linearity corrections ", time.time()-ta, " seconds")
        
            #******************************************************************************
            #Combine data cube into single image
            fluxImg, zptnImg, varImg = combData.upTheRampCL(inttime, data, satFrame, 32)
            #fluxImg = combData.upTheRampCRRejectCL(inttime, data, satFrame, 32)

            data = 0

            #get uncertainties for each pixel
            sigma = wifisUncertainties.getUTR(inttime, fluxImg, satFrame)

            #add additional header information here
        
            wifisIO.writeFits([fluxImg, sigma, satFrame], savename+'_dark.fits', hdr=hdr)

            out = 0
        else:
            #read in file instead
            fluxImg, sigma, satFrame = wifisIO.readImgsFromFile(savename+'_dark.fits')[0]
            
        procDark.append(fluxImg)
        procSigma.append(sigma)
        procSatFrame.append(satFrame)

    #now combine all dark images into master dark, propagating uncertainties as needed
    masterDark, masterSigma  = wifisUncertainties.compMedian(np.array(procDark),np.array(procSigma), axis=0)

    #combine satFrame
    masterSatFrame = np.median(np.array(procSatFrame), axis=0).astype('int')
    
    #******************************************************************************
    #******************************************************************************

    #add/modify header information here
    hdr['INTTIME']=iTime

    #save file
    wifisIO.writeFits([masterDark, masterSigma, masterSatFrame], masterSave)

    #extract any information of interest
    #RON, etc.
    
else:
     print('No processing necessary')   

print ("Total time to run entire script: ",time.time()-t0)
