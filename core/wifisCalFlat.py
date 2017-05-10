"""

Calibrate flat field images

Produces:
- master flat image
- slitlet traces
- ??

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
import wifisSlices as slices
import wifisUncertainties
import wifisBadPixels as badPixels
import astropy.io.fits as fits

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0' # Used to show compile errors for debugging, can be removed
os.environ['PYOPENCL_CTX'] = '1' # Used to specify which OpenCL device to target. Should be uncommented and pointed to correct device to avoid future interactive requests

t0 = time.time()

#*****************************************************************************
#******************************* Required input ******************************
fileList = 'flat.lst' 
nlFile = '/data/WIFIS/H2RG-G17084-ASIC-08-319/UpTheRamp/20170504201819/processed/master_detLin_NLCoeff.fits'        
satFile = '/data/WIFIS/H2RG-G17084-ASIC-08-319/UpTheRamp/20170504201819/processed/master_detLin_satCounts.fits'
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

procFlux = []
procSigma = []
procSatFrame = []

#go through list and process each file individually

#first check master flat and limits exists
if(os.path.exists('processed/master_flat.fits') and os.path.exists('processed/master_flat_limits.fits') and os.path.exists('processed/master_flat_slices.fits')):

    cont = wifisIO.userInput('Master flat, slices and limits files already exists, do you want to continue processing (y/n)?')

    if (cont.lower() == 'y'):
        contProc = True
    else:
        contProc = False
else:
    contProc = True
    
if (contProc):
    
    for folder in lst:
        
        folder = folder.tostring()
        savename = 'processed/'+folder

        if(os.path.exists(savename+'_flat.fits')):
            cont = wifisIO.userInput('Processed flat field file already exists for ' +folder+', do you want to continue processing (y/n)?')
            if (cont.lower() == 'n'):
                print('Reading image'+savename+'_flat.fits instead')
                fluxImg, sigmaImg, satFrame= wifisIO.readImgsFromFile(savename+'_flat.fits')[0]
                
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

            #convert data to float32 for future processing
            data = data.astype('float32')
            
            #******************************************************************************
            #Correct data for reference pixels
            ta = time.time()
            print("Subtracting reference pixel channel bias")
            refCor.channelCL(data, 32)
            print("Subtracting reference pixel row bias")
            refCor.rowCL(data, 4,1)
            print("time to apply reference pixel corrections ", time.time()-ta, " seconds")       
            #******************************************************************************
            #find if any pixels are saturated to avoid use in future calculations

            satCounts = wifisIO.readImgsFromFile(satFile)[0]
            satFrame = satInfo.getSatFrameCL(data, satCounts,32)
                
            #******************************************************************************
            #apply non-linearity correction
            ta = time.time()
            print("Correcting for non-linearity")
            
            #find NL coefficient file
            nlCoeff = wifisIO.readImgsFromFile(nlFile)[0]
            NLCor.applyNLCorCL(data, nlCoeff, 32)
            print("time to apply non-linearity corrections ", time.time()-ta, " seconds")
        
            #******************************************************************************
            #Combine data cube into single image
            fluxImg = combData.upTheRampCL(inttime, data, satFrame, 32)[0]
            #fluxImg = combData.upTheRampCRRejectCL(inttime, data, satFrame, 32)

            data = 0
            
            #get uncertainties for each pixel
            sigmaImg = wifisUncertainties.compUTR(inttime, fluxImg, satFrame)
            
            #write image to a file, saving saturation info as additional extension

            #****************************************************************************
            #add additional header information here

            #****************************************************************************
            
            wifisIO.writeFits([fluxImg, sigmaImg, satFrame], savename+'_flat.fits', hdr=hdr)
            
            out = 0
        else:
            #read in file instead
            fluxImg,sigmaImg, satFrame = wifisIO.readImgsFromFile(savename+'_flat.fits')[0]

        procFlux.append(fluxImg)
        procSigma.append(sigmaImg)
        procSatFrame.append(satFrame)


    #*************************************************************************************
    #*************************************************************************************
    #*************************************************************************************
    #DONE PROCESSING INDIVIDUAL FLAT FIELD RAMPS, NOW CREATE MASTER IMAGES
    
    #now combine all flatfield images into master flat, propagating uncertainties as needed
    masterFlat, masterSigma = wifisUncertainties.compMedian(np.array(procFlux), np.array(procSigma),axis=0)

    #combine satFrame
    masterSatFrame = np.median(np.array(procSatFrame), axis=0).astype('int')
    
    #******************************************************************************
    # CORRECT BAD PIXELS

    #check for BPM and read, if exists
    if(os.path.exists(bpmFile)):
        masterFlatCor = badPixels.corBadPixelsAll(masterFlat, BPM, dispAxis=0, mxRng=2, MP=True)
        masterSigmaCor = badPixels.corBadPixelsAll(mastSigma, BPM, dispAxis=0, mxRng=2, MP=True, sigma=True)
    else:
        masterFlatCor = masterFlat
        masterSigmaCor = masterSigma
        
        print('*** WARNING: No bad pixel mask provided ***')
    #******************************************************************************
    
    #find limits of each slice
    limits = slices.findLimits(masterFlatCor, dispAxis=0, winRng=51, imgSmth=5, limSmth=10)

    #write limits to file
    wifisIO.writeFits(limits,'processed/master_flat_limits.fits')

    #now extract the individual slices
    fluxSlices = slices.extSlices(masterFlatCor, limits, dispAxis=0)

    #extract uncertainty slices
    sigmaSlices = slices.extSlices(masterSigmaCor, limits, dispAxis=0)

    #extract saturation slices
    satSlices = slices.extSlices(masterSatFrame, limits, dispAxis=0)

    #now get smoothed and normalized response function
    masterRes = slices.getResponseAll(fluxSlices, 0, 0.1)
    masterSig = slices.ffCorrectAll(sigmaSlices, masterRes)
        
    #write master image to file
    wifisIO.writeFits([masterFlatCor,masterSigmaCor, masterSatFrame],'processed/master_flat.fits')

    #write master image slices to file
    wifisIO.writeFits(masterRes + masterSig + satSlices,'processed/master_flat_slices.fits')
else:
    print('No processing necessary')
    
print ("Total time to run entire script: ",time.time()-t0)
