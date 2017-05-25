"""

Calibrates Ronchi mask and zero-point file

Requires:
- 

Produces:
-


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
import wifisSpatialCor as spatialCor
import wifisSlices as slices

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0' # Used to show compile errors for debugging, can be removed
os.environ['PYOPENCL_CTX'] = '1' # Used to specify which OpenCL device to target

#*****************************************************************************
#************************** Required input ***********************************
ronchifoldername = 'test1'
zpntfoldername = 'test1'
nlFile = 'processed/master_detLin_NLCoeff.fits'        
satFile = 'processed/master_detLin_satCounts.fits'
bpmFile = 'processed/bad_pixel_mask.fits'
limitsFile = ''
#*****************************************************************************
#*****************************************************************************

savename = '/processed/'+filename

t0 = time.time()

if(os.path.exists(savename+'_distCal.fits') and (os.path.exists(savename+'_distMap.fits')) and (os.path.exists(savename+'_ronchiTraces.fits'))):
    cont = wifisIO.userInput('Processed ronchi calibration files already exist for ' +foldername+', do you want to continue processing (y/n)?')
    if (cont.lower() == 'y'):
        contProc = True
    else:
        contProc = False
else:
    contProc = True
    
if (contProc):

    #Read in data
    ta = time.time()
    data, inttime, hdr = wifisIO.readRampFromFolder(foldername)
    print("time to read all files took", time.time()-ta, " seconds")

    nFrames = inttime.shape[0]
    nx = data.shape[1]
    ny = data.shape[0]
    #******************************************************************************

    #Correct data for reference pixels
    ta = time.time()
    print("Subtracting reference pixel channel bias")
    refCor.channelCL(data, 32)
    print("Subtracting reference pixel row bias")
    refCor.rowCL(data, 4,5)
    print("time to apply reference pixel corrections ", time.time()-ta, " seconds")
    #******************************************************************************

    #find if any pixels are saturated to avoid use in future calculations
    satCounts = wifisIO.readImgsFromFile(satFile)
    satFrame = satInfo.getSatFrameCL(data, satCounts,32)
    #******************************************************************************

    #apply non-linearity correction
    ta = time.time()
    print("Correcting for non-linearity")

    #find NL coefficient file
    nlCoeff = wifisIO.readImgsFromFile(nlFile)
    NLCor.applyNLCorCL(data, nlCoeff, 32)
    print("time to apply non-linearity corrections ", time.time()-ta, " seconds")

    #******************************************************************************

    #Combine data into single image
    fluxImg = combData.upTheRampCL(inttime, data, satFrame, 32)[0]
    #sigmaImg = wifisUncertainties.getUTR(inttime, fluxImg, satFrame)
    data = 0

    #******************************************************************************
    #Correct for dark current
    #Identify appropriate dark image for subtraction
    iTime = inttime[-1]-inttime[0]
    darkName = 'processed/master_dark_I'+str(iTime)+'.fits'
    if (os.path.exists(darkName)):
        darkImg,darkSig = wifisIO.readImgsFromFile(darkName)[0][0,1] #get the first two extensions
        fluxImg -= darkImg
        #sigmaImg = np.sqrt(sigmaImg**2 + darkSig**2)
    else:
        cont = wifisIO.userInput('No corresponding master dark image could be found, do you want to proceed without dark subtraction (y/n)?')
        if (cont.lower() == 'n'):
            exit()

    #******************************************************************************    
    #write image to a file

    wifisIO.writeFits([fluxImg, satFrame], savename+'_spatCal.fits')
        
    #******************************************************************************

    #read in limits file
    limits = wifisIO.readImgsFromFile(limitsFile)[0]
    
    #extract ronchi slices
    ronchiSlices = slices.extSlices(fluxImg, limits, dispAxis=0)
    
    #Now trace Ronchi masks
    #traces, widths = spatialCor.traceRonchiAll(ronchiSlices, nbin=2, winRng=5, mxWidth=2,smth=10, bright=False)

    #build fake ronchi mask for now
    
    
    #check if solution already exists.
    if(os.path.exists(savename+'_waveFitResults.pkl') and (os.path.exists(savename+'_waveMap.fits'))):
        cont = wifisIO.userInput('Dispersion solution and wavemap already exists for ' +foldername+', do you want to continue and replace (y/n)?')
        if (cont.lower() == 'n'):
            exit()

    #first extract the slices
    limits = wifisIO.readImgsFromFile(limitsFile)
    waveSlices = slices.extSlices(fluxImg, limits, dispAxis=0)
    
    result = waveSol.getWaveSol(waveSlices, template, atlasname, 3, prevSol, winRng=9, mxCcor=30, weights=False, buildSol=False, sigmaClip=2, allowLower=True, lngthConstraint=True)

    dispSolLst = result[0]
    #Save all results
    wifisIO.writePickle(savename+'_waveFitResults.pkl', results)

    #Create wavemap
    waveMapLst = waveSol.buildWaveMap(dispSolLst, waveSlices[0].shape[1])

    #save wavemap solution
    wifisIO.writeFits(waveMapLst, savename+'_waveMap.fits')

print ("Total time to run entire script: ",time.time()-t0)

