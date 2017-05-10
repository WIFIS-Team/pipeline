"""

Calibrates arc lamp images

Requires:
- 

Produces:
- per pixel wavelength solution


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

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0' # Used to show compile errors for debugging, can be removed
os.environ['PYOPENCL_CTX'] = '1' # Used to specify which OpenCL device to target

#*****************************************************************************
#************************** Required input ***********************************
foldername = 'test1'
templateFile = 'template'
prevResultsFile = 'prevSolName'
nlFile = 'processed/master_detLin_NLCoeff.fits'        
satFile = 'processed/master_detLin_satCounts.fits'
bpmFile = 'processed/bad_pixel_mask.fits'
limitsFile = ''
#*****************************************************************************
#*****************************************************************************

#first check if required input exists
if not (os.path.exists(nlFile) and os.path.exists(satFile) and os.path.exists(limitsFile)):
    if not (os.path.exists(satFile)):
        print ('*** ERROR: Cannot continue, file ' + satFile + ' does not exist. Please process a detector linearity calibration sequence or provide the necessary file ***')
    if not (os.path.exists(nlFile)):
        print ('*** ERROR: Cannot continue, file ' + nlFile + ' does not exist. Please process a detector linearity calibration sequence or provide the necessary file ***')
    if not (os.path.exists(limitsFile)):
        print ('*** ERROR: Cannot continue, file ' + limitsFile + ' does not exist. Please process flat field calibration sequence or provide the necessary file ***')
    raise SystemExit('*** Missing required calibration files, exiting ***')

#create processed directory, in case it doesn't exist
wifisIO.createDir('processed')

savename = '/processed/'+filename

t0 = time.time()

if(os.path.exists(savename+'_waveCal.fits') and (os.path.exists(savename+'_waveMap.fits')) and (os.path.exists(savename+'_waveFitResuls.pkl'))):
    cont = wifisIO.userInput('Processed wavelength calibration files already exists for ' +foldername+', do you want to continue processing (y/n)?')
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
    sigmaImg = wifisUncertainties.getUTR(inttime, fluxImg, satFrame)
    data = 0

    #******************************************************************************
    #Correct for dark current
    #Identify appropriate dark image for subtraction
    iTime = inttime[-1]-inttime[0]
    darkName = 'processed/master_dark_I'+str(iTime)+'.fits'
    if (os.path.exists(darkName)):
        darkImg,darkSig = wifisIO.readImgsFromFile(darkName)[0][0,1] #get the first two extensions
        fluxImg -= darkImg
        sigmaImg = np.sqrt(sigmaImg**2 + darkSig**2)
    else:
        cont = wifisIO.userInput('No corresponding master dark image could be found, do you want to proceed without dark subtraction (y/n)?')
        if (cont.lower() == 'n'):
            exit()

    #******************************************************************************    
    #write image to a file

    wifisIO.writeFits([fluxImg, sigmaImg, satFrame], savename+'_waveCal.fits')
        
    #******************************************************************************
    #Determine dispersion solution

    #read in template
    template = wifisIO.readImgsFromFile(templateFile)

    #read in template results to extract lambda -> wavelength solution
    prevResults = wifisIO.readPickle(prevResultsFile)
    prevSol = prevResults[5]
    
    #provide line atlas file
    atlasname = 'external_data/best_lines2.dat'

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

