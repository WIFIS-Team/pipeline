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
import wifisUncertainties
import wifisBadPixels as badPixels
import wifisSlices as slices

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0' # Used to show compile errors for debugging, can be removed
os.environ['PYOPENCL_CTX'] = '1' # Used to specify which OpenCL device to target

#*****************************************************************************
#************************** Required input ***********************************
foldername = '20170510234217'
templateFile = '/data/pipeline/external_data/templateSlices.fits'
prevResultsFile = '/data/pipeline/external_data/prevSol.pkl'
nlFile = '/data/WIFIS/H2RG-G17084-ASIC-08-319/UpTheRamp/20170504201819/processed/master_detLin_NLCoeff.fits'        
satFile = '/data/WIFIS/H2RG-G17084-ASIC-08-319/UpTheRamp/20170504201819/processed/master_detLin_satCounts.fits'
bpmFile = 'bpm_noref.fits'
limitsFile = 'processed/master_flat_limits.fits'
atlasname = '/data/pipeline/external_data/best_lines2.dat'

#*****************************************************************************
#*****************************************************************************

#first check if required input exists
if not (os.path.exists(nlFile) and os.path.exists(satFile) and os.path.exists(limitsFile) and os.path.exists(atlasname)):
    if not (os.path.exists(satFile)):
        print ('*** ERROR: Cannot continue, file ' + satFile + ' does not exist. Please process a detector linearity calibration sequence or provide the necessary file ***')
    if not (os.path.exists(nlFile)):
        print ('*** ERROR: Cannot continue, file ' + nlFile + ' does not exist. Please process a detector linearity calibration sequence or provide the necessary file ***')
    if not (os.path.exists(limitsFile)):
        print ('*** ERROR: Cannot continue, file ' + limitsFile + ' does not exist. Please process flat field calibration sequence or provide the necessary file ***')
    if not (os.path.exists(atlasname)):
        print ('*** ERROR: Cannot continue, file ' + atlasname + ' does not exist. Please provide the necessary Atlas file***')
    raise SystemExit('*** Missing required calibration files, exiting ***')

#create processed directory, in case it doesn't exist
wifisIO.createDir('processed')

savename = 'processed/'+foldername

t0 = time.time()

if(os.path.exists(savename+'_wave.fits') and (os.path.exists(savename+'_waveMap.fits')) and (os.path.exists(savename+'_waveFitResuls.pkl'))):
    cont = wifisIO.userInput('Processed wavelength calibration files already exists for ' +foldername+', do you want to continue processing (y/n)?')
    if (cont.lower() == 'y'):
        contProc = True
    else:
        contProc = False
else:
    contProc = True
    
if (contProc):

    if (os.path.exists(savename+'_wave.fits')):
        cont = wifisIO.userInput('Processed flat field file already exists for ' +foldername+', do you want to continue processing (y/n)?')
        if (cont.lower() == 'n'):
            print('Reading image'+savename+'_wave.fits instead')
            fluxImg, sigmaImg, satFrame= wifisIO.readImgsFromFile(savename+'_wave.fits')[0]
                
            contProc2 = False
        else:
            contProc2 = True
    else:
        contProc2 = True
        
    if (contProc2):
            
        #Read in data
        ta = time.time()
        data, inttime, hdr = wifisIO.readRampFromFolder(foldername)
        print("time to read all files took", time.time()-ta, " seconds")
    
        data = data.astype('float32')
        nFrames = data.shape[2]
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
        satFrame = satInfo.getSatFrameCL(data, satCounts,1)
        #******************************************************************************

        #apply non-linearity correction
        ta = time.time()
        print("Correcting for non-linearity")

        #find NL coefficient file
        nlCoeff = wifisIO.readImgsFromFile(nlFile)[0]
        NLCor.applyNLCorCL(data, nlCoeff, 1)
        print("time to apply non-linearity corrections ", time.time()-ta, " seconds")

        #******************************************************************************

        #Combine data into single image
        #fluxImg = combData.upTheRampCL(inttime, data, satFrame, 32)[0]
        #sigmaImg = wifisUncertainties.compUTR(inttime, fluxImg, satFrame)
        fluxImg = combData.fowlerSamplingCL(inttime, data, satFrame,1)
        sigmaImg = wifisUncertainties.compFowler(inttime, fluxImg, satFrame, nFrames)
        data = 0

        #******************************************************************************
        #Correct for dark current
        #Identify appropriate dark image for subtraction
        print('Correcting for dark current')
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
        #trim image to remove reference pixels
        fluxImg = fluxImg[4:2044, 4:2044]
        sigmaImg = sigmaImg[4:2044, 4:2044]
        satFrame = satFrame[4:2044,4:2044]
    
        #******************************************************************************

        # CORRECT BAD PIXELS
        print('Correcting bad pixels')
        #check for BPM and read, if exists
        if(os.path.exists(bpmFile)):
            BPM = wifisIO.readImgsFromFile(bpmFile)[0]
                
            waveCor = badPixels.corBadPixelsAll(fluxImg, BPM, dispAxis=0, mxRng=2, MP=True)
            sigmaCor = badPixels.corBadPixelsAll(sigmaImg, BPM, dispAxis=0, mxRng=2, MP=True, sigma=True)

            fluxImg = waveCor
            sigmaImg = sigmaCor
        else:
            print('*** WARNING: No bad pixel mask provided ***')

        #******************************************************************************
        #add header info here
    
        #write image to a file

        wifisIO.writeFits([fluxImg, sigmaImg, satFrame], savename+'_wave.fits', hdr = hdr)
        
        #******************************************************************************
    
    #Determine dispersion solution
    print('Determining dispersion solution')
    #read in template
    template = wifisIO.readImgsFromFile(templateFile)[0]

    #read in template results to extract lambda -> wavelength solution
    #prevResults = wifisIO.readPickle(prevResultsFile)
    #prevSol = prevResults[5]
    prevSol = wifisIO.readPickle(prevResultsFile)
    
    #check if solution already exists.
    if(os.path.exists(savename+'_waveFitResults.pkl') and (os.path.exists(savename+'_waveMap.fits'))):
        cont = wifisIO.userInput('Dispersion solution and wavemap already exists for ' +foldername+', do you want to continue and replace (y/n)?')
        if (cont.lower() == 'n'):
            exit()

    #first extract the slices
    limits = wifisIO.readImgsFromFile(limitsFile)[0]
    waveSlices = slices.extSlices(fluxImg, limits, dispAxis=0)
    
    result = waveSol.getWaveSol(waveSlices, template, atlasname, 3, prevSol, winRng=9, mxCcor=30, weights=False, buildSol=False, sigmaClip=2, allowLower=True, lngthConstraint=True)

    dispSolLst = result[0]
    #Save all results
    wifisIO.writePickle(savename+'_waveFitResults.pkl', results)

    print('Creating wavelength map')
    #Create wavemap
    waveMapLst = waveSol.buildWaveMap(dispSolLst, waveSlices[0].shape[1])

    #save wavemap solution
    wifisIO.writeFits(waveMapLst, savename+'_waveMap.fits')

print ("Total time to run entire script: ",time.time()-t0)

