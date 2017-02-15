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

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1' # Used to show compile errors for debugging, can be removed
os.environ['PYOPENCL_CTX'] = ':1' # Used to specify which OpenCL device to target

#*****************************************************************************
#************************** Required input ***********************************
foldername = 'test1'
templatename = 'template'
#*****************************************************************************
#*****************************************************************************

savename = '/processed/'+filename

t0 = time.time()

if(os.path.exists(savename+'_waveCal.fits')):
    cont = wifisIO.userInput('Processed waveCal file already exists for ' +foldername+', do you want to continue processing (y/n)?')
    if (cont.lower() == 'y'):
        contProc = True
    else:
        contProc = False
else:
    contProc = True
    
if (contProc):
    
    #Read in data
    t1 = time.time()
    data, inttime, hdr = wifisIO.readFromFolder(foldername)
    print("time to read all files took", time.time()-t1, " seconds")

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

    #Combine data into single image
    fluxImg = combData.upTheRampCL(inttime, data, satFrame, 32)

    data = 0

    #******************************************************************************
    #Correct for dark current
    #Identify appropriate dark image for subtraction
    darkName = 'processed/master_dark_I'+str(inttime[-1])+'.fits'
    if (os.path.exists(darkName)):
        darkImg = wifisIO.readImgFromFile(darkName)[:,:,0] #only use the first HDU, the second contains saturation information
        flxImg = flxImg - darkImg
    else:
        cont = wifisIO.userInput('No corresponding master dark image could be found, do you want to proceed without dark subtraction (y/n)?')
        if (cont.lower() == 'n'):
            exit()

    #******************************************************************************    
    #write image to a file

    if(os.path.exists(savename+'_waveCal.fits')):
        cont = wifisIO.userInput('Processed waveCal file already exists for ' +foldername+', do you want to replace (y/n)?')
        if (cont.lower() == 'y'):
            contWrite = True
        else:
            contWrite = False
    else:
        contWrite = True
        

    if (contWrite):
        #add or modify fits header here

        out = np.zeros((ny,nx,2))
        out[:,:,0] = fluxImg
        out[:,:,1] = satFrame
        wifisIO.writeFits(out, savename+'_waveCal.fits', hdr=hdr,'Total integration time'])
        out = 0
        
    #******************************************************************************
    #Determine dispersion solution

    #read in template
    template = wifisIO.readImgFromFile(templatename+'_waveCal.fits')

    #provide line atlas file
    atlasname = 'external_data/best_lines2.dat'

    #read in dispersion solution corresponding to template
    prevSol = wifisIO.readTable(templatename+'_waveSol.dat')

    #check if solution already exists.
    if(os.path.exists(savename+'_waveSol.dat')):
        cont = wifisIO.userInput('Dispersion solution already exists for ' +foldername+', do you want to continue and replace (y/n)?')
        if (cont.lower() == 'n'):
            exit()

    #data image, template, atlas file, max fitting order, list of prev. solutions, dispersion axis direction, window range fro line fitting, maximum allowable cross-correlation pixel offset
    result = waveSol.getWaveSol(fluxImg, template, atlasname, 1, prevSol, dispAxis=1, winRng=7, mxCcor=30)

    #Save dispersion solution
    dispSol = []
    for sol in result:
        dispSol.append(sol[0])

    wifisIO.writeTable(savename+'_waveSol.dat', dispSol)

t1 = time.time()
print ("Total time to run entire script: ",t1-t0)

