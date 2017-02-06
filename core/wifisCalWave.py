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

#Read in data
t1 = time.time()
out = wifisIO.readFromFolder(foldername)
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
    darkImg = wifisIO.readImgFromFile(darkName)
    flxImg = flxImg - darkImg
else:
    cont = wifisIO.userInput('No corresponding master dark image could be found, do you want to proceed without dark subtraction (y/n)?')
    if (cont.lower() == 'n'):
        exit()

#******************************************************************************    
#write image to a file
# *** STILL TO DO - SAVE SATURATION INFO AS ANOTHER HDU OR DIFFERENT FILE ***

if(os.path.exists(savename+'_waveCal.fits')):
    cont = wifisIO.userInput('Processed waveCal file already exists for ' +foldername+', do you want to replace (y/n)?')
    if (cont.lower() == 'y'):
        wifisIO.writeFits(fluxImg, savename+'_waveCal.fits', hdr=['INTTIME',inttime[-1],'Total integration time'])
else:
    wifisIO.writeFits(fluxImg, savename+'_waveCal.fits', hdr=['INTTIME',inttime[-1],'Total integration time'])

#******************************************************************************
#Determine dispersion solution

#read in template
template = wifisIO.readImgFromFile(templatename+'_waveCal.fits')

#provide line atlas file
atlasname = 'external_data/best_lines2.dat'

#read in dispersion solution corresponding to template
prevSol = wifisIO.readTable(templatename+'_waveSol.dat')

#data image, template, atlas file, max fitting order, list of prev. solutions, dispersion axis direction, window range fro line fitting, maximum allowable cross-correlation pixel offset
result = waveSol.getWaveSol(fluxImg, template, atlasname, 1, prevSol, dispAxis=0, winRng=7, mxCcor=30)

#*** NEED TO SAVE RESULTS OF FITTING ***
