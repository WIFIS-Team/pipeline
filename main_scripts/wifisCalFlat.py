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
import wifisHeaders as headers
import wifisProcessRamp as processRamp
from matplotlib.backends.backend_pdf import PdfPages

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0' # Used to show compile errors for debugging, can be removed
os.environ['PYOPENCL_CTX'] = '1' # Used to specify which OpenCL device to target. Should be uncommented and pointed to correct device to avoid future interactive requests

t0 = time.time()

#*****************************************************************************
#******************************* Required input ******************************
fileList = 'flat.lst' # a simple ascii file containing a list of the folder names that contain the ramp data
nlFile = '/data/WIFIS/H2RG-G17084-ASIC-08-319/UpTheRamp/20170504201819/processed/master_detLin_NLCoeff.fits' # the non-linearity correction coefficients file        
satFile = '/data/WIFIS/H2RG-G17084-ASIC-08-319/UpTheRamp/20170504201819/processed/master_detLin_satCounts.fits' # the saturation limits file
bpmFile = 'bpm.fits' # the bad pixel mask
distMapLimitsFile = ''
plot = True #whether to plot the traces
#*****************************************************************************

#NOTES
#file list can be a 2D table: where all files to be coadded are on the same row and files to be processed separately are on different rows. *** NOT YET IMPLEMENTED

#first check if required input exists
if not (os.path.exists(nlFile) and os.path.exists(satFile)):
    if not (os.path.exists(satFile)):
        print ('*** ERROR: Cannot continue, file saturation file ' + satFile + ' does not exist. Please process the a detector linearity calibration sequence and provide the necessary file ***')
    if not (os.path.exists(nlFile)):
        print ('*** ERROR: Cannot continue, file NL coefficient file ' + nlFile + ' does not exist. Please process the a detector linearity calibration sequence and provide the necessary file ***')
    raise SystemExit('*** Missing calibration files required, exiting ***')

#create processed directory, in case it doesn't exist
wifisIO.createDir('processed')

if (plot):
    wifisIO.createDir('quality_control')

print('Reading in calibration files')
#open calibration files
nlCoeff = wifisIO.readImgsFromFile(nlFile)[0]
satCounts = wifisIO.readImgsFromFile(satFile)[0]

if (os.path.exists(bpmFile)):
    BPM = wifisIO.readImgsFromFile(bpmFile)[0]
else:
    BPM = None

#read file list
lst= wifisIO.readAsciiList(fileList)

if lst.ndim == 0:
    lst = np.asarray([lst])

procFlux = []
procSigma = []
procSatFrame = []

#go through list and process each file individually
#************
#eventually need to add capability to create master flat from groups
#************
for lstNum in range(len(lst)):

    if (lst.ndim>1):
        folder = lst[lstNum,0]
    else:
        folder = lst[lstNum].tostring()

    savename = 'processed/'+folder
    
    #first check master flat and limits exists
    
    if(os.path.exists(savename+'_flat.fits') and os.path.exists(savename+'_flat_limits.fits') and os.path.exists(savename+'_flat_slices.fits') and os.path.exists(savename+'_flat_slices_norm.fits')):
        cont = 'n'
        cont = wifisIO.userInput('All processed flat field files already exists for ' +folder+', do you want to continue processing (y/n)?')
    else:
        cont = 'y'
        
    if (cont.lower() == 'y'):
        print('*** Working on folder ' + folder + ' ***')

        if (os.path.exists(savename+'_flat.fits')):
            cont = wifisIO.userInput('Processed flat field file already exists for ' +folder+', do you want to continue processing (y/n)?')
            if (cont.lower() == 'n'):
                print('Reading image'+savename+'_flat.fits instead')
                flatCor, sigmaImg, satFrame= wifisIO.readImgsFromFile(savename+'_flat.fits')[0]
                contProc2 = False
            else:
                contProc2 = True
        else:
            contProc2 = True
        
        if (contProc2):
            flatCor, sigmaCor, satFrame = processRamp.fromUTR(folder, savename+'_flat.fits', satCounts, nlCoeff, BPM, nChannel=32, rowSplit=1, nlSplit=32, combSplit=32, crReject=False, bpmCorRng=20)

        print('Finding slice limits and extracting slices')
        #find limits of each slice with the reference pixels, but the returned limits exclude them
        limits = slices.findLimits(flatCor, dispAxis=0, winRng=51, imgSmth=5, limSmth=20, rmRef=True)

        #get smoother limits, if desired, using polynomial fitting
        polyLimits = slices.polyFitLimits(limits, degree=3)

        if os.path.exists(distMapLimitsFile):
            distMapLimits = wifisIO.readImgsFromFile(distMapLimitsFile)[0]
            shft = np.nanmedian(polyLimits[1:-1,:] - distMapLimits[1:-1,:])
            polyLimits = distMapLimits + shft
            
        #save figures of tracing results for quality control purposes
        if (plot):
            print('Plotting results')
            #with PdfPages('quality_control/'+folder+'_flat_slices_traces.pdf') as pdf:
            plt.ioff()
            fig = plt.figure()
            med1= np.nanmedian(flatCor)
                
            plt.imshow(flatCor[4:2044,4:2044], aspect='auto', cmap='jet', clim=[0,2.*med1], origin='lower')
            plt.colorbar()
            for l in range(limits.shape[0]):
                plt.plot(limits[l]+0.5, np.arange(limits.shape[1]),'k', linewidth=1)
                plt.plot(polyLimits[l]+0.5, np.arange(limits.shape[1]),'r--', linewidth=1)
            plt.savefig('quality_control/'+folder+'_flat_slices_traces.png', dpi=300)
            plt.close(fig)
                    
        limits = polyLimits

        #write limits to file
        wifisIO.writeFits(limits,savename+'_flat_limits.fits')

        #get rid of reference pixels
        flatCor = flatCor[4:2044, 4:2044]
        sigmaCor = sigmaCor[4:2044, 4:2044]
        satFrame = satFrame[4:2044,4:2044]

        #now extract the individual slices
        flatSlices = slices.extSlices(flatCor, limits, dispAxis=0)

        #extract uncertainty slices
        sigmaSlices = slices.extSlices(sigmaCor, limits, dispAxis=0)

        #extract saturation slices
        satSlices = slices.extSlices(satFrame, limits, dispAxis=0)

        print('Getting normalized flat field')
        #now get smoothed and normalized response function
        flatNorm = slices.getResponseAll(flatSlices, 0, 0.1)
        sigmaNorm = slices.ffCorrectAll(sigmaSlices, flatNorm)
        
        #write slices to file
        wifisIO.writeFits(flatSlices+sigmaSlices+satSlices,savename+'_flat_slices.fits')

        #write normalized images to file
        wifisIO.writeFits(flatNorm + sigmaNorm + satSlices,savename+'_flat_slices_norm.fits')
        print('*** Finished processing ' + folder + ' in ' + str(time.time()-t1) + ' seconds ***')
        
print('Total time to run entire script: ' + str(time.time()-t0) + ' seconds')
