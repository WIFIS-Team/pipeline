"""

Calibrate flat field images

Produces:
- master flat image
- slitlet traces
- ??

"""
import matplotlib
matplotlib.use('pdf')
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
plt.ioff()

t0 = time.time()

#*****************************************************************************
#******************************* Required input ******************************
fileList = 'flat.lst' # a simple ascii file containing a list of the folder names that contain the ramp data

#mostly static input from here
rootFolder = '/data/WIFIS/H2RG-G17084-ASIC-08-319'
nlFile = '/home/jason/wifis/data/non-linearity/may/processed/master_detLin_NLCoeff.fits' # the non-linearity correction coefficients file        
satFile = '/home/jason/wifis/data/non-linearity/may/processed/master_detLin_satCounts.fits' # the saturation limits file
bpmFile = '/data/pipeline/external_data/bpm.fits' # the bad pixel mask
distMapLimitsFile = '/data/pipeline/external_data/ronchiMap_limits.fits'

#optional behaviour of pipeline
plot = True #whether to plot the traces
crReject = False
skipObsinfo = False
skipMissing = True

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
    t1 = time.time()
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
                flatImgs, hdr= wifisIO.readImgsFromFile(savename+'_flat.fits')
                flatImg, sigmaImg, satFrame = flatImgs
                if (type(hdr) is list):
                    hdr = hdr[0]
                contProc2=False
            else:
                contProc2=True
        else:
            contProc2=True

        if contProc2:
            print('Processing ramp')
            flatImg, sigmaImg, satFrame, hdr = processRamp.auto(folder, rootFolder,savename+'_flat.fits', satCounts, nlCoeff, BPM, nChannel=32, rowSplit=1, nlSplit=32, combSplit=32, crReject=crReject, bpmCorRng=20, skipObsinfo=skipObsinfo, skipMissing=skipMissing)
            
        if os.path.exists(savename+'_flat_limits.fits'):
            cont = wifisIO.userInput('Limits file already exists for ' +folder+', do you want to continue processing (y/n)?')
            if (cont.lower() == 'n'):
                print('Reading limits'+savename+'_flat_limits.fits instead')
                limits, limitsHdr= wifisIO.readImgsFromFile(savename+'_flat_limits.fits')
                shft = limitsHdr['LIMSHIFT']
                contProc2 = False
            else:
                contProc2 = True
        else:
            contProc2= True

        if (contProc2):
            print('Finding slice limits and extracting slices')
            #find limits of each slice with the reference pixels, but the returned limits exclude them
            limits = slices.findLimits(flatImg, dispAxis=0, winRng=51, imgSmth=5, limSmth=20, rmRef=True)

            #get smoother limits, if desired, using polynomial fitting
            polyLimits = slices.polyFitLimits(limits, degree=3)

            if os.path.exists(distMapLimitsFile):
                print('Finding slice limits relative to distortion map file')
                distMapLimits = wifisIO.readImgsFromFile(distMapLimitsFile)[0]
                shft = int(np.nanmedian(polyLimits[1:-1,:] - distMapLimits[1:-1,:]))
                finalLimits = distMapLimits
            else:
                finalLimits = polyLimits
                shft = 0

            #write distMapLimits + shft to file
            hdr.set('LIMSHIFT',shft, 'Limits shift relative to Ronchi slices')
            wifisIO.writeFits(finalLimits,savename+'_flat_limits.fits', hdr=hdr)
            
            #save figures of tracing results for quality control purposes
            if (plot):
                print('Plotting results')
                plt.ioff()
                wifisIO.createDir('quality_control')
                with PdfPages('quality_control/'+folder+'_flat_slices_traces.pdf') as pdf:
                    fig = plt.figure()
                    med1= np.nanmedian(flatImg)
                    
                    plt.imshow(flatImg[4:-4,4:-4], aspect='auto', cmap='jet', clim=[0,2.*med1], origin='lower')
                    plt.colorbar()
                    for l in range(limits.shape[0]):
                        plt.plot(limits[l], np.arange(limits.shape[1]),'k', linewidth=1)
                        plt.plot(finalLimits[l]+shft, np.arange(limits.shape[1]),'r--', linewidth=1)
                    pdf.savefig()
                    plt.close(fig)

        #get rid of reference pixels
        flatImg = flatImg[4:-4, 4:4-4]
        sigmaImg = sigmaImg[4:-4, 4:-4]
        satFrame = satFrame[4:-4,4:-4]

        if os.path.exists(savename+'_flat_slices.fits'):
            cont = wifisIO.userInput('Flat slices file already exists for ' +folder+', do you want to continue processing (y/n)?')
            if (cont.lower() == 'n'):
                print('Reading slices file '+savename+'_flat_slices.fits instead')
                flatSlices = wifisIO.readImgsFromFile(savename+'_flat_slices.fits')[0]
                contProc2 = False
            else:
                contProc2 = True
        else:
            contProc2= True
            
        if (contProc2):
            print('Extracting slices')           
            #now extract the individual slices
            flatSlices = slices.extSlices(flatImg, finalLimits, dispAxis=0, shft=shft)
            
            #extract uncertainty slices
            sigmaSlices = slices.extSlices(sigmaImg, finalLimits, dispAxis=0, shft=shft)

            #extract saturation slices
            satSlices = slices.extSlices(satFrame, finalLimits, dispAxis=0)

            #write slices to file
            wifisIO.writeFits(flatSlices+sigmaSlices+satSlices,savename+'_flat_slices.fits',hdr=hdr)

        if os.path.exists(savename+'_flat_slices_norm.fits'):
            cont = wifisIO.userInput('Normalized flat slices file already exists for ' +folder+', do you want to continue processing (y/n)?')
            if (cont.lower() == 'n'):
                contProc2 = False
            else:
                contProc2 = True
        else:
            contProc2= True
            
        if (contProc2):
            print('Getting normalized flat field')
            #now get smoothed and normalized response function
            flatNorm = slices.getResponseAll(flatSlices, 0, 0.1)
            sigmaNorm = slices.ffCorrectAll(sigmaSlices, flatNorm)
            
            #write normalized images to file
            wifisIO.writeFits(flatNorm + sigmaNorm + satSlices,savename+'_flat_slices_norm.fits',hdr=hdr)
        print('*** Finished processing ' + folder + ' in ' + str(time.time()-t1) + ' seconds ***')
        
print('Total time to run entire script: ' + str(time.time()-t0) + ' seconds')
