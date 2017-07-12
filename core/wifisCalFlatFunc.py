"""

Calibrate flat field images

Produces:
- master flat image
- slitlet traces
- ??

"""
import matplotlib
matplotlib.use('gtkagg')
import numpy as np
import time
import matplotlib.pyplot as plt
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
import warnings

def runCalFlat(lst, hband=False, darkLst=None, rootFolder='', nlCoef=None, satCounts=None, BPM=None, distMapLimitsFile='', plot=True, nChannel=32, nRowAverage=4,rowSplit=1,nlSplit=32, combSplit=32,bpmCorRng=100, crReject=False, skipObsinfo=False,winRng=51, polyFitDegree=3, imgSmth=5, avgRamps=False):
    
    """
    Flat calibration function which can be used/called from another script.
    """

    plt.ioff()

    t0 = time.time()
    
    #create processed directory, in case it doesn't exist
    wifisIO.createDir('processed')
    wifisIO.createDir('quality_control')

    #NOTES
    #file list can be a 2D table: where all files to be coadded are on the same row and files to be processed separately are on different rows. *** NOT YET IMPLEMENTED

    if hband:
        print('***WORKING ON H-BAND DATA***')
    
    #create processed directory, in case it doesn't exist
    wifisIO.createDir('processed')

    if (plot):
        wifisIO.createDir('quality_control')

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
            folder = lst[lstNum]

        if darkLst is not None:
            if darkLst.ndim>1:
                darkFile = darkLst[lstNum]
            else:
                darkFile = darkLst[lstNum]
        else:
            darkFile = None

        #check if folder contains multiple ramps
        nRamps = wifisIO.getNumRamps(folder, rootFolder=rootFolder)
        nRampsAll = nRamps

        #set a new variable to indicate the number of ramps while setting nRamps = 1 to avoid repeating
        if nRamps > 1 and avgRamps:
            nRamps = 1
        
        for rampNum in range(1,nRamps+1):
            t1 = time.time()

            if nRamps > 1:
                savename = 'processed/'+folder+'_R'+'{:02d}'.format(rampNum)
            else:
                savename = 'processed/'+folder

            #first check master flat and limits exists
    
            if(os.path.exists(savename+'_flat.fits') and os.path.exists(savename+'_flat_limits.fits') and os.path.exists(savename+'_flat_slices.fits') and os.path.exists(savename+'_flat_slices_norm.fits')):
                cont = 'n'
                if nRamps > 1:
                    cont = wifisIO.userInput('All processed flat field files already exists for ' +folder+', ramp ' + '{:02d}'.format(rampNum)+', do you want to continue processing (y/n)?')
                else:
                    cont = wifisIO.userInput('All processed flat field files already exists for ' +folder+', do you want to continue processing (y/n)?')

            else:
                cont = 'y'
        
            if (cont.lower() == 'y'):
                if nRamps > 1:
                    print('*** Working on folder ' + folder + ', ramp ' + '{:02d}'.format(rampNum)+' ***')
                else:
                    print('*** Working on folder ' + folder + ' ***')

                if (os.path.exists(savename+'_flat.fits')):
                    cont = 'n'
                    if nRamps > 1:
                        cont = wifisIO.userInput('Processed flat field file already exists for ' +folder+', ramp ' +  '{:02d}'.format(rampNum)+', do you want to continue processing (y/n)?')
                    else:
                        cont = wifisIO.userInput('Processed flat field file already exists for ' +folder+', do you want to continue processing (y/n)?')

                    if (not cont.lower() == 'y'):
                        print('Reading image '+savename+'_flat.fits instead')
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
                    if nRampsAll > 1 and avgRamps:
                        flatImgAll = []
                        sigmaImgAll = []
                        satFrameAll = []
                        for rampNum2 in range(1,nRampsAll+1):
                            print('Processing ramp ' + str(rampNum2))
                            
                            flatImg, sigmaImg, satFrame, hdr = processRamp.auto(folder, rootFolder,savename+'_flat.fits', satCounts, nlCoef, BPM, nChannel=nChannel, rowSplit=rowSplit, nlSplit=nlSplit, combSplit=combSplit, crReject=crReject, bpmCorRng=bpmCorRng, skipObsinfo=skipObsinfo, nRows=nRowAverage, rampNum=rampNum2)
                            flatImgAll.append(flatImg)
                            sigmaImgAll.append(sigmaImg)
                            satFrameAll.append(satFrame)

                        #now combine all images
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', RuntimeWarning)
                            flatImg = np.nanmean(np.asarray(flatImgAll),axis=0)
                            sigmaImg = np.zeros(sigmaImg.shape,dtype=sigmaImg.dtype)
                            for k in range(len(sigmaImgAll)):
                                sigmaImg += sigmaImgAll[k]**2
                            sigmaImg = np.sqrt(sigmaImg)/len(sigmaImgAll)
                            satFrame = np.nanmean(np.asarray(satFrameAll),axis=0).round().astype('int')
                        del flatImgAll
                        del sigmaImgAll
                        del satFrameAll
                    else:
                        flatImg, sigmaImg, satFrame, hdr = processRamp.auto(folder, rootFolder,savename+'_flat.fits', satCounts, nlCoef, BPM, nChannel=nChannel, rowSplit=rowSplit, nlSplit=nlSplit, combSplit=combSplit, crReject=crReject, bpmCorRng=bpmCorRng, skipObsinfo=skipObsinfo, nRows=nRowAverage, rampNum=rampNum)

                    #carry out dark subtraction
                    if darkFile is not None:
                        print('Subtracting dark ramp')
                        dark, darkSig, darkSat = wifisIO.readImgsFromFile(darkFile)[0]
                        flatImg -= dark
                        sigmaImg = np.sqrt(sigmaImg**2 + darkSig**2)
                    else:
                        warnings.warn('*** NO DARK RAMP PROVIDED SKIPPING DARK SUBTRACTION ***')

                if os.path.exists(savename+'_flat_limits.fits'):
                    if nRamps > 1:
                        cont = wifisIO.userInput('Limits file already exists for ' +folder+ ', ramp '+'{:02d}'.format(rampNum)+', do you want to continue processing (y/n)?')
                    else:
                        cont = wifisIO.userInput('Limits file already exists for ' +folder+ ', do you want to continue processing (y/n)?')
                        
                    if (not cont.lower() == 'y'):
                        print('Reading limits '+savename+'_flat_limits.fits instead')
                        finalLimits, limitsHdr= wifisIO.readImgsFromFile(savename+'_flat_limits.fits')
                        shft = limitsHdr['LIMSHIFT']
                        contProc2 = False
                    else:
                        contProc2 = True
                else:
                    contProc2= True
                    
                if (contProc2):
                    print('Finding slice limits and extracting slices')
                    #find limits of each slice with the reference pixels, but the returned limits exclude them
                    limits = slices.findLimits(flatImg, dispAxis=0, winRng=51, imgSmth=imgSmth, limSmth=20, rmRef=True)
                    
                    #get smoother limits, if desired, using polynomial fitting
                    polyLimits = slices.polyFitLimits(limits, degree=polyFitDegree, sigmaClipRounds=2)

                    if os.path.exists(distMapLimitsFile):
                        print('Finding slice limits relative to distortion map file')
                        distMapLimits = wifisIO.readImgsFromFile(distMapLimitsFile)[0]

                        if hband:
                            #only use region with suitable flux
                            flatImgMed = np.nanmedian(flatImg[4:-4,4:-4], axis=1)
                            flatImgMedGrad = np.gradient(flatImgMed)
                            medMax = np.nanargmax(flatImgMed)
                            lim1 = np.nanargmax(flatImgMedGrad[:medMax])
                            lim2 = np.nanargmin(flatImgMedGrad[medMax:])+medMax
                            shft = int(np.nanmedian(polyLimits[1:-1,lim1:lim2+1] - distMapLimits[1:-1,lim1:lim2+1]))
                        else:
                            shft = int(np.nanmedian(polyLimits[1:-1,:] - distMapLimits[1:-1,:]))
                            
                        finalLimits = distMapLimits
                    else:
                        finalLimits = polyLimits
                        shft = 0
                
                    #write distMapLimits + shft to file
                    hdr.set('LIMSHIFT',shft, 'Limits shift relative to Ronchi slices')
                    wifisIO.writeFits(finalLimits,savename+'_flat_limits.fits', hdr=hdr, ask=False)
            
                    #save figures of tracing results for quality control purposes
                    if (plot):
                        print('Plotting results')
                        plt.ioff()
                        wifisIO.createDir('quality_control')
                            
                        if nRamps > 1:
                            pdfName = 'quality_control/'+folder+'_R'+'{:02d}'.format(rampNum)+'_flat_slices_traces.pdf'
                        else:
                            pdfName = 'quality_control/'+folder+'_flat_slices_traces.pdf'

                        with PdfPages(pdfName) as pdf:
                            fig = plt.figure()
                            med1= np.nanmedian(flatImg)
                            
                            plt.imshow(flatImg[4:-4,4:-4], aspect='auto', cmap='jet', clim=[0,2.*med1], origin='lower')
                            plt.xlim=(0,2040)
                            plt.colorbar()
                            for l in range(limits.shape[0]):
                                plt.plot(limits[l], np.arange(limits.shape[1]),'k', linewidth=1) #drawn limits
                                plt.plot(np.clip(finalLimits[l]+shft,0, flatImg[4:-4,4:-4].shape[0]-1), np.arange(limits.shape[1]),'r--', linewidth=1) #shifted ronchi limits, if provided, or polynomial fit
                            pdf.savefig()
                            plt.close(fig)

                    #get rid of reference pixels
                    flatImg = flatImg[4:-4, 4:-4]
                    sigmaImg = sigmaImg[4:-4, 4:-4]
                    satFrame = satFrame[4:-4,4:-4]

                    if os.path.exists(savename+'_flat_slices.fits'):
                        cont='n'
                        if nRamps > 1:
                            cont = wifisIO.userInput('Flat slices file already exists for ' +folder+ ', ramp '+'{:02d}'.format(rampNum)+', do you want to continue processing (y/n)?')
                        else:
                            cont = wifisIO.userInput('Flat slices file already exists for ' +folder+ ', do you want to continue processing (y/n)?')

                        if (not cont.lower() == 'y'):
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
                        satSlices = slices.extSlices(satFrame, finalLimits, dispAxis=0, shft=shft)

                        #write slices to file
                        wifisIO.writeFits(flatSlices+sigmaSlices+satSlices,savename+'_flat_slices.fits',hdr=hdr, ask=False)

                    if os.path.exists(savename+'_flat_slices_norm.fits'):
                        cont = 'n'
                        if nRamps> 1:
                            cont = wifisIO.userInput('Normalized flat slices file already exists for ' +folder+', ramp ' + '{:02d}'.format(rampNum)+ ', do you want to continue processing (y/n)?')
                        else:
                            cont = wifisIO.userInput('Normalized flat slices file already exists for ' +folder+', do you want to continue processing (y/n)?')

                        if (not cont.lower() == 'y'):
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
                        wifisIO.writeFits(flatNorm + sigmaNorm + satSlices,savename+'_flat_slices_norm.fits',hdr=hdr, ask=False)
                    if nRamps > 1:    
                        print('*** Finished processing ' + folder + ', ramp ' + '{:02d}'.format(rampNum)+' in ' + str(time.time()-t1) + ' seconds ***')
                    else:
                        print('*** Finished processing ' + folder + ' in ' + str(time.time()-t1) + ' seconds ***')

                        
    print('Total time to run entire script: ' + str(time.time()-t0) + ' seconds')
   
    return

