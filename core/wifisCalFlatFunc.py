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

def runCalFlat(lst, hband=False, darkLst=None, rootFolder='', nlCoef=None, satCounts=None, BPM=None, distMapLimitsFile='', plot=True, nChannel=32, nRowsAvg=0,rowSplit=1,nlSplit=32, combSplit=32,bpmCorRng=100, crReject=False, skipObsinfo=False,winRng=51, polyFitDegree=3, imgSmth=5, avgRamps=False,nlFile='',bpmFile='', satFile='',darkFile='',flatCutOff=0.1,flatSmooth=0, logfile=None, gain=1., ron=1., dispAxis=0,limSmth=20, ask=True):
    
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
        print('*** WORKING ON H-BAND DATA ***')
    
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
                if ask:
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
                            if logfile is not None:
                                logfile.write('Processing ' + folder + ' ramp ' + str(rampNum)+'\n')

                            flatImg, sigmaImg, satFrame, hdr = processRamp.auto(folder, rootFolder,savename+'_flat.fits', satCounts, nlCoef, BPM, nChannel=nChannel, rowSplit=rowSplit, nlSplit=nlSplit, combSplit=combSplit, crReject=crReject, bpmCorRng=bpmCorRng, skipObsinfo=skipObsinfo, nRows=nRowsAvg, rampNum=rampNum2,nlFile=nlFile,satFile=satFile,bpmFile=bpmFile, gain=gain, ron=ron,logfile=logfile)
                            flatImgAll.append(flatImg)
                            sigmaImgAll.append(sigmaImg)
                            satFrameAll.append(satFrame)

                        #now combine all images
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', RuntimeWarning)
                            flatImg = np.nanmedian(np.asarray(flatImgAll),axis=0)
                            #sigmaImg = np.zeros(sigmaImg.shape,dtype=sigmaImg.dtype)
                            #for k in range(len(sigmaImgAll)):
                            #sigmaImg += sigmaImgAll[k]**2
                            sigmaImg = np.sqrt(np.nansum(np.asarray(sigmaImg)**2,axis=0))/len(sigmaImgAll)
                            #sigmaImg = np.nanmedian(np.asarray(sigmaImgAll),axis=0)
                            satFrame = np.nanmedian(np.asarray(satFrameAll),axis=0).round().astype('int')

                        if logfile is not None:
                            logfile.write('Median combined data of ' + str(nRampsAll) + ' ramps\n')
                            
                        hdr.add_history('Data is median average of ' +str(nRampsAll) + ' ramps')
                        wifisIO.writeFits([flatImg,sigmaImg,satFrame],savename+'_flat.fits',hdr=hdr,ask=False)
                        
                        del flatImgAll
                        del sigmaImgAll
                        del satFrameAll
                    else:
                        flatImg, sigmaImg, satFrame, hdr = processRamp.auto(folder, rootFolder,savename+'_flat.fits', satCounts, nlCoef, BPM, nChannel=nChannel, rowSplit=rowSplit, nlSplit=nlSplit, combSplit=combSplit, crReject=crReject, bpmCorRng=bpmCorRng, skipObsinfo=skipObsinfo, nRows=nRowsAvg, rampNum=rampNum, nlFile=nlFile, satFile=satFile, bpmFile=bpmFile, gain=gain, ron=ron, logfile=logfile)

                    #carry out dark subtraction
                    if darkLst is not None and darkLst[0] is not None:
                        print('Subtracting dark ramp')
                        dark, darkSig = darkLst
                        flatImg -= dark
                        sigmaImg = np.sqrt(sigmaImg**2 + darkSig**2)
                        hdr.add_history('Dark image subtracted using file:')
                        hdr.add_history(darkFile)
                        if logfile is not None:
                            logfile.write('Subtracted dark image using file:\n')
                            logfile.write(darkFile+'\n')
                    else:
                        warnings.warn('*** No dark image provide, or file does not exist ***')
                        if logfile is not None:
                            logfile.write('*** WARNING: No dark image provide, or file ' + str(darkFile)+' does not exist ***')

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

                    #remove comment about contents of file
                    hdrTmp = hdr[::-1]
                    hdrTmp.remove('COMMENT')
                    hdr = hdrTmp[::-1]
                    
                    #find limits of each slice with the reference pixels, but the returned limits exclude them
                    limits = slices.findLimits(flatImg, dispAxis=0, winRng=winRng, imgSmth=imgSmth, limSmth=limSmth, rmRef=True)

                    if logfile is not None:
                        logfile.write('Identified slice limits using the following parameters:\n')
                        logfile.write('dispAxis: '+str(dispAxis)+'\n')
                        logfile.write('winRng: ' + str(winRng)+'\n')
                        logfile.write('imgSmth: ' + str(imgSmth)+'\n')
                        logfile.write('limSmth: ' + str(limSmth)+'\n')
                        
                    #get smoother limits, if desired, using polynomial fitting
                    polyLimits = slices.polyFitLimits(limits, degree=polyFitDegree, sigmaClipRounds=2)

                    if logfile is not None:
                        logfile.write('Fit polynomial to slice edge traces using:\n')
                        logfile.write('Polynomial degree: ' + str(polyFitDegree)+'\n')
                        logfile.write('sigmaClipRounds: ' + str(2)+'\n')
                        
                    if os.path.exists(distMapLimitsFile):
                        print('Finding slice limits relative to distortion map file')
                        hdr.add_history('Slice limits are relative to the following file:')
                        hdr.add_history(distMapLimitsFile)
                        distMapLimits = wifisIO.readImgsFromFile(distMapLimitsFile)[0]
                        if logfile is not None:
                            logfile.write('Finding slice limits relative to distortion map file:\n')
                            logfile.write(distMapLimitsFile+'\n')
                            
                        if hband:
                            #only use region with suitable flux
                            if dispAxis == 0:
                                flatImgMed = np.nanmedian(flatImg[4:-4,4:-4], axis=1)
                            else:
                                flatImgMed = np.nanmedian(flatImg[4:-4,4:-4], axis=0)
                                
                            flatImgMedGrad = np.gradient(flatImgMed)
                            medMax = np.nanargmax(flatImgMed)
                            lim1 = np.nanargmax(flatImgMedGrad[:medMax])
                            lim2 = np.nanargmin(flatImgMedGrad[medMax:])+medMax
                            shft = int(np.nanmedian(polyLimits[1:-1,lim1:lim2+1] - distMapLimits[1:-1,lim1:lim2+1]))
                            if logfile is not None:
                                logfile.write('Only used pixels between ' + str(lim1) +' and ' + str(lim2)+'\n')
                        else:
                            shft = int(np.nanmedian(polyLimits[1:-1,:] - distMapLimits[1:-1,:]))
                        
                        if logfile is not None:
                            logfile.write('Median pixel shift using all inner edge limits is ' + str(shft)+'\n')
                        finalLimits = distMapLimits
                    else:
                        finalLimits = polyLimits
                        shft = 0
                        if logfile is not None:
                            logfile.write('*** WARNING:No slice limits provided for distortion map. Finding independent slice limits ***\n')
                            logfile.write(distMapLimitsFile+'\n')
 
                
                    #write distMapLimits + shft to file
                    hdr.set('LIMSHIFT',shft, 'Limits shift relative to Ronchi slices')
                    hdr.add_comment('File contains the edge limits for each slice')
                    wifisIO.writeFits(finalLimits,savename+'_flat_limits.fits', hdr=hdr, ask=False)

                    #remove comment about contents of file
                    hdrTmp = hdr[::-1]
                    hdrTmp.remove('COMMENT')
                    hdr = hdrTmp[::-1]
                    
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

                    if logfile is not None:
                        logfile.write('Removing reference pixels\n')
                        
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
                        if logfile is not None:
                            logfile.write('Extracted flat slices\n')
                        
                        #extract uncertainty slices
                        sigmaSlices = slices.extSlices(sigmaImg, finalLimits, dispAxis=0, shft=shft)
                        if logfile is not None:
                            logfile.write('Extracted uncertainty slices\n')
                            
                        #extract saturation slices
                        satSlices = slices.extSlices(satFrame, finalLimits, dispAxis=0, shft=shft)
                        if logfile is not None:
                            logfile.write('Extracted saturation info slices\n')
                            
                        #write slices to file

                        hdr.add_comment('File contains each slice image as separate extension')
                        wifisIO.writeFits(flatSlices+sigmaSlices+satSlices,savename+'_flat_slices.fits',hdr=hdr, ask=False)
                        
                        #remove comment about contents of file
                        hdrTmp = hdr[::-1]
                        hdrTmp.remove('COMMENT')
                        hdr = hdrTmp[::-1]
                        
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
                        flatNorm = slices.getResponseAll(flatSlices, flatSmooth, flatCutOff)
                        hdr.add_comment('File contains the normalized flat-field response function')
                        hdr.add_history('Smoothed using Gaussian with 1-sigma width of ' + str(flatSmooth) + ' pixels')
                        hdr.add_history('Normalized cutoff threshold is ' + str(flatCutOff))

                        if logfile is not None:
                            logfile.write('Computed normalized response function from flat slices using the following parameters:\n')
                            logfile.write('flatSmooth: ' + str(flatSmooth)+'\n')
                            logfile.write('flatCutoff: ' + str(flatCutOff)+'\n')
                            
                        sigmaNorm = slices.ffCorrectAll(sigmaSlices, flatNorm)
                        if logfile is not None:
                            logfile.write('Computed uncertainties for normalized response function for each slice\n')
                        
                        
                        #write normalized images to file
                        wifisIO.writeFits(flatNorm + sigmaNorm + satSlices,savename+'_flat_slices_norm.fits',hdr=hdr, ask=False)
                    if nRamps > 1:    
                        print('*** Finished processing ' + folder + ', ramp ' + '{:02d}'.format(rampNum)+' in ' + str(time.time()-t1) + ' seconds ***')
                    else:
                        print('*** Finished processing ' + folder + ' in ' + str(time.time()-t1) + ' seconds ***')
                
   
    return

