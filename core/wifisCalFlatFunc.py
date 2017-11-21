"""

Function used to process a set of raw flat field ramps

Usage: runCalFlat(lst, hband=False, darkLst=None, rootFolder='', nlCoef=None, satCounts=None, BPM=None, distMapLimitsFile='', plot=True, nChannel=32, nRowsAvg=0,rowSplit=1,nlSplit=32, combSplit=32,bpmCorRng=100, crReject=False, skipObsinfo=False,winRng=51, polyFitDegree=3, imgSmth=5,nlFile='',bpmFile='', satFile='',darkFile='',flatCutOff=0.1,flatSmooth=0, logfile=None, gain=1., ron=None, dispAxis=0,limSmth=20, ask=True, obsCoords=None,satSplit=32)

lst - list of folder names corresponding to each flat field observation
hband - boolean flag to specify if the observation was taken with the H-band filter
darkLst - is a reduced and processed dark image (or a list containing the dark image and associated uncertainty)
rootFolder - specifies the root location of the raw data
nlCoef - the non-linearity coefficient correction array for non-linearity corrections
satCounts - the map/image of the saturation levels of each pixel
BPM - a bad pixel mask to be used to mark bad pixels
distMapLimitsFile - the file location corresponding to a distortion map for each slice
plot - a boolean flag to indicate if quality control plots should be plotted and saved
nchannel - number of read channels used for detector readout
nRowsAvg - number of rows to be averaged for row reference correction (using a moving average)
rowSplit - number of instances to split the task of row reference correction (a higher value reduces memory limitations at the expense of longer processing times). Must be an integer number of the number of frames in the ramp sequence
nlSplit -  number of instances to split the task of non-linearity correction (a higher value reduces memory limitations at the expense of longer processing times). Must be integer number of the number of columns in the detector image.
combSlit - number of instances to split the task of creating a single ramp image (a higher value reduces memory limitations at the expense of longer processing times). Must be integer number of the number of columns in the detector image.
bpmCorRng - the maximum separation between the bad pixel and the nearest good pixel for bad pixel corrections.
crReject - boolean flag to use routine suited to reject cosmic ray events for creating ramp image
skipObsinfo - boolean flag to allow skipping of warning/failure if obsinfo.dat file is not present.
winRng - window range for identifying slice edge limits, relative to defined position
polyFitDegree - degree of polynomial fit to be used for tracing the slice edge limits
imgSmth - a keyword specifying the Gaussian width of the smoothing kernel for finding the slice-edge limits
nlFile - the name/path of the non-linearity coefficient file to be specified in fits header
bpmFile - the name/path of the bad pixel mask file to be specified in fits header
satFile - the name/path of the saturation info file to be specified in fits header
darkFile - the name/path of the dark image file to be specified in fits header
flatCutOff -  cutoff value for which all pixels with normalized values less than this value are set to NaN
flatSmooth - a keyword specifying the Gaussian width of the smoothing kernel to be used for determining the response function
logfile - file object corresponding to the logfile
gain - gain conversion factor needed if and only if RON image is given in units of e- not counts
ron - readout noise image/map of detector
dispAxis - integer specifying the dispersion axis (0 - along the y-axis, 1 - along the x-axis)
limSmth - integer specifying the Gaussian width of the smoothing kernel used to smooth the traced slice-edge limits
ask - boolean flag used to specify if the user should be prompted if files already exist. If set to no, no reprocessing is done
obsCoords - list containing the observatory coordinates [longitude (deg), latitude (deg), altitude (m)]
satSplit - number of instances to split the task of carrying out saturation correction (a higher value reduces memory limitations at the expense of longer processing times). Must be integer number of the number of columns in the detector image.
"""

import matplotlib
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import wifisIO 
import wifisSlices as slices
import wifisProcessRamp as processRamp
from matplotlib.backends.backend_pdf import PdfPages
import warnings
from astropy import time as astrotime, coordinates as coord, units
import colorama
from astropy.visualization import ZScaleInterval

def runCalFlat(lst, hband=False, darkLst=None, rootFolder='', nlCoef=None, satCounts=None, BPM=None, distMapLimitsFile='', plot=True, nChannel=32, nRowsAvg=0,rowSplit=1,nlSplit=32, combSplit=32,bpmCorRng=100, crReject=False, skipObsinfo=False,winRng=51, polyFitDegree=3, imgSmth=5,nlFile='',bpmFile='', satFile='',darkFile='',flatCutOff=0.1,flatSmooth=0, logfile=None, gain=1., ron=None, dispAxis=0,limSmth=20, ask=True, obsCoords=None,satSplit=32):
    
    """
    Flat calibration function which can be used/called from another script.
    """

    colorama.init()
    
    plt.ioff()

    t0 = time.time()
    
    #create processed directory, in case it doesn't exist
    wifisIO.createDir('processed')
    wifisIO.createDir('quality_control')

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

        t1 = time.time()

        savename = 'processed/'+folder

        #first check master flat and limits exists
    
        if(os.path.exists(savename+'_flat.fits') and os.path.exists(savename+'_flat_limits.fits') and os.path.exists(savename+'_flat_slices.fits') and os.path.exists(savename+'_flat_slices_norm.fits')):
            cont = 'n'
            if ask:
                cont = wifisIO.userInput('All processed flat field files already exists for ' +folder+', do you want to continue processing (y/n)?')
        else:
            cont = 'y'
            
        if (cont.lower() == 'y'):
            print('*** Working on folder ' + folder + ' ***')

            if (os.path.exists(savename+'_flat.fits')):
                cont = 'n'
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
                flatImg, sigmaImg, satFrame, hdr = processRamp.auto(folder, rootFolder,savename+'_flat.fits', satCounts, nlCoef, BPM, nChannel=nChannel, rowSplit=rowSplit, nlSplit=nlSplit, combSplit=combSplit, crReject=crReject, bpmCorRng=bpmCorRng, skipObsinfo=skipObsinfo, nRows=nRowsAvg, rampNum=None, nlFile=nlFile, satFile=satFile, bpmFile=bpmFile, gain=gain, ron=ron, logfile=logfile, obsCoords=obsCoords, avgAll=True, satSplit=satSplit)
            
            #carry out dark subtraction
            if darkLst is not None and darkLst[0] is not None:
                print('Subtracting dark ramp')
                if len(darkLst)>1:
                    dark, darkSig = darkLst[:2]
                    sigmaImg = np.sqrt(sigmaImg**2 + darkSig**2)
                else:
                    dark = darkLst[0]
                    logfile.write('*** Warning: No uncertainty associated with dark image ***\n')
                    print(colorama.Fore.RED+'*** WARNING: No uncertainty associated with dark image ***'+colorama.Style.RESET_ALL)

                flatImg -= dark
                hdr.add_history('Dark image subtracted using file:')
                hdr.add_history(darkFile)
                if logfile is not None:
                    logfile.write('Subtracted dark image using file:\n')
                    logfile.write(darkFile+'\n')
            else:
                print(colorama.Fore.RED+'*** WARNING: No dark image provided, or file does not exist ***'+colorama.Style.RESET_ALL)
                if logfile is not None:
                    logfile.write('*** WARNING: No dark image provided, or file ' + str(darkFile)+' does not exist ***')

            if os.path.exists(savename+'_flat_limits.fits'):
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
                limits = slices.findLimits(flatImg, dispAxis=dispAxis, winRng=winRng, imgSmth=imgSmth, limSmth=limSmth, rmRef=True)

                if logfile is not None:
                    logfile.write('Identified slice limits using the following parameters:\n')
                    logfile.write('dispAxis: '+str(dispAxis)+'\n')
                    logfile.write('winRng: ' + str(winRng)+'\n')
                    logfile.write('imgSmth: ' + str(imgSmth)+'\n')
                    logfile.write('limSmth: ' + str(limSmth)+'\n')
                    
                if hband:
                    print('Using suitable region of detector to determine flat limits')
                    if logfile is not None:
                        logfile.write('Using suitable region of detector to determine flat limits:\n')

                    #only use region with suitable flux
                    if dispAxis == 0:
                        flatImgMed = np.nanmedian(flatImg[4:-4,4:-4], axis=1)
                    else:
                        flatImgMed = np.nanmedian(flatImg[4:-4,4:-4], axis=0)
                            
                    flatImgMedGrad = np.gradient(flatImgMed)
                    medMax = np.nanargmax(flatImgMed)
                    lim1 = np.nanargmax(flatImgMedGrad[:medMax])
                    lim2 = np.nanargmin(flatImgMedGrad[medMax:])+medMax

                    if logfile is not None:
                        logfile.write('Using following detector limits to set slice limits:\n')
                        logfile.write(str(lim1)+ ' ' + str(lim2)+'\n')
                        
                    polyLimits = slices.polyFitLimits(limits, degree=2, sigmaClipRounds=2, constRegion=[lim1,lim2])
                else:
                    #get smoother limits, if desired, using polynomial fitting
                    polyLimits = slices.polyFitLimits(limits, degree=polyFitDegree, sigmaClipRounds=2)

                if logfile is not None:
                    logfile.write('Fit polynomial to slice edge traces using:\n')
                    logfile.write('Polynomial degree: ' + str(polyFitDegree)+'\n')
                    logfile.write('sigmaClipRounds: ' + str(2)+'\n')

                    if hband:
                        logfile.write('Only used pixels between ' + str(lim1) +' and ' + str(lim2)+'\n')
                     
                if os.path.exists(distMapLimitsFile):
                    print('Finding slice limits relative to distortion map file')
                    hdr.add_history('Slice limits are relative to the following file:')
                    hdr.add_history(distMapLimitsFile)
                    distMapLimits = wifisIO.readImgsFromFile(distMapLimitsFile)[0]
                    if logfile is not None:
                        logfile.write('Finding slice limits relative to distortion map file:\n')
                        logfile.write(distMapLimitsFile+'\n')

                    if hband:
                        shft = int(np.nanmedian(polyLimits[1:-1,lim1:lim2+1] - distMapLimits[1:-1,lim1:lim2+1]))
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

                wifisIO.writeFits(finalLimits.astype('float32'),savename+'_flat_limits.fits', hdr=hdr, ask=False)

                #remove comment about contents of file
                hdrTmp = hdr[::-1]
                hdrTmp.remove('COMMENT')
                hdr = hdrTmp[::-1]
                    
                #save figures of tracing results for quality control purposes
                if (plot):
                    print('Plotting results')
                    plt.ioff()
                    wifisIO.createDir('quality_control')
                            
                    pdfName = 'quality_control/'+folder+'_flat_slices_traces.pdf'
                    with PdfPages(pdfName) as pdf:
                        fig = plt.figure()
                        #med1= np.nanmedian(flatImg)
                        interval = ZScaleInterval()
                        lims = interval.get_limits(flatImg[4:-4,4:-4])
                        #plt.imshow(flatImg[4:-4,4:-4], aspect='auto', cmap='jet', clim=[0,2.*med1], origin='lower')
                        plt.imshow(flatImg[4:-4,4:-4], aspect='auto', cmap='jet', clim=lims, origin='lower')
                        
                        plt.xlim=(0,2040)
                        plt.colorbar()
                        for l in range(limits.shape[0]):
                            if dispAxis==0:
                                plt.plot(limits[l], np.arange(limits.shape[1]),'k', linewidth=1) #drawn limits
                                plt.plot(np.clip(finalLimits[l]+shft,0, flatImg[4:-4,4:-4].shape[0]-1), np.arange(limits.shape[1]),'r--', linewidth=1) #shifted ronchi limits, if provided, or polynomial fit
                            else:
                                plt.plot(np.arange(limits.shape[1]),limits[l],'k', linewidth=1) #drawn limits
                                plt.plot(np.arange(limits.shape[1]),np.clip(finalLimits[l]+shft,0, flatImg[4:-4,4:-4].shape[0]-1),'r--', linewidth=1) #shifted ronchi limits

                        if hband:
                            if dispAxis==0:
                                plt.plot([0,flatImg[4:-4,4:-4].shape[1]-1],[lim1,lim1],'b:',linewidth=1)
                                plt.plot([0,flatImg[4:-4,4:-4].shape[1]-1],[lim2,lim2],'b:',linewidth=1)
                            else:
                                plt.plot([lim1,lim1],[0,flatImg[4:-4,4:-4].shape[1]-1],'b:',linewidth=1)
                                plt.plot([lim2,lim2],[0,flatImg[4:-4,4:-4].shape[1]-1],'b:',linewidth=1)

                        plt.tight_layout()
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
                flatSlices = slices.extSlices(flatImg, finalLimits, dispAxis=dispAxis, shft=shft)
                for slc in flatSlices:
                    slc = slc.astype('float32')
                    
                if logfile is not None:
                    logfile.write('Extracted flat slices\n')
                        
                #extract uncertainty slices
                sigmaSlices = slices.extSlices(sigmaImg, finalLimits, dispAxis=dispAxis, shft=shft)
                for slc in sigmaSlices:
                    slc = slc.astype('float32')
                    
                if logfile is not None:
                    logfile.write('Extracted uncertainty slices\n')
                            
                #extract saturation slices
                satSlices = slices.extSlices(satFrame, finalLimits, dispAxis=dispAxis, shft=shft)
                for slc in satSlices:
                    slc = slc.astype('float32')
                    
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
                for slc in flatNorm:
                    slc = slc.astype('float32')
                    
                hdr.add_comment('File contains the normalized flat-field response function')
                hdr.add_history('Smoothed using Gaussian with 1-sigma width of ' + str(flatSmooth) + ' pixels')
                hdr.add_history('Normalized cutoff threshold is ' + str(flatCutOff))

                if logfile is not None:
                    logfile.write('Computed normalized response function from flat slices using the following parameters:\n')
                    logfile.write('flatSmooth: ' + str(flatSmooth)+'\n')
                    logfile.write('flatCutoff: ' + str(flatCutOff)+'\n')
                    
                sigmaNorm = slices.ffCorrectAll(sigmaSlices, flatNorm)
                for slc in sigmaNorm:
                    slc = slc.astype('float32')
                
                if logfile is not None:
                    logfile.write('Computed uncertainties for normalized response function for each slice\n')
                        
                        
                #write normalized images to file
                wifisIO.writeFits(flatNorm + sigmaNorm + satSlices,savename+'_flat_slices_norm.fits',hdr=hdr, ask=False)
                print('*** Finished processing ' + folder + ' in ' + str(time.time()-t1) + ' seconds ***')
                   
    return

