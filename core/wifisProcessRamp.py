"""

tools used to process ramps

"""

import numpy as np
import wifisGetSatInfo as satInfo
import wifisNLCor as NLCor
import wifisRefCor as refCor
import wifisIO 
import wifisCombineData as combData
import wifisUncertainties
import wifisBadPixels as badPixels
import wifisHeaders as headers
import os
import time
import glob
import warnings

def process(folder, saveName, satCounts, nlCoeff, BPM,nChannel=32, nRows=0,rowSplit=1, satSplit=32, nlSplit=32, combSplit=32, crReject=False, bpmCorRng=1, saveAll=True, ignoreBPM=False,skipObsinfo=False, rampNum=None, satFile='', nlFile='', bpmFile='',logfile=None, fowler=False, gain=1., ron=None, obsCoords=None, avgAll=False):
    """
    Process a set of ramp images to create a single ramp image. Carries out all related routines (non-linearity correction, bad pixel correction, etc.)
    Usage: imgComb, sigComb, satComb, hdr = process(folder, saveName, satCounts, nlCoeff, BPM,nChannel=32, nRows=0,rowSplit=1, satSplit=32, nlSplit=32, combSplit=32, crReject=False, bpmCorRng=1, saveAll=True, ignoreBPM=False,skipObsinfo=False, rampNum=None, satFile='', nlFile='', bpmFile='',logfile=None, fowler=False, gain=1., ron=None, obsCoords=None, avgAll=False)

    folder - the name of the input folder containing the ramp to process
    saveName - the file name which to save the resulting image and corresponding attributes
    satCounts - the map/image of the saturation levels of each pixel
    nlCoeff - the non-linearity coefficient correction array for non-linearity corrections
    BPM - a bad pixel mask to be used to mark bad pixels
    nChannel - number of read channels used for detector readout
    nRows - number of rows to be averaged for row reference correction (using a moving average)
    rowSplit - number of instances to split the task of row reference correction (a higher value reduces memory limitations at the expense of longer processing times). Must be an integer number of the number of frames in the ramp sequence
    satSplit - number of instances to split the task of carrying out saturation correction (a higher value reduces memory limitations at the expense of longer processing times). Must be integer number of the number of columns in the detector image.
    nlSplit -  number of instances to split the task of non-linearity correction (a higher value reduces memory limitations at the expense of longer processing times). Must be integer number of the number of columns in the detector image.
    combSlit - number of instances to split the task of creating a single ramp image (a higher value reduces memory limitations at the expense of longer processing times). Must be integer number of the number of columns in the detector image.
    crReject - boolean flag to use routine suited to reject cosmic ray events for creating ramp image
    bpmCorRng - the maximum separation between the bad pixel and the nearest good pixel for bad pixel corrections.
    saveAll - a boolean keyword used to specify if all attributes should be saved along with the processed ramp image (saturation info and uncertainties)
    ignoreBPM - a boolean keyword used to indicate if missing a bad pixel mask should show a warning or not
    skipObsinfo - boolean flag to allow skipping of warning/failure if obsinfo.dat file is not present.
    rampNum - an integer indicating which ramp to use if the corresponding observation has multiple ramps present
    satFile - the name/path of the saturation info file to be specified in fits header
    nlFile - the name/path of the non-linearity coefficient file to be specified in fits header
    bpmFile - the name/path of the bad pixel mask file to be specified in fits header
    logfile - file object corresponding to the logfile
    fowler - a boolean keyword to indicate if the observation to be processed is a Fowler sampling ramp
    gain - gain conversion factor needed if and only if RON image is given in units of e- not counts
    ron - readout noise image/map of detector
    obsCoords -  list containing the observatory coordinates [longitude (deg), latitude (deg), altitude (m)]
    avgAll - boolean keyword indicating whether or not to median-average the results if multiple ramps are present in a single observation
    imgComb - the final ramp image providing the flux of the ramp sequence
    sigComb - the estimated uncertainties for each pixel
    satComb - the final saturation info about each pixel
    hdr - an astropy header object containing information about the observation
    """

    print('Processing folder ' + folder)
    if logfile is not None:
        logfile.write('Processing folder ' + folder+'\n')
        
    #first determine number of ramps
    rampLst = glob.glob(folder+'/*N01.fits')

    #check if gzipped files exist instead
    if len(rampLst)==0:
        rampLst = glob.glob(folder+'/*N01.fits.gz')
    
    nRamps = len(rampLst)
    del rampLst
    
    if avgAll:
        if nRamps > 1:
            print('Processing ' + str(nRamps)+' ramps')
            if logfile is not None:
                logfile.write('Processing ' + str(nRamps)+' ramps\n')
        rampLst = np.arange(1,nRamps+1)
    else:
        if rampNum is None:
            if nRamps > 1:
                raise Warning('*** More than one set of ramps present in folder ' + folder + '. Specify which ramp to use, or  set keyword avgAll to True ***')
            else:
                rampLst = [None]
        else:
            rampLst = [rampNum]
            
    #create lists to hold all processed images
    obsAll = []
    sigAll = []
    satAll = []

    for rampNum in rampLst:
        #Read in data
        print('Reading FITS images into cube')
        data, inttime, hdr = wifisIO.readRampFromFolder(folder, rampNum=rampNum)
        if logfile is not None:
            logfile.write('Reading folder ' + str(folder)+'\n')

            
        #convert data to float32 for future processing
        data = data.astype('float32')
        hdr.add_history('Processed using WIFIS pyPline')
        hdr.add_history('on '+time.strftime("%c"))

        #******************************************************************************
        #Correct data for reference pixels
        if nChannel >0:
            print("Subtracting reference pixel channel bias")
            refCor.channelCL(data, nChannel)
            hdr.add_history('Channel reference pixel corrections applied using '+ str(nChannel) +' channels')
            if logfile is not None:
                logfile.write('Subtracted reference pixel channel bias using ' + str(nChannel) + ' channels\n')

        if nRows > 0:
            print("Subtracting reference pixel row bias")
            refCor.rowCL(data, nRows,rowSplit)
            hdr.add_history('Row reference pixel corrections applied using '+ str(int(nRows+1))+ ' pixels')
            if logfile is not None:
                logfile.write('Subtraced row reference pixel bias using moving average of ' + str(int(nRows)+1) + ' rows\n')

        if satCounts is None:
            satFrame = np.empty((data.shape[0],data.shape[1]),dtype='uint32')
            satFrame[:] = data.shape[2]
            hdr.add_history('No saturation levels determined. Using all ramp frames')
        else:
            satFrame = satInfo.getSatFrameCL(data, satCounts,satSplit, ignoreRefPix=True)
            hdr.add_history('Saturation levels determined from file:')
            hdr.add_history(satFile)
            if logfile is not None:
                logfile.write('Determined first saturated frame based on saturation limits\n')

        #******************************************************************************
        #apply non-linearity correction
    
        #find NL coefficient file
        if nlCoeff is not None:
            print("Correcting for non-linearity")
            NLCor.applyNLCorCL(data, nlCoeff, nlSplit)
            hdr.add_history('Non-linearity corrections applied using file:')
            hdr.add_history(nlFile)
            if logfile is not None:
                logfile.write('Non-linearity corrections applied\n')
       
        #******************************************************************************
        #Combine data cube into single image
        if fowler:
            fluxImg = combData.fowlerSamplingCL(inttime, data, satFrame, combSplit)
            hdr.add_history('Flux determined from mean average of Fowler reads in openCL')
    
            #free up some memory
            del data

            #get uncertainties for each pixel
            if ron is not None:
                sigmaImg = wifisUncertainties.compFowler(inttime, fluxImg, satFrame, ron, gain=gain)
            else:
                sigmaImg = None
        else:
            if (crReject):
                fluxImg = combData.upTheRampCRRejectCL(inttime, data, satFrame, combSplit)[0]
                sigmaImg = None
                hdr.add_history('Flux determined from median gradient')
                if logfile is not None:
                    logfile.write('Determined flux using median gradient of the ramp\n')

            else:
                fluxImg  = combData.upTheRampCL(inttime, data, satFrame, combSplit)[0]
                hdr.add_history('Flux determined through linear regression')
                if logfile is not None:
                    logfile.write('Determined flux using linear regression\n')
            
            #free up some memory
            del data
            #get uncertainties for each pixel
            if ron is not None:
                sigmaImg = wifisUncertainties.compUTR(inttime, fluxImg, satFrame, ron=ron, gain=gain)
            else:
                sigmaImg = None
            #****************************************************************************
        #add additional header information here
        if not skipObsinfo:
            headers.addTelInfo(hdr, folder+'/obsinfo.dat',logfile=logfile, obsCoords=obsCoords)
        hdr.add_comment('File contains flux, sigma, and sat info as multi-extensions')
        #****************************************************************************

        #mark bad pixels
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            fluxImg[satFrame < 2] = np.nan
            #fluxImg[fluxImg < 0] = np.nan
            if sigmaImg is not None:
                sigmaImg[~np.isfinite(fluxImg)] = np.nan

            #mark pixels with flux > than maximum achievable flux as bad pixels
            dT = np.median(np.gradient(inttime))
            mxFlux = 65535./dT # assumes 16-bit electronics
            fluxImg[fluxImg>mxFlux] = np.nan
            fluxImg[fluxImg<-mxFlux] = np.nan
       
        #check for BPM and read, if exists
        if(BPM is not None):
            #assumes BPM is same dimensions as raw image file
            fluxImg[BPM.astype(bool)] = np.nan
        else:
            if not ignoreBPM:
                cont = wifisIO.userInput('*** WARNING: No bad pixel mask provided. Do you want to continue? *** (y/n)?')
                if (cont.lower()!='y'):
                    logfile.write('*** WARNING: MISSING BAD PIXEL MASK ***')
                    raise Warning('*** Missing bad pixel mask ***')

        # CORRECT BAD PIXELS
        if bpmCorRng > 0:
            print('Correcting for bad pixels')
            #try and correct all pixels, but not the reference pixels
            imgCor = np.empty(fluxImg.shape, dtype = fluxImg.dtype)
            imgCor[4:-4,4:-4] = badPixels.corBadPixelsAll(fluxImg[4:-4,4:-4], dispAxis=0, mxRng=int(bpmCorRng), MP=True)
            if sigmaImg is not None:
                sigmaCor = np.empty(sigmaImg.shape, dtype= sigmaImg.dtype)
                sigmaCor[4:-4, 4:-4]  = badPixels.corBadPixelsAll(sigmaImg[4:-4,4:-4], dispAxis=0, mxRng=int(bpmCorRng), MP=True, sigma=True)
            else:
                sigmaCor = sigmaImg
                
            hdr.add_history('Bad pixel mask used:')
            hdr.add_history(bpmFile)
            hdr.add_history('Bad pixels corrected using nearest pixels within ' + str(bpmCorRng) + ' pixel range')
            if logfile is not None:
                logfile.write('Bad pixels corrected using nearest pixels within ' + str(bpmCorRng) + ' pixel range\n')

            
        elif(bpmCorRng==0):
            imgCor = fluxImg
            sigmaCor = sigmaImg
            
        obsAll.append(imgCor)
        sigAll.append(sigmaCor)
        satAll.append(satFrame)

    if (nRamps>1):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            imgComb = np.nanmedian(np.asarray(obsAll),axis=0)

            if sigmaImg is not None:
                sigComb = np.sqrt(np.nansum(np.asarray(sigAll)**2,axis=0))/nRamps
            else:
                sigComb = np.empty(imgComb.shape,dtype='float32')
                sigComb[:]=np.nan

            satComb = np.nanmedian(np.asarray(satAll),axis=0)

        hdr.add_history('Image is median combination of ' + str(nRamps)+ ' ramps')
        if logfile is not None:
            logfile.write('Median combined data of ' + str(nRamps) + ' ramps\n')
    else:
        imgComb = obsAll[0]
        sigComb = sigAll[0]
        satComb = satAll[0]
        
    if saveAll:
        if sigComb is None:
            sigComb = np.zeros(imgComb.shape, dtype=imgComb.dtype)
        wifisIO.writeFits([imgComb, sigComb, satComb], saveName, hdr=hdr, ask=False)
    else:
        wifisIO.writeFits(imgComb, saveName, hdr=hdr, ask=False)
 
    return imgComb, sigComb, satComb, hdr

def auto(folder, rootFolder, saveName, satCounts, nlCoeff, BPM,nChannel=32, nRows=0,rowSplit=1, satSplit=32, nlSplit=32, combSplit=32, crReject=False, bpmCorRng=1, saveAll=True, ignoreBPM=False, skipObsinfo=False, rampNum=None, bpmFile='', satFile='', nlFile='', gain=1., ron=None, logfile=None,obsCoords=None, avgAll=False):
    """
    Routine to set up processing of a set of ramp images to create a single ramp image. Carries out all related routines (non-linearity correction, bad pixel correction, etc.). The location of the observation will be used to determine the type of ramp sequence and the corresponding steps to take.

    Usage: imgCor, sigCor, satFrame, hdr = auto(folder, rootFolder, saveName, satCounts, nlCoeff, BPM,nChannel=32, nRows=0,rowSplit=1, satSplit=32, nlSplit=32, combSplit=32, crReject=False, bpmCorRng=1, saveAll=True, ignoreBPM=False,skipObsinfo=False, rampNum=None, satFile='', nlFile='', bpmFile='',logfile=None, fowler=False, gain=1., ron=None, obsCoords=None, avgAll=False)

    folder - the name of the input folder containing the ramp to process
    rootFolder - the location of the root folder
    saveName - the file name which to save the resulting image and corresponding attributes
    satCounts - the map/image of the saturation levels of each pixel
    nlCoeff - the non-linearity coefficient correction array for non-linearity corrections
    BPM - a bad pixel mask to be used to mark bad pixels
    nChannel - number of read channels used for detector readout
    nRows - number of rows to be averaged for row reference correction (using a moving average)
    rowSplit - number of instances to split the task of row reference correction (a higher value reduces memory limitations at the expense of longer processing times). Must be an integer number of the number of frames in the ramp sequence
    satSplit - number of instances to split the task of carrying out saturation correction (a higher value reduces memory limitations at the expense of longer processing times). Must be integer number of the number of columns in the detector image.
    nlSplit -  number of instances to split the task of non-linearity correction (a higher value reduces memory limitations at the expense of longer processing times). Must be integer number of the number of columns in the detector image.
    combSlit - number of instances to split the task of creating a single ramp image (a higher value reduces memory limitations at the expense of longer processing times). Must be integer number of the number of columns in the detector image.
    crReject - boolean flag to use routine suited to reject cosmic ray events for creating ramp image
    bpmCorRng - the maximum separation between the bad pixel and the nearest good pixel for bad pixel corrections.
    saveAll - a boolean keyword used to specify if all attributes should be saved along with the processed ramp image (saturation info and uncertainties)
    ignoreBPM - a boolean keyword used to indicate if missing a bad pixel mask should show a warning or not
    skipObsinfo - boolean flag to allow skipping of warning/failure if obsinfo.dat file is not present.
    rampNum - an integer indicating which ramp to use if the corresponding observation has multiple ramps present
    satFile - the name/path of the saturation info file to be specified in fits header
    nlFile - the name/path of the non-linearity coefficient file to be specified in fits header
    bpmFile - the name/path of the bad pixel mask file to be specified in fits header
    logfile - file object corresponding to the logfile
    fowler - a boolean keyword to indicate if the observation to be processed is a Fowler sampling ramp
    gain - gain conversion factor needed if and only if RON image is given in units of e- not counts
    ron - readout noise image/map of detector
    obsCoords -  list containing the observatory coordinates [longitude (deg), latitude (deg), altitude (m)]
    avgAll - boolean keyword indicating whether or not to median-average the results if multiple ramps are present in a single observation
    imgComb - the final ramp image providing the flux of the ramp sequence
    sigComb - the estimated uncertainties for each pixel
    satComb - the final saturation info about each pixel
    hdr - an astropy header object containing information about the observation


    """

    #check if file already processed
    
    #check file type    
    #CDS
    if os.path.exists(rootFolder+'/CDSReference/'+folder):
        folderType = '/CDSReference/'

        imgCor, sigmaCor, satFrame, hdr = process(rootFolder+folderType+folder, saveName, satCounts, nlCoeff, BPM,nChannel=nChannel, nRows=nRows,rowSplit=1, satSplit=1, nlSplit=1, combSplit=1, crReject=False, bpmCorRng=bpmCorRng, ignoreBPM=ignoreBPM, saveAll=saveAll, skipObsinfo=skipObsinfo, rampNum=rampNum, bpmFile=bpmFile, satFile=satFile, nlFile=nlFile, fowler=True, gain=gain, ron=ron, logfile=logfile, obsCoords=obsCoords, avgAll=avgAll)
    #Fowler
    elif os.path.exists(rootFolder+'/FSRamp/'+folder):
        folderType = '/FSRamp/'
        
        imgCor, sigmaCor, satFrame, hdr = process(rootFolder+folderType+folder, saveName, satCounts, nlCoeff, BPM,nChannel=nChannel, nRows=nRows,rowSplit=1, satSplit=1, nlSplit=1, combSplit=1, crReject=False, bpmCorRng=bpmCorRng, ignoreBPM=ignoreBPM, saveAll=saveAll, skipObsinfo=skipObsinfo, rampNum=rampNum, bpmFile=bpmFile, satFile=satFile, nlFile=nlFile, fowler=True, gain=gain, ron=ron, logfile=logfile, obsCoords=obsCoords, avgAll=avgAll)

    elif os.path.exists(rootFolder + '/UpTheRamp/'+folder):
        folderType = '/UpTheRamp/'
        imgCor, sigmaCor, satFrame, hdr = process(rootFolder+folderType+folder, saveName, satCounts, nlCoeff, BPM,nChannel=nChannel, nRows=nRows,rowSplit=rowSplit, satSplit=satSplit, nlSplit=nlSplit, combSplit=combSplit, crReject=crReject, bpmCorRng=bpmCorRng, ignoreBPM=ignoreBPM, saveAll=saveAll, skipObsinfo=skipObsinfo, rampNum=rampNum, bpmFile=bpmFile, satFile=satFile, nlFile=nlFile, fowler=False, gain=gain, ron=ron, logfile=logfile, obsCoords=obsCoords, avgAll=avgAll)

    else:
        raise Warning('*** Ramp folder ' + folder + ' does not exist ***')
    
    return imgCor, sigmaCor, satFrame, hdr
