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

def process(folder, saveName, satCounts, nlCoeff, BPM,nChannel=32, nRows=0,rowSplit=1, satSplit=32, nlSplit=32, combSplit=32, crReject=False, bpmCorRng=2, saveAll=True, ignoreBPM=False,skipObsinfo=False, rampNum=None, satFile='', nlFile='', bpmFile='',logfile=None, fowler=False, gain=1., ron=1., obsCoords=None):
    """
    """
    
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
    print("Subtracting reference pixel channel bias")
    refCor.channelCL(data, nChannel)
    hdr.add_history('Channel reference pixel corrections applied using '+ str(nChannel) +' channels')
    if logfile is not None:
        logfile.write('Subtracted reference pixel channel bias using ' + str(nChannel) + ' channels\n')

    print("Subtracting reference pixel row bias")
    if nRows > 0:
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
    print("Correcting for non-linearity")
    
    #find NL coefficient file
    if nlCoeff is not None:
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
        sigmaImg = wifisUncertainties.compFowler(inttime, fluxImg, satFrame)

    else:
        if (crReject):
            fluxImg = combData.upTheRampCRRejectCL(inttime, data, satFrame, combSplit)[0]
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
        sigmaImg = wifisUncertainties.compUTR(inttime, fluxImg, satFrame, ron=ron, gain=gain)
               
    #write image to a file - first extension is the flux, second is uncertainties, third is saturation info

    #****************************************************************************
    #add additional header information here
    if not skipObsinfo:
        headers.addTelInfo(hdr, folder+'/obsinfo.dat',logfile=logfile, obsCoords=obsCoords)
    hdr.add_comment('File contains flux, sigma, and sat info as multi-extensions')
    #****************************************************************************
    
    if saveAll:
        wifisIO.writeFits([fluxImg, sigmaImg, satFrame], saveName, hdr=hdr, ask=False)
    else:
        wifisIO.writeFits(fluxImg, saveName, hdr=hdr, ask=False)
        
    # CORRECT BAD PIXELS
    #check for BPM and read, if exists
    if(BPM is not None and bpmCorRng > 0):
        print('Correcting for bad pixels')

        #assumes BPM is same dimensions as raw image file
        
        fluxImg[BPM.astype(bool)] = np.nan
        fluxImg[satFrame < 2] = np.nan
        #fluxImg[fluxImg < 0] = np.nan
        sigmaImg[~np.isfinite(fluxImg)] = np.nan
    
        #try and correct all pixels, but not the reference pixels
        imgCor = np.empty(fluxImg.shape, dtype = fluxImg.dtype)
        sigmaCor = np.empty(sigmaImg.shape, dtype= sigmaImg.dtype)
    
        imgCor[4:-4,4:-4] = badPixels.corBadPixelsAll(fluxImg[4:-4,4:-4], dispAxis=0, mxRng=bpmCorRng, MP=True) 
        sigmaCor[4:-4, 4:-4]  = badPixels.corBadPixelsAll(sigmaImg[4:-4,4:-4], dispAxis=0, mxRng=bpmCorRng, MP=True, sigma=True)

        hdr.add_history('Bad pixel mask used:')
        hdr.add_history(bpmFile)
        hdr.add_history('Bad pixels corrected using nearest pixels within ' + str(bpmCorRng) + ' pixel range')
        if logfile is not None:
            logfile.write('Bad pixels corrected using nearest pixels within ' + str(bpmCorRng) + ' pixel range\n')
    elif(BPM is None):
        if not ignoreBPM:
            cont = wifisIO.userInput('*** WARNING: No bad pixel mask provided. Do you want to continue? *** (y/n)?')
            if (cont.lower()!='y'):
                raise Warning('*** Missing bad pixel mask ***')
                
        imgCor = fluxImg
        sigmaCor = sigmaImg
    elif(bpmCorRng==0):
        imgCor = fluxImg
        sigmaCor = sigmaImg
        
    return imgCor, sigmaCor, satFrame, hdr


def auto(folder, rootFolder, saveName, satCounts, nlCoeff, BPM,nChannel=32, nRows=0,rowSplit=1, satSplit=32, nlSplit=32, combSplit=32, crReject=False, bpmCorRng=2, saveAll=True, ignoreBPM=False, skipObsinfo=False, rampNum=None, bpmFile='', satFile='', nlFile='', gain=1., ron=1., logfile=None,obsCoords=None):
    """
    """

    #check if file already processed
    
    #check file type    
    #CDS
    if os.path.exists(rootFolder+'/CDSReference/'+folder):
        folderType = '/CDSReference/'

        imgCor, sigmaCor, satFrame, hdr = process(rootFolder+folderType+folder, saveName, satCounts, nlCoeff, BPM,nChannel=nChannel, nRows=nRows,rowSplit=1, satSplit=1, nlSplit=1, combSplit=1, crReject=False, bpmCorRng=bpmCorRng, ignoreBPM=ignoreBPM, saveAll=saveAll, skipObsinfo=skipObsinfo, rampNum=rampNum, bpmFile=bpmFile, satFile=satFile, nlFile=nlFile, fowler=True, gain=gain, ron=ron, logfile=logfile, obsCoords=obsCoords)
    #Fowler
    elif os.path.exists(rootFolder+'/FSRamp/'+folder):
        folderType = '/FSRamp/'
        
        imgCor, sigmaCor, satFrame, hdr = process(rootFolder+folderType+folder, saveName, satCounts, nlCoeff, BPM,nChannel=nChannel, nRows=nRows,rowSplit=1, satSplit=1, nlSplit=1, combSplit=1, crReject=False, bpmCorRng=bpmCorRng, ignoreBPM=ignoreBPM, saveAll=saveAll, skipObsinfo=skipObsinfo, rampNum=rampNum, bpmFile=bpmFile, satFile=satFile, nlFile=nlFile, fowler=True, gain=gain, ron=ron, logfile=logfile, obsCoords=obsCoords)

    elif os.path.exists(rootFolder + '/UpTheRamp/'+folder):
        folderType = '/UpTheRamp/'
        imgCor, sigmaCor, satFrame, hdr = process(rootFolder+folderType+folder, saveName, satCounts, nlCoeff, BPM,nChannel=nChannel, nRows=nRows,rowSplit=rowSplit, satSplit=satSplit, nlSplit=nlSplit, combSplit=combSplit, crReject=crReject, bpmCorRng=bpmCorRng, ignoreBPM=ignoreBPM, saveAll=saveAll, skipObsinfo=skipObsinfo, rampNum=rampNum, bpmFile=bpmFile, satFile=satFile, nlFile=nlFile, fowler=False, gain=gain, ron=ron, logfile=logfile, obsCoords=obsCoords)

    else:
        raise Warning('*** Ramp folder ' + folder + ' does not exist ***')
    
    return imgCor, sigmaCor, satFrame, hdr
