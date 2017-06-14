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

def fromUTR(folder, saveName, satCounts, nlCoeff, BPM,nChannel=32, rowSplit=1, satSplit=32, nlSplit=32, combSplit=32, crReject=False, bpmCorRng=2, saveAll=True):
    """
    """
    
    #Read in data
    print('Reading FITS images into cube')
    data, inttime, hdr = wifisIO.readRampFromFolder(folder)
    
    #convert data to float32 for future processing
    data = data.astype('float32')
            
    #******************************************************************************
    #Correct data for reference pixels
    print("Subtracting reference pixel channel bias")
    refCor.channelCL(data, nChannel)
    print("Subtracting reference pixel row bias")
    refCor.rowCL(data, 4,rowSplit)
        
    satFrame = satInfo.getSatFrameCL(data, satCounts,satSplit)
                
    #******************************************************************************
    #apply non-linearity correction
    print("Correcting for non-linearity")
    
    #find NL coefficient file
    NLCor.applyNLCorCL(data, nlCoeff, nlSplit)
    
    #******************************************************************************
    #Combine data cube into single image
    if (crReject):
        fluxImg = combData.upTheRampCRRejectCL(inttime, data, satFrame, combSplit)
    else:
        fluxImg  = combData.upTheRampCL(inttime, data, satFrame, combSplit)[0]
        
    #free up some memory
    del data
    
    #get uncertainties for each pixel
    sigmaImg = wifisUncertainties.compUTR(inttime, fluxImg, satFrame)
            
    #write image to a file - first extension is the flux, second is uncertainties, third is saturation info

    #****************************************************************************
    #add additional header information here
    headers.addTelInfo(hdr, folder+'/obsinfo.dat')

    #****************************************************************************
    
    if (saveAll):
        wifisIO.writeFits([fluxImg, sigmaImg, satFrame], saveName, hdr=hdr, ask=False)
    else:
        wifisIO.writeFits(fluxImg, saveName, hdr=hdr, ask=False)
        
    # CORRECT BAD PIXELS
    #check for BPM and read, if exists
    if(BPM is not None):
        print('Correcting for bad pixels')

        #assumes BPM is same dimensions as raw image file
        
        fluxImg[BPM.astype(bool)] = np.nan
        fluxImg[satFrame < 2] = np.nan
        #fluxImg[fluxImg < 0] = np.nan
        sigmaImg[~np.isfinite(fluxImg)] = np.nan
    
        #try and correct all pixels, but not the reference pixels
        imgCor = np.empty(fluxImg.shape, dtype = fluxImg.dtype)
        sigmaCor = np.empty(sigmaImg.shape, dtype= sigmaImg.dtype)
    
        imgCor[4:2044,4:2044] = badPixels.corBadPixelsAll(fluxImg[4:2044,4:2044], dispAxis=0, mxRng=bpmCorRng, MP=True) 
        sigmaCor[4:2044, 4:2044]  = badPixels.corBadPixelsAll(sigmaImg[4:2044,4:2044], dispAxis=0, mxRng=bpmCorRng, MP=True, sigma=True)
    else:
        cont = wifisIO.userInput('*** WARNING: No bad pixel mask provided. Do you want to continue? *** (y/n)?')
        if (cont.lower()!='y'):
            raise SystemExit('*** Missing bad pixel mask. Exiting ***')
        else:
            imgCor = fluxImg
            sigmaCor = sigmaImg
   
    return imgCor, sigmaCor, satFrame, hdr


def fromFowler(folder, saveName, satCounts, nlCoeff, BPM,nChannel=32, rowSplit=1, satSplit=32, nlSplit=32, combSplit=32, crReject=False, bpmCorRng=2, saveAll=True):
    """
    """
    
    #Read in data
    print('Reading FITS images into cube')
    data, inttime, hdr = wifisIO.readRampFromFolder(folder)
    
    #convert data to float32 for future processing
    data = data.astype('float32')
            
    #******************************************************************************
    #Correct data for reference pixels
    print("Subtracting reference pixel channel bias")
    refCor.channelCL(data, nChannel)
    print("Subtracting reference pixel row bias")
    refCor.rowCL(data, 4,rowSplit)
        
    satFrame = satInfo.getSatFrameCL(data, satCounts,satSplit)
    
    #******************************************************************************
    #apply non-linearity correction
    print("Correcting for non-linearity")
            
    #find NL coefficient file
    NLCor.applyNLCorCL(data, nlCoeff, nlSplit)
        
    #******************************************************************************
    #Combine data cube into single image
    fluxImg = combData.fowlerSamplingCL(inttime, data, satFrame, combSplit)

    #free up some memory
    del data

    #get uncertainties for each pixel
    sigmaImg = wifisUncertainties.compFowler(inttime, fluxImg, satFrame)
            
    #write image to a file - first extension is the flux, second is uncertainties, third is saturation info

    #****************************************************************************
    #add additional header information here
    headers.addTelInfo(hdr, folder+'/obsinfo.dat')

    #****************************************************************************
    if saveAll:
        wifisIO.writeFits([fluxImg, sigmaImg, satFrame], saveName, hdr=hdr, ask=False)
    else:
        wifisIO.writeFits(fluxImg, saveName, hdr=hdr, ask=False)
    
    # CORRECT BAD PIXELS
    #check for BPM and read, if exists
    if(BPM is not None):
        print('Correcting for bad pixels')

        #assumes BPM is same dimensions as raw image file
        
        fluxImg[BPM.astype(bool)] = np.nan
        fluxImg[satFrame < 2] = np.nan
        #fluxImg[fluxImg < 0] = np.nan
        sigmaImg[~np.isfinite(fluxImg)] = np.nan

        #try and correct all pixels, but not the reference pixels
        imgCor = np.empty(fluxImg.shape, dtype = fluxImg.dtype)
        sigmaCor = np.empty(sigmaImg.shape, dtype= sigmaImg.dtype)
        
        imgCor[4:2044,4:2044] = badPixels.corBadPixelsAll(fluxImg[4:2044,4:2044], dispAxis=0, mxRng=bpmCorRng, MP=True) 
        sigmaCor[4:2044, 4:2044]  = badPixels.corBadPixelsAll(sigmaImg[4:2044,4:2044], dispAxis=0, mxRng=bpmCorRng, MP=True, sigma=True)
    else:
        cont = wifisIO.userInput('*** WARNING: No bad pixel mask provided. Do you want to continue? *** (y/n)?')
        if (cont.lower()!='y'):
            raise SystemExit('*** Missing bad pixel mask. Exiting ***')
        else:
            imgCor = fluxImg
            sigmaCor = sigmaImg
   
    return imgCor, sigmaCor, satFrame, hdr


def auto(folder, rootFolder, saveName, satCounts, nlCoeff, BPM,nChannel=32, rowSplit=1, satSplit=32, nlSplit=32, combSplit=32, crReject=False, bpmCorRng=2, saveAll=True):
    """
    """

    #check if file already processed
    
    #check file type    
    #CDS
    if os.path.exists(rootFolder+'/CDSReference/'+folder):
        folderType = '/CDSReference/'

        imgCor, sigmaCor, satFrame, hdr = fromFowler(rootFolder+folderType+folder, saveName, satCounts, nlCoeff, BPM,nChannel=nChannel, rowSplit=1, satSplit=1, nlSplit=1, combSplit=1, crReject=False, bpmCorRng=bpmCorRng)
    #Fowler
    elif os.path.exists(rootFolder+'/FSRamp/'+folder):
        folderType = '/FSRamp/'
        
        imgCor, sigmaCor, satFrame, hdr = fromFowler(rootFolder+folderType+folder, saveName, satCounts, nlCoeff, BPM,nChannel=nChannel, rowSplit=1, satSplit=1, nlSplit=1, combSplit=1, crReject=False, bpmCorRng=bpmCorRng)

    elif os.path.exists(rootFolder + '/UpTheRamp/'+folder):
        folderType = '/UpTheRamp/'
        imgCor, sigmaCor, satFrame, hdr = fromUTR(rootFolder+folderType+folder, saveName, satCounts, nlCoeff, BPM,nChannel=nChannel, rowSplit=rowSplit, satSplit=satSplit, nlSplit=nlSplit, combSplit=combSplit, crReject=crReject, bpmCorRng=bpmCorRng)

    else:
        raise SystemExit('*** Ramp folder ' + folder + ' does not exist ***')

    return imgCor, sigmaCor, satFrame, hdr
