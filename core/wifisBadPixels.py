"""

A set of functions to determine and correct bad pixels

"""

import numpy as np
import multiprocessing as mp
import wifisIO
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
import colorama
colorama.init()

def corBadPixelsAll(data,dispAxis=0,mxRng=2,MP=True, ncpus=None, sigma=False):
    """
    Used to corrected all bad pixels in a given 2D image, through linear interpolation of the nearest pixels in a given pixel range, along the dispersion axis
    Usage: outData = corBadPixels(data, dispAxis=0, mxRng=2, MP=True, ncpus=None, sigma=False)
    data is the input image (with bad pixels marked as NaNs in the image,
    dispAxis lists the dispersion direction (0 -> along the Y-axis, 1 -> along the X-axis)
    mxRng is the maximum search range to be used to identify good pixels
    MP is a flag to enable multiprocessing
    ncpus determines the number of processes to run, when in multiprocessing mode
    sigma is a flag that specifis if the input image is an image of uncertainties, and if so treats them accordingly during the linear interpolation scheme.
    """

    #run through list of all bad pixels and correct them through linear interpolation along dispersion axis

    if mxRng == 0:
        return data
    else:
        mxRng = int(mxRng)
        
    #create list of bad pixels
    bad = np.where(~np.isfinite(data))

    if (MP == False):
        corrections=[]

        #run separate algorithm on uncertainty data
        if (sigma):
            for i in range(bad[0].shape[0]):
                if (dispAxis==0):
                    #only provide the correction routine with the disperion vector for that particular bad pixel
                    corrections.append(corBadPixelSigma([data[:,bad[1][i]],  bad[0][i], mxRng]))
                else:
                    corrections.append(corBadPixelSigma([data[bad[0][i],:], bad[1][i], mxRng]))
        else:
            for i in range(bad[0].shape[0]):
                #only provide the correction routine with the disperion vector for that particular bad pixel
                if (dispAxis==0):
                    corrections.append(corBadPixel([data[:,bad[1][i]], bad[0][i], mxRng]))
                else:
                    corrections.append(corBadPixel([data[bad[0][i],:], bad[1][i], mxRng]))
    else:
        #setup input list to feed MP routine
        lst = []
        
        for i in range(bad[0].shape[0]):
            if (dispAxis==0):
                lst.append([data[:,bad[1][i]], bad[0][i], mxRng])
            else:
                lst.append([data[bad[0][i],:], bad[1][i], mxRng])

        #run MP routine
        if (ncpus == None):
            ncpus =mp.cpu_count()
        pool = mp.Pool(ncpus)
        if (sigma):
            corrections = pool.map(corBadPixelSigma, lst)
        else:
            corrections = pool.map(corBadPixel, lst)
        pool.close()

    #create new array with corrected pixels
    outData = np.empty(data.shape, dtype=data.dtype)
    np.copyto(outData, data)
    
    for i in range(len(corrections)):
        outData[bad[0][i],bad[1][i]] = corrections[i]

    return outData

def corBadPixel(input):
    """
    Used to correct individual bad pixels by interpolating of the nearest good pixels in the search window, along dispersion axis.
    Usage: corr = corBadPixel(input)
    input is a list containing:
    data - the image containing the bad pixels marked by NaNs
    badPix - the location of the bad pixel
    mxRng - the maximum separation (in pixels) allowed between the bad pixel and the nearest good pixel to be used for correction.
    """

    data = input[0]
    badPix = input[1]
    mxRng = input[2]

    nx = data.shape[0]
    #zero pad current vector
    ytmp = np.zeros(nx+2)
    np.copyto(ytmp[1:-1],data)

    badPix += 1

    if mxRng > 0:
        #determine if any good pixels within range
        xa = badPix
        keepxa = False
        for xa in xrange(badPix-1,badPix-mxRng-1,-1):
            if(np.isfinite(ytmp[xa])):
                keepxa = True
                break

        xb = badPix
        keepxb = False
        for xb in xrange(badPix+1,badPix+mxRng+1,1):
            if (np.isfinite(ytmp[xb])):
                keepxb = True
                break

        #interpolate between values, if useable, else put in an NaN
        if (keepxa and keepxb):
            ia = (xb-badPix)/float(xb-xa)*ytmp[xa]*keepxa
            ib = (badPix-xa)/float(xb-xa)*ytmp[xb]*keepxa
            corr = ia + ib
        else:
            corr = np.nan
    else:
        corr = np.nan
        
    #restore pixx to input value, in case it modifies the input
    badPix -= 1
    
    return corr
    
def corBadPixelSigma(input):
    """
    Used to correct individual bad pixels by interpolating of the nearest good pixels in the search window, along dispersion axis. This function is to be used on images containing uncertainties such that the uncertainties are properly propagated
    Usage: corr = corBadPixelSigma(input)
    input is a list containing:
    data - the uncertainty image containing the bad pixels marked by NaNs
    badPix - the location of the bad pixel
    mxRng - the maximum separation (in pixels) allowed between the bad pixel and the nearest good pixel to be used for correction.
    """
    data = input[0]
    badPix = input[1]
    mxRng = input[2]

    nx = data.shape[0]
    #zero pad current vector
    ytmp = np.zeros(nx+2)
    np.copyto(ytmp[1:-1],data)

    badPix += 1

    #determine if any good pixels within range
    xa = badPix
    keepxa = False
    for xa in xrange(badPix-1,badPix-mxRng-1,-1):
        if(np.isfinite(ytmp[xa])):
            keepxa = True
            break

    xb = badPix
    keepxb = False
    for xb in xrange(badPix+1,badPix+mxRng+1,1):
        if (np.isfinite(ytmp[xb])):
            keepxb = True
            break

    #interpolate between values, if useable, else put in a NaN
    if (keepxa and keepxb):
        ia = (xb-badPix)/float(xb-xa)*ytmp[xa]*keepxa
        ib = (badPix-xa)/float(xb-xa)*ytmp[xb]*keepxa
        corr = np.sqrt(ia**2 + ib**2)
    else:
        corr = np.nan

    #restore pixx to input value, in case it modifies the input
    badPix -= 1
    
    return corr

def getBadPixelsFromNLCoeff(nlCoeff,hdr,saveFile = '',cutoff=1e-5):
    """
    Used to determine bad pixels from a the first two non-linearity correction coefficients.
    Usage: bpm,hdr = getBadPixelsFromNLCoeff(nlCoeff, hdr, saveFile='',cutoff=1e-5)
    bpm is the output bad pixel mask (zeros for good pixels, ones for bad pixels)
    hdr is the updated output astropy hdr object
    nlCoeff is the input array specifying the correction coefficient for each pixel (nY, nX, nCoeffs)
    hdr is an astropy hdr object to be updated
    saveFile is the name of the file name to save the corresponding summary images
    cutoff is the cutoff limit in the normalized PDF to identify outliers. Outliers are identified if the coefficient has a probability of < cutoff or > 1-cutoff.
    """

    print('Determining bad pixels from NL Coeff term1')
    #based on nlCoeff
    term1 = nlCoeff[:,:,0]
    med = np.nanmedian(term1)
    #std = np.nanstd(term1)
    hist = np.histogram(term1.flatten(), bins=1000, range=[0.85,1.15])
    fig=plt.figure()

    plt.hist(term1.flatten(), bins=1000, range=[0.85,1.15])
    cumsum = np.cumsum(hist[0])
    cumsum = cumsum/cumsum.max().astype('float32')*hist[0].max()

    histx = np.linspace(0.85,1.15, num=1000)+(1.15-0.85)/2000.
    plt.plot(histx, cumsum)
    cumsum/=cumsum.max()

    whr = np.where(cumsum <cutoff)[0]
    rng1 = histx[whr[-1]]
    whr = np.where(cumsum >1-cutoff)[0]
    rng2 = histx[whr[0]]
    
    plt.plot([rng1, rng1], [0, hist[0].max()])
    plt.plot([rng2, rng2],[0,hist[0].max()])
    plt.title('First NL Coefficient term')
    plt.savefig(saveFile+'_hist_term1.png',dpi=300)
    plt.close()
    
    bpm = np.zeros(nlCoeff[:,:,0].shape,dtype='uint8')

    bpm[term1>=rng2] = 1
    bpm[term1<=rng1] = 1
    refFrame=np.ones(nlCoeff[:,:,0].shape,dtype=bool)
    refFrame[4:-4,4:-4]=False
    bpm[refFrame] = 0

    #****************************************************************
    print('Determining bad pixels from NL Coeff term2')

    term2 = nlCoeff[:,:,1]
    med = np.nanmedian(term2)
    #std = np.nanstd(term2)
    hist = np.histogram(term2.flatten(), bins=1000, range=[-1e-5,1e-5])
    fig=plt.figure()
    plt.hist(term2.flatten(), bins=1000, range=[-1e-5,1e-5])
    cumsum = np.cumsum(hist[0])
    cumsum = cumsum/cumsum.max().astype('float32')*hist[0].max()

    histx = np.linspace(-1e-5,1e-5, num=1000)+(2e-5)/2000.
    plt.plot(histx, cumsum)
    cumsum/=cumsum.max()

    whr = np.where(cumsum < cutoff)[0]
    rng3 = histx[whr[-1]]
    whr = np.where(cumsum > 1-cutoff)[0]
    rng4 = histx[whr[0]]

    plt.plot([rng3, rng3], [0, hist[0].max()])
    plt.plot([rng4, rng4],[0,hist[0].max()])
    plt.title('Second NL Coefficient term')
    plt.savefig(saveFile+'_hist_term2.png',dpi=300)
    plt.close()

    hdr.add_history('Determined bad pix from first two terms of non-linearity coefficients')
    hdr.add_history('with normalized probability density of ' + str(cutoff) + ' or worse')
    hdr.add_history('Term 1 good range: ' + str(rng1) + ' - ' + str(rng2))
    hdr.add_history('Term 2 good range: ' + str(rng3) + ' - ' + str(rng4))

    hdr.set('QC_NBADL',len(np.where(bpm ==1)[0]),'Number of bad pixels based on non-linearity')
    hdr.set('QC_NBAD',len(np.where(bpm ==1)[0]),'Number of bad pixels in total')

    return bpm, hdr

def getBadPixelsFromDark(dark,hdr,darkFile='',saveFile = '',cutoff=1e-5, BPM=None):
    """
    Used to determine bad pixels from a dark image.
    Usage: BPM,hdr = getBadPixelsFromDark(dark, hdr, darkFile='',saveFile='',cutoff=1e-5, BPM=None)
    BPM is the output bad pixel mask (zeros for good pixels, ones for bad pixels)
    hdr is the output updated astropy hdr object
    dark is the input dark image used to determine the bad pixels
    hdr is an astropy hdr object to be updated
    darkFile is the name/path of the dark to be specified in the hdr object
    saveFile is the name of the file name to save the corresponding summary images
    cutoff is the the cutoff limit in the normalized PDF to identify outliers. Outliers are identified if the coefficient has a probability of < cutoff or > 1-cutoff.
    BPM is an optional bad pixel mask image to be updated. I.e. if provided, the bad pixels are a combination of those in BPM and those found in this function.
    """

    print('Determining bad pixels from dark frame')
    #based on nlCoeff
    med = np.nanmedian(dark)
    hist = np.histogram(dark.flatten(), bins=1000, range=[-1,1])
    fig=plt.figure()

    plt.hist(dark.flatten(), bins=1000, range=[-1,1])
    cumsum = np.cumsum(hist[0])
    cumsum = cumsum/cumsum.max().astype('float32')*hist[0].max()

    histx = np.linspace(-1,1, num=1000)+1/1000.
    plt.plot(histx, cumsum)
    cumsum/=cumsum.max()

    whr = np.where(cumsum <=cutoff)[0]
    if len(whr) == 0:
        print(colorama.Fore.RED+'*** WARNING: NO PIXELS WITH DARK CURRENT <1 CNT/S. SKIPPING BAD PIXEL IDENTIFICATION DUE TO HIGH DARK CURRENT ***'+colorama.Style.RESET_ALL)
        return BPM,hdr
    else:
        rng1 = histx[whr[-1]]
        whr = np.where(cumsum >=1-cutoff)[0]
        rng2 = histx[whr[0]]
    
        plt.plot([rng2, rng2],[0,hist[0].max()])
        plt.title('Histogram of dark current')
        plt.xlabel('Dark value [counts/s]')
        plt.savefig(saveFile+'_hist.png',dpi=300)
        plt.close()

        if BPM is None:
            BPM = np.zeros(dark.shape,dtype='uint8')

        bpmTmp = np.zeros(dark.shape,dtype='uint8')
        BPM[dark>rng2] = 1
        bpmTmp[dark>rng2] = 1
    
        refFrame=np.ones(dark.shape,dtype=bool)
        refFrame[4:-4,4:-4]=False
        BPM[refFrame] = 0
        bpmTmp[refFrame] = 0
    
        hdr.add_history('Determined bad pix from dark frame:')
        hdr.add_history(darkFile)
        hdr.add_history('with normalized probability density of worse than ' + str(cutoff))
        hdr.add_history('Dark level cutoff: ' + str(rng2))
        
        hdr.set('QC_NBADD',len(np.where(bpmTmp ==1)[0]),'Number of bad pixels based on dark current')
        hdr.set('QC_NBAD',len(np.where(BPM ==1)[0]),'Total number of bad pixels')
        

    return BPM, hdr

def getBadPixelsFromRON(ron,hdr,ronFile='',saveFile = '',cutoff=1e-5, BPM=None):
    """
    Used to determine bad pixels from an image containing the read-out noise for each pixel.
    Usage: bpm, hdr = getBadPixelsFromRON(ron, hdr, ronFile='',saveFile='',cutoff=1e-5, BPM=None)
    BPM is the output bad pixel mask (zeros for good pixels, ones for bad pixels)
    hdr is the output updated astropy hdr object
    ron is the input ron image used to determine the bad pixels
    hdr is an astropy hdr object to be updated
    ronFile is the name/path of the ron image to be specified in the hdr object
    saveFile is the name of the file name to save the corresponding summary images
    cutoff is the cutoff limit in the normalized PDF to identify outliers. Outliers are identified if the coefficient has a probability of < cutoff or > 1-cutoff.
    BPM is an optional bad pixel mask image to be updated. I.e. if provided, the bad pixels are a combination of those in BPM and those found in this function.
    """

    print('Determining bad pixels from RON frame')
    #based on nlCoeff
    med = np.nanmedian(ron)

    interval =  ZScaleInterval()
    range = np.asarray(interval.get_limits(ron))
    range[0]=0
    range[1]*=2
    
    hist = np.histogram(ron.flatten(), bins=1000, range=range)
    fig=plt.figure()

    plt.hist(ron.flatten(), bins=1000, range=range)
    cumsum = np.cumsum(hist[0])
    cumsum = cumsum/cumsum.max().astype('float32')*hist[0].max()

    histx = np.linspace(0,range[1], num=1000)+1/1000.
    plt.plot(histx, cumsum)
    cumsum/=cumsum.max()

    whr = np.where(cumsum >=1-cutoff)[0]
    rng2 = histx[whr[0]]
    
    plt.plot([rng2, rng2],[0,hist[0].max()])
    plt.title('Histogram of RON')
    plt.xlabel('RON value [counts]')
    plt.savefig(saveFile+'_hist.png',dpi=300)
    plt.close()

    if BPM is None:
        BPM = np.zeros(ron.shape,dtype='uint8')

    bpmTmp = np.zeros(ron.shape,dtype='uint8')
    BPM[ron>rng2] = 1
    bpmTmp[ron>rng2] = 1
    
    refFrame=np.ones(ron.shape,dtype=bool)
    refFrame[4:-4,4:-4]=False
    BPM[refFrame] = 0
    bpmTmp[refFrame] = 0
    
    hdr.add_history('Determined bad pix from RON frame:')
    hdr.add_history(ronFile)
    hdr.add_history('with normalized probability density of worse than ' + str(cutoff))
    hdr.add_history('RON level cutoff: ' + str(rng2))

    hdr.set('QC_NBADR',len(np.where(bpmTmp ==1)[0]),'Number of bad pixels based on RON')
    hdr.set('QC_NBAD',len(np.where(BPM ==1)[0]),'Total number of bad pixels')


    return BPM, hdr
