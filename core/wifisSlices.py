"""

Tools used to separate slitlets 

"""

import numpy as np
import multiprocessing as mp
import wifisIO
import astropy.convolution as conv
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift
import warnings

def limFit1(input):
    """
    Used to determine slice edges for a single column, assuming dispersion axis is aligned along rows.
    Usage: limMeas = limFit1(input)
    input is a list, with the first item being the spatial spectrum and the second item being the window range for finding the limits
    limMeas is a list the limits of the slices
    """
    
    #from third commissioning run, better alignment
    centGuess = [25, 138, 252, 366, 479, 594, 706, 820, 935, 1048, 1163, 1278, 1390, 1506, 1620, 1735, 1849, 1965, 2044]
    y = input[0]
    nRng = input[1]
   
    winRng = np.arange(nRng)-nRng/2
    limMeas = []

    #get the edges of the middle slices
    for cent in centGuess:
        xtmp = (winRng+cent).astype('int')
        xtmp = xtmp[np.where(xtmp >=0)[0]]
        xtmp = xtmp[np.where(xtmp < len(y))[0]]
        ytmp = y[xtmp]
        d1 = np.gradient(ytmp)
        d2 = np.gradient(d1)

        limMeas.append(xtmp[np.argmax(d2)])

    #may be able to remove the following code
    #now do the last slice edge
    #cent = 4095
    #xtmp = winRng+cent
    #xtmp = xtmp[np.where(xtmp >=0)[0]]
    #xtmp = xtmp[np.where(xtmp < len(y))[0]]
    #ytmp = y[xtmp]
    #d1 = np.gradient(ytmp)
    #d2 = np.gradient(d1)
    #limMeas.append(xtmp[np.argmin(d1)])

    return limMeas

def findLimits(data, dispAxis=0, winRng=51, imgSmth=5, limSmth=10, ncpus=None, rmRef=False):
    """
    Used to determine slice limits from a full flat-field image
    Usage: limits = findLimits(data, dispAxis=,winRng=,imgSmth=,limSmth= )
    data is the input flat field image from which the limits will be found
    dispAxis is keyword specifying the dispersion axis (0-> along the y-axis, 1-> along the x-axis)
    winRng is a keyword specifying the number of pixels to use when finding the slice edge limits
    imgSmth is a keyword specifying the Gaussian width of the smoothing kernel for finding the slice-edge limits
    limSmth is a keyword specigying the Gaussian width of the smoothing kernel for smoothing the found limits
    ncpus is a keyword indicating the number of processes to spawn
    rmRef is a keyword indicating if the image includes the reference pixels and if they should be removed from the limits
    returns an array containing the pixel limits for each slice edge along the dispersion axis
    """

    if (dispAxis == 0):
        dTmp = data.T
    else:
        dTmp = data

    #first cutoff limits along the dispersion direction
    if (rmRef):
        dTmp = dTmp[:, 4:dTmp.shape[1]-4]

    nx = dTmp.shape[0]
    ny = dTmp.shape[1]

    #go through and identify slice limits
    inpLst = []

    #create kernel for smoothing along the spatial direction
    gKern = conv.Gaussian1DKernel(stddev=imgSmth) 

    #next smooth the image and add to input list
    for i in range(ny):
        y = conv.convolve(dTmp[:,i],gKern, boundary='extend', normalize_kernel=True)
        inpLst.append([y, winRng])

    #setup and run the MP code for finding the limits
    if (ncpus == None):
        ncpus =mp.cpu_count()
    pool = mp.Pool(ncpus)
    result = pool.map(limFit1, inpLst)
    pool.close()

    result = np.array(result).T

    #now smooth limits to avoid issues (maybe decide to fit polynomial instead)
    gKern = conv.Gaussian1DKernel(stddev=limSmth)
    limSmth = np.zeros(result.shape)

    for i in range(result.shape[0]):
        y = conv.convolve(result[i,:], gKern, boundary='extend', normalize_kernel=True)
        limSmth[i,:] = y

    #now go through and shift all limits along the spatial direction to account for reference removed image
    if (rmRef):
        limSmth = np.clip(limSmth-4,0, dTmp.shape[1]-1)

    #lastly round all limits to nearest integer to avoid potential pixel differences
    limSmth = np.round(limSmth).astype(int)
    
    return limSmth

#def extSlices(data, limits, dispAxis=0):
#    """
#    Extract a list of slices (sub-images) from the given image.
#    Usage: slices = extSlices(data, limits, dispAxis=)
#    data is the input data image from which the slices will be extracted
#    limits is an array specifying the slice-edge limits of each slice
#    dispAxis is a keyword specifying the dispersion direction (0-> along the y-axis, 1-> along the x-axis)
#    """
#
#    if (dispAxis == 0):
#        dTmp = data.T
#    else:
#        dTmp = data
#
#    slices = []
#    n=limits.shape[1]
#    nSlices = limits.shape[0]
#    
#    for i in range(0,nSlices-1):
#        mn = np.floor(np.min(limits[i,:])).astype('int')
#        mx = np.ceil(np.max(limits[i+1,:])).astype('int')
#
#        slice = np.empty((mx-mn,n), dtype=data.dtype)
#        slice[:] = np.nan
#        
#        for j in range(mn,mx):
#            keep = np.ones(n, dtype=bool)
#            whr = np.where(np.floor(limits[i,:]) > j)[0]
#            keep[whr] = False
#            whr = np.where(np.ceil(limits[i+1,:]) < j)[0]
#            keep[whr] = False
#            slice[j-mn,keep] = dTmp[j,keep] 
#  
#        slices.append(slice)
#    return slices


def getResponseFuncPoly(input):
    """
    NOT CURRENTLY USED FOR ANYTHING
    """

    slc = input[0]
    nOrd = input[1]

    pInit = models.Polynomial2D(degree=nOrd)
    fitP = fitting.LinearLSQFitter()
    x,y = np.mgrid[:slc.shape[0], :slc.shape[1]]

    whr = np.where(~np.isfinite(slc))
    p = fitP(pInit,x,y,slc)
    
    return p.parameters

def getResponse2D(input):
    """
    Returns a possibly smoothed and normalized response function for the provided slice
    Usage: norm = getResponse2D(input)
    input is a list containing the image slice, the width of the Gaussian kernel to use for smoothing, and the cutoff for which the normalized response function is just set to 1.
    """

    slice = input[0]
    sigma = input[1]
    cutoff = input[2]
    nrmValue = input[3]
    
    if (sigma>0):
        gKern = conv.Gaussian2DKernel(stddev=sigma) 
        sliceSmth=conv.convolve(slice,gKern,boundary='extend',normalize_kernel=True)
    else:
        sliceSmth = slice
    
    norm = sliceSmth/nrmValue
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        norm[norm<cutoff] = np.nan
    #norm[np.isfinite(norm)][norm[np.isfinite(norm)]<cutoff] = np.nan
    #norm[np.isfinite(norm)<cutoff]=np.nan
    norm[~np.isfinite(norm)]=np.nan

    return norm

def getResponseAll(flatSlices, sigma, cutoff, MP=True, ncpus=None):
    """
    Returns a possibly smoothed and normalized response function for all slices in the provided list
    Usage: result = getResponseAll(slices, sigma, cutoff, MP=)
    slices is a list of image slices from which to derive the response function
    sigma is the Gaussian width of the kernel to use for smoothing the input data
    cutoff is a value for which all pixels with normalized values less than this value are set to 1.
    MP is a keyword used to enable multiprocessing routines that may improve performance
    """

    #first determine the normalization weight, based on the maximum median value along each slice
    medSlice = getMedLevelAll(flatSlices, MP=True, ncpus=None)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        nrmValue = np.nanmax(medSlice)
        
    #multiprocessing may improve performance
    if (MP):
        #set up the input list

        lst = []
        for s in flatSlices:
            lst.append([s, sigma, cutoff, nrmValue])

        #setup and run the MP code for finding the limits
        if (ncpus == None):
            ncpus =mp.cpu_count()
        pool = mp.Pool(ncpus)
        result = pool.map(getResponse2D, lst)
        pool.close()
    else:
        result=[]
        for s in slices:
            result.append(getResponse2D([s,sigma, cutoff,nrmValue]))
            
    return result

def getPoly2D(input):
    """
    NOT CURRENTLY USED FOR ANYTHING
    """
    
    params = input[0]
    degree = input[1]
    x = input[2]
    y = input[3]

    #initialize array
    z = np.zeros(x.shape)

    z += params[0]
    
    cntr = 1
    #terms for Sum x^n
    for i in range(1,degree+1):
        z += params[cntr]*x**np.float(i)
        #print('x^'+str(i))
        cntr+=1

    #terms for Sum y^n
    for i in range(1,degree+1):
        z+=params[cntr]*y**np.float(i)
        #print('y^'+str(i))
        cntr+=1

    #if (cntr < len(params)):
    #terms for Sum x*y^n
    for i in range(1,degree):
        z+=params[cntr]*x*y**np.float(i)
        #print('x*y^'+str(i))
        cntr+=1

    #if (cntr < len(params)):
    for i in range(2,degree):
        z+=params[cntr]*y*x**np.float(i)
        #print('y*x^'+str(i))
        cntr+=1

    z /= np.max(z)

    #z[np.where(z<=0)]=1.
    return z

def ffCorrectAll(slices, response, MP=False, ncpus=None):
    """
    Returns flat-field corrected image slices determined from the provided data
    Usage: result = ffCorrectAll(slices, response)
    slices is a list of image slices that you wish to flat-field
    response is the normalized flat-field image responses functions to use for flat-fielding
    MP is a keyword used to enable multiprocessing routines that may improve performance  
    """

    #only use MP if many slices. Serial version (non-MP) is faster for WIFIS
    #need to confirm on better CPU
    
    if (MP):
        
        #set up the input list
        lst = []
    
        for i in range(len(slices)):
            lst.append([slices[i], response[i]])
  
        #setup and run the MP code for finding the limits
        if (ncpus == None):
            ncpus =mp.cpu_count()
        pool = mp.Pool(ncpus)
        result = pool.map(ffCorrectSlice,lst)
        pool.close()
    else:
        result=[]

        for i in range(len(slices)):
            whr = np.where(response[i] != 0)
            tmp = np.empty(slices[i].shape, dtype=slices[i].dtype)
            tmp[:] = np.nan
            tmp[whr[0],whr[1]] = slices[i][whr[0],whr[1]]/response[i][whr[0],whr[1]]
            result.append(tmp)
        
    return result

def ffCorrectSlice(input):
    """
    Returns flat-field corrected image determined from the provided data
    Usage: result = ffCorrectSlice(input)
    input is a list containing the image slice and the normalized flat-field image
    """

    slc = input[0]
    response = input[1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        #whr = np.where(response != 0)
        #tmp = np.empty(slices.shape, dtype=slices.dtype)
        #tmp[:] = np.nan
        #tmp[whr[0],whr[1]] = slice=[whr[0],whr[1]]/response[whr[0],whr[1]]
        result = slc/response
        result[~np.isfinite] = np.nan

    return result

def getTrimLimsAll(flatSlices, threshold=0.1, plot=False,MP=False, ncpus=None):
    """
    Routine to find the trim limits for a distortion corrected image.
    Usage: lims = getTrimLimsAll(flatSlices, threshold=0.1, plot=False, MP=False, ncpus=None)
    flatSlices is a list of distortion corrected flat image slices
    threshold is an optional keyword to set the threshold at which the edge limits are determine, where threshold is the fraction of the maximum pixel in the image. A value of None allows the code to determine this cutoff itself.
    plot is a keyword used to plot the results for debugging
    MP is a keyword used to enable/disable multiprocessing (MP)
    ncpus is a keyword that is used to set the number of simultaneously running processes in MP mode
    lims is a list of the returned limits
    """
    if (MP):
        lst = []
        for f in flatSlices:
            lst.append([f,threshold,plot])

        if (ncpus==None):
            ncpus =mp.cpu_count()
        pool = mp.Pool(ncpus)
        lims = pool.map(getTrimLimsSlice, lst)
        pool.close()
        
    else:
    
        lims = []
        for f in flatSlices:
            
            lims.append(getTrimLimsSlice([f,threshold,plot]))
            
    return lims

def getTrimLimsSlice(input):
    """
    Task used to find the trim limits for a single distortion corrected image slice.
    Usage: [y1,y2] = getTrimLimSlice(input)
    where input is a list containing:
    slc - the distortion corrected flatfield slice image
    threshold - the threshold value from where to set the limits, or a None can be provided, which allows the code to determine this threshold itself
    plot - a boolean value used to plot the results, for debugging purposes.
    [y1, y2] is a list of the upper and lower limits.
    """

    slc = input[0]
    threshold = input[1]
    plot = input[2]
    
    #if no threshold is given, use derivatives to identify limits
    #else, use given threshold

    #first, replace all NaN by zero to allow summation
    #slc[np.where(np.isnan(slc))] = 0
    
    if threshold is None:
        #use gradient to define cutoff
        
        #work on axis 1, assumes that wavetrim already took care of other axis
        ytmp = np.nansum(slc, axis=1)
        n = ytmp.shape[0]
        
        gKern = conv.Gaussian1DKernel(stddev=1) #needs to be optimized, possibly read in
        y = conv.convolve(ytmp,gKern, boundary='extend', normalize_kernel=True)
        d1 = np.abs(np.gradient(y))
        #whr = np.where(y > 0.5*np.max(y))[0]
        #y1 = np.argmax(d1[0:whr[0]])
        #y2 = np.argmax(d1[whr[-1]:])+whr[-1]
        y1 = np.nanargmax(d1[0:n/2])
        y2 = np.nanargmax(d1[n/2:])+n/2
        ysmth = ytmp
    else:
        ytmp = np.nanmedian(slc, axis=1)

        #smooth spectrum first to avoid hot/cold pixels
        gKern = conv.Gaussian1DKernel(10)
        ysmth = conv.convolve(ytmp, gKern, boundary='extend', normalize_kernel=True)
        yout = ytmp/np.nanmax(ysmth)
       
        whr = np.where(yout >= threshold)[0]
        y1 = whr[0]
        y2 = whr[-1]

    if(plot):
        fig = plt.figure()
        plt.plot(ytmp)
        plt.plot(ysmth, '--')
        plt.plot([y1,y1],[np.nanmin(ytmp),np.nanmax(ytmp)],'r--')
        plt.plot([y2,y2],[np.nanmin(ytmp),np.nanmax(ytmp)],'r--')
        plt.show()
        
    return [y1, y2]

def trimSlice(input):
    """
    Routine to trim a given slice image.
    Usage out = trimSlice(input), where input is a list including:
    slc - the slice image which needs to be trimmed
    lims - a list of the upper and lower trim limits
    out is the returned imaged trimmed to smaller size
    """

    slc = input[0]
    lims = input[1]
    ylim = [lims[0], lims[1]]

    ny = np.abs(ylim[0]-ylim[1])+1
    
    #create output array

    out = np.empty((ny,slc.shape[1]), dtype=slc.dtype)
    out[:] = np.nan
    np.copyto(out, slc[ylim[0]:ylim[1]+1,:])

    return out

def trimSliceAll(extSlices, limits, MP=False, ncpus=None):
    """
    Routine to trim all slices in a provided list of slices
    Usage outSlices = trimSliceAll(extSlices, limits, MP=False, ncpus=None)
    extSlices is the list of input slices that you want to trim
    limits is a list of the trim limits to be used to trim each slice
    MP is a keyword used to determine if multiprocessing should be used
    ncpus is a keyword to control the maximum number of simultaneously running processes, when in MP mode
    outSlices is the returned list of trimmed slices 

    """

    if (MP):
        if (ncpus == None):
            ncpus =mp.cpu_count()
        pool = mp.Pool(ncpus)

        #build input list
        lst = []
        for i in range(len(extSlices)):
            lst.append([extSlices[i],limits[i]])
            
        outSlices = pool.map(trimSlice, lst)
        pool.close()
        
    else:
    
        outSlices = []

        for i in range(len(extSlices)):
            outSlices.append(trimSlice([extSlices[i],limits[i]]))
            
    return outSlices

def extSlices(data, limits,shft=0, dispAxis=0):
    """
    Extract a list of slices (sub-images) from the given image based on relative slice limits
    Usage: slices = extSlices(data, limits, shft,dispAxis=)
    data is the input data image from which the slices will be extracted
    limits is an array specifying the slice-edge limits of each slice
    shft is the shift needed to apply to the limits to match the current image
    dispAxis is a keyword specifying the dispersion direction (0-> along the y-axis, 1-> along the x-axis)
    """

    #modify dispersion direction to fit with routine
    if (dispAxis == 0):
        dTmp = data.T
    else:
        dTmp = data

    #compute the relative limits based on the provided shift
    limNew = np.clip(limits + shft, 0, dTmp.shape[0]-1)
    
    #initialize slice list
    slices = []
    n=limits.shape[1]
    nSlices = limits.shape[0]

    if (np.issubdtype(data.dtype,np.integer)):
        dtype = 'float32'
    else:
        dtype = data.dtype
        
    for i in range(0,nSlices-1):
        #initialize output slices
        mnOut = np.floor(np.min(limits[i,:])).astype('int')
        mxOut = np.ceil(np.max(limits[i+1,:])).astype('int')
        slice = np.empty((mxOut-mnOut,n), dtype=dtype)
        slice[:] = np.nan

        #now go through input image and copy into output slices
        mnIn = np.clip(int(mnOut + shft),0, dTmp.shape[0]-1)
        mxIn= np.clip(int(mxOut + shft), 0, dTmp.shape[0]-1)

        for j in range(mnIn,mxIn):
            keep = np.ones(n, dtype=bool)
            whr = np.where(np.floor(limNew[i,:]) > j)[0]
            keep[whr] = False
            whr = np.where(np.ceil(limNew[i+1,:]) < j)[0]
            keep[whr] = False
            slice[j-mnIn,keep] = dTmp[j,keep] 
  
        slices.append(slice)
    return slices

def polyFitLimits(limits, degree=2,constRegion=None, sigmaClipRounds=0):
    """
    constRegion are limits to constrain the fit between two cutoff points. Useful for Hband.
    """

    polyLimits = []

    x = np.arange(limits.shape[1])
    for i in range(limits.shape[0]):
        y = limits[i,:]
        if constRegion is not None:
            xfit = x[constRegion[0]:constRegion[1]]
            yfit = y[constRegion[0]:constRegion[1]]
        else:
            xfit = x
            yfit = y

        polyCoef = np.polyfit(xfit,yfit, degree)
        poly = np.poly1d(polyCoef)

        if sigmaClipRounds > 0:
            for rnd in range(sigmaClipRounds):
                ypoly = poly(xfit)
                diff = np.abs(ypoly - yfit)
                med = np.nanmedian(yfit-ypoly)
                std = np.nanstd(yfit-ypoly)
                if (std > 1e-3):
                    whr = np.where(diff <= med + std)[0]
                    xfitRnd = xfit[whr]
                    yfitRnd = yfit[whr]
                    pcoef = np.polyfit(xfitRnd, yfitRnd,degree)
                    poly = np.poly1d(pcoef)
               
        polyFit = poly(x)
        polyFit = np.clip(polyFit, 0, x.shape[0]-1)
        polyLimits.append(np.round(polyFit).astype(int))
    return np.asarray(polyLimits)
        
def medSmoothSlices(extSlices, nPix, MP=True, ncpus=None):
    """
    """
    
    
    if (MP):
        #set up input list
        lst = []

        for slc in extSlices:
            lst.append([slc, nPix])
            
        if (ncpus == None):
            ncpus = mp.cpu_count()
        pool = mp.Pool(ncpus)
        result = pool.map(medSmoothSlice,lst)
        pool.close()
    else:
        result = []
        for slc in extSlices:
            result.append(medSmoothSlice(slc, nPix))

    return result

def medSmoothSlice(input):
    """
    """

    slc = input[0]
    nPix = input[1]
    win = np.arange(nPix) - int(nPix/2)

    out = np.empty(slc.shape)
    for i in range(slc.shape[0]):
        xrng = win + i
        xrng = xrng[np.logical_and(xrng>0,xrng<slc.shape[0])]

        out[i,:] = np.nanmedian(slc[xrng,:], axis=0)

    return out

def getMedLevelAll(extSlices, MP=True, ncpus=None):
    """
    """
    
    
    if (MP):
        #set up input list
        lst = []

        if (ncpus == None):
            ncpus = mp.cpu_count()
        pool = mp.Pool(ncpus)
        result = pool.map(getMedLevelSlice,extSlices)
        pool.close()
    else:
        result = []
        for slc in extSlices:
            result.append(getMedLevelSlice(slc))

    return result

def getMedLevelSlice(slc):
    """
    """

    out = np.empty(slc.shape[1])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        for i in range(slc.shape[1]):
            y = slc[:,i]
            out[i] = np.nanmedian(y)

    return out

def getResponse2DMed(input):
    """
    Returns a possibly smoothed and normalized response function for the provided slice
    Usage: norm = getResponse2D(input)
    input is a list containing the image slice, the width of the Gaussian kernel to use for smoothing, and the cutoff for which the normalized response function is just set to 1.
    """

    slc = input[0]
    sigma = input[1]
    cutoff = input[2]
    nrmValue = input[3]
    winRng2 = int(np.round(sigma/2.))
    x = np.arange(slc.shape[1])
    y = np.arange(slc.shape[0])
    
    if (sigma>0):
        #use a moving window to compute median average in box about pixel
        sliceSmth = np.empty(slc.shape, dtype=slc.dtype)
        sliceSmth[:] = np.nan
        
        for i in range(slc.shape[0]):
            for j in range(slc.shape[1]):
                x1 =  np.clip(j-winRng2,0,slc.shape[1]-1)
                x2 =  np.clip(j+winRng2,0,slc.shape[1]-1)
                y1 =  np.clip(i-winRng2,0,slc.shape[0]-1)
                y2 =  np.clip(i+winRng2,0,slc.shape[0]-1)

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    sliceSmth[i,j] = np.nanmedian(slc[y1:y2, x1:x2])
    else:
        sliceSmth = slc

    norm = sliceSmth/nrmValue
    norm[np.isfinite(norm)][norm[np.isfinite(norm)]<cutoff] = np.nan

    return norm

def getResponseAllMed(flatSlices, sigma, cutoff, MP=True, ncpus=None):
    """
    Returns a possibly smoothed and normalized response function for all slices in the provided list
    Usage: result = getResponseAll(slices, sigma, cutoff, MP=)
    slices is a list of image slices from which to derive the response function
    sigma is the Gaussian width of the kernel to use for smoothing the input data
    cutoff is a value for which all pixels with normalized values less than this value are set to 1.
    MP is a keyword used to enable multiprocessing routines that may improve performance
    """

    #first determine the normalization weight, based on the maximum median value along each slice
    medSlice = getMedLevelAll(flatSlices, MP=True, ncpus=None)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        nrmValue = np.nanmax(medSlice)
        
    #multiprocessing may improve performance
    if (MP):
        #set up the input list

        lst = []
        for s in flatSlices:
            lst.append([s, sigma, cutoff, nrmValue])

        #setup and run the MP code for finding the limits
        if (ncpus == None):
            ncpus =mp.cpu_count()
        pool = mp.Pool(ncpus)
        result = pool.map(getResponse2DMed, lst)
        pool.close()
    else:
        result=[]
        for s in slices:
            result.append(getResponse2DMed([s,sigma, cutoff,nrmValue]))
            
    return result
