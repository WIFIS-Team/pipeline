"""

Tools used to separate slitlets 

"""

import numpy as np
import multiprocessing as mp
import wifisIO
import astropy.convolution as conv
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt

def limFit1(input):
    """
    Used to determine slice edges for a single column, assuming dispersion axis is aligned along rows.
    Usage: limMeas = limFit1(input)
    input is a list, with the first item being the spatial spectrum and the second item being the window range for finding the limits
    limMeas is a list the limits of the slices
    """
    
    #this should be updated once WIFIS is in near final alignment/state.
    #values are currently for 4kx4k optical CCD

    #centGuess=[100,311,530,754,978,1198,1420,1650,1875,2097,2323,2551,2774,3002,3222,3462,3669,3907]
    #centGuess = [34, 255, 466, 694, 915, 1142, 1363, 1593, 1812, 2041,2265,2492,2715, 2943,3167, 3392,3610,3834,4044]
    #centGuess = [85, 195,308,422,535,654,763,877,986,1100,1218,1332, 1442,1564,1669, 1792,1896,2015]
    centGuess = [13, 122, 232, 345, 458, 570, 683, 795, 908, 1019, 1132, 1245, 1359, 1469, 1582, 1693, 1807, 1919, 2037]
    
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

def findLimits(data, dispAxis=1, winRng=51, imgSmth=5, limSmth=10, ncpus=None):
    """
    Used to determine slice limits from a full flat-field image
    Usage: limits = findLimits(data, dispAxis=,winRng=,imgSmth=,limSmth= )
    data is the input flat field image from which the limits will be found
    dispAxis is keyword specifying the dispersion axis (0-> along the y-axis, 1-> along the x-axis)
    winRng is a keyword specifying the number of pixels to use when finding the slice edge limits
    imgSmth is a keyword specifying the Gaussian width of the smoothing kernel for finding the slice-edge limits
    limSmth is a keyword specigying the Gaussian width of the smoothing kernel for smoothing the found limits
    returns an array containing the pixel limits for each slice edge along the dispersion axis
    """

    if (dispAxis == 0):
        dTmp = data.T
    else:
        dTmp = data

    nx = dTmp.shape[0]
    ny = dTmp.shape[1]
    
    #go through and identify slice limits

    inpLst = []

    #create kernel for smoothing along the spatial direction
    gKern = conv.Gaussian1DKernel(stddev=imgSmth) 

    #next smooth the image and add to input list
    for i in range(ny):
        y = conv.convolve(dTmp[:,i],gKern)
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
        y = conv.convolve(result[i,:], gKern, boundary='extend')
        limSmth[i,:] = y
    
    return limSmth

def extSlices(data, limits, dispAxis=1):
    """
    Extract a list of slices (sub-images) from the given image.
    Usage: slices = extSlices(data, limits, dispAxis=)
    data is the input data image from which the slices will be extracted
    limits is an array specifying the slice-edge limits of each slice
    dispAxis is a keyword specifying the dispersion direction (0-> along the y-axis, 1-> along the x-axis)
    """

    if (dispAxis == 0):
        dTmp = data.T
    else:
        dTmp = data

    slices = []
    n=limits.shape[1]
    nSlices = limits.shape[0]
    
    for i in range(0,nSlices-1):
        mn = np.floor(np.min(limits[i,:])).astype('int')
        mx = np.ceil(np.max(limits[i+1,:])).astype('int')

        slice = np.zeros((mx-mn,n))
        #slice[:] = np.nan
        
        for j in range(mn,mx):
            keep = np.ones(n, dtype=bool)
            whr = np.where(np.floor(limits[i,:]) > j)[0]
            keep[whr] = False
            whr = np.where(np.ceil(limits[i+1,:]) < j)[0]
            keep[whr] = False
            slice[j-mn,keep] = dTmp[j,keep] 
  
        slices.append(slice)
    return slices


def getResponseFuncPoly(input):
    """
    NOT CURRENTLY USED FOR ANYTHING
    """

    slice = input[0]
    nOrd = input[1]
    
    pInit = models.Polynomial2D(degree=nOrd)
    fitP = fitting.LevMarLSQFitter()
    x,y = np.mgrid[:s.shape[0], :s.shape[1]]
    p = fitP(pInit,x,y,z)
    
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

    if (sigma>0):
        gKern = conv.Gaussian2DKernel(stddev=sigma) 
        sliceSmth=conv.convolve(slice,gKern,boundary='extend')
    else:
        sliceSmth = slice

    norm = sliceSmth/np.max(sliceSmth)
    norm[np.where(norm < cutoff)] = 1.

    return norm

def getResponseAll(slices, sigma, cutoff, MP=True, ncpus=None):
    """
    Returns a possibly smoothed and normalized response function for all slices in the provided list
    Usage: result = getResponseAll(slices, sigma, cutoff, MP=)
    slices is a list of image slices from which to derive the response function
    sigma is the Gaussian width of the kernel to use for smoothing the input data
    cutoff is a value for which all pixels with normalized values less than this value are set to 1.
    MP is a keyword used to enable multiprocessing routines that may improve performance
    """

    #multiprocessing may improve performance
    if (MP):
        #set up the input list

        lst = []
        for s in slices:
            lst.append([s, sigma, cutoff])

        #setup and run the MP code for finding the limits
        if (ncpus == None):
            ncpus =mp.cpu_count()
        pool = mp.Pool(ncpus)
        result = pool.map(getResponse2D, lst)
        pool.close()
    else:
        result=[]
        for s in slices:
            result.append(getResponse2D([s,sigma, cutoff]))
            
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
            result.append(slices[i]/response[i])
        
    return result

def ffCorrectSlice(input):
    """
    Returns flat-field corrected image determined from the provided data
    Usage: result = ffCorrectSlice(input)
    input is a list containing the image slice and the normalized flat-field image
    """

    slice = input[0]
    response = input[1]

    result = slice/response

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

    slc = np.empty((input[0].shape))
    np.copyto(slc, input[0])
    threshold = input[1]
    plot = input[2]
    
    #if no threshold is given, use derivatives to identify limits
    #else, use given threshold

    #first, replace all NaN by zero to allow summation
    slc[np.where(np.isnan(slc))] = 0
    
    if threshold is None:
        #use gradient to define cutoff
        
        #work on axis 1, assumes that wavetrim already took care of other axis
        ytmp = np.sum(slc, axis=1)
        n = ytmp.shape[0]
        
        gKern = conv.Gaussian1DKernel(stddev=1) #needs to be optimized, possibly read in
        y = conv.convolve(ytmp,gKern)
        d1 = np.abs(np.gradient(y))
        #whr = np.where(y > 0.5*np.max(y))[0]
        #y1 = np.argmax(d1[0:whr[0]])
        #y2 = np.argmax(d1[whr[-1]:])+whr[-1]
        y1 = np.argmax(d1[0:n/2])
        y2 = np.argmax(d1[n/2:])+n/2

    else:

        ytmp = np.sum(slc, axis=1)/float(slc.shape[1])
        whr = np.where(ytmp >= threshold*np.max(ytmp))[0]
        y1 = whr[0]
        y2 = whr[-1]

    if(plot):
        plt.figure()
        plt.plot(ytmp)
        plt.plot([y1,y1],[np.min(ytmp),np.max(ytmp)],'r--')
        plt.plot([y2,y2],[np.min(ytmp),np.max(ytmp)],'r--')
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

    out = np.empty((ny,slc.shape[1]))
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
    
