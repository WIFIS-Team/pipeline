"""
"""

import numpy as np
import warnings

def compFowler(inttime, fluxImg, satFrame, ron, gain=1.):
    """
    Routine to compute the per-pixel uncertainties associated with a Fowler sampling ramp or a CDS (correlated-double sample) frame.
    Usage: sigma = compFowler(inttime, fluxImg, satFrame, ron, gain=1.)
    inttime is array of integration times
    fluxImg is processed flux image, typically in units of counts/s
    satFrame is array indicating the frame number of first saturated frame
    ron is array/image/map indicating the read-out noise per pixel, typically in units of counts
    gain in units of e-/Count - only necessary if ron is given in e- (not ADU, counts or DN)
    sigma is the returned array/image indicating the computed 1-sigma uncertainty at each pixel.
    """

    nFrames = len(inttime)
    deltaT = inttime[-1] - inttime[int(inttime.shape[0]/2)-1] #integration time
    dT = inttime[1]-inttime[0] # readout time per frame
    nReads = satFrame - int(nFrames/2)

    #convert ron to units of counts (if gain != 1)
    ron /= gain 
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        eff_ron = np.sqrt(2)*ron/np.sqrt(nReads) #effective read noise, assuming no co-additions

        #compute variance, without co-adds
        var = fluxImg/(gain*deltaT) * (1.-(1./3.)*(dT/deltaT)*(nReads**2-1.)/nReads) + eff_ron**2/(gain**2*deltaT**2)

        #compute uncertainty from variance, return as a 32-bit float
        sigma = np.sqrt(var).astype('float32')

    return sigma


def compUTR(inttime, fluxImg, satFrame, ron, gain = 1.):
    """
    Routine to compute the per-pixel uncertainties associated with an up-the-ramp sequence.
    inttime is array of integration times
    fluxImg is processed flux image, in units of counts/s
    satFrame is array indicating the frame number of first saturated frame
    ron is array/image/map indicating the read-out noise per pixel, typically in units of counts
    gain in units of e-/Count - only necessary if ron is given in e- (not ADU, counts or DN)
    sigma is the returned array/image indicating the computed 1-sigma uncertainty at each pixel.
    """

    dT = np.mean(np.gradient(inttime)) # mean readout time per frame
    deltaT = (satFrame - 1)*dT 
    nReads = satFrame

    #convert RON into units of counts, in case gain != 1
    ron /= gain
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        eff_ron = np.sqrt(2)*ron/np.sqrt(nReads) #effect read noise, assuming no co-additions

        #compute variance, assuming no co-adds
        var = 6./5. * fluxImg/(gain *nReads *deltaT)* (nReads**2 + 1.)/(nReads + 1) + 6.*eff_ron**2/(gain**2*deltaT**2)*(nReads - 1.)/(nReads + 1.)

        #compute uncertainty from variance, return as a 32-bit float
        sigma = np.sqrt(var).astype('float32')

    return sigma

def compMedian(data, sigma, axis=None):
    """
    Compute the median and associated 1-sigma uncertainty of an input array and the associated uncertainty array, along the specified axis. The uncertainty is computed using standard error propogation, but for computation of the mean, not median.    
    
    Returns the median of the array elements.
    
    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : {int, None}, optional
        Axis or axes along which the medians are computed. The default
        is to compute the median along a flattened version of the array.
    Modified version of Numpy's nanmedian function
    """

    a = np.ma.masked_array(np.asanyarray(data),np.isnan(data))
    s = np.ma.masked_array(np.asanyarray(sigma),np.isnan(data))
    
    if (axis is None):
        a = a.ravel()
        s = s.ravel()
        
        order = np.argsort(a)
        a = a[order]
        s = s[order]

        sz = a.shape[0]

        #if even number of elements, take average of middle two elements
        if (sz % 2 == 0):
            outD =  (a[sz/2-1] + a[sz/2])/2.
            #outS = np.sqrt(s[sz/2-1]**2 + s[sz/2]**2)
        else:
            outD = a[int(sz/2)]
            #outS = s[int(sz/2)]
        outS = np.sqrt(np.sum(s**2))/sz

    else:

        index = list(np.ix_(*[np.arange(i) for i in a.shape]))
        index[axis] = a.argsort(axis)
        a = a[index]
        s = s[index]
        
        if (axis > 0):
            a = a.swapaxes(0,axis)
            s = s.swapaxes(0,axis)
            
        sz = a.shape[0]
        
        #if even number of elements, take average of middle two elements
        if (sz % 2 == 0):
            outD = (a[sz/2-1, ...] + a[sz/2,...])/2.
            #outS = np.sqrt(s[sz/2-1,...]**2 + s[sz/2,...]**2)
        else:
           # outS = s[int(sz/2),...]
            outD = a[int(sz/2),...]
        outS = np.sqrt(np.sum(s**2,axis=0))/sz
            
        #swap axis back to original location
        if (axis - 1 > 0):
            outD = outD.swapaxes(0,axis-1)
            outS = outS.swapaxes(0,axis-1)
    
    return np.asarray(outD), np.asarray(outS)

def addSlices(sigma1, sigma2):
    """
    Routine to add uncertainties for two lists of slice images containing the 1-sigma values, using standard error propogation rules
    Usage: sigma = addSlices(sigma1, sigma2)
    sigma1 is the first list of 1-sigma uncertainty images
    sigma2 is the second list of 1-sigma uncertainty images
    sigma is the returned list of the added uncertainties
    """

    sigma = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)

        for i in range(len(sigma1)):
            sigmaSlc = np.sqrt(sigma1[i]**2 + sigma2[i]**2)
            sigma.append(sigmaSlc)

    return sigma


def multiplySlices(slice1, sigma1, slice2,sigma2):
    """
    Routine to compute the uncertainty for multiplication of two lists of slice images containing the 1-sigma values, using standard error propogation rules
    Usage: sigma = addSlices(sigma1, sigma2)
    sigma1 is the first list of 1-sigma uncertainty images
    sigma2 is the second list of 1-sigma uncertainty images
    sigma is the returned list of the added uncertainties
    """

    sigma = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)

        for i in range(len(sigma1)):
            sigmaSlc = slice1[i]*slice2[i]*np.sqrt((sigma1[i]/slice1[i])**2 + (sigma2[i]/slice2[i])**2)
            sigma.append(sigmaSlc)

    return sigma


