"""
"""

import numpy as np

def compFowler(inttime, fluxImg, satFrame, nFrames, gain = 1., ron=1.):
    """
    inttime is array of integration times
    fluxImg is processed flux image. ***UNITS UNKNOWN AT THIS POINT***
    satFrame is array indicating the frame number of first saturated frame
    gain in units of e-/Count
    ron = read out noise, in units of electrons
    """

    deltaT = inttime[-1] - inttime[int(inttime.shape[0]/2)-1] #integration time
    dT = inttime[1]-inttime[0] # readout time per frame
    nReads = satFrame - int(nFrames/2)
    eff_ron = np.sqrt(2)*ron/np.sqrt(nReads) #effect read noise, assuming no co-additions

    #compute variance, assuming no co-adds
    var = fluxImg/(gain*deltaT) * (1.-(1./3.)*(dT/deltaT)*(nReads**2-1.)/nReads) + eff_ron**2/(gain**2*deltaT**2)

    #compute uncertainty from variance as a 32-bit float
    sigma = np.sqrt(var).astype('float32')

    return sigma


def compUTR(inttime, fluxImg, satFrame, gain = 1., ron=1.):
    """
    inttime is array of integration times
    fluxImg is processed flux image. ***UNITS UNKNOWN AT THIS POINT***
    satFrame is array indicating the frame number of first saturated frame
    gain in units of e-/Count
    ron = read out noise, in units of electrons
    """

    dT = np.mean(np.gradient(inttime)) # mean readout time per frame
    deltaT = (satFrame - 1)*dT 
    nReads = satFrame
    eff_ron = np.sqrt(2)*ron/np.sqrt(nReads) #effect read noise, assuming no co-additions

    #compute variance, assuming no co-adds
    var = 6./5. * fluxImg/(gain *nReads *deltaT)* (nReads**2 + 1.)/(nReads + 1) + 6.*eff_ron**2/(gain**2*deltaT**2)*(nReads - 1.)/(nReads + 1.)

    #compute uncertainty from variance as a 32-bit float
    sigma = np.sqrt(var).astype('float32')

    return sigma

def compMedian(data, sigma, axis=None):
    """
    Compute the median along the specified axis, but with standard error propagation rules. Uses np.ma.masked_array to ignore NaNs, but median is not properly computed in presence of NaNs (i.e. NaNs still contribute to length of axis).
    
    Returns the median of the array elements.
    
    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : {int, None}, optional
        Axis or axes along which the medians are computed. The default
        is to compute the median along a flattened version of the array.
        A sequence of axes is supported since version 1.9.0.
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
            outS = np.sqrt(s[sz/2-1]**2 + s[sz/2]**2)
        else:
            outD = a[int(sz/2)]
            outS = s[int(sz/2)]
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
            outS = np.sqrt(s[sz/2-1,...]**2 + s[sz/2,...]**2)
        else:
            outS = s[int(sz/2),...]
            outD = a[int(sz/2),...]
                        
        #swap axis back to original location
        if (axis - 1 > 0):
            outD = outD.swapaxes(0,axis-1)
            outS = outS.swapaxes(0,axis-1)
    
    return np.asarray(outD), np.asarray(outS)
