"""

Set of routines to help with post-processing of data cube

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splrep, splev
from scipy.interpolate import interp1d
import multiprocessing as mp
from scipy.optimize import curve_fit
import warnings
from astropy.convolution import convolve_fft
from astropy.convolution import Gaussian2DKernel
import time
import pyopencl as cl
import os
import wifisCreateCube as createCube
import wifisHeaders as headers
import copy

def splineContFit(x,y,regions, lineRegions=None,order=3,winRng=10.):
    """
    Use scipy spline fitting to determine continuum shape of input spectrum.
    Usage: xfit, yfit, contFit = splineContFit(x,y,regions, lineRegions,order=3, winRng=10)
    x is the input wavelength array
    y is the input flux array
    regions is a list of starting/stopping wavelength coordinates, used to confine the continuum regions (e.g. regions = [[x1, x2], [x3, x4], [x5,x6]])
    lineRegions is an optional array that specifies line regions to be ignored in the fitting process (NOT CURRENTLY IMPLEMENTED)
    order is a keyword used to specify the order of the spline fitting to use (1 = linear, 2 = quadratic, 3 = cubic, etc.)
    winRng is a keyword used to set the wavelength range from which to identify the pixel with the maximum flux, which will be used to determine the cubic spline fitting.
    xfit, yfit, and contFit are the output, where xfit and yfit are the x and y values of the points used for fitting and contFit is the continuum fit to the original x positions.
    """
    
    #use scipy cubic spline interpolate to get fit to given control points

    xfit=[x[0]]
    yfit=[y[0]]

    for reg in regions:
        strt = reg[0]
        end = np.min([reg[0] + winRng,reg[1]])
        while (strt < reg[1]):
            whr = np.where(np.logical_and(x>=strt,x<end))[0]
            mx = np.nanargmax(y[whr])+whr[0]
            xfit.append(x[mx])
            yfit.append(y[mx])
            strt += winRng
            end = np.min([end+winRng,reg[1]])

    xfit.append(x[-1])
    yfit.append(y[-1])
    xfit = np.concatenate(xfit)
    yfit = np.concatenate(yfit)
    
    splfit = splrep(xfit,yfit,x,k=order)
    contFit = splev(x, splfit)

    return xfit, yfit, contFit

def crossCorCube(wave, cube1, cube2, regions=None, oversample=20, absorption=False, ncpus=None, mode='idl', contFit1=True, contFit2=True,nContFit=50, contFitOrder=1, mxShift=4,reject=0, velocity=False):
    """
    Determine the pixel shift (if velocity=False) between input cubes cube1 and cube2, expected to be on the same wavelength grid/coordinate system x. Determines velocity difference, if velocity=True.
    Usage: shiftOut = crossCorCube(wave, cube1, cube2, regions=None, oversample=20, absorption=False, ncpus=None, mode='idl', contFit1=True, contFit2=True,nContFit=50, contFitOrder=1, mxShift=4,reject=0, velocity=False)
    wave is in a numpy array providing the wavelength coordinates
    cube1 is the first input cube
    cube2 is the second input cube
    regions is an optional parameter that provides specific regions to confine the cross correlation process. It is a list of lists (i.e. [[x1, x2],[x3,x4],...]
    oversample is the oversampling factor to use for determining subpixel shifts
    absorption is a boolean keyword to specify if the input cubes contain absorption lines
    ncpus sets the number of processes to use
    mode is a string keyword to specify how the cross-correlation is carried out ('idl' or 'fft')
    contFit1 is a boolean keyword to indicate if continuum fitting should be carried out on the first cube
    contFit2 is a boolean keyword to indicate if continuum fitting should be carried out on the second cube
    nContFit is an integer that specifies the number of pixels used to identify continuum points
    contFitOrder is the spline order to use for continuum fitting (1 - linear splint, 2 - quadratic spline, 3 - cubic spline)
    mxShift is the maximum allowed shift between cubes, in units of pixels or velocity (if velocity = True)
    reject is sigma-threshold to use for carrying out sigma-clipping when doing the continuum fitting
    velocity is a boolean keyword to specify if the measured shift is in pixels (False) or velocity units (True)
    shiftOut is an image containing the measured shift at each spatial coordinate
    """

    cube1tmp = np.empty(cube1.shape, dtype=cube1.dtype)
    cube2tmp = np.empty(cube1.shape, dtype=cube1.dtype)

    np.copyto(cube1tmp, cube1)
    np.copyto(cube2tmp,cube2)
      
    #build input list
    
    inpLst = []

    if velocity:
        v = (wave-wave[int(wave.shape[0]/2)])/wave[int(wave.shape[0]/2)]*2.99792458e5 # in km/s
        vconst = np.linspace(v[0],v[-1],num=v.shape[0]*oversample)
        
        for i in range(cube1.shape[0]):
            for j in range(cube1.shape[1]):
                inpLst.append([vconst,v, cube1tmp[i,j,:], cube2tmp[i,j,:],oversample, absorption,mode, contFit1, contFit2,nContFit,contFitOrder,mxShift, False, reject, regions, wave])
    else:
        for i in range(cube1.shape[0]):
            for j in range(cube1.shape[1]):
                inpLst.append([cube1tmp[i,j,:], cube2tmp[i,j,:],oversample, absorption,mode, contFit1, contFit2,nContFit,contFitOrder,mxShift, False, reject, regions,wave])

    if ncpus is None:
        ncpus = mp.cpu_count()
    pool = mp.Pool(ncpus)

    if velocity:
        shiftLst = pool.map(crossCorVelMP,inpLst)
    else:
        shiftLst = pool.map(crossCorPixMP,inpLst)

    pool.close()

    shiftOut = np.empty((cube1.shape[0],cube1.shape[1]), dtype=cube1.dtype)

    k=0
    for i in range(shiftOut.shape[0]):
        for j in range(shiftOut.shape[1]):
            shiftOut[i,j] = shiftLst[k]
            k+=1
            

    return shiftOut
    
def crossCorPixMP(input):
    """
    Determine the pixel difference between input spectra y1 and y2 on the same wavelength grid/coordinate system x.
    Usage: shiftOut = crossCorPixMP(inpu)
    input is a list containing:
    y1 - the first spectrum
    y2 - the second spectrum
    oversample - the oversampling rate
    mode - the mode to use for determining the shift
    contFit1 - boolean value indicating if the first spectrum is to be continuum rectified
    contFit2 - boolean value indicating if the second spectrum is to be continuum rectified
    nContFit - the number of pixels to use for determining continuum level/points
    contFitOrder - the spline order of the continuum fit
    mxShift - the maximum allowed pixel shift, which defines search range
    plot - boolean value to specify if plotting should be carried out (for debugging purposes)
    reject - sigma-threshold to use for clipping pixels during the continuum fitting stage
    regions - a list of pixel ranges indicating the regions to use for cross-correlation
    wave - a numpy array providing the wavelength coordinates at each pixel
    shiftOut is the measured pixel shift
    """

    y1= input[0]
    y2 = input[1]
    oversample=input[2]
    absorption= input[3]
    mode = input[4]
    contFit1 = input[5]
    contFit2 = input[6]
    nContFit = input[7]
    contFitOrder = input[8]
    mxShift = input[9]
    plot = input[10]
    reject =input[11]
    regions=input[12]
    wave = input[13]
    
    #quit if either input arrays are all NaN
    if np.all(~np.isfinite(y1)) or np.all(~np.isfinite(y2)):
        return np.nan

    if contFit1:
        y1tmp = y1 - splineContFitMP([y1,contFitOrder,nContFit, reject])
    else:
        y1tmp = np.empty(y1.shape,dtype=y1.dtype)
        np.copyto(y1tmp,y1)

    if contFit2:
        y2tmp = y2 - splineContFitMP([y2,contFitOrder,nContFit, reject])    
    else:
        y2tmp = np.empty(y2.shape,dtype=y2.dtype)
        np.copyto(y2tmp,y2)

    #now get rid of NaNs
    y1tmp[~np.isfinite(y1)] = 0.
    y2tmp[~np.isfinite(y2)] = 0.
        
    #y1Tmp = np.zeros(y1tmp.shape)
    #y2Tmp = np.zeros(y2tmp.shape)
    
    #now only use regions of interest
    #by building a new spectrum with only the good regions, padded by a 0 in between
    
    if regions is not None:
        y1Tmp = np.asarray([])
        y2Tmp = np.asarray([])
        
        for reg in regions:
            whr = np.where(np.logical_and(wave>=reg[0],wave<=reg[1]))[0]
            y1Tmp= np.append(y1Tmp,y1tmp[whr])
            y1Tmp = np.append(y1Tmp,np.zeros(mxShift))

            y2Tmp=np.append(y2Tmp,y2tmp[whr])
            y2Tmp=np.append(y2Tmp,np.zeros(mxShift))
            #y1Tmp[whr] = y1tmp[whr]
            #y2Tmp[whr] = y2tmp[whr]
            
    else:
        y1Tmp = np.empty(y1tmp.shape)
        y2Tmp = np.empty(y2tmp.shape)

        np.copyto(y1Tmp,y1tmp)
        np.copyto(y2Tmp,y2tmp)

    x = np.arange(y1Tmp.shape[0])
    xInt = np.linspace(0,x[-1],num=y1Tmp.shape[0]*oversample)
    
    y1Int = np.interp(xInt,x, y1Tmp)
    y2Int = np.interp(xInt, x, y2Tmp)
    
    if absorption:
        y1tmp = 1. - y1tmp
        y2tmp = 1. - y2tmp

    if mxShift is not None:
        rng = np.arange(-mxShift*oversample, mxShift*oversample+1)
    else:
        rng = np.arange(-xInt.shape[0]-2, xInt.shape[0]-1)

    yCC = crossCorIDL([y1Int, y2Int, rng])
    shiftOut = rng[np.argmax(yCC)]/np.float(oversample)

    if plot:
        fig = plt.figure()
        plt.plot(rng/np.float(oversample), yCC)
        plt.show()
 
    return shiftOut

def crossCorSpec(wave, spec1, spec2, regions=None, oversample=20, absorption=False, mode='idl', contFit1=True, contFit2=True,nContFit=50,contFitOrder=1, mxVel=200, plot=False, reject=0, velocity=True):
    """
    Routine to determine the velocity difference between two spectra, expected to be on the same wavelength grid.
    Usage: rvOut = crossCorSpec(wave, spec1, spec2, regions=None, oversample=20, absorption=False, mode='idl', contFit1=True, contFit2=True,nContFit=50,contFitOrder=1, mxVel=200, plot=False, reject=0, velocity=True)
    wave is the numpy wavelength array specifying the wavelength coordinates at each pixel
    spec1 is the first spectrum
    spec2 is the second spectrum
    regions is a list of wavelength coordinates specifying regions to use in the cross-correlation process
    oversample is the oversampling rate used to determine sub-pixel shifts
    absorption is a boolean keyword to specify if the input spectra contain absorption lines
    mode is a keyword specifying the mode of cross-correlation ('idl', 'fft')
    contFit1 is a boolean keyword indicating if the first spectrum should be continuum rectified
    contFit2 is a boolean keyword indicating if the second spectrum should be continuum rectified
    contFitOrder is the spline order to use for continuum fitting (1 - linear splint, 2 - quadratic spline, 3 - cubic spline)
    mxVel is the maximum allowed velocity shift between spectra
    reject is sigma-threshold to use for carrying out sigma-clipping when doing the continuum fitting
    velocity is a boolean keyword to specify if the measured shift is in pixels (False) or velocity units (True)
    shiftOut is an image containing the measured shift at each spatial coordinate
    """

      
    if velocity:
        #compute velocity grid and interpolate velocity grid
        v = (wave-wave[int(wave.shape[0]/2)])/wave[int(wave.shape[0]/2)]*2.99792458e5 # in km/s
        vconst = np.linspace(v[0],v[-1],num=v.shape[0]*oversample)

        rvOut = crossCorVelMP([vconst, v, spec1, spec2, oversample, absorption, mode,contFit1,contFit2,nContFit,contFitOrder,mxVel, plot, reject,regions,wave])
    else:
        rvOut = crossCorPixMP([spec1, spec2, oversample, absorption, mode,contFit1,contFit2,nContFit,contFitOrder,mxVel, plot, reject,regions,wave])

    return rvOut
    
def splineContFitMP(input):
    """
    Use scipy spline fitting to determine continuum shape of input spectrum.
    Usage: contFit = splineContFitMP(input)
    input is a list containing:
    y - the input flux array
    order - the order of the spline fitting to use (1 = linear, 2 = quadratic, 3 = cubic, etc.)
    winRng - the number of pixels to use for the fitting
    contFit is the output numpy array containing the continuum fit
    """

    y = input[0]
    order = input[1]
    winRng=input[2]
    reject = input[3]
    
    #use scipy cubic spline interpolate to get fit to given control points
 
    whr = np.where(np.isfinite(y))[0]

    if len(whr) < order:
        return np.nan
    
    x = np.arange(y.shape[0])

    ytmp = y[whr]
    xtmp = x[whr]

    #use median of 0-> winRng/2 to ancor starting point
    whr = np.where(np.logical_and(xtmp >= xtmp[0], xtmp<=xtmp[0]+winRng/2))[0]

    xfit=[xtmp[0]]

    if reject > 0:
        yrej = ytmp[whr]
        for i in range(3):
            whr = np.where(np.abs(yrej)<=reject*np.nanstd(yrej)+np.nanmedian(yrej))
            yrej = yrej[whr]
        yfit=[np.nanmedian(yrej)]
    else:
        yfit=[np.nanmedian(ytmp[whr])]

    end = 0
    strt = xtmp[0]
    while end < y.shape[0]:
        whr = np.where(np.logical_and(xtmp >= strt, xtmp <= strt+winRng))[0]
        if (len(whr)>0):
            xfit.append(np.nanmedian(xtmp[whr]))
                        
            if reject > 0:
                yrej = ytmp[whr]
                for i in range(3):
                    whr = np.where(np.abs(yrej)<=reject*np.nanstd(yrej)+np.nanmedian(yrej))
                    yrej = yrej[whr]
                yfit.append(np.nanmedian(yrej))
            else:
                yfit.append(np.median(ytmp[whr]))
        end = strt + winRng
        strt+=winRng
        
    #for i in range(xtmp[0],ytmp.shape[0]-winRng,winRng):
    #    whr = np.where(np.logical_and(xtmp[0]
    #    xfit.append(xtmp[0]+i+winRng/2.)
    #    yfit.append(np.nanmedian(ytmp[i:i+winRng]))

    #use median of end-winRng/2 -> end to ancor ending point
    whr = np.where(np.logical_and(xtmp >= xtmp[-1]-winRng/2, xtmp<=xtmp[-1]))[0]

    if reject > 0:
        yrej = ytmp[whr]
        for i in range(3):
            whr = np.where(np.abs(yrej)<=reject*np.nanstd(yrej)+np.nanmedian(yrej))
            yrej = yrej[whr]
        yfit.append(np.nanmedian(yrej))
    else:
        yfit.append(np.nanmedian(ytmp[whr]))

    xfit.append(xtmp[-1])

    xfit = np.asarray(xfit)
    yfit = np.asarray(yfit)
    
    splfit = splrep(xfit,yfit,x,k=order)
    contFit = splev(x, splfit)

    return contFit

def crossCorIDL(input):
    """
    Routine to compute the cross-correlation between two spectra using the IDL method
    Usage: p = crossCorIDL(input)
    input is a list containing:
    y1 - spectrum 1
    y2 - spectrum 2
    rng - a numpy array specifying the pixel shifts at which to compute the cross-correlation
    p is the p unnormalized p value corresponding to the cross-correlation
    """

    y1 = input[0]
    y2 = input[1]
    rng = input[2]

    p = np.zeros(rng.shape[0])
    y1sub = y1 - np.mean(y1)
    y2sub = y2 - np.mean(y2)
    n = y1.shape[0]
    
    for i in range(len(rng)):
        lag = int(rng[i])
        if lag >= 0:
            p[i] = np.sum(y1sub[:n-lag]*y2sub[lag:])
        else:
            p[i] = np.sum(y2sub[:n+lag]*y1sub[-lag:])
            
        #p[i] /= np.sqrt(np.sum(y1sub**2)*np.sum(y2sub**2))

    return p
        
def crossCorVelMP(input):
    """
    Determine the velocity difference between input spectra y1 and y2 on the same wavelength grid.
    Usage: rvOut = crossCorVelMP(input)
    input is a list containing:
    vconst - the linearized velocity grid
    y1 - the first input spectrum
    y2 - the second input spectrum
    oversample - the oversampling rate to measure sub-pixel corrections
    mode - the mode to use to carry out the cross-correlation measurement ('idl', 'fft')
    contFit1 - a boolean value indicating whether the first spectrum should be continuum rectified
    contFit2 - a boolean value indicating whether the second spectrum should be continuum rectified
    nContFit - the number of pixels to use for determining continuum points
    contFitOrder - the spline fitting order to use for continuum fitting
    mxVel - the maximum velocity used to specify search range
    plot - a boolean value indicating whether plotting shoud be carried out (for debugging purposes)
    reject - the sigma-clipping threshold to use for rejecting pixels in the continuum fitting routine
    regions - a list of wavelength coordinates specifying the regions to use for cross-correlation
    wave - the input wavlength array giving the wavelength coordinates for the input spectra
    rvOut is the measured radial velocity shift between the two spectra
    """

    vconst = input[0]
    v = input[1]
    y1= input[2]
    y2 = input[3]
    oversample=input[4]
    absorption= input[5]
    mode = input[6]
    contFit1 = input[7]
    contFit2 = input[8]
    nContFit = input[9]
    contFitOrder = input[10]
    mxVel = input[11]
    plot = input[12]
    reject =input[13]
    regions=input[14]
    wave = input[15]
    
    #quit if either input arrays are all NaN
    if np.all(~np.isfinite(y1)) or np.all(~np.isfinite(y2)):
        return np.nan
    
    #go through regions list and construct new arrays from subsets

    if contFit1:
        y1tmp = y1 - splineContFitMP([y1,contFitOrder,nContFit, reject])
    else:
        y1tmp = np.empty(y1.shape,dtype=y1.dtype)
        np.copyto(y1tmp,y1)
    if contFit2:
        y2tmp = y2 - splineContFitMP([y2,contFitOrder,nContFit, reject])    
    else:
        y2tmp = np.empty(y2.shape,dtype=y2.dtype)
        np.copyto(y2tmp,y2)
        
    #now get rid of NaNs
    y1tmp[~np.isfinite(y1)] = 0.
    y2tmp[~np.isfinite(y2)] = 0.

    y1Tmp = np.zeros(y1tmp.shape)
    y2Tmp = np.zeros(y2tmp.shape)
    
    #now only use regions of interest
    if regions is not None:
        for reg in regions:
            whr = np.where(np.logical_and(wave>=reg[0],wave<=reg[1]))[0]
            y1Tmp[whr] = y1tmp[whr]
            y2Tmp[whr] = y2tmp[whr]
    else:
        np.copyto(y1Tmp,y1tmp)
        np.copyto(y2Tmp,y2tmp)

    #now compute and interpolate onto constant velocity grid, and converting to line intensity (i.e. 1 - continuum)
    y1Const = np.interp(vconst,v,y1Tmp,left=0,right=0)
    y2Const = np.interp(vconst, v, y2Tmp, left=0, right=0)
    dv = (vconst[1]-vconst[0])

    if absorption:
        y1Const = 1. - y1Const#finter(vconst)
        y2Const = 1. - y2Const

    if mode.lower() == 'fft':
        #use FFT to compute cross-correlation
        fft1 = np.fft.fft(y1Const)
        fft2 = np.fft.fft(y2Const)
        den = np.abs(fft1)*np.abs(fft2)
        num = fft1*fft2.conjugate()
        r = num/den
        m = np.abs(np.fft.ifft(r))
                
        if mxVel is not None:
            m2 = np.fft.fftshift(m)
            whr = np.where(vconst < -mxVel)[0]
            m2[whr] = 0
            whr = np.where(vconst > mxVel)[0]
            m2[whr] = 0
            
            m = np.fft.ifftshift(m2)

        mx =np.nanargmax(m)
        if mx > m.shape[0]//2:
            mx -= m.shape[0]
        rvOut = -mx*(vconst[1]-vconst[0])
        if plot:
            fig = plt.figure()
            plt.plot(m)
            plt.show()
    elif (mode == 'idl') :
        if mxVel is not None:
            rng = np.arange(-np.ceil(mxVel/dv), np.ceil(mxVel/dv)+1)
        else:
            rng = np.arange(-vconst.shape[0]-2, vconst.shape[0]-1)

        yCC = crossCorIDL([y1Const, y2Const, rng])
        rvOut = rng[np.argmax(yCC)]*dv

        if plot:
            fig = plt.figure()
            plt.plot(rng*dv, yCC)

            fig = plt.figure()
            plt.plot(v, y1)
            plt.plot(v-rvOut ,y2)
            plt.show()

            
    else:
        #Use numpy correlate to determine cross-correlation between input spectra
        yCC = np.correlate(y1Const, y2Const, mode="full")
        xCC = (np.arange(y1Const.shape[0]*2-1)-y1Const.shape[0]+1) #computes the velocity shift for each oversampled pixel in the cross-correlation output
        if mxVel is not None:
            whr = np.where(np.logical_or(xCC >= -mxVel*dv, xCC <= mxVel*dv))[0]
            yCC = yCC[whr]
            xCC = xCC[whr]
        if plot:
            fig = plt.figure()
            plt.plot(xCC*dv, yCC)
            plt.show()
        #now determine the RV offset, from the maximum of the cross-correlation function
        rvOut = -xCC[np.nanargmax(yCC)]*dv

    return rvOut

def scaleSky(y,f):
    """
    Function used to multiply input vector y by scalar value f
    """

    return y*f

def fitSky(y1, y2, bounds=[-np.inf, np.inf]):
    """
    Routine used to determine optimal line scaling between given inputs
    Usage: popt = fitSky(y1,y2, bounds=[-np.inf,np.inf])
    y1 is the first spectrum to be matched
    y2 is the second spectrum 
    bounds specify the relative scaling factor bounds
    popt is the returned value of the optimization fit
    """

    popt, pcov = curve_fit(scaleSky,y2, y1, p0=[1.], bounds=bounds)

    return popt

def subScaledSkyPix(input):
    """
    Routine to scale and subtract the sky spectrum from an observed spectrum.
    Usage: specOut, fOut = subScaledskyPix(input)
    input is a list containing the following:
    wave - a numpy array containing the wavelength coordinates of the input spectra
    obs - a numpy array containing the observed spectrum
    sky - a numpy array containing the sky spectrum
    regions - a list of wavelength coordinates providing the limits of the regions to use for sky subtraction
    sigmaClip - the sigma-clipping threshold at which to reject pixels in the sky scaling process
    sigmaClipRounds - defines the number of sigma-clipping iterations to carry out
    useMaxOnly - a relative fraction of the maximum flux to specify if only pixels with relative flux greater than this value should be used for sky scaling and subtracting. A value of zero uses all pixels.
    nContFit - the number of pixels to use for identifying continuum points for continuum subtraction
    specOut is the sky-scaled subtracted spectrum
    fOut provides a list of the scaling used for each region
    """
    wave = input[0]
    obs = input[1]
    sky = input[2]
    regions = input[3]
    bounds = input[4]
    sigmaClip = input[5]
    sigmaClipRounds = input[6]
    useMaxOnly = input[7]
    nContFit = input[8]
    
    obsCont = splineContFitMP([obs, 1, nContFit, 1])
    skyCont = splineContFitMP([sky, 1, nContFit, 1])
    skyScaled = np.zeros(sky.shape,dtype=sky.dtype)

    if regions is None:
        regions = [[wave.min(), wave.max()]]
        
    if np.all(~np.isfinite(obs)) or np.all(~np.isfinite(sky)):
        fOut = []
        for reg in regions:
            fOut.append(np.nan)
        return obs, fOut
    
    outside = np.ones(sky.shape).astype(bool)

    #list to hold all scalings
    fOut = []
    
    #construct the sky spectrum outside of the scaled region
    for reg in regions:
        if len(reg) ==2:
            whr = np.where(np.logical_and(wave>=reg[0],wave<=reg[1]))[0]
        elif len(reg)==4:
            whr = np.where(np.logical_or(np.logical_and(wave>=reg[0],wave<=reg[1]),np.logical_and(wave>=reg[2],wave<=reg[3])))[0]

        outside[whr] = False
    skyScaled[outside] = sky[outside]
    
    for reg in regions:
        whr = np.where(np.logical_and(wave>=reg[0],wave<=reg[1]))[0]
        #outside.append([whr[0],whr[-1]])
        
        otmp = (obs-obsCont)[whr]
        stmp = (sky-skyCont)[whr]

        otmp[~np.isfinite(otmp)] = 0.
        stmp[~np.isfinite(stmp)] = 0.

        f = fitSky(otmp, stmp)
        if (f < bounds[0]):
            f = bounds[0]
        elif (f > bounds[1]):
            f = bounds[1]

        skyCor = stmp*f
        
        if sigmaClipRounds>0:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                for i in range(sigmaClipRounds):
                    y = otmp-stmp*f
                    med = np.nanmedian(y)
                    std = np.nanstd(y)
                    goodPix = np.where(np.abs(y-med)<=sigmaClip*std)[0]
                    #badPix = np.where(np.abs(y-med)>sigmaClip*std)[0]
                    badPix = np.ones(stmp.shape[0])
                    #use only pixels with value useMax of maximum flux for fitting and subtracting
                    if useMaxOnly > 0:
                        #badPix2 = np.where(stmp < useMaxOnly*stmp[goodPix].max())[0]
                        goodPix = np.where(stmp >= useMaxOnly*stmp[goodPix].max())[0]
                        badPix[goodPix] = 0
                        #badPixAll = np.concatenate((badPix,badPix2))
                        #goodPixAll = np.concatenate((goodPix,goodPix2))
                        #badPixAll = np.unique(badPixAll)
                        #goodPixAll = np.unique(goodPixAll)

                    badPix[goodPix]=0
                        
                    f = fitSky(otmp[goodPix], stmp[goodPix])

                    if (f < bounds[0]):
                        f = bounds[0]
                    elif (f > bounds[1]):
                        f = bounds[1]

            skyCor[goodPix] = stmp[goodPix]*f
            skyCor[badPix.astype(bool)] = stmp[badPix.astype(bool)]
            
        skyScaled[whr] = skyCont[whr] + skyCor
        fOut.append(f)
    return obs - skyScaled, fOut
    
def subScaledSkyCube(wave, obs, sky, mxScale=0.5, regions=None, MP=True, ncpus=None,sigmaClip=3, sigmaClipRounds=1, useMaxOnly=0., nContFit=50):
    """
    Routine used to scale sky lines and subtract from an entire cube.
    Usage: subCube, fImg = subScaledSkyCube(wave, obs, sky, mxScale=0.5,  regions=None, MP=True, ncpus=None,sigmaClip=3, sigmaClipRounds=1, useMaxOnly=0., nContFit=50)
    wave is a numpy array specifying the wavelength coordinates of both cubes
    obs is the observation cube
    sky is the sky cube
    mxScale is the maximum allowable scaled difference between sky/obs spectrum (e.g. 0.2 -> sky can be scaled in range of 0.8-1.2)
    regions is a list of wavelength coordinates specifying the different sky regions/bands
    MP is a boolean keyword specifying if multiprocessing should be used
    ncpus is an integer variable specfying the number of processes to use in MP mode
    sigmaClip is the sigma-clipping threshold to use for determining the sky scaling fraction
    sigmaClipRounds is the number of sigma-clipping iterations to run
    useMaxOnly - a relative fraction of the maximum flux to specify if only pixels with relative flux greater than this value should be used for sky scaling and subtracting. A value of zero uses all pixels.
    nContFit is the number of pixels to use during continuum fitting
    subCube is the output scaled-sky subtracted cube
    fImg is an image providing the scaling factor at each pixel
    """

    #get bounds based on input
    if mxScale is not None:
        bounds = [1.-mxScale, 1.+mxScale]
    else:
        bounds = [-np.inf, np.inf]

    #build input list
    inpLst = []
    for i in range(sky.shape[0]):
        for j in range(sky.shape[1]):
            inpLst.append([wave,obs[i,j,:],sky[i,j,:],regions, bounds,sigmaClip,sigmaClipRounds,useMaxOnly,nContFit])
    
    if MP:
    #setup multithreading

        if ncpus is None:
            ncpus = mp.cpu_count()
        pool = mp.Pool(ncpus)
        subLst = pool.map(subScaledSkyPix,inpLst)
        pool.close()
        
        k=0
        subCube = np.empty(obs.shape, dtype=obs.dtype)
        fImg = np.empty((obs.shape[0], obs.shape[1], len(regions)), dtype=obs.dtype)
        
        for i in range(sky.shape[0]):
            for j in range(sky.shape[1]):
                subCube[i,j,:] = subLst[k][0]
                for l in range(fImg.shape[2]):
                    fImg[i,j,l] = subLst[k][1][l]
                k+=1
    else:
        k=0
        subCube = np.empty(obs.shape, dtype=obs.dtype)
        fImg = np.empty((obs.shape[0], obs.shape[1], len(regions)), dtype=obs.dtype)

        for i in range(sky.shape[0]):
            for j in range(sky.shape[1]):
                subLst = subScaledSkyPix(inpLst[k])
                subCube[i,j,:] = subLst[0]
                for l in range(fImg.shape[2]):
                    fImg[i,j,l] = subLst[1][l]
                k+=1

    return subCube, fImg

def subScaledSkySlices(waveMap, obsSlices, skySlices, mxScale=0.5, regions=None, MP=True, ncpus=None,sigmaClip=3, sigmaClipRounds=1, useMaxOnly=0., nContFit=50):
    """
    Routine used to scale sky lines and subtract from image slices.
    Usage: subSlices, fSlices = subScaledSkySlices(waveMap, obsSlices, skySlices, mxScale=0.5,  regions=None, MP=True, ncpus=None,sigmaClip=3, sigmaClipRounds=1, useMaxOnly=0., nContFit=50)
    waveMap is a list of wavelength mapping images for each slice
    obsSlices is the observation slices
    skySlices is the sky slices
    mxScale is the maximum allowable scaled difference between sky/obs spectrum (e.g. 0.2 -> sky can be scaled in range of 0.8-1.2)
    regions is a list of wavelength coordinates specifying the different sky regions/bands
    MP is a boolean keyword specifying if multiprocessing should be used
    ncpus is an integer variable specfying the number of processes to use in MP mode
    sigmaClip is the sigma-clipping threshold to use for determining the sky scaling fraction
    sigmaClipRounds is the number of sigma-clipping iterations to run
    useMaxOnly - a relative fraction of the maximum flux to specify if only pixels with relative flux greater than this value should be used for sky scaling and subtracting. A value of zero uses all pixels.
    nContFit is the number of pixels to use during continuum fitting
    subCube is the output scaled-sky subtracted cube
    fImg is an image providing the scaling factor at each pixel
    mxScale is maximum allowable scaled difference between sky/obs spectrum (e.g. 0.2 or 20%, implies that the sky emission strength can be scaled in range of +/-20% or 0.8-1.2).
    """

    #get bounds based on input
    if mxScale is not None:
        bounds = [1.-mxScale, 1.+mxScale]
    else:
        bounds = [-np.inf, np.inf]

    #build input list
    inpLst = []
    for i in range(len(skySlices)):
        for j in range(skySlices[i].shape[0]):
            inpLst.append([waveMap[i][j,:],obsSlices[i][j,:],skySlices[i][j,:],regions, bounds,sigmaClip,sigmaClipRounds,useMaxOnly, nContFit])
    
    if MP:
    #setup multithreading

        if ncpus is None:
            ncpus = mp.cpu_count()
        pool = mp.Pool(ncpus)
        subLst = pool.map(subScaledSkyPix,inpLst)
        pool.close()
        
        k=0
        subSlices = []
        fSlices=[]
        
        for i in range(len(skySlices)):
            fSlices.append([])
            subSlices.append(np.empty(skySlices[i].shape, dtype=skySlices[i].dtype))
            subSlices[i][:]=np.nan
            for j in range(skySlices[i].shape[0]):
                subSlices[i][j,:] = subLst[k][0]
                fSlices[i].append(subLst[k][1])
                k+=1
    else:
        k=0
        subSlices = []
        fSlices=[]

        for i in range(len(skySlices)):
            fSlices.append([])
            subSlices.append(np.empty(skySlices[i].shape, dtype=skySlices[i].dtype))
            subSlices[i][:]=np.nan
            for j in range(skySlices[i].shape[0]):
                subLst = subScaledSkyPix(inpLst[k])
                subSlices[i][j,:] = subLst[0]
                fSlices[i].append(subLst[1])
                k+=1

    return subSlices, fSlices
  
def crossCorSlices(waveMap, slices1, slices2, regions=None, oversample=20, absorption=False, ncpus=None, mode='idl', contFit1=True, contFit2=True,nContFit=50, contFitOrder=1, mxShift=4,reject=0, velocity=False):
    """
    Determine the pixel shift (if velocity=False) between input cubes cube1 and cube2, expected to be on the same wavelength grid/coordinate system x. Determines velocity difference, if velocity=True. mxShift is in pixels or km/s.
    Usage: shiftOut = crossCorSlices(waveMap, slices1, slices2, regions=None, oversample=20, absorption=False, ncpus=None, mode='idl', contFit1=True, contFit2=True,nContFit=50, contFitOrder=1, mxShift=4,reject=0, velocity=False)
    waveMap is a list of wavelength maps for each slice
    slices1 is the slices of the first image
    slices2 is the slices of the second image
    regions is a list of wavelength coordinates specifying the regions to use for cross-correlation
    oversample is the oversampling rate to measure sub-pixel corrections
    absorption is a boolean keyword to specify if the input images contain absorption lines
    ncpus specifies the number of processes to run in parallel
    mode is the mode to use to carry out the cross-correlation measurement ('idl', 'fft')
    contFit1 is a boolean keyword indicating whether the first spectrum should be continuum rectified
    contFit2 is a boolean keyword indicating whether the second spectrum should be continuum rectified
    nContFit is the number of pixels to use for determining continuum points
    contFitOrder is the spline fitting order to use for continuum fitting
    mxShift is the maximum shift used to specify search range (in pixels, if velocity = False; in velocity if velocity=True)
    reject is the sigma-clipping threshold to use for rejecting pixels in the continuum fitting routine
    velocity is a boolean keyword indicating if the measured shift is done in pixels (False) or velocity (True)
    shiftOut is a list of the measured shift between the each slice
    """

    #build input list
    inpLst = []

    if velocity:
        
        for i in range(len(slices1)):
            for j in range(slices1[i].shape[0]):
                v = (waveMap[i][j,:]-waveMap[i][j,:][int(waveMap[i][j,:].shape[0]/2)])/waveMap[i][j,:][int(waveMap[i][j,:].shape[0]/2)]*2.99792458e5 # in km/s
                vconst = np.linspace(v[0],v[-1],num=v.shape[0]*oversample)
                inpLst.append([vconst,v, slices1[i,j,:], slices2[i,j,:],oversample, absorption,mode, contFit1, contFit2,nContFit,contFitOrder,mxShift, False, reject, regions,waveMap[i][j,:]])
    else:
        for i in range(len(slices1)):
            for j in range(slices1[i].shape[0]):
                inpLst.append([slices1[i][j,:], slices2[i][j,:],oversample, absorption,mode, contFit1, contFit2,nContFit,contFitOrder,mxShift, False, reject, regions,waveMap[i][j,:]])

    if ncpus is None:
        ncpus = mp.cpu_count()
    pool = mp.Pool(ncpus)

    if velocity:
        shiftLst = pool.map(crossCorVelMP,inpLst)
    else:
        shiftLst = pool.map(crossCorPixMP,inpLst)

    pool.close()

    shiftOut = []
    for i in range(len(slices1)):
        shiftOut.append(np.zeros(slices1[i].shape, dtype=slices1[i].dtype))

    k=0
    for i in range(len(shiftOut)):
        for j in range(shiftOut[i].shape[0]):
            shiftOut[i][j,:] = shiftLst[k]
            k+=1            

    return shiftOut
    
def buildfSlicesMap(fSlices):
    """
    Routine to build a map of the scaling factors from all the slices.
    Usage: outMap = buildfSlicesMap(fSlices)
    fSlices contains the list of scaling factors
    outMap if the derived map
    """

    outMap =[]

    for slc in fSlices:
        #get number of regions

        mxRegs = 0
        for i in range(len(slc)):
            nRegs = len(slc[i])
            if nRegs > mxRegs:
                mxRegs = nRegs

        #now construct the individual maps
        tmpMap = np.zeros((len(slc),mxRegs),dtype='float32')

        for i in range(len(slc)):
            for j in range(mxRegs):
                tmpMap[i,j] = slc[i][j]
        outMap.append(tmpMap)

    return outMap

def shiftSlicesAll(inpSlices, pixShift, MP=False, ncpus=None,axis=0):
    """
    Routine to shift all slices by the indicated pixel shift.
    Usage outSlices = shiftSlicesAll(inpSlices, pixShift, MP=False, ncpus=None)
    inpSlices is the input slices to shift
    pixShift is the shift in pixels to apply
    MP is a boolean keyword that specifies if the routine should be split into multiple processes
    ncpus specifies the number of processes
    outSlices is a list of the shifted image slices, on the same coordinate system as the input slices
    """

    #create input list
    inpLst = []
    for slc in inpSlices:
        inpLst.append([slc,pixShift,axis])

    if MP:
        if ncpus is None:
            ncpus = mp.cpu_count()
        pool = mp.Pool(ncpus)
        outSlices = pool.map(shiftSlice,inpLst)
        pool.close()
    else:
        outSlices =[]
        for lst in inpLst:
            outSlices.append(shiftSlice(lst))
    return outSlices
        
    
def shiftSlice(input):
    """
    Routine to apply a pixel shift to an image slice
    Usage: slcNew = shiftSlice(input)
    input is a list containing:
    slc - the input image slice
    pixShift - the shift to apply
    axis - the axis along which to shift the image
    slcNew is the shifted image on the same coordinate system as the input
    """

    slc = input[0]
    pixShift = input[1]
    axis=input[2]

    slcNew = np.empty(slc.shape,dtype=slc.dtype)

    if axis==1:
        xOrg = np.arange(slc.shape[1]).astype(float)-pixShift
        xNew = np.arange(slc.shape[1])

        for i in range(slc.shape[0]):
            slcNew[i,:] = np.interp(xNew,xOrg,slc[i,:])
    else:
        xOrg = np.arange(slc.shape[0]).astype(float)-pixShift
        xNew = np.arange(slc.shape[0])

        for i in range(slc.shape[1]):
            slcNew[:,i] = np.interp(xNew,xOrg,slc[:,i])

    return slcNew

def shiftImage(img, pixShift,dispAxis=0):
    """
    Routine to shift an image along the dispersion axis
    Usage outImg = shiftImage(img, pixShift, dispAxis=0)
    img is the input image
    pixShift is the shift to apply
    dispAxis specifies the dispersion direction
    outImg is the shifted image on the same coordinate system as the input image
    """

    #img = input[0]
    #pixShift = input[1]
    #dispAxis=input[2]

    if dispAxis==0:
        imgTmp = np.empty(img.shape, dtype=img.dtype)
        np.copyto(imgTmp, img)
    else:
        imgTmp = np.empty(img.T.shape, dtype=img.dtype)
        np.copyto(imgTmp, img.T)

    xOrg = np.arange(imgTmp.shape[0]).astype(float)-pixShift
    xNew = np.arange(imgTmp.shape[0])
    outImg = np.empty(imgTmp.shape,dtype=imgTmp.dtype)
    
    for i in range(imgTmp.shape[1]):
        outImg[:,i] = np.interp(xNew,xOrg,imgTmp[:,i])
        
    if dispAxis!=0:
        outImg = outImg.T

    return outImg

def getSmoothedImage(img,kernSize=4):
    """
    Routine to apply a Gaussian smoothing to an input image
    Usage: imgSmth = getSmoothedImage(img, kernSize=4)
    img is the input image
    kernSize is the width of the Gaussian kernel used for smoothing
    imgSmth is the resulting smoothed image
    """

    kernel = Gaussian2DKernel(kernSize)

    imgSmth = convolve_fft(img, kernel, nan_treatment='fill',normalize_kernel=True)

    return imgSmth
    
def crossCorImage(img1, img2, regions=None, oversample=20, absorption=False, ncpus=None, mode='idl', contFit1=True, contFit2=True,nContFit=50, contFitOrder=1, mxShift=4,reject=0, dispAxis=0., maxFluxLevel=0, position=None):
    """
    Determine the pixel shift between input images img1 and img2, expected to be have the same dimensions.
    Usage: shiftOut = crossCorImage(img1,img2 regions=None, oversample=20, absorption=False, ncpus=None, mode='idl', contFit1=True, contFit2=True,nContFit=50, contFitOrder=1, mxShift=4,reject=0, dispAxis=0,maxFluxLevel=0, position=None)
    img1 is the first image
    img2 is the second image
    regions is a list of pixel coordinates specifying the regions to use for cross-correlation (along the dispersion axis)
    oversample is the oversampling rate to measure sub-pixel corrections
    absorption is a boolean keyword to specify if the input images contain absorption lines
    ncpus specifies the number of processes to run in parallel
    mode is the mode to use to carry out the cross-correlation measurement ('idl', 'fft')
    contFit1 is a boolean keyword indicating whether the first spectrum should be continuum rectified
    contFit2 is a boolean keyword indicating whether the second spectrum should be continuum rectified
    nContFit is the number of pixels to use for determining continuum points
    contFitOrder is the spline fitting order to use for continuum fitting
    mxShift is the maximum shift used to specify search range (in pixels, if velocity = False; in velocity if velocity=True)
    reject is the sigma-clipping threshold to use for rejecting pixels in the continuum fitting routine
    dispAxis specifies the dispersion direction (0 - along the y-axis, 1 - along the x-axis)
    maxFluxLevel specifies the relative flux threshold (1 being the max, 0 being the min) to specify which regions of the detector are used during the cross-correlation routine
    shiftOut is a list of the measured shift between the each slice
    """

    #build input list
    inpLst = []

    if dispAxis != 0:
        img1Tmp = np.empty(img1.T.shape, dtype=img1.dtype)
        img2Tmp = np.empty(img2.T.shape, dtype=img2.dtype)
        np.copyto(img1Tmp, img1.T)
        np.copyto(img2Tmp, img2.T)
    else:
        img1Tmp = img1
        img2Tmp = img2


    if maxFluxLevel > 0:
        csSpec = np.nansum(img1Tmp,axis=0)

        #rescale
        csSpec = (csSpec-np.nanmin(csSpec))/(np.nanmax(csSpec)-np.nanmin(csSpec))
        whr = np.where(csSpec<maxFluxLevel)[0]
        for j in range(len(whr)):
            inpLst.append([img1Tmp[:,whr[j]], img2Tmp[:,whr[j]],oversample, absorption,mode, contFit1, contFit2,nContFit,contFitOrder,mxShift, False, reject, None,position])
    else:
        for j in range(img1Tmp.shape[1]):
            inpLst.append([img1Tmp[:,j], img2Tmp[:,j],oversample, absorption,mode, contFit1, contFit2,nContFit,contFitOrder,mxShift, False, reject, None,position])

    if ncpus is None:
        ncpus = mp.cpu_count()
    pool = mp.Pool(ncpus)

    shiftLst = pool.map(crossCorPixMP,inpLst)

    pool.close()

    if dispAxis==0:
        shiftOut = np.empty(img1.shape[1])
    else:
        shiftOut = np.empty(img1.shape[0])
    shiftOut[:] = np.nan
    
    if maxFluxLevel >0:
        for i in range(len(whr)):
            shiftOut[whr[i]] = shiftLst[i]
    else:
        for i in range(len(shiftLst)):
            shiftOut[i] = shiftLst[i]
    
    return shiftOut
    
def crossCorImageCL(img1, img2, regions=None, oversample=20, absorption=False, ncpus=None, contFit1=True, contFit2=True,nContFit=50, contFitOrder=1, mxShift=4,reject=0, dispAxis=0., maxFluxLevel=0, position=None):
    """
    Determine the pixel shift between input images img1 and img2, expected to be have the same dimensions.
    Usage: shiftOut = crossCorImage(img1,img2 regions=None, oversample=20, absorption=False, ncpus=None, mode='idl', contFit1=True, contFit2=True,nContFit=50, contFitOrder=1, mxShift=4,reject=0, dispAxis=0,maxFluxLevel=0, position=None)
    img1 is the first image
    img2 is the second image
    regions is a list of pixel coordinates specifying the regions to use for cross-correlation (along the dispersion axis)
    oversample is the oversampling rate to measure sub-pixel corrections
    absorption is a boolean keyword to specify if the input images contain absorption lines
    ncpus specifies the number of processes to run in parallel
    mode is the mode to use to carry out the cross-correlation measurement ('idl', 'fft')
    contFit1 is a boolean keyword indicating whether the first spectrum should be continuum rectified
    contFit2 is a boolean keyword indicating whether the second spectrum should be continuum rectified
    nContFit is the number of pixels to use for determining continuum points
    contFitOrder is the spline fitting order to use for continuum fitting
    mxShift is the maximum shift used to specify search range (in pixels, if velocity = False; in velocity if velocity=True)
    reject is the sigma-clipping threshold to use for rejecting pixels in the continuum fitting routine
    dispAxis specifies the dispersion direction (0 - along the y-axis, 1 - along the x-axis)
    maxFluxLevel specifies the relative flux threshold (1 being the max, 0 being the min) to specify which regions of the detector are used during the cross-correlation routine
    shiftOut is a list of the measured shift between the each slice
    """

    if dispAxis != 0:
        img1tmp = np.empty(img1.T.shape, dtype=img1.dtype)
        img2tmp = np.empty(img2.T.shape, dtype=img2.dtype)
        np.copyto(img1tmp, img1.T)
        np.copyto(img2tmp, img2.T)
    else:
        img1tmp = img1
        img2tmp = img2


    #now only use regions of interest
    #by building a new spectrum with only the good regions, padded by a 0 in between
    
    if regions is not None:
        for reg in regions:
            if 'img1Tmp' in locals():
                img1Tmp= np.append(img1Tmp,img1tmp[reg[0]:reg[1],:],axis=0)
            else:
                img1Tmp = img1tmp[reg[0]:reg[1],:]
            img1Tmp = np.append(img1Tmp,np.zeros((mxShift,img1tmp.shape[1])),axis=0)

            if 'img2Tmp' in locals():
                img2Tmp=np.append(img2Tmp,img2tmp[reg[0]:reg[1],:],axis=0)
            else:
                img2Tmp = img2tmp[reg[0]:reg[1],:]

            img2Tmp=np.append(img2Tmp,np.zeros((mxShift, img2tmp.shape[1])),axis=0)
           
    else:
        img1Tmp = img1tmp
        img2Tmp = img2tmp
                
    t1 = time.time()
    #first subtract continuum
    #build input list
    inpLst = []

    for i in range(img1Tmp.shape[1]):
        inpLst.append([img1Tmp[:,i],1,50,1])
        
    if ncpus is None:
        ncpus = mp.cpu_count()
    pool = mp.Pool(ncpus)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        contLst = pool.map(splineContFitMP,inpLst)
        pool.close()
    
        for i in range(img1Tmp.shape[1]):
            img1Tmp[:,i] = img1Tmp[:,i] - contLst[i]

        for i in range(img1Tmp.shape[1]):
            inpLst.append([img2Tmp[:,i],1,50,1])
        
        pool = mp.Pool(ncpus)
        contLst = pool.map(splineContFitMP,inpLst)
        pool.close()
    
        for i in range(img1Tmp.shape[1]):
            img2Tmp[:,i] = img2Tmp[:,i] - contLst[i]

    
    img1Tmp[~np.isfinite(img1Tmp)]=0
    img2Tmp[~np.isfinite(img2Tmp)]=0

    xOld = np.arange(img1Tmp.shape[0])
    xNew = np.linspace(0,xOld[-1],num=img1Tmp.shape[0]*oversample)

    fInterp = interp1d(xOld,img1Tmp, kind='linear',bounds_error=False,fill_value=0,axis=0)
    img1Int = fInterp(xNew)
    fInterp = interp1d(xOld,img2Tmp, kind='linear',bounds_error=False,fill_value=0,axis=0)
    img2Int = fInterp(xNew)
    
    img1Int -= np.mean(img1Int)
    img2Int -= np.mean(img2Int)
    
    nx = img1Int.shape[0]
    ny = img2Int.shape[1]
        
    lag = np.arange(-mxShift*oversample, mxShift*oversample+1).astype(np.int32)
    path = os.path.dirname(__file__)
    clCodePath = path+'/opencl_code'

    ctx = cl.create_some_context(interactive=True)
    queue = cl.CommandQueue(ctx)
    
    filename = clCodePath+'/crosscor.cl'
    f = open(filename, 'r')
    fstr = "".join(f.readlines())
    program = cl.Program(ctx, fstr).build()
    mf = cl.mem_flags

    program.xcorPosLag.set_scalar_arg_dtypes([np.uint32, np.uint32, None, None,None,None])
    img1_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img1Int.astype('float32'))
    img2_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img2Int.astype('float32'))
    p_pos = np.zeros(lag.shape[0]/2+1, dtype='float32')
    p_pos_buf = cl.Buffer(ctx, mf.WRITE_ONLY, p_pos.nbytes)
    lag_pos = lag[lag.shape[0]/2:]
    lag_pos_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = lag_pos)
    program.xcorPosLag(queue,(p_pos.shape),None,np.uint32(nx), np.uint32(ny),img1_buf, img2_buf,lag_pos_buf, p_pos_buf)
    cl.enqueue_copy(queue, p_pos_buf, p_pos).wait()

    #now negative lags
    p_neg = np.zeros(lag.shape[0]/2, dtype='float32')
    p_neg_buf = cl.Buffer(ctx, mf.WRITE_ONLY, p_neg.nbytes)
    lag_neg = lag[:lag.shape[0]/2]
    lag_neg_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = lag_neg)
    program.xcorNegLag(queue,(p_neg.shape),None,np.uint32(nx), np.uint32(ny),img1_buf, img2_buf,lag_neg_buf, p_neg_buf)
    cl.enqueue_copy(queue, p_neg_buf, p_neg).wait()

    p = np.append(p_neg,p_pos)
    shiftOut = lag[np.nanargmax(p)]/np.float(oversample)

    return shiftOut

def subScaledSkySlices2(waveMap, obsSlices, skySlices, waveGridProps,hdr,mxScale=0.5, regions=None, MP=True, ncpus=None,nContFit=50, fluxThresh=0.1,fitInd=False, saveFile='', logfile=None, missing_left_slice=False, missing_right_slice=False):
    """
    Routine used to scale sky lines and subtract from image slices.
    Usage: subSlices = subScaledSkySlices2(waveMap, obsSlices, skySlices, waveGridProps, hdr, mxScale=0.5,  regions=None, MP=True, ncpus=None,nContFit=50,fluxThresh=0.1,fitInd=False, saveFile='', logfile=None)
    waveMap is a list of wavelength mapping images for each slice
    obsSlices is the observation slices
    skySlices is the sky slices
    waveGridProps is a list of the grid properties associated with the waveMap ([min wave, max wave, number of grid points])
    hdr is an astropy header object corresponding to the header associated with the observation
    mxScale is the maximum allowable scaled difference between sky/obs spectrum (e.g. 0.2 -> sky can be scaled in range of 0.8-1.2)
    regions is a list of wavelength coordinates specifying the different sky regions/bands
    MP is a boolean keyword specifying if multiprocessing should be used
    ncpus is an integer variable specfying the number of processes to use in MP mode
    nContFit is the number of pixels to use during continuum fitting
    fluxThresh is the relative flux ratio of lines to find and scale (relative to the line with the maximum flux in each region)
    fitInd is a boolean keyword to support scaling/fitting of individual lines (True) or of regions as a whole (False)
    saveFile is the name of the output file that will contain some QC plots
    logfile is a file object corresponding to the log file
    """

    #create fully gridded slices from input slices
    print('Placing sky and science slices on uniform wavelength grid for sky subtraction')
    if logfile is not None:
        logfile.write('Placing sky and science slices on uniform wavelength grid for sky subtractiton\n')
    dataTmpGrid = createCube.waveCorAll(obsSlices, waveMap, waveGridProps=waveGridProps)
    skyTmpGrid = createCube.waveCorAll(skySlices, waveMap, waveGridProps=waveGridProps)

    #get temporary cubes
    dataTmpCube = createCube.mkCube(dataTmpGrid, ndiv=0, missing_left=missing_left_slice, missing_right=missing_right_slice).astype('float32')
    skyTmpCube = createCube.mkCube(skyTmpGrid, ndiv=0, missing_left=missing_left_slice, missing_right=missing_right_slice).astype('float32')
    hdrTmp = copy.copy(hdr[:])
    headers.getWCSCube(dataTmpCube, hdrTmp, 1, 1, waveGridProps)
            
    print('Getting median spectra for skyline scaling')
    if logfile is not None:
        logfile.write('Getting median spectra for skyline scaling\n')
    obsSpec = np.nanmedian(dataTmpCube, axis=[0,1])
    skySpec = np.nanmedian(skyTmpCube,axis=[0,1])
    wave = 1e9*(np.arange(skyTmpCube.shape[2])*hdrTmp['CDELT3'] +hdrTmp['CRVAL3'])
    del dataTmpGrid
    del dataTmpCube
    del skyTmpGrid
    del skyTmpCube
    del hdrTmp
    
    #find continuum levels
    obs = obsSpec - splineContFitMP([obsSpec, 1, nContFit, 1])
    sky = skySpec - splineContFitMP([skySpec, 1, nContFit, 1])

    obs[~np.isfinite(obs)]=0
    sky[~np.isfinite(sky)]=0
    
    if regions is None:
        regions = [[wave.min(), wave.max()]]
        
    if np.all(~np.isfinite(obs)) or np.all(~np.isfinite(sky)):
        return obsSlices

    #get bounds based on input
    if mxScale is not None:
        bounds = [1.-mxScale, 1.+mxScale]
    else:
        bounds = [-np.inf, np.inf]

    #plot continuum subtracted sky spectrum for QC purposes
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(wave,sky)
    plt.xlabel('Wavelength')
    plt.ylabel('Continuum subtracted flux')
    plt.plot([wave.min(), wave.max()],[fluxThresh,fluxThresh],'--')

    #plot region windows
    ylim = ax.get_ylim()
    for reg in regions:
        plt.plot([reg[0],reg[0]],[ylim[0],ylim[1]],'k--')
        plt.plot([reg[1],reg[1]],[ylim[0],ylim[1]],'k--')

    plt.tight_layout()
    plt.savefig(saveFile+'_skyThresh.pdf',dpi=600)
    plt.close()

    fig = plt.figure() # plot QC scalings
    ax = fig.add_subplot(111)
    
    print('Finding scaling factors')
    if logfile is not None:
        logfile.write('Finding scaling factors\n')


    #list to hold all scalings
    fOut = []
    rngOut = []

    #notes
    #need to improve line identification/ranges

    #now find scaling factors for each line/blend/region
    for reg in regions:

        rngOut.append([])
        fOut.append([])
        
        #concatenated line arrays
        otmp = np.asarray([])
        stmp = np.asarray([])
        wtmp = np.asarray([])
        
        #go through region and find every line with strength >=fluxThresh
        whrReg = np.where(np.logical_and(wave >= reg[0], wave<=reg[1]))[0]
        wReg = wave[whrReg]
        sReg = sky[whrReg]
        oReg = obs[whrReg]
        
        strt = 0
        end=0
        while strt < sReg.shape[0]:
            if not sReg[strt] >= fluxThresh:
                strt+=1
                end=strt+1
            else:
                #count how many pixels > threshold
                cnt = 1
                end = strt+1

                if end < sReg.shape[0]:
                    val = sReg[end]
                    
                    while val >= fluxThresh:
                        cnt +=1
                        end+=1

                        if end >= sReg.shape[0]:
                            break
                        else:
                            val = sReg[end]
                                    
                #only consider a true line if >2 pixels
                if cnt <3:
                    strt=end
                else:
                    #add line to concatenated spectrum for fitting and add wavelength range to list
                    #include one extra pixel on either side
                    if strt >0:
                        st =strt - 1
                    else:
                        st = strt
                        
                    if end <sReg.shape[0]:
                        nd= end + 1
                    else:
                        nd = end
                    
                    rngOut[-1].append([wReg[st],wReg[nd-1]])
                    otmp = np.append(otmp,oReg[st:nd])
                    wtmp = np.append(wtmp,wReg[st:nd])
                                            
                    #fit this line region on its own, if desired
                    if fitInd:
                        f = fitSky(oReg[st:nd],sReg[st:nd])
                        if (f < bounds[0]):
                            f = bounds[0]
                        elif (f > bounds[1]):
                            f = bounds[1]
                        fOut[-1].append(f)
                        stmp = np.append(stmp,sReg[st:nd]*f)
                        plt.plot(wtmp.mean(),f,'bo')

                    else:
                        stmp = np.append(stmp,sReg[st:nd])
                                            
  
                strt = end 
                cnt = 0

        if not fitInd:
            f = fitSky(otmp, stmp)[0]
            if (f < bounds[0]):
                f = bounds[0]
            elif (f > bounds[1]):
                f = bounds[1] 
            fOut[-1].append(f)
            plt.plot((reg[0]+reg[1])/2.,f, 'bo')

    #plot region windows

    ylim=ax.get_ylim()
    for reg in regions:
        plt.plot([reg[0],reg[0]],[ylim[0],ylim[1]],'k--')
        plt.plot([reg[1],reg[1]],[ylim[0],ylim[1]],'k--')

    plt.tight_layout()
    plt.savefig(saveFile+'_sky_scalings.pdf')
    plt.close()

    print('Subtracting scaled sky slices from science slices')
    #now set up list to run sky subtraction on individual spectra
    inpLst = []
    for wave,oSlice,sSlice in zip(waveMap,obsSlices,skySlices):
        for i in range(len(oSlice)):
            inpLst.append([wave[i],oSlice[i,:],sSlice[i,:],rngOut,nContFit,fOut])

    if MP:
        if ncpus is None:
            ncpus = mp.cpu_count()
        pool = mp.Pool(ncpus)
        subLst = pool.map(subScaledSkySpec,inpLst)
        pool.close()
    else:
        subLst = []
        for i in range(len(inpLst)):
            subLst.append(subScaledSkySpec(inpLst[i]))

    #reconstruct output as slices
    outSlices = []
    k=0
    for i in range(len(obsSlices)):
        slcTmp = np.empty(obsSlices[i].shape,dtype=obsSlices[i].dtype)
        for j in range(slcTmp.shape[0]):
            slcTmp[j,:] = subLst[k]
            k+=1
        outSlices.append(slcTmp)
    return outSlices
        
def subScaledSkySpec(input):
    """
    Routine used to subtract a scaled sky emission line spectrum from the observed spectrum
    Usage: specOut = subScaledSkySpec(input)
    input is a list containing:
    wave - a numpy array containing the wavelength at each point
    obs - a numpy array containing the observed spectrum
    sky - a numpy array containing the sky spectrum
    regions - a list of wavelength coordinates indicating the starting and ending wavelength of the regions to carry out the sky scaling
    nContFit - an integer indicating the number of pixels used to identify continuum points
    factors - the scaling factors used to scale individual lines or regions
    specOut is the scaled, sky-subtracted spectrum
    """
    wave = input[0]
    obs = input[1]
    sky = input[2]
    regions = input[3]
    nContFit = input[4]
    factors = input[5]
    
    skyCont = splineContFitMP([sky, 1, nContFit, 1])
    skyScaled = np.zeros(sky.shape,dtype=sky.dtype)

    if regions is None:
        regions = [[wave.min(), wave.max()]]
        
    if np.all(~np.isfinite(obs)) or np.all(~np.isfinite(sky)):
        return obs
    
    outside = np.ones(sky.shape).astype(bool)

    #construct the sky spectrum outside of the scaled region
    for reg in regions:
        if len(reg)>1:
            for rng in reg:
                rng = np.asarray(rng)
                whr = np.where(np.logical_and(wave>=rng.min(),wave<=rng.max()))[0]
                outside[whr] = False
        else:
            whr = np.where(np.logical_and(wave>=reg[0],wave<=reg[1]))[0]
            outside[whr] = False

    skyScaled[outside] = sky[outside]
    
    for reg,f in zip(regions,factors):
        for i in range(len(reg)):
            reg = np.asarray(reg)
            whr = np.where(np.logical_and(wave>=reg[i].min(),wave<=reg[i].max()))[0]
            stmp = (sky-skyCont)[whr]
            #plt.plot(obs[whr])
            #plt.plot(sky[whr])
            if len(f)>1:
                skyScaled[whr]=f[i]*stmp+skyCont[whr]
            else:
                skyScaled[whr]=f*stmp+skyCont[whr]
            #plt.plot(skyScaled[whr])
            #plt.show()
    return obs - skyScaled
    
