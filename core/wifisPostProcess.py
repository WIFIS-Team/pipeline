import wifisIO
import matplotlib.pyplot as plt
import numpy as np
import astropy.convolution as convolution
from scipy.interpolate import spline
import wifisWaveSol
from scipy.interpolate import interp1d
import multiprocessing as mp
from scipy.optimize import curve_fit
import warnings

def splineContFit(x,y,regions, lineRegions=None,order=3,winRng=10.):
    """
    Use scipy spline fitting to determine continuum shape of input spectrum.
    Usage: xfit, yfit, contFit = splineContFit(x,y,regions, order=3, winRng=10)
    x is the input wavelength array
    y is the input flux array
    regions is a list of starting/stopping wavelength coordinates, used to confine the continuum regions (e.g. regions = [[x1, x2], [x3, x4], [x5,x6]])
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
    
    contFit = spline(xfit,yfit,x,order=order)

    return xfit, yfit, contFit

def crossCorCube(wave, cube1, cube2, regions=None, oversample=20, absorption=False, ncpus=None, mode='idl', contFit=True, nContFit=50, contFitOrder=1, mxShift=4,reject=0, velocity=False):
    """
    Determine the pixel shift (if velocity=False) between input cubes cube1 and cube2, expected to be on the same wavelength grid/coordinate system x. Determines velocity difference, if velocity=True. mxShift is in pixels or km/s.
    Usage: 
    """

    cube1tmp = np.empty(cube1.shape, dtype=cube1.dtype)
    cube2tmp = np.empty(cube1.shape, dtype=cube1.dtype)

    cube1tmp[:] = np.nan
    cube2tmp[:] = np.nan
     
    if regions is not None:
        for reg in regions:
            whr = np.where(np.logical_and(wave>=reg[0],wave<=reg[1]))[0]
            cube1tmp[:,:,whr] = cube1[:,:,whr]
            cube2tmp[:,:,whr] = cube2[:,:,whr]
    else:
        cube1tmp = cube1tmp
        cube2tmp = cube2tmp
        
    #build input list
    
    inpLst = []

    if velocity:
        v = (wave-wave[int(wave.shape[0]/2)])/wave[int(wave.shape[0]/2)]*2.99792458e5 # in km/s
        vconst = np.linspace(v[0],v[-1],num=v.shape[0]*oversample)
        
        for i in range(cube1.shape[0]):
            for j in range(cube1.shape[1]):
                inpLst.append([vconst,v, cube1tmp[i,j,:], cube2tmp[i,j,:],oversample, absorption,mode, contFit, nContFit,contFitOrder,mxShift, False, reject])
    else:
        for i in range(cube1.shape[0]):
            for j in range(cube1.shape[1]):
                inpLst.append([cube1tmp[i,j,:], cube2tmp[i,j,:],oversample, absorption,mode, contFit, nContFit,contFitOrder,mxShift, False, reject])

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
    Usage: 
    """

    y1= input[0]
    y2 = input[1]
    oversample=input[2]
    absorption= input[3]
    mode = input[4]
    contFit = input[5]
    nContFit = input[6]
    contFitOrder = input[7]
    mxShift = input[8]
    plot = input[9]
    reject =input[10]
    
    #quit if either input arrays are all NaN
    if np.all(~np.isfinite(y1)) or np.all(~np.isfinite(y2)):
        return np.nan
    
    if contFit:
        y1tmp = y1 - splineContFitMP([y1,contFitOrder,nContFit, reject])
        y2tmp  = y2 - splineContFitMP([y2,contFitOrder,nContFit, reject])
    else:
        y1tmp = np.empty(y1.shape)
        np.copyto(y1tmp,y1)
        y2tmp = np.empty(y2.shape)
        np.copyto(y2tmp,y2)

    x = np.arange(y1.shape[0])
    xInt = np.linspace(0,x[-1],num=y1.shape[0]*oversample)
    
    y1Int = np.interp(xInt,x, y1tmp)
    y2Int = np.interp(xInt, x, y2tmp)
    
    if absorption:
        y1tmp = 1. - y1tmp
        y2tmp = 1. - y2tmp

    if mxShift is not None:
        rng = np.arange(-mxShift*oversample, mxShift*oversample)
    else:
        rng = np.arange(-xInt.shape[0]-2, xInt.shape[0]-2)

    yCC = crossCorIDL([y1Int, y2Int, rng])
    shiftOut = rng[np.argmax(yCC)]/oversample

    if plot:
        fig = plt.figure()
        plt.plot(rng/oversample, yCC)
        plt.show()
 
    return shiftOut

def crossCorSpec(wave, spec1, spec2, regions=None, oversample=20, absorption=False, mode='idl', contFit=True, nContFit=50,contFitOrder=1, mxVel=200, plot=False, reject=0, velocity=True):
    """
    Determine the velocity difference between input spectra cube1 and cube2, expected to be on the same wavelength grid/coordinate system x.
    Usage: 
    """

    #go through regions list and construct whrLst
    #whrLst = []

    y1tmp = np.empty(spec1.shape, dtype=spec1.dtype)
    y2tmp = np.empty(spec1.shape, dtype=spec1.dtype)

    y1tmp[:] = np.nan
    y2tmp[:] = np.nan
    
    if regions is not None:
        for reg in regions:
            whr = np.where(np.logical_and(wave>=reg[0],wave<=reg[1]))[0]
            y1tmp[whr] = spec1[whr]
            y2tmp[whr] = spec2[whr]
    else:
        y1tmp = spec1
        y2tmp = spec2
        
    #now compute velocity grid and interpolate velocity grid
    v = (wave-wave[int(wave.shape[0]/2)])/wave[int(wave.shape[0]/2)]*2.99792458e5 # in km/s
    vconst = np.linspace(v[0],v[-1],num=v.shape[0]*oversample)

    if velocity:
        rvOut = crossCorVelMP([vconst, v, y1tmp, y2tmp, oversample, absorption, mode,contFit,nContFit,contFitOrder,mxVel, plot, reject])
    else:
        rvOut = crossCorPixMP([y1tmp, y2tmp, oversample, absorption, mode,contFit,nContFit,contFitOrder,mxVel, plot, reject])

    return rvOut
    
def splineContFitMP(input):
    """
    Use scipy spline fitting to determine continuum shape of input spectrum.
    Usage: xfit, yfit, contFit = splineContFit(x,y,regions, order=3, winRng=10)
    x is the input wavelength array
    y is the input flux array
    regions is a list of starting/stopping wavelength coordinates, used to confine the continuum regions (e.g. regions = [[x1, x2], [x3, x4], [x5,x6]])
    order is a keyword used to specify the order of the spline fitting to use (1 = linear, 2 = quadratic, 3 = cubic, etc.)
    winRng is a keyword used to set the wavelength range from which to identify the pixel with the maximum flux, which will be used to determine the cubic spline fitting.
    xfit, yfit, and contFit are the output, where xfit and yfit are the x and y values of the points used for fitting and contFit is the continuum fit to the original x positions.
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
    
    contFit = spline(xfit,yfit,x,order=order)

    return contFit

def crossCorIDL(input):
    """
    lags in range must integer values
    """

    y1 = input[0]
    y2 = input[1]
    rng = input[2]

    p = np.zeros(rng.shape[0])
    y1sub = y1 - np.nanmean(y1)
    y2sub = y2 - np.nanmean(y2)
    n = y1sub.shape[0]
    
    for i in range(len(rng)):
        lag = int(rng[i])
        if lag >= 0:
            p[i] = np.nansum(y1sub[:n-lag]*y2sub[lag:])
        else:
            p[i] = np.nansum(y2sub[:n+lag]*y1sub[-lag:])
            
        p[i] /= np.sqrt(np.nansum(y1sub**2)*np.nansum(y2sub**2))

    return p
        
def crossCorVelMP(input):
    """
    Determine the velocity difference between input spectra y1 and y2 on the same wavelength grid/coordinate system x.
    Usage: 
    """

    vconst = input[0]
    v = input[1]
    y1= input[2]
    y2 = input[3]
    oversample=input[4]
    absorption= input[5]
    mode = input[6]
    contFit = input[7]
    nContFit = input[8]
    contFitOrder = input[9]
    mxVel = input[10]
    plot = input[11]
    reject =input[12]
    
    #quit if either input arrays are all NaN
    if np.all(~np.isfinite(y1)) or np.all(~np.isfinite(y2)):
        return np.nan
    
    #go through regions list and construct new arrays from subsets

    if contFit:
        y1 -= splineContFitMP([y1,contFitOrder,nContFit, reject])
        y2 -= splineContFitMP([y2,contFitOrder,nContFit, reject])    

    if mode != 'idl':    
        #now get rid of NaNs
        y1[~np.isfinite(y1)] = 0.
        y2[~np.isfinite(y2)] = 0.

    #now compute and interpolate onto constant velocity grid, and converting to line intensity (i.e. 1 - continuum)
    y1Const = np.interp(vconst,v,y1,left=0,right=0)
    y2Const = np.interp(vconst, v, y2, left=0, right=0)
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
            rng = np.arange(-np.ceil(mxVel/dv), np.ceil(mxVel/dv))
        else:
            rng = np.arange(-vconst.shape[0]-2, vconst.shape[0]-2)

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
    """

    popt, pcov = curve_fit(scaleSky,y2, y1, p0=[1.], bounds=bounds)

    return popt

def subScaledSkyPix(input):
    """
    """
    wave = input[0]
    obs = input[1]
    sky = input[2]
    regions = input[3]
    bounds = input[4]
    sigmaClip = input[5]
    sigmaClipRounds = input[6]
    
    obsCont = splineContFitMP([obs, 1, 50, 1])
    skyCont = splineContFitMP([sky, 1, 50, 1])
    skyScaled = np.zeros(sky.shape,dtype=sky.dtype)

    if np.all(~np.isfinite(obs)) or np.all(~np.isfinite(sky)):
        fOut = []
        for reg in regions:
            fOut.append(np.nan)
        return obs, fOut
    
    #outside = []
    fOut = []

    outside = np.ones(sky.shape).astype(bool)
    
    #construct the sky spectrum outside of the scaled region
    for reg in regions:
        whr = np.where(np.logical_and(wave>=reg[0],wave<=reg[1]))[0]
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
        
        if sigmaClipRounds>0:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                for i in range(sigmaClipRounds):
                    y = otmp-stmp*f
                    med = np.nanmedian(y)
                    std = np.nanstd(y)
                    badPix = np.where(np.abs(y-med)>sigmaClip*std)[0]
                    otmp[badPix] = 0
                    stmp[badPix] = 0
                    f = fitSky(otmp, stmp)

                    if (f < bounds[0]):
                        f = bounds[0]
                    elif (f > bounds[1]):
                        f = bounds[1]
             
        fOut.append(f)
        skyScaled[whr] = skyCont[whr] + (sky[whr]-skyCont[whr])*f

    #now add in regions of sky not part of scaling algorithm
    #skyScaled[:outside[0][0]] = sky[:outside[0][0]]

    #for i in range(1,len(outside)):
    #    skyScaled[outside[i-1][1]+1:outside[i][0]] = sky[outside[i-1][1]+1:outside[i][0]]

    #skyScaled[outside[-1][1]+1:] = sky[outside[-1][1]+1:]

    return obs - skyScaled, fOut
    
def subScaledSkyCube(wave, obs, sky, mxScale=0.5, regions=None, MP=True, ncpus=None,sigmaClip=3, sigmaClipRounds=1):
    """
    mxScale is maximum allowable scaled difference between sky/obs spectrum (e.g. 0.2 -> sky can be scaled in range of 0.8-1.2).
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
            inpLst.append([wave,obs[i,j,:],sky[i,j,:],regions, bounds,sigmaClip,sigmaClipRounds])
    
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

def subScaledSkySlices(waveMap, obsSlices, skySlices, mxScale=0.5, regions=None, MP=True, ncpus=None,sigmaClip=3, sigmaClipRounds=1):
    """
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
            inpLst.append([waveMap[i][j,:],obsSlices[i][j,:],skySlices[i][j,:],regions, bounds,sigmaClip,sigmaClipRounds])
    
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
            subSlices.append(np.empty(skySlices[i].shape, dtype=skySlices[i].dtype))
            subSlices[i][:]=np.nan
            for j in range(skySlices[i].shape[0]):
                subLst = subScaledSkyPix(inpLst[k])
                subSlices[i][j,:] = subLst[0]
                fSlices[i].append(subLst[k][1])
                k+=1

    return subSlices, fSlices
  
def crossCorSlices(waveMap, slices1, slices2, regions=None, oversample=20, absorption=False, ncpus=None, mode='idl', contFit=True, nContFit=50, contFitOrder=1, mxShift=4,reject=0, velocity=False):
    """
    Determine the pixel shift (if velocity=False) between input cubes cube1 and cube2, expected to be on the same wavelength grid/coordinate system x. Determines velocity difference, if velocity=True. mxShift is in pixels or km/s.
    Usage: 
    """

    slices1tmp = []
    slices2tmp = []
 
    #go through each slice and copy needed regions
    if regions is not None:
        for i in range(len(slices1)):
            slices1tmp.append(np.zeros(slices1[i].shape,dtype=slices1[i].dtype))
            slices2tmp.append(np.zeros(slices2[i].shape,dtype=slices2[i].dtype))
            for reg in regions:
                whr = np.where(np.logical_and(waveMap[i]>=reg[0],waveMap[i]<=reg[1]))
                slices1tmp[i][whr[0],whr[1]] = slices1[i][whr[0],whr[1]]
                slices2tmp[i][whr[0],whr[1]] = slices2[i][whr[0],whr[1]]
    else:
        for i in range(len(slices1)):
            slices1tmp.append(np.zeros(slices1[i].shape,dtype=slices1[i].dtype))
            slices2tmp.append(np.zeros(slices2[i].shape,dtype=slices2[i].dtype))
            np.copyto(slices1tmp[i],slices1[i])
            np.copyto(slices2tmp[i],slices2[i])


    #build input list
    inpLst = []

    if velocity:
        
        for i in range(len(slices1)):
            for j in range(slices1[i].shape[0]):
                v = (waveMap[i][j,:]-waveMap[i][j,:][int(waveMap[i][j,:].shape[0]/2)])/waveMap[i][j,:][int(waveMap[i][j,:].shape[0]/2)]*2.99792458e5 # in km/s
                vconst = np.linspace(v[0],v[-1],num=v.shape[0]*oversample)
                inpLst.append([vconst,v, slices1tmp[i,j,:], slices2tmp[i,j,:],oversample, absorption,mode, contFit, nContFit,contFitOrder,mxShift, False, reject])
    else:
        for i in range(len(slices1)):
            for j in range(slices1[i].shape[0]):
                inpLst.append([slices1tmp[i][j,:], slices2tmp[i][j,:],oversample, absorption,mode, contFit, nContFit,contFitOrder,mxShift, False, reject])

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
        shiftOut.append(slices1[i].shape, dtype=slices1[i].dtype)

    k=0
    for i in range(len(shiftOut)):
        for j in range(shiftOut[i].shape[0]):
            shiftOut[i][j,:] = shiftLst[k]
            k+=1            

    return shiftOut
    
