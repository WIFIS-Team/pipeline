"""

Tools to create final data cube

"""

import numpy as np
import multiprocessing as mp
import wifisIO
import time
from scipy.interpolate import griddata
import astropy.convolution as conv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def compSpatGrid(distTrimSlices):
    """
    Routine to compute the maximum, minimum and average resolution element for spatial coordinates from a list of trimmed distortion corrected spatial map image slices.
    Usage: [spatMin, spatMax, dSpat] = compSpatGrid(distTrimSlices)
    distTrimSlices should be a list of image slices containing the spatial maps for each slice, that have already been distortion corrected and trimmed.
    spatMin is the minimum spatial coordinate
    spatMax is the maximum spatial coordinate
    dSdpat is the mean dispersion or resolution of all slices
    """
    
    dMin = []
    dMax = []
    deltaD = []
    for d in distTrimSlices:
        dmin=np.nanmax(d[0,:])
        dmax= np.nanmin(d[-1,:])

        dMin.append(dmin)
        dMax.append(dmax)

        n = d.shape[0]
        deltaD.append((dmax-dmin)/(n-1))
        
    spatMin = np.min(dMin)
    spatMax = np.max(dMax)
    dSpat = np.mean(deltaD)
    return([spatMin,spatMax,dSpat])

def distCorAll(dataSlices, distMapSlices, method='linear', ncpus=None):
    """
    Routine to distortion correct a list of slices.
    Usage: outLst = distCorAll(dataSlices, distMapSlices, method='linear', ncpus=None)
    dataSlices is the list of the data slices to be distortion corrected
    distMapSlices is the list of slices containing the distortion mapping
    method is a keyword to control how the interpolation is carried out (default is using bilinear interpolation)
    ncpus is a keyword that allows to control the number of multi-processesing processes
    returned is a list of distortion corrected slices.
    """

    lst = []
    # setup input list
    for i in range(len(dataSlices)):
        lst.append([dataSlices[i], distMapSlices[i],method])

    #setup multiprocessing
    if (ncpus==None):
        ncpus =mp.cpu_count()

    pool = mp.Pool(ncpus)
    
    #run multiprocessing of the code
    outLst = pool.map(distCorSlice, lst)
    pool.close()

    return outLst

def distCorSlice(input):
    """
    Routine to distortion correct individual slices.
    Usage out = distCorSlice(input)
    input is a list that contains:
    dataSlc - is the image slice of the input data to be distortion corrected,
    distSlc - is the distortion mapping for the specific slice
    method - is a string indicating the interpolation method to use ("linear", "cubic", or "nearest")
    """

    #rename input
    dataSlc = input[0]
    distSlc = input[1]
    method = input[2]
    
    #get spatial grid properties
    nSpat = dataSlc.shape[0]
    dSpat = (np.max(distSlc) - np.min(distSlc))/(nSpat-1)
    minSpat = np.min(distSlc)
    maxSpat = np.max(distSlc)

    #setup linearized output grid
    gX, gY = np.mgrid[minSpat:maxSpat+dSpat:dSpat, :dataSlc.shape[1]]

    #use wavelength and desired spatial grid size to map onto new grid
    points = []
    vals = []

    for i in range(dataSlc.shape[0]):
        for j in range(dataSlc.shape[1]):
            if (~np.isnan(distSlc[i,j])):
                points.append([distSlc[i,j],j])
                vals.append(dataSlc[i,j])

    vals = np.array(vals)
    
    out = griddata(points, vals, (gX, gY), method=method)

    return out

def compWaveGrid(waveTrim, dispSol):
    """
    Determines mean dispersion solution to use for placing all slices on a uniform wavelength grid. 
    Usage: result = compWaveGrid(waveTrim, dispSol)
    waveTrim is a list of all trimmed wave map slices, providing the wavelength at each pixel for a given slice
    dispSol is an array containing the wavelength solution derived for each vector along the dispersion axis on the detector.
    """
    
    #compute mean dispersion with 3-sigma clipping
    fnte = np.where(np.isfinite(dispSol[:,1]))[0]
    gridDisp = np.median(dispSol[fnte,1])
    whr = np.where(np.abs(dispSol[fnte,1]-gridDisp) < 1*np.std(dispSol[fnte,1]))[0]
    gridDisp = np.median(dispSol[fnte[whr],1])
  
    mnLst = []
    mxLst = []
    
    for w in waveTrim:
        for i in range(w.shape[0]):
            y = w[np.where(w>0)]
            mnLst.append(np.nanmin(y))
            mxLst.append(np.nanmax(y))

    #compute min and maximum wavelengths
    gridMin = np.min(mnLst)
    gridMax = np.max(mxLst)
    
    return gridMin, gridMax, np.abs(gridDisp)

def collapseCube(cube):
    """
    Simple tool to collapse a complete image cube along the dispersion direction.
    Usage: out = collapseCube(cube)
    cube is the input image cube, which assumes the last index is the wavelength dispersion direction
    out is the return 2D collapsed image. NaN values are not propagated.
    """
    
    out = np.nansum(cube, axis=2)

    return out

def mkCube(corSlices, ndiv=1, MP=True, ncpus=None):
    """
    Routine to create a complete image cube from a list of grided (for both spatial and wavelength) image slices
    Usage: outCube = mkCube(corSlices, ndiv=1, MP=True, ncpus=None)
    corSlices is the list of slices placed on a uniform spatial and wavelength grid
    ndiv is a keyword to specify how to subdivide the original slices to artificially increase the output resolution along the y-axis (the between slices). A value of 0 will not increase the number of pixels between each slice. A value of 1 will introduce a single pixel between each slice and linearly interpolate to fill in that value. A value of n will introduce n pixels between each slice.
    MP is a keyword to specify whether multi-processing should be used. For some machines, setting MP=False may result in a speed improvement.
    ncpu is a keyword to specify the number of processes to run simultaneously when running in MP mode. The default is None, which allows python to set this value automatically based on the number of threads supported by your CPU.
    """
    
    #initialize output cube

    tmpCube = np.empty((len(corSlices), corSlices[0].shape[0],corSlices[0].shape[1]))
    tmpCube[:] = np.nan
    
    for i in range(tmpCube.shape[0]):
        tmpCube[i,:,:] = corSlices[i]
        
    #interpolate cube onto different y-grid (x-grid already done in previous step)
    out = interpCube(tmpCube, ndiv=ndiv, ncpus=ncpus)

    outCube = np.empty((out[0].shape[0],out[0].shape[1],len(out)))
    outCube[:] = np.nan
    for i in range(len(out)):
        outCube[:,:,i] = out[i]
        
    return outCube

def interpCube(inCube, ndiv=1, MP=True, ncpus=None):
    """
    Routine to interpolate between image slices
    Usage: out = interpCube(inCube, ndiv=1, MP=True, ncpus=None)
    inCube is the original image cube without
    ndiv is a keyword to specify how to subdivide the original slices to artificially increase the output resolution along the y-axis (the between slices). A value of 0 will not increase the number of pixels between each slice. A value of 1 will introduce a single pixel between each slice and linearly interpolate to fill in that value. A value of n will introduce n pixels between each slice.
    MP is a keyword to specify whether multi-processing should be used. For some machines, setting MP=False may result in a speed improvement.
    ncpu is a keyword to specify the number of processes to run simultaneously when running in MP mode. The default is None, which allows python to set this value automatically based on the number of threads supported by your CPU.
    """

    if (MP):
        if (ncpus == None):
            ncpus =mp.cpu_count()
        pool = mp.Pool(ncpus)

        #build input list
        lst = []
        for i in range(inCube.shape[2]):
            lst.append([inCube[:,:,i],ndiv])
            
        out = pool.map(interpFrame,lst)
        pool.close()
    else:
        out = []
        for i in range(inCube.shape[2]):
            out.append(interpFrame(inCube[i], ndiv))
        
    return out

    
def interpFrame(input):
    """
    Routine to carry out interpolation between pixels (along the y-axis) for individual frames
    Usage: out = interpFrame(input)
    input is a list containing:
    img - the input image frame to be interpolated onto new grid
    ndiv - the number of subdivisions to increase the pixel count (along the y-axis) between the original pixels
    """

    #get input
    img = input[0]
    ndiv = input[1]
    ny = (ndiv)*(img.shape[0]-1)+img.shape[0]
    
    #old grid points
    yold = np.arange(img.shape[0])
    
    #compute new grid points, ensuring the endpoints are kept
    dy = (img.shape[0])/np.float(ny)
    y = np.linspace(0,img.shape[0]-1,num=ny)
    
    out = np.empty((ny,img.shape[1]))
    out[:] = np.nan

    #now move through frame, interpolating along the y-axis.
    #********************************************************

    for i in range(0,y.shape[0]-1):
        
        ia = np.floor(y[i]).astype('int')
        ib = np.clip(ia+1,0,img.shape[0]-1)
        whrA = np.where(~np.isnan(img[ia,:]))[0]
        whrB = np.where(~np.isnan(img[ib,:]))[0]
        out[i,whrA] = (ib-y[i])*img[ia,whrA]
        out[i,whrB] += (y[i]-ia)*img[ib,whrB]

    return out

def mkWaveSpatGridAll(dataSlices, waveMapSlices, distMapSlices,waveGridProps, spatGridProps,method='linear', ncpus=None):
    """
    Routine to place all image slices on a uniform spatial and wavelength grid.
    Usage: mkWaveSpatGridAll(dataSlices, waveMapSlices, distMapSlices, waveGridProps, spatGridProps,methd='linear', ncpus=None)
    dataSlices is a list of the original image slices to be place on the uniform grid
    waveMapSlices is a list containing slices that provide the wavelength mapping for each pixel in the slices
    distMapSlices is a list containing slices that provide the spatial mapping for each pixel in the slices
    waveGridProps is a list specifying the properties of the wavelength grid - the minimum wavelength coordinate, the maximum wavelength coordinate, and the dispersion along the wavelength axis
    spatGridProps is a list specifying the properties of the spatial grid - the minimum spatial coordinate, the maximum spatial coordinate, and the dispersion along the spatial direction
    method is a keyword indicating what type of interpolation to use ("nearest" neighbour, bi-"linear" interpolation, or "cubic" interpolation)
    ncpus is a keyword allowing one to specify the maximum number of processes to be run simultaneously when carrying out this task.
    """

    #setup input list for multiprocessing
    lst = []
    for i in range(len(dataSlices)):
        lst.append([dataSlices[i], waveMapSlices[i], distMapSlices[i],waveGridProps, spatGridProps,method])

    #setup multiprocessing
    if (ncpus == None):
        ncpus =mp.cpu_count()
    pool = mp.Pool(ncpus)
    
    #run the multiprocessing code
    outLst = pool.map(mkWaveSpatGridSlice, lst)
    pool.close()

    return outLst

def mkWaveSpatGridSlice(input):
    """ 
    Routine to place image slice on a uniform spatial and wavelength grid.
    Usage: out= mkWaveSpatGridSlice(input) 
    input is a list containing:
    dataSlc - the original image slice to be place on the uniform grid
    waveSlc - the wavelength mapping for each pixel in the slice
    distSlc - the spatial mapping for each pixel in the slice
    waveGridProps - a list specifying the properties of the wavelength grid - the minimum wavelength coordinate, the maximum wavelength coordinate, and the dispersion along the wavelength axis
    spatGridProps - a list specifying the properties of the spatial grid - the minimum spatial coordinate, the maximum spatial coordinate, and the dispersion along the spatial direction
    method - indicating what type of interpolation to use ("nearest" neighbour, bi-"linear" interpolation, or "cubic" interpolation)
    """
    
    #rename input
    dataSlc = input[0]
    waveSlc = input[1]
    distSlc = input[2]
    waveGridProps = input[3]
    spatGridProps = input[4]
    method = input[5]
    
    #get spatial grid properties
    minSpat = spatGridProps[0]
    maxSpat = spatGridProps[1]
    dSpat = spatGridProps[2]

    #get wavelength grid properties
    maxWave =waveGridProps[1]
    minWave = waveGridProps[0]
    dWave = waveGridProps[2]

    #setup linearized output grid
    gX, gY = np.mgrid[minSpat:maxSpat+dSpat:dSpat, minWave:maxWave+dWave:dWave]

    #use wavelength and desired spatial grid size to map onto new grid
    points = []
    vals = []

    for i in range(dataSlc.shape[0]):
        for j in range(dataSlc.shape[1]):
            if (~np.isnan(waveSlc[i,j]) and waveSlc[i,j]!=0 and ~np.isnan(distSlc[i,j])):
                points.append([distSlc[i,j],waveSlc[i,j]])
                vals.append(dataSlc[i,j])

    vals = np.array(vals)
    
    out = griddata(points, vals, (gX, gY), method=method)

    return out
