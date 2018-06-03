"""

Set of tools used during the creation of the final data cube

"""

import numpy as np
import multiprocessing as mp
import wifisIO
import time
from scipy.interpolate import griddata
from scipy import interpolate
import astropy.convolution as conv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def compSpatGrid(distTrimSlices):
    """
    Routine to compute the grid properties (maximum, minimum and total number of spatial coordinates) from a list of trimmed distortion corrected spatial map image slices.
    Usage: [spatMin, spatMax, N] = compSpatGrid(distTrimSlices)
    distTrimSlices should be a list of image slices containing the spatial maps for each slice, that have already been distortion corrected and trimmed.
    spatMin is the minimum spatial coordinate
    spatMax is the maximum spatial coordinate
    N is the final number of grid points to be used during spatial rectification (gridding)
    """
    
    dMin = []
    dMax = []
    deltaD = []
    for d in distTrimSlices:
        #all pixels should have the same coordinate, but just in case
        dmin=np.nanmedian(d[0,:]) 
        dmax= np.nanmedian(d[-1,:])

        dMin.append(dmin)
        dMax.append(dmax)

        n = d.shape[0]
        deltaD.append((dmax-dmin)/(n-1))
        
    spatMin = np.min(dMin)
    spatMax = np.max(dMax)
    dSpat = np.median(deltaD)
    N = int((spatMax-spatMin)/dSpat)
    return([spatMin,spatMax,N])

def distCorAll(dataSlices, distMapSlices, method='akima', ncpus=None, spatGridProps=None, MP=True,smooth=0,nMult=1, ):
    """
    Routine to distortion correct a list of slices.
    Usage: outLst = distCorAll(dataSlices, distMapSlices, method='linear', ncpus=None,spatGridProps=None, MP=True)
    dataSlices is the list of the data slices to be distortion corrected
    distMapSlices is the list of slices containing the distortion mapping
    method is a keyword to control how the interpolation is carried out (default is using akima interpolation along the spatial axis). Options are: 'linear', 'cubic', 'akima', or 'nearest'
    ncpus is a keyword that allows to control the number of multi-processesing processes
    spatGridProps  is the an optional keyword that provides the properties for the output spatial grid
    MP is a boolean keyword used to specify if multiprocessing should be used.
    returned is a list of distortion corrected slices.
    """

    lst = []
    # setup input list
    for i in range(len(dataSlices)):
        lst.append([dataSlices[i], distMapSlices[i],method,spatGridProps,smooth,nMult])

    if (MP):
        
        #setup multiprocessing
        if (ncpus==None):
            ncpus =mp.cpu_count()
            
        pool = mp.Pool(ncpus)
    
        #run multiprocessing of the code
        outLst = pool.map(distCorSlice1D, lst)
        pool.close()
    else:
        outLst = []

        for i,l in enumerate(lst):
            print('Working on slice ', i)
            outLst.append(distCorSlice1D(l))
            
    return outLst

def distCorSlice1D(input):
    """
    Routine to distortion correct individual slices.
    Usage out = distCorSlice1D(input)
    input is a list that contains:
    dataSlc - is the image slice of the input data to be distortion corrected,
    distSlc - is the distortion mapping for the specific slice
    method - is a string indicating the interpolation method to use ("linear", or "akima"). Not used for anything other than compatability with older function.
    spatGridProps - the final output properties of the spatial grid. If set to None, then the code automatically determines the grid properties for the slice
    Returned is the distortion corrected image slice placed on a grid.
    """

    dataSlc = np.copy(input[0])
    distSlc = np.copy(input[1])
    method = input[2] 
    spatGridProps = input[3]
    smooth = input[4]
    nMult = input[5]
    
    #get spatial grid properties if not provided
    if (spatGridProps is not None):
        minSpat = float(spatGridProps[0])
        maxSpat = float(spatGridProps[1])
        nSpat = float(spatGridProps[2])
    else:
        nSpat = dataSlc.shape[0]
        minSpat = np.nanmin(distSlc)
        maxSpat = np.nanmax(distSlc)

    #get output coordinate array
    xout = np.linspace(minSpat,maxSpat, num=int(nSpat*nMult))

    #get output density
    dSpat = ((maxSpat-minSpat)/float(nSpat-1))
    
    #initialize output distortion corrected map
    out = np.empty((xout.shape[0], dataSlc.shape[1]), dtype=dataSlc.dtype)
    out[:] = np.nan

    #determine gradient of coordinate map for converting flux to flux density
    gradMap = np.gradient(distSlc,axis=0)
   
    if smooth>0:
        gKern = conv.Gaussian1DKernel(smooth)
    
    for i in range(dataSlc.shape[1]):
        if smooth>0:
            y = conv.convolve(dataSlc[:,i],gKern,normalize_kernel=True, boundary='extend')
        else:
            y = dataSlc[:,i]
            
        #get input coordinates and coordinate span per pixel
        x = distSlc[:,i]
        d = gradMap[:,i]

        #convert flux to flux density
        y /=d
        whr = np.where(~np.isnan(y))[0]
        whrNan = np.where(np.isnan(y))[0]

        if str(method).lower() == 'akima':
            try:
                fInt = interpolate.Akima1DInterpolator(x[whr],y[whr])
                out[:,i] = fInt(xout)*dSpat
            except:
                raise Warning('Akima interpolation method failed at column '+str(i))
        elif str(method).lower() == 'linear':
            
        #the old method
            out[:,i] = np.interp(xout,x,y, right=np.nan,left=np.nan)*dSpat
        else:
            raise Warning("*** INTERPOLATION METHOD MUST BE SET TO AKIMA OR LINEAR ***")
            
        #************************************
        #NOW DEAL WTIH NANS
        #************************************

        if len(whrNan)>0:
            for j in range(len(whrNan)):
                rng = np.where(np.logical_and(xout>=x[whrNan[j]]-d[whrNan[j]]/2.,xout<=x[whrNan[j]]+d[whrNan[j]]/2.))[0]

                out[rng,i] = np.nan
                
        
    return out    

def distCorSlice(input):
    """
    Routine to distortion correct individual slices.
    Usage out = distCorSlice(input)
    input is a list that contains:
    dataSlc - is the image slice of the input data to be distortion corrected,
    distSlc - is the distortion mapping for the specific slice
    method - is a string indicating the interpolation method to use ("linear", "cubic", or "nearest")
    spatGridProps - the final output properties of the spatial grid. If set to None, then the code automatically determines the grid properties for the slice.
    Returned is a distortion corrected image.
    """

    #rename input
    dataSlc = input[0]
    distSlc = input[1]
    method = input[2]
    spatGridProps = input[3]
    
    #get spatial grid properties if not provided
    if (spatGridProps is not None):
        minSpat = float(spatGridProps[0])
        maxSpat = float(spatGridProps[1])
        nSpat = float(spatGridProps[2])
        dSpat = (maxSpat-minSpat)/(nSpat-1)
    else:
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
            if (np.isfinite(distSlc[i,j])):
                points.append([distSlc[i,j],j])
                vals.append(dataSlc[i,j])

    vals = np.array(vals)
    
    out = griddata(points, vals, (gX, gY), method=method)

    return out

def compWaveGrid(waveTrimSlices):
    """
    Determines mean dispersion solution to use for placing all slices on a uniform wavelength grid. 
    Usage: [gridMin, gridMax, N] = compWaveGrid(waveTrimSlices)
    waveTrim is a list of all trimmed wave map slices, providing the wavelength at each pixel for a given slice
    gridMin is the minimum wavelength of the grid
    gridMax is the maximum wavelength of the grid
    N is the total number of grid points between the min and maximum wavelength of the grid
    """


    wMin = []
    wMax = []
    deltaW = []
    for w in waveTrimSlices:
        #get a version of collapsed slice
        wCol = np.nanmedian(w, axis=0)

        #find first non-NaN column
        whr = np.where(np.isfinite(wCol))[0]
        wTmp = wCol[whr]
        
        wmin=np.nanmin(w[:,whr])
        wmax= np.nanmax(w[:,whr])

        wMin.append(wmin)
        wMax.append(wmax)

        n = wTmp.shape[0]
        deltaW.append((wmax-wmin)/(n-1))
        
    waveMin = np.min(wMin)
    waveMax = np.max(wMax)
    waveDisp = np.abs(np.median(deltaW))
    N = int((waveMax-waveMin)/waveDisp)
    
    return waveMin, waveMax, N#np.abs(waveDisp)

def collapseCube(cube):
    """
    Simple tool to collapse a complete image cube along the dispersion direction.
    Usage: out = collapseCube(cube)
    cube is the input image cube, which assumes the last index is the wavelength dispersion direction
    out is the return 2D collapsed image. NaN values are not propagated.
    """
    
    out = np.nansum(cube, axis=2)

    return out

def mkCube(corSlices, ndiv=1, MP=True, ncpus=None, missing_left=False, missing_right=False):
    """
    Routine to create a complete image cube from a list of grided (for both spatial and wavelength) image slices
    Usage: outCube = mkCube(corSlices, ndiv=1, MP=True, ncpus=None)
    corSlices is the list of slices placed on a uniform spatial and wavelength grid (all having the same dimensions/properties)
    ndiv is a keyword to specify how to subdivide the original slices to artificially increase the output resolution along the y-axis (the between slices). A value of 0 will not increase the number of pixels between each slice. A value of 1 will introduce a single pixel between each slice and linearly interpolate to fill in that value. A value of n will introduce n pixels between each slice.
    MP is a keyword to specify whether multi-processing should be used. For some machines, setting MP=False may result in a speed improvement.
    ncpu is a keyword to specify the number of processes to run simultaneously when running in MP mode. The default is None, which allows python to set this value automatically based on the number of threads supported by your CPU.
    Returns an image cube with the first axis corresponding to the y-axis of the image, the 2nd to the x-axis, and the third to the wavelength axis.    """
    
    #initialize output cube

    tmpCube = np.empty((18, corSlices[0].shape[0],corSlices[0].shape[1]),dtype=corSlices[0].dtype)
    tmpCube[:] = np.nan

    #first check how many slices exist

    if len(corSlices)==18:
        #if all 18 slices present use the following
        #place slices in correct order!
        tmpCube[0,:,:] = corSlices[16]
        tmpCube[1,:,:] = corSlices[14]
        tmpCube[2,:,:] = corSlices[12]
        tmpCube[3,:,:] = corSlices[10] 
        tmpCube[4,:,:] = corSlices[8] 
        tmpCube[5,:,:] = corSlices[6]  
        tmpCube[6,:,:] = corSlices[4] 
        tmpCube[7,:,:] = corSlices[2]  
        tmpCube[8,:,:] = corSlices[0]  
        tmpCube[9,:,:] = corSlices[1] 
        tmpCube[10,:,:] = corSlices[3]
        tmpCube[11,:,:] = corSlices[5]
        tmpCube[12,:,:] = corSlices[7]
        tmpCube[13,:,:] = corSlices[9] 
        tmpCube[14,:,:] = corSlices[11] 
        tmpCube[15,:,:] = corSlices[13] 
        tmpCube[16,:,:] = corSlices[15]   
        tmpCube[17,:,:] = corSlices[17]   

    elif len(corSlices)==17:
        #if only 17 slices present
        #if the left-most (centre-field) slice is missing, replace it with NaNs
        print('Found ' + str(len(corSlices))+' slices to reconstruct cube')

        if missing_left:
            tmpCube[0,:,:] = corSlices[15]
            tmpCube[1,:,:] = corSlices[13]
            tmpCube[2,:,:] = corSlices[11]
            tmpCube[3,:,:] = corSlices[9] 
            tmpCube[4,:,:] = corSlices[7] 
            tmpCube[5,:,:] = corSlices[5]  
            tmpCube[6,:,:] = corSlices[3] 
            tmpCube[7,:,:] = corSlices[1]
            tmpSlice = np.empty(corSlices[0].shape,dtype=corSlices[0].dtype)
            tmpSlice[:] = np.nan
            tmpCube[8,:,:] = tmpSlice  
            tmpCube[9,:,:] = corSlices[0] 
            tmpCube[10,:,:] = corSlices[2]
            tmpCube[11,:,:] = corSlices[4]
            tmpCube[12,:,:] = corSlices[6]
            tmpCube[13,:,:] = corSlices[8] 
            tmpCube[14,:,:] = corSlices[10] 
            tmpCube[15,:,:] = corSlices[12] 
            tmpCube[16,:,:] = corSlices[14]   
            tmpCube[17,:,:] = corSlices[16]

        elif missing_right:
            #if only 17 slices present
            #if the right-most (centre-field) slice is missing, replace it with NaNs
            tmpSlice = np.empty(corSlices[0].shape,dtype=corSlices[0].dtype)
            tmpSlice[:] = np.nan
            tmpCube[0,:,:] = corSlices[16]
            tmpCube[1,:,:] = corSlices[14]
            tmpCube[2,:,:] = corSlices[12]
            tmpCube[3,:,:] = corSlices[10] 
            tmpCube[4,:,:] = corSlices[8] 
            tmpCube[5,:,:] = corSlices[6]  
            tmpCube[6,:,:] = corSlices[4] 
            tmpCube[7,:,:] = corSlices[2]  
            tmpCube[8,:,:] = corSlices[0]  
            tmpCube[9,:,:] = corSlices[1] 
            tmpCube[10,:,:] = corSlices[3]
            tmpCube[11,:,:] = corSlices[5]
            tmpCube[12,:,:] = corSlices[7]
            tmpCube[13,:,:] = corSlices[9] 
            tmpCube[14,:,:] = corSlices[11] 
            tmpCube[15,:,:] = corSlices[13] 
            tmpCube[16,:,:] = corSlices[15]   
            tmpCube[17,:,:] = tmpSlice

        else:
            raise Warning('*** ONLY 17 SLICES DETECTED. THE MISSING SLICE MUST BE SPECIFIED USING THE CORRECT KEYWORD ***')
    else:
        raise Warning('*** MORE THAN 18 SLICES DETECTED. ONLY 18 SLICES EXPECTED ***')

    #interpolate cube onto different y-grid (x-grid already done in previous step)
    out = interpCube(tmpCube, ndiv=ndiv, ncpus=ncpus)

    outCube = np.empty((out[0].shape[0],out[0].shape[1],len(out)))
    outCube[:] = np.nan
    for i in range(len(out)):
        outCube[:,:,i] = out[i]
        
    return outCube

def interpCube(inCube, ndiv=1, MP=True, ncpus=None):
    """
    Routine to linearly interpolate between image slices
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
        whrA = np.where(np.isfinite(img[ia,:]))[0]
        whrB = np.where(np.isfinite(img[ib,:]))[0]
        out[i,whrA] = (ib-y[i])*img[ia,whrA]
        out[i,whrB] += (y[i]-ia)*img[ib,whrB]

    return out

def mkWaveSpatGridAll(dataSlices, waveMapSlices, distMapSlices,waveGridProps, spatGridProps,method='linear', ncpus=None):
    """
    *** THIS ROUTINE IS NO LONGER USED BY THE PIPELINE ***
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
    *** THIS ROUTINE IS NO LONGER USED BY THE PIPELINE ***
    Routine to place image slice on a uniform spatial and wavelength grid.
    Usage: out= mkWaveSpatGridSlice(input) 
    input is a list containing:
    dataSlc - the original image slice to be place on the uniform grid
    waveSlc - the wavelength mapping for each pixel in the slice
    distSlc - the spatial mapping for each pixel in the slice
    waveGridProps - a list specifying the properties of the wavelength grid - the minimum wavelength coordinate, the maximum wavelength coordinate, and the dispersion along the wavelength axis
    spatGridProps - a list specifying the properties of the spatial grid - the minimum spatial coordinate, the maximum spatial coordinate, and the number of grid points along the spatial direction
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
            if (np.isfinite(waveSlc[i,j]) and waveSlc[i,j]!=0 and np.isfinite(distSlc[i,j])):
                points.append([distSlc[i,j],waveSlc[i,j]])
                vals.append(dataSlc[i,j])

    vals = np.array(vals)
    
    out = griddata(points, vals, (gX, gY), method=method)

    return out

def waveCorAll(dataSlices, waveMapSlices, method='linear', ncpus=None, waveGridProps=None, MP=True):
    """
    Routine to place slices on uniform wavelength grid
    Usage: outLst = waveCorAll(dataSlices, waveMapSlices, method='linear', ncpus=None)
    dataSlices is the list of the data slices to be distortion corrected
    waveMapSlices is the list of slices containing the wavelength mapping
    method is a keyword to control how the interpolation is carried out (default is using linear interpolation)
    ncpus is a keyword that allows to control the number of multi-processesing processes
    waveGridProps is a list containing the (optional) grid properties to be used by all slices.
    MP is a boolean keyword to specify if multiprocessing should be used.
    returned is a list of distortion corrected slices.
    """

    lst = []
    # setup input list
    for i in range(len(dataSlices)):
        lst.append([dataSlices[i], waveMapSlices[i],method,waveGridProps])

    if (MP):
        #setup multiprocessing
        if (ncpus==None):
            ncpus =mp.cpu_count()

        pool = mp.Pool(ncpus)
    
        #run multiprocessing of the code
        outLst = pool.map(waveCorSlice1D, lst)
        pool.close()
    else:
        outLst = []
        for l in lst:
            outLst.append(waveCorSlice1D(l))
            
    return outLst

def waveCorSlice1D(input):
    """
    Routine to place individual slices on uniform wavelength grid using linear interpolation along the wavelength axis.
    Usage out = waveCorSlice(input)
    input is a list that contains:
    dataSlc - is the image slice of the input data to be distortion corrected,
    waveSlc - is the wavelength mapping for the specific slice
    method - is a string indicating the interpolation method to use ("linear", "cubic", or "nearest"). This is only present for compatability reasons.
    waveGridProps is a list specifying the properties of the wavelength grid - the minimum and maximum wavelength coordinate, and the number of grid points along the wavelength direction. If None is given, the code automatically determines these properties for the given slice.
    Returned is a slice image placed on a uniform linearized wavelength grid.
    """

    #rename input
    dataSlc = input[0]
    waveSlc = input[1]
    method = input[2] #only present for compatability with older version
    waveGridProps = input[3]
    
    if waveGridProps is not None:
        minWave = float(waveGridProps[0])
        maxWave = float(waveGridProps[1])
        nWave = float(waveGridProps[2])
    else:
        #get wave grid properties
        nWave = dataSlc.shape[1]
        minWave = np.nanmin(waveSlc)
        maxWave = np.nanmax(waveSlc)

    #get the parameters for the gridded data
    xout = np.linspace(minWave,maxWave, num=int(nWave))
    dWave = (maxWave-minWave)/float(nWave-1.)
    out = np.empty((dataSlc.shape[0], xout.shape[0]),dtype=dataSlc.dtype)
    out[:] = np.nan

    #determine gradient of coordinate map for converting flux to flux density
    diffMap = np.abs(np.gradient(waveSlc,axis=1))
 
    for i in range(dataSlc.shape[0]):
        srt = np.argsort(waveSlc[i,:])
        x = waveSlc[i,srt]
        y = dataSlc[i,srt]/diffMap[i,srt]

        #fInt = interpolate.Akima1DInterpolator(x[whr],y[whr])
        out[i,:] = np.interp(xout,x,y, right=np.nan,left=np.nan)*dWave
        #out[i,:] = fInt(xout)*dWave
        
    return out    
           

def waveCorSlice(input):
    """
    *** THIS ROUTINE IS NO LONGER USED BY THE PIPELINE ***
    Routine to place individual slices on uniform wavelength grid.
    Usage out = waveCorSlice(input)
    input is a list that contains:
    dataSlc - is the image slice of the input data to be distortion corrected,
    waveSlc - is the wavelength mapping for the specific slice
    method - is a string indicating the interpolation method to use ("linear", "cubic", or "nearest")
    Returned is a list of distortion corrected images.
    """

    #rename input
    dataSlc = input[0]
    waveSlc = input[1]
    method = input[2]
    waveGridProps = input[3]
    
    if waveGridProps is not None:
        minWave = float(waveGridProps[0])
        maxWave = float(waveGridProps[1])
        nWave = float(waveGridProps[2])
        dWave = (maxWave-minWave)/(nWave-1)
    else:
        #get wave grid properties
        nWave = dataSlc.shape[1]
        dWave = (np.nanmax(waveSlc) - np.nanmin(waveSlc))/(nWave-1)
        minWave = np.nanmin(waveSlc)
        maxWave = np.nanmax(waveSlc)
    
    #setup linearized output grid
    gX, gY = np.mgrid[:dataSlc.shape[0], minWave:maxWave+dWave:dWave]

    #use wavelength and desired spatial grid size to map onto new grid
    points = []
    vals = []

    for i in range(dataSlc.shape[0]):
        for j in range(dataSlc.shape[1]):
            if (np.isfinite(waveSlc[i,j])):
                points.append([i,waveSlc[i,j]])
                vals.append(dataSlc[i,j])

    vals = np.array(vals)
    
    out = griddata(points, vals, (gX, gY), method=method)

    return out
