"""

Tools to help determine spatial or distortion corrections

"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import multiprocessing as mp
import wifisIO
import time
import astropy.convolution as conv
from scipy.interpolate import griddata
from scipy.interpolate import RectBivariateSpline
from astropy.modeling import models, fitting
from scipy.interpolate import interp1d

def gaussFit(x, y, plot=False):
    """
    Routine to fit a Gaussian to provided x and y data points and return the fitted coefficients.
    Usage: params = gaussFit(x, y, guessWidth, plot=True/False)
    x is the 1D array of coordinates
    y is the 1D array of data values at the given coordinates
    plot is a keyword to allow for plotting of the fit (for debug purposes)
    params is a list containing in the fitted parameters in the following order: offset, amplitude, centre, width
    """
 
    #use scipy to do fitting
    offsetGuess = (y[0]+y[-1])/2. # use edges of input to range to guess at offset
    #ampGuess = np.min(y)- offsetGuess #if absorption profile
    ampGuess =  np.max(y) - offsetGuess
    
    popt, pcov = curve_fit(gaussian, x, y, p0=[offsetGuess, ampGuess, np.mean(x),1.])

    if (plot == True):
        plt.close('all')
        plt.plot(x,y)
        plt.plot(x, y, 'ro')
        plt.plot(x, gaussian(x, popt[0], popt[1], popt[2], popt[3]))
        xtmp = np.mgrid[x[0]:x[-1]:0.1]
        plt.plot(xtmp, gaussian(xtmp, popt[0], popt[1], popt[2], popt[3]))
        plt.show()
        
    return popt

def gaussian(x,offset,amp,cen,wid):
    """
    Returns a Gaussian function of the form y = A + b*exp(-z^2/2), where z=(x-cen)/wid
    Usage: output = gaussian(x, offset, amp, cen, wid)
    x is the input 1D array of coordinates
    offset is the vertical offset of the Gaussian
    amp is the amplitude of the Gaussian
    cen is the centre of the Gaussian
    wid is the 1-sigma width of the Gaussian
    returns an array with same size as x corresponding to the Gaussian solution
    """
    
    z = (x-cen)/wid    
    return offset + amp*np.exp(-z**2/2.)

def getFit2(x, y, mxWidth=1,plot=False):
    """
    Task used to determine centroid of provided sample
    Usage c = getFit2(x,y, mxWidth=1, plot=False)
    x is the x-coordinates of the sample
    y is the y-coordinates (or values) of the provided sample
    mxWidth is a keyword used to set the expected maximum width of a Gaussian function, for quality of fit control
    plot is keyword to plot the results for debugging
    """
    try:
        gfit = gaussFit2(x,y,plot=plot)
        c = gfit[1]
        width = np.abs(gfit[2])
        
        if (c < x[0] or c > x[-1]) or (width > mxWidth):
            mid = int(len(y)/2)
            mx = np.argmax(y[mid-1:mid+2])+mid-1
            cc = np.nansum(x[mx-1:mx+2]*y[mx-1:mx+2])/np.nansum(y[mx-1:mx+2])
            width = np.nan
            
            if (plot):
                plt.plot(x,y)
                plt.plot([cc,cc],[np.min(y), np.max(y)])
                plt.plot([c,c], [np.min(y),np.max(y)],'--')
                plt.show()
            c = cc
    except (RuntimeError, ValueError):
        mid = len(y)/2
        mx = np.nanargmax(y[int(mid-1):int(mid+2)])+mid-1
        c = np.nansum(x[int(mx-1):int(mx+2)]*y[int(mx-1):int(mx+2)])/np.nansum(y[int(mx-1):int(mx+2)])
        width = np.nan
    return c,width

def fitColumn(pos,slce,allTrace, winRng, reverse=False, plot=False, prnt=False, mxWidth=1,bright=False):
    """
    Task to trace Ronchi mask
    Usage: trace = fitColumn(pos, slc, allTrace, winRng, reverse=False, plot=False, prnt=False, mxWidth=1, bright=False)
    pos is the current column number that is being traced
    slce is the slice corresponding to the Ronchi image that needs tracing
    allTrace is an array that stores the results of the tracing
    winRng is the maximum range for which the fit can deviate about the initial guess
    reverse is a keyword used to control which direction the tracing occurs (reverse=False -> the trace proceeds to higher coordinates; reverse=True -> the fit proceeds to lower coordinates)
    plot is a keyword to plot the fitting, for debugging purposes
    prnt is a keyword to print the results of the fitting, for debugging purposes
    mxWidth is a keyword used to control the quality of fit of the Gaussian used to determine the centring
    bright is a keyword used to select which part of the Ronchi mask to trace (bright=True is used to trace the bright bands, bright=False traces the dark bands)
    trace is an array containing the values of trace for each Ronchi band at the particular position
    """

    #extract Ronchi information at given position
    y = slce[:,pos]
    nDips = allTrace.shape[0]
    trace =np.zeros(nDips)
    traceWidth = np.empty(nDips)
    traceWidth[:] = np.nan

    #compute 2nd derivative for identifying centre of bright or dark bands for tracing
    d2 = np.gradient(np.gradient(y))
    #d2 = y 
    x = np.arange(len(y))
    winRng2 = int(winRng/2)
    
    #find the centroid of each band/dip
    for j in range(0,nDips):

        prevMeas = allTrace[j,:]

        #exclude bad measurements
        whr1 = np.where(prevMeas != 0)[0]
        whr2 = np.where(np.isfinite(prevMeas[whr1]))[0]

        if (reverse):
            prevFit = prevMeas[whr1[whr2[0]]]
        else:
            prevFit = prevMeas[whr1[whr2[-1]]]
       
        #find dip
        oldDip = np.round(prevFit).astype('int')
        xtmp = (np.arange(winRng)+ oldDip - winRng2).astype('int')
        xtmp = xtmp[np.where(xtmp < len(y))]
      
        #update centre position for better fitting, but only allow for 1-pixel shift
        ytmp = d2[xtmp]
        
        if (bright):
            try:
                dipPos = xtmp[np.nanargmin(ytmp[winRng2-1:winRng2+2])+winRng2-1]
            except(ValueError):
                dipPos=np.nan
        else:
            try:
                dipPos = xtmp[(np.nanargmax(ytmp[winRng2-1:winRng2+2])+winRng2-1).astype('int')]
            except(ValueError):
                dipPos = np.nan

        if (dipPos is not np.nan):
            xtmp = (np.arange(winRng)+dipPos - winRng2).astype('int') #can remove once know quality
            xtmp = xtmp[np.where(xtmp < len(y))]

            if (len(xtmp)>2):
                #fit Gaussian or parabola to region to determine line centre
                if (bright):
                    yfit = np.nanmin(d2[xtmp])-d2[xtmp]
                    yfit -= np.nanmin(yfit)
                else:
                    yfit = d2[xtmp] - np.nanmin(d2[xtmp])
                    
                if (len(xtmp)>2):
                    trace[j],traceWidth[j]  =  getFit2(xtmp, yfit, plot=plot, mxWidth=mxWidth)
                    
                else:
                    trace[j] = np.nan

                if (prnt):
                    print(trace[j], prevFit-trace[j])
            else:
                trace[j] = np.nan
                yfit = [np.nan]
        else:
            trace[j] = np.nan
            yfit = [np.nan]
            
        #avoid bad fits

        if (np.isnan(trace[j])):
            badFit = True
        else:
            if ((np.abs(trace[j]-prevFit)) > mxWidth/2.):
                badFit = True
            else:
                badFit = False
                
        if (badFit): 

            #compute centre as weighted average instead
            try:
                if (len(xtmp)>2 and len(yfit)>2):
                    mxPos = np.nanargmax(yfit[winRng2-1:winRng2+2])
                    #xtmp = xtmp[int(np.arange(int(winRng/2))-1+mxPos)]
                    xtmp = xtmp[int(mxPos-winRng2):int(mxPos+winRng2+1)]
                    ytmp = np.abs(d2[xtmp])
                    c = np.nansum(xtmp*ytmp)/np.nansum(ytmp)
                else:
                    c = np.nan
            except(ValueError):
                c = np.nan
           
            if (prnt):
                print(pos, j, trace[j], prevFit, prevFit-trace[j], c, prevFit-c)

            if (np.isnan(c)):
                trace[j] = np.nan
            elif (np.abs(c-prevFit) > mxWidth):
                trace[j] = np.nan
            else:
                trace[j] = c
                                
        if (plot):
            plt.figure()
            plt.plot(d2)
            plt.plot(xtmp, d2[xtmp], 'ro')
            plt.plot([trace[j],trace[j]], [np.min(d2),np.max(d2)],'--')
            plt.show()   


    return trace, traceWidth


def gaussian2(x,amp,cen,wid):
    """
    Returns a Gaussian function of the form y = A*exp(-z^2/2), where z=(x-cen)/wid
    Usage: output = gaussian(x, amp, cen, wid)
    x is the input 1D array of coordinates
    amp is the amplitude of the Gaussian
    cen is the centre of the Gaussian
    wid is the 1-sigma width of the Gaussian
    returns an array with same size as x corresponding to the Gaussian solution
    """
    
    z = (x-cen)/wid    
    return amp*np.exp(-z**2/2.)

def gaussFit2(x, y, plot=False):
    """
    Routine to fit a Gaussian to provided x and y data points and return the fitted coefficients.
    Usage: params = gaussFit(x, y, guessWidth, plot=True/False)
    x is the 1D array of coordinates
    y is the 1D array of data values at the given coordinates
    guessWidth is the initial guess for the width of the Gaussian
    plot is a keyword to allow for plotting of the fit (for debug purposes)
    params is a list containing in the fitted parameters in the following order: amplitude, centre, width
    """
 
    #use scipy to do fitting
    ampGuess = np.min(y)-np.max(y)
    popt, pcov = curve_fit(gaussian2, x, y, p0=[ampGuess, np.mean(x),1.])

    if (plot == True):
        plt.close('all')
        plt.plot(x,y)
        plt.plot(x, y, 'ro')
        plt.plot(x, gaussian2(x, popt[0], popt[1], popt[2]))
        xtmp = np.mgrid[x[0]:x[-1]:0.1]
        plt.plot(xtmp, gaussian2(xtmp, popt[0], popt[1], popt[2]))
        plt.show()
        
    return popt

def traceRonchiSlice(input):
    """
    Routine used to trace all bands for a given Ronchi slice
    Usage: outTrace = traceRonchiSlice(input), where input is a list containing the following:
    img - the Ronchi slice image to trace
    nbin - an integer specifying the number of pixels to bin together to improve the contrast between dark/bright bands and to reduce the computation time for tracing
    winRng - the range used fit the individual bands
    plot - a boolean value indicating whether the results should be plotted - for debugging purposes only
    mxWidth - the maximum width of a Gaussian fit, which is used for quality control purposes
    smth - an integer indicating the width of the Gaussian filter used for smoothing the traces
    bright - a boolean value indicating whether the bright bands (True) or dark bands (False) should be fit
    outTrace is a 2D array providing the position of each band along the slice
    """

    img = input[0]
    nbin = input[1]
    winRng = input[2]
    winRng2 = int(winRng/2)
    maxPix  = input[3] # *** maxPix IS FOR TESTING PURPOSES ONLY ***
    plot = input[4]
    mxWidth = input[5]
    smth = input[6]
    bright = input[7]
    flatSlc = input[8]
    threshold=input[9]
    
    # REMOVE once testing complete    
    
    #first bin the image, if requested
    if (nbin > 1):
        
        #tmp = np.zeros((img.shape[0], img.shape[1])) #*** THIS IS THE PROPER CODE ***
        tmp = np.zeros((img.shape[0], maxPix/nbin))
        
        for i in range(tmp.shape[1]-1):
            tmp[:,i] = np.nansum(img[:,nbin*i:nbin*(i+1)],axis=1)#+img[:,nbin*(i+1)]
    else:
        tmp  = img

    #find column with maximum signal in median along column
    if (flatSlc is None):
        medArray = np.zeros(tmp.shape[1])
        for i in range(medArray.shape[0]):
            y = tmp[:,i]
            medArray[i] = np.nanmedian(y)
    else:
        medArray = np.zeros(tmp.shape[1])
        for i in range(medArray.shape[0]):
            y = flatSlc[:,i]
            medArray[i] = np.nanmedian(y)

    #find column with the maximum signal to identify position and number of Ronchi dips
    #m1, m2 = np.unravel_index(np.nanargmax(tmp), tmp.shape)
    m2 = np.nanargmax(medArray)

    #extract signal from this column
    y=tmp[:,m2]
    x=np.arange(len(y))
    d2 = np.gradient(np.gradient(y))
    
    #only use regions where signal in all pixels > threshold of the median signal (of the max portion)
    if (flatSlc is None):
        mx = np.nanmedian(y[y>0.05*np.nanmax(y)])
        whr = np.where(y > threshold*mx)[0]
    else:
        yFlat = flatSlc[:,m2]
        mx = np.nanmedian(yFlat[yFlat>0.05*np.nanmax(yFlat)])
        whr = np.where(yFlat > threshold*mx)[0]
       
    strt = whr[0]
    mxPix = whr[-1]

    #now start counting the # of dips
    #and get their positions

    trace =[]
    traceWidth = []
    xtmp = (np.arange(winRng)+strt).astype('int')
    ytmp = y[xtmp]
    
    if (bright):
        dipPos = xtmp[np.nanargmin(d2[xtmp])] #position of first dip
    else:
        dipPos = xtmp[np.nanargmax(d2[xtmp])] #position of first dip

    #stop when dip position
    while (dipPos < mxPix):

        xtmp = (np.arange(winRng)+dipPos - winRng2).astype('int')
        xtmp = xtmp[np.where(np.logical_and(xtmp >0,xtmp<len(y)))]
        
        #fit function to region to determine line centre
        if (bright):
            yfit = np.nanmin(d2[xtmp])-d2[xtmp]
            yfit -= np.nanmin(yfit)
        else:
            yfit = d2[xtmp] - np.nanmin(d2[xtmp])
            
        fitTmp = getFit2(xtmp, yfit, plot=plot, mxWidth=mxWidth)
        trace.append(fitTmp[0])
        traceWidth.append(fitTmp[1])
        
        if (plot):
            plt.figure()
            plt.plot(y)
            plt.plot([whr[0]-1, mxPix], [0.5*mx, 0.5*mx], 'g--')
            plt.plot(xtmp, y[xtmp], 'ro')
            plt.plot([trace[-1],trace[-1]], [np.nanmin(y),np.nanmax(y)],'--')
            plt.show()   
                
        #now start search for next dip
        strt = xtmp[-1]
        xtmp = np.arange(winRng)+strt
        xtmp = xtmp[np.where(xtmp < len(y))]
        if (bright):
            with np.errstate(invalid='raise'):
                try:
                    dipPos = xtmp[np.nanargmin(d2[xtmp])]
                except(ValueError):
                    dipPos = np.nan
        else:
            with np.errstate(invalid='raise'):
                try:
                    dipPos = xtmp[np.nanargmax(d2[xtmp])]
                except(ValueError):
                    dipPos=np.nan
                
    #count the number of dips
    nDips = len(trace)
    
    #initialize array to hold positions of each dip across detector
    #allTrace = np.zeros((tmp.shape[1],nDips))
    allTrace = np.empty((nDips, int(maxPix/nbin)))
    allWidth = np.empty((nDips, int(maxPix/nbin)))
    allTrace[:] = np.nan
    allWidth[:] = np.nan
    
    #fill in first set of measurements
    allTrace[:,m2] = trace

    #now do the rest of the columns
    #first work backwards from starting position
    for i in range(m2-1,0,-1):
        allTrace[:,i],allWidth[:,i] = fitColumn(i,tmp,allTrace,winRng=winRng, reverse=True,mxWidth=mxWidth, bright=bright)
    
    #now work forwards
    for i in range(int(m2+1),int(maxPix/nbin)):
        allTrace[:,i],allWidth[:,i] = fitColumn(i,tmp,allTrace, winRng=winRng, mxWidth=mxWidth,bright=bright)

    #now smooth all traces and fill in missing values due to binning

    outTrace = np.empty((allTrace.shape[0], allTrace.shape[1]*nbin))
    outTrace[:] = np.nan

    outWidth = np.empty((allWidth.shape[0], allWidth.shape[1]*nbin))
    outWidth[:] = np.nan
    
    xTrace = np.arange(allTrace.shape[1])*nbin
    xOut = np.arange(outTrace.shape[1])

    if (smth > 0):
        for j in range(allTrace.shape[0]):
            yTrace = allTrace[j,:]
            wTrace = allWidth[j,:]
            
            #remove badly fit regions to be replaced by smoothed curve
            ytmp = np.empty(xOut.shape)
            ytmp[:] = np.nan
            ytmp[xTrace] = yTrace
            gKern = conv.Gaussian1DKernel(smth)
            outTrace[j,:] = conv.convolve(ytmp, gKern, boundary='extend', normalize_kernel=True)

            #remove badly fit regions and replace with linear interpolation
            wtmp = np.empty(xOut.shape)
            wtmp[:]=np.nan
            wtmp[xTrace] = wTrace
            whrBad = np.where(~np.isfinite(wtmp))[0]

            if(len(whrBad)>0):
                whrGood = np.where(np.isfinite(wtmp))[0]
                finter = interp1d(xOut[whrGood],wtmp[whrGood], kind='linear', bounds_error=False)
                wtmp[whrBad] =finter(xOut[whrBad])
                outWidth[j,:] = wtmp
            else:
                outWidth[j,:] = wtmp
    else:
        outTrace = allTrace
        outWidth = allWidth
        
    if (plot):
        #tmp = np.sqrt(img)
        #mn = np.nanmin(img[np.where(tmp != 0)])
        mx = np.nanmax(img)
        #plt.imshow(img, aspect='auto', clim=[mn, np.max(img)])
        plt.imshow(img, aspect='auto', clim=[mx*0.05, mx])

        for i in range(outTrace.shape[0]):
            if (i%2 == 0):
                plt.plot(outTrace[i,:], 'k')
            else:
                plt.plot(outTrace[i,:], 'k--')
            plt.plot(np.repeat(m2, outTrace.shape[0]),outTrace[:,m2],'ro')
        plt.show()

    return outTrace, outWidth

def traceRonchiAll(extSlices, nbin=2, winRng=5, mxWidth=1,smth=5,bright=False, MP=True,ncpus=None, flatSlices=None, threshold=0.5):
    """
    Routine used to trace all bands for a all Ronchi slices
    Usage: result = traceRonchiAll(extSlices, nbin=2, winRng=5, mxWidth=1, smth=5, bright=False, ncpus=None)
    extSlices is a list containing the Ronchi image slices to trace
    nbin is an integer specifying the number of pixels to bin together to improve the contrast between dark/bright bands and to reduce the computation time for tracing
    winRng is the range used fit the individual bands
    mxWidth is the maximum width of a Gaussian fit, which is used for quality control purposes
    smth is an integer indicating the width of the Gaussian filter used for smoothing the traces
    bright is a boolean value indicating whether the bright bands (True) or dark bands (False) should be fit
    MP is a boolean keyword used to indicate if multiprocessing is used
    ncpus is an integer indicating the number of simultanesously run processes, when in multiprocessing mode. An input of None allows the code to automatically determine this value
    result is a list of 2D arrays providing the position of each band along the individual slice
    """

    if (MP):
        #set up input list
        lst = []

        for i in range(len(extSlices)):
            if (flatSlices is None):
                flatSlc = None
            else:
                flatSlc = flatSlices[i]
                
            lst.append([extSlices[i], nbin, winRng, extSlices[i].shape[1], False, mxWidth,smth,bright, flatSlc])

        if (ncpus == None):
            ncpus = mp.cpu_count()
        pool = mp.Pool(ncpus)
        result = pool.map(traceRonchiSlice,lst)
        pool.close()
    else:
        result = []
        for i in range(len(extSlices)):
            if (flatSlices is None):
                flatSlc = None
            else:
                flatSlc = flatSlices[i]
 
            result.append(traceRonchiSlice([slc, nbin, winRng, extSlices[i].shape[1],False, mxWidth, smth, bright, flatSlc]))

    #now organize results
    traces = []
    widths = []

    for r in result:
        traces.append(r[0])
        widths.append(r[1])
        
    return traces, widths

def extendTraceSlice(input):
    """
    Routine to interpolate the Ronchi traces onto the provided pixel grid and extrapolate the fit towards regions that fall outside the Ronchi traces for a particular slice
    Usage: z = extendTraceSlice(input), where input is a list containing:
    trace - the tracing of the Ronchi mask
    slc - the grid onto which the interpolation/extrapolation of the trace is applied
    space - the physical spacing between Ronchi bands, in any unit
    zero - the tracing of the zero-point of each slice, to provide better alignment between different slices
    method - to determine the type of interpolation ("nearest" neighbour, bi-"linear" interpolation, "cubic" interpolation)
    z is the interpolation/extrapolation of the Ronchi traces onto the grid
    """

    #rename input
    trace = input[0]
    slc = input[1]
    space = input[2]
    zero = input[3]
    method=input[4]
    order =input[5]
    
    #setup grid points for interpolation
    gridY, gridX = np.mgrid[:slc.shape[0], :trace.shape[1]]

    #get coordinates for all useful (i.e. non-NaN) traces
    points = []
    vals = []
    for i in range(trace.shape[0]):
        for j in range(trace.shape[1]):
            if (~np.isnan(trace[i,j])):
                points.append([j,trace[i,j]])
                vals.append(i*space)
    vals = np.array(vals)

    #use scipy griddata to interpolate onto grid
    z = griddata(points, vals, (gridX, gridY), method=method)

    #check if the zero points are provided
    if (zero is not None):
        #use zeropoint trace to place grid on an absolute scale
        #first determine spatial coordinate for each trace point
        zeroInterp = np.empty(zero.shape)
        zeroInterp[:] = np.nan

        #identify corresponding spatial coordinate from Ronchi interpolation
        #using linear interpolation for increased precision (as trace likely falls between two grid points)
        for i in range(zero.shape[0]):
            ia = np.floor(zero[i]).astype('int')
            ib = ia+1
            za= z[ia,i]
            zb = z[ib,i]
            zeroInterp[i] = (zero[i]-ia)*zb + (ib-zero[i])*za

        #now subtract the zero value from each point to place on absoluate grid
        z -= zeroInterp
        
    #create new array with bad pixels/rows removed
    xgood=[]
    ygood=[]
    zgood=[]

    for i in range(z.shape[0]):
        if(len(np.where(np.isnan(z[i,:]))[0])==0):
            xgood.append(gridX[i,:])
            ygood.append(gridY[i,:])
            zgood.append(z[i,:])

    xgood = np.array(xgood)
    ygood = np.array(ygood)
    zgood = np.array(zgood)

    whr = np.where(np.isnan(z))
    xbad = gridX[whr]
    ybad = gridY[whr]

    #use simple polynomial fit to interpolated Ronchi grid to extrapolate outside of grid
    pinit = models.Polynomial2D(degree=order)
    fitP = fitting.LinearLSQFitter()
    p = fitP(pinit, xgood, ygood, zgood)
    zbad = p(xbad, ybad)
    
    z[whr] = zbad
    
    return z

def extendTraceAll(traceLst, extSlices, zeroTraces,space=1/15.,order=4,method='linear', ncpus=None, MP=True):
    """
    Routine to interpolate the Ronchi traces onto the provided pixel grid and extrapolate the fit towards regions that fall outside the Ronchi traces for all slices
    Usage: interpLst = extendTraceAll(traceLst, extSlices, zeroTraces, space=5., method='linear', ncpus=None, MP=True)
    traceLst is a list providing the traces of the Ronchi mask for each individual slice
    extSlices is a list providing a slice images that will be used as a grid onto which the interpolation/extrapolation of the trace is applied
    space is the physical spacing between Ronchi bands, in any unit
    zeroTraces is a list providing the tracing of the zero-point of each slice, to provide better alignment between different slices
    method is a string indicating the type of interpolation to use ("nearest" neighbour, bi-"linear" interpolation, "cubic" interpolation)
    ncpus is an integer indicating the number of simultanesously run processes, when in multiprocessing mode. An input of None allows the code to automatically determine this value
    MP is a boolean indicating whether multiprocessing should be used
    interpLst is the list of all traces interpolated onto the provided grid
    """

    #setup input list

    if (MP ==False):
        interpLst = []

        for i in range(len(traceLst)):

            if (zeroTraces is None):
                interpLst.append(extendTraceSlice([traceLst[i], extSlices[i], space, None, method,order]))
            else:
                interpLst.append(extendTraceSlice([traceLst[i], extSlices[i], space, zeroTraces[i], method,order]))
    else:
        lst = []
        for i in range(len(traceLst)):
            if (zeroTraces is None):
                lst.append([traceLst[i], extSlices[i], space, None, method, order])
            else:        
                lst.append([traceLst[i], extSlices[i], space, zeroTraces[i],method,order])

        if (ncpus == None):
            ncpus =mp.cpu_count()
        pool = mp.Pool(ncpus)
        interpLst = pool.map(extendTraceSlice, lst)
        pool.close()

    return interpLst

def traceWireFrameAll(zeroSlices, nbin=2,winRng=31,smooth=5,bright=False,MP=True, plot=False, mxChange=5, ncpus=None):
    """
    Routine used to determine/trace zero-point images (e.g. wireframe) of each slice
    Usage: cent = traceZeroPointAll(zeroSlices, nbin=2, winRng=31, smooth=5, bright=False, MP=True, plot=False, mxChange=5, ncpus=None)
    zeroSlices is a list containing the zero-point image slices
    nbin is an integer used to specify the number of pixels used for binning each slice to improve the contrast of the zeropoint slices and reduce the computation time of the tracing
    winRng is the maximum range about the centre of the slice used to find and fit for the zeropoint centre
    smooth is an integer specifying the width of the Gaussian kernel used for smoothing the trace
    bright is a boolean used to indicate whether the feature used for tracing is brigher (True) or darker (False) then its surrounding
    MP is a boolean used to indicate whether multiprocessing should be used
    plot is a boolean used to indicate whether the results should be plotted - for debugging purposes only
    mxChange is a variable that indicates the maximum allowable change between pixels for the tracing, for quality control purposes
    ncpus is an integer specifying the number of simultaneously running processes when in MP mode
    cent is a list containing the zero-point traces of each slice
    """

    if (MP):
        if (ncpus == None):
            ncpus =mp.cpu_count()
        pool = mp.Pool(ncpus)

        #build input list
        lst = []
        for i in range(len(zeroSlices)):
            lst.append([zeroSlices[i],nbin,winRng,plot,smooth,bright,mxChange])
            
        cent = pool.map(traceWireFrameSlice, lst)
        pool.close()
        
    else:
    
        cent = []

        for i in range(len(zeroSlices)):
            cent.append(traceWireFrameSlice([zeroSlices[i],nbin,winRng,plot,smooth,bright,mxChange]))
            
    return cent

def traceWireFrameSlice(input):
    """
    Routine used to determine/trace zero-point image (e.g. wireframe) of a single slice
    Usage: centSmth = traceZeroPointSlice(input), where input is a list containing: 
    img - the zeropoint image slice
    nbin - an integer used to specify the number of pixels used for binning each slice to improve the contrast of the zeropoint slices and reduce the computation time of the tracing
    winRng - the maximum range from which to carry out the fitting
    plot - a boolean used to indicate whether the results should be plotted - for debugging purposes only
    smth - an integer specifying the width of the Gaussian kernel used for smoothing the trace
    bright - a boolean used to indicate whether the feature used for tracing is brigher (True) or darker (False) then its surrounding
    mxChange - a variable that indicates the maximum allowable change between pixels for the tracing, for quality control purposes
    centSmth is the zero-point trace for the entire slice    
    """

    #rename variables
    img = input[0]
    nbin = input[1]
    winRng = input[2]
    winRng2 = int(winRng/2)
    plot = input[3]
    smth = input[4]
    bright = input[5]
    mxChange = input[6]
    
    #first bin the image, if requested
    if (nbin > 1):
        
        tmp = np.zeros((img.shape[0], img.shape[1]/nbin))

        for i in range(tmp.shape[1]-1):
            tmp[:,i] = np.nansum(img[:,nbin*i:nbin*(i+1)],axis=1)
    else:
        tmp = img

    #find column with the maximum signal to find first zeropoint measurement
    m1, m2 = np.unravel_index(np.argmax(tmp), tmp.shape)

    #extract signal from this column

    x=np.arange(tmp.shape[0])

    #start search from the middle pixel
    mid = int(tmp.shape[0]/2.)

    #find feature in window range, and recentre on this feature
    y = tmp[:,m2]
    xrng = np.arange(winRng) - winRng2 + mid
    xrng = xrng[np.where(xrng>0)[0]]
    xrng = xrng[np.where(xrng<len(y))[0]].astype('int')

    cent = np.empty(tmp.shape[1])
    cent[:] = np.nan
    
    if (xrng.shape[0] > 0):
        yrng = y[xrng]
    
        if (bright):
            mid = xrng[np.argmax(yrng)]
        else:
            mid = xrng[np.argmin(yrng)]

        #now fit Gaussian to feature to identify centre
        xrng = np.arange(winRng) - winRng2 + mid
        xrng = xrng[np.where(xrng>0)[0]]
        xrng = xrng[np.where(xrng<len(y))[0]].astype('int')

        if (xrng.shape[0] > 0):
            yrng = y[xrng]

            #remove NaNs
            whrFinite = np.where(np.isfinite(yrng))[0]
            if len(whrFinite)>0:
                xrng = xrng[whrFinite]
                yrng = yrng[whrFinite]
                
                fit = gaussFit(xrng, yrng, plot=plot)
                
                if (fit[2] >= xrng[0] and fit[2] <= xrng[-1]):
                    cent[m2]= fit[2]

    #now go through rest of pixels to find centre as well
    #first work backwards from starting position

    midStrt = mid
    
    for i in range(m2-1,0,-1):
        y = tmp[:,i]
        xrng = np.arange(winRng) - winRng2 + mid
        xrng = xrng[np.where(xrng>0)[0]]
        xrng = xrng[np.where(xrng<len(y))[0]].astype('int')

        if (xrng.shape[0] > 0):
            yrng = y[xrng]
            midOld = mid
        
            if (bright):
                mid = xrng[np.argmax(yrng)]
            else:
                mid = xrng[np.argmin(yrng)]

            if (np.abs(mid-midOld)>mxChange):
                mid = midOld
            
            #now fit Gaussian to feature to identify centre
            xrng = np.arange(winRng) - winRng2 + mid
            xrng = xrng[np.where(xrng>0)[0]]
            xrng = xrng[np.where(xrng<len(y))[0]].astype('int')
        
            if (xrng.shape[0] > 0):
                yrng = y[xrng]

                whrFinite = np.where(np.isfinite(yrng))[0]

                if len(whrFinite)>1:

                    xrng = xrng[whrFinite]
                    yrng = yrng[whrFinite]
                    
                    try:
                        fit = gaussFit(xrng, yrng, plot=plot)

                        if (fit[2] >= xrng[0] and fit[2] <= xrng[-1]):
                            cent[i] = fit[2]
                        
                    except (RuntimeError):
                        pass
            
                 
    #now work forwards
    mid = midStrt
    
    for i in range(m2+1,tmp.shape[1]):
        y = tmp[:,i]
        xrng = np.arange(winRng) - winRng2 + mid
        xrng = xrng[np.where(xrng>0)[0]]
        xrng = xrng[np.where(xrng<len(y))[0]].astype('int')

        if (xrng.shape[0] > 0):
            yrng = y[xrng]
            midOld = mid
        
            if (bright):
                mid = xrng[np.argmax(yrng)]
            else:
                mid = xrng[np.argmin(yrng)]

            if (np.abs(mid-midOld)>mxChange):
                mid = midOld
            
            #now fit Gaussian to feature to identify centre
            xrng = np.arange(winRng) - winRng2 + mid
            xrng = xrng[np.where(xrng>0)[0]]
            xrng = xrng[np.where(xrng<len(y))[0]].astype('int')
        
            if (xrng.shape[0] > 0):
                yrng = y[xrng]
        
                try:
                    whrFinite = np.where(np.isfinite(yrng))[0]
                    if len(whrFinite)>0:
                        xrng = xrng[whrFinite]
                        yrng = yrng[whrFinite]
                        
                        fit = gaussFit(xrng, yrng, plot=plot)
                        
                        if (fit[2] >= 0 and fit[2] < len(y)):
                            cent[i] = fit[2]

                except (RuntimeError):
                    pass


    xTrace = np.arange(cent.shape[0])*nbin
       
    centOut = np.empty(img.shape[1])
    centOut[:] = np.nan
    centOut[xTrace] = cent

    if (smth > 0):
        gKern = conv.Gaussian1DKernel(smth)
        centSmth = conv.convolve(centOut, gKern, boundary='extend',normalize_kernel=True)
    else:
        centSmth = centOut
        
    return centSmth

def buildWidthMap(traceLst, widthLst, slicesLst):
    """
    """

    mapLst = []

    for i in range(len(traceLst)):

        gX, gY = np.mgrid[:slicesLst[i].shape[0],:slicesLst[i].shape[1]]

        points = []
        vals = []

        for j in range(traceLst[i].shape[0]):
            for k in range(traceLst[i].shape[1]):
                if (np.isfinite(traceLst[i][j,k]) and np.isfinite(widthLst[i][j,k])):
                    points.append([k,traceLst[i][j,k]])
                    vals.append(widthLst[i][j,k])
        vals = np.array(vals)
        mapLst.append(griddata(points, vals, (gY, gX), method='linear'))
    return mapLst

def traceCentreFlatAll(flatSlices, cutoff=0.5,limSmth=10,MP=True, plot=False, ncpus=None):
    """
    """

    if (MP):
        if (ncpus == None):
            ncpus =mp.cpu_count()
        pool = mp.Pool(ncpus)

        #build input list
        lst = []
        for i in range(len(flatSlices)):
            lst.append([flatSlices[i],cutoff,limSmth,plot])
            
        cent = pool.map(traceCentreFlatSlice, lst)
        pool.close()
        
    else:
    
        cent = []

        for i in range(len(flatSlices)):
            cent.append(traceCentreFlatSlice([flatSlices[i],cutoff, limSmth,plot]))
            
    return cent

def traceCentreFlatSlice(input):
    """
    """

    #rename input
    flatSlice = input[0]
    cutoff = input[1]
    limSmth = input[2]
    plot = input[3]

    centre = np.empty(flatSlice.shape[1])
    centre[:] = np.nan
    
    for i in range(flatSlice.shape[1]):
    
        #if no threshold is given, use derivatives to identify limits
        #else, use given threshold

        ytmp = flatSlice[:,i]
        xtmp = np.arange(flatSlice.shape[0])
    
        if cutoff is None:
            #use gradient to define cutoff

            #search for edges in the first and last 20 useable pixels
            whr = np.where(np.isfinite(ytmp))[0]
            if len(whr) > 50:
                x1 = xtmp[whr[:50]]
                y1 = ytmp[whr[:50]]
                x2 = xtmp[whr[-50:]]
                y2 = ytmp[whr[-50:]]
            else:
                x1 = xtmp[whr]
                x2 = xtmp[whr]
                y1 = ytmp[whr]
                y2 = ytmp[whr]
        
            #find edges
            g1 = np.abs(np.gradient(y1))
            g2 = np.abs(np.gradient(y2))

            m1 = x1[np.nanargmax(g1)]
            m2 = x2[np.nanargmax(g2)]

        else:
            #smooth spectrum first to avoid hot pixels, when determining the cutoff threshold
            gKern = conv.Gaussian1DKernel(10)
            ysmth = conv.convolve(ytmp, gKern, boundary='extend',normalize_kernel=True)
            yout = ytmp/np.nanmax(ysmth)

            whr = np.where(np.isfinite(yout))[0]
            if len(whr) > 50:
                x1 = xtmp[whr[:50]]
                y1 = yout[whr[:50]]
                x2 = xtmp[whr[-50:]]
                y2 = yout[whr[-50:]]
            else:
                x1 = xtmp[whr]
                x2 = xtmp[whr]
                y1 = yout[whr]
                y2 = yout[whr]

            whr1 = np.where(y1 < cutoff)[0]
            if (len(whr1)>0):
                #good pixel is last bad pixel + 1
                m1 = np.clip(x1[whr1[-1]]+1, 0, ytmp.shape[0])
            else:
                m1 = np.clip(x1[0]-1,0, ytmp.shape[0])
                
            whr2 = np.where(y2 < cutoff)[0]
            if (len(whr2)>0):
                #good pixel is first bad pixel -1
                m2 = np.clip(x2[whr2[0]]-1, 0, ytmp.shape[0])
            else:
                m2 = np.clip(x2[-1]+1,0,ytmp.shape[0])
                
        centre[i] = (m1+m2)/2.
    
    gKern = conv.Gaussian1DKernel(limSmth)
    centSmth = conv.convolve(centre, gKern, boundary='extend',normalize_kernel=True)

    if (plot):
        plt.clf()
        plt.imshow(flatSlice, aspect='auto')
        #plt.plot(np.arange(flatSlice.shape[1]), centre)
        plt.plot(np.arange(flatSlice.shape[1]), centSmth, linewidth=2)
        plt.show()

    return centSmth

def polyFitRonchiTrace(trace, goodReg, order=3, lngthConstraint=False):
    """
    """

    polyTrace = np.empty(trace.shape)
    polyTrace[:] = np.nan

    x = np.arange(trace.shape[1])
    xFit =[]

    if (len(goodReg)>1):
        #assume different pixels to use for each trace
        for reg in goodReg:
            #assume all regions are defined in pairs
            xTmp = np.asarray([])
            
            for j in range(0,len(reg),2):
                xTmp = np.append(xTmp,x[np.logical_and(x>=reg[j],x<=reg[j+1])])
            xFit.append(xTmp)
    else:
        reg = goodReg[0]
        
        for i in range(trace.shape[0]):
            xTmp = np.asarray([])
            
            for j in range(0,len(goodReg),2):
                xTmp = np.append(xTmp,x[np.logical_and(x>=reg[j],x<=reg[j+1])])
            xFit.append(xTmp)

    #now carry out the fits
    for j in range(trace.shape[0]):
        rng = xFit[j].astype('int')
        fitRng = np.where(np.isfinite(trace[j,rng]))[0]
        xfit = x[rng[fitRng]]
        yfit = trace[j,rng[fitRng]]

        pcoef = np.polyfit(xfit,yfit,order)
        poly = np.poly1d(pcoef)

        if(lngthConstraint):
            if (xfit.max()-xfit.min() < 1000):
                xTmp = np.arange(xfit.min(), xfit.max()+1)
                polyTrace[j,xTmp] = poly(xTmp)
            else:
                polyTrace[j,:] = poly(x)
        else:
            polyTrace[j,:] = poly(x)


    return polyTrace



