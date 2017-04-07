"""

Tools to help determine dispersion solution and line fitting

"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import multiprocessing as mp
import wifisIO

def crossCor (spectrum, template,mx):
    """
    Carries out cross-correlation between input spectrum and template and returns the relative shift between the two.
    Usage: shift = crossCor(spectrum, template, mx)
    spectrum is a 1D numpy array
    template is a 1D numpy array
    mx sets the maximum window size from which the pixel-shift is determined
    Returns the shift in pixels
    """
    
    rng = np.arange(spectrum.shape[0]*2-1)-spectrum.shape[0]
    lag = np.correlate(spectrum, template, mode="full")
    whr = np.where(np.abs(rng) < mx)
    rng = rng[whr[0]]
    lag = lag[whr[0]]
    mx = np.argmax(lag)
    return rng[mx]

def gaussian(x,amp,cen,wid):
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

def gaussFit(x, y, guessWidth, plot=False):
    """
    Routine to fit a Gaussian to provided x and y data points and return the fitted coefficients.
    Usage: params = gaussFit(x, y, guessWidth, plot=True/False)
    x is the 1D array of coordinates
    y is the 1D array of data values at the given coordinates
    guessWidth is the initial guess for the width of the Gaussian
    plot is a keyword to allow for plotting of the fit (for debug purposes)
    params is a list containing in the fitted parameters in the following order: amplitude, centre, width
    """

    ytmp = y - np.min(y)
    #use scipy to do fitting
    popt, pcov = curve_fit(gaussian, x, ytmp, p0=[np.max(ytmp), np.mean(x),guessWidth])

    if (plot):
        plt.close('all')
        plt.plot(x,ytmp)
        plt.plot(x, ytmp, 'ro')
        plt.plot(x, gaussian(x, popt[0], popt[1], popt[2]))
        plt.show()
        
    return popt

def getSolQuick(input):
    """
    Determine dispersion solution for given dataset
    Usage solution = getSolQuick(input)
    input is a list containing the following items: spectrum, template, line atlas, maximum polynomial order for fitting, dispersion solution of the template, window range for line fitting, flag to use a weighted fitting or not, window size for carrying out the cross-correlation, flag to plot the results of each step (for debugging).
    spectrum is a 1D array of the data values at each pixel
    template is the corresponding 1D array for the template spectrum
    the line atlas is a list of line centres (in units of wavelength) and corresponding predicted strengths
    the polynomial is of the form y = Sum_i^mx a_i*x^i, where mx is the maximum polynomial order (e.g. mx = 1 -> y = a0 + a1*x).
    the dispersion solution is a list containing the polynomial coefficients in increasing order (e.g. [a0, a1, ..., an])
    the window range is an integer value setting the window for the maximum searchable pixel shift between the template and spectrum
    setting the weight flag to 'True' will weight the line centres by their fitted amplitudes when carrying out the polynomial fitting
    setting the plot flag to 'True' will plot individual steps of the fitting (use for debugging purposes only)
    """
    
    yRow = input[0]
    template = input[1]
    atlas = input[2]
    mxorder = input[3]
    templateSol = input[4]
    winRng = input[5]
    mxCcor = input[6]
    useWeights = input[7]
    plot = input[8]
    buildSol = input[9]
    totPix = len(yRow)

    #get cross-correlation correction to correct any offsets to within 1 pixel
    #to improve search window and line identification
    if (mxCcor > 0):
        pixOffset = crossCor(yRow, template, mxCcor)
    else:
        pixOffset = 0

    #try to subtract any offset such that the noise level is at 0
    #using a recursive sigma clipping routine
    #may be unecessary if the proper processing already deals with this
    
    tmp = yRow
    flr = np.median(tmp)
    for i in range(10):
        whr = np.where((tmp-flr) < 3.*np.std(tmp))
        tmp = tmp[whr[0]]
        flr = np.median(tmp)

    #measure the noise level - may be uneccessary if we can provide this from other routines
    nse = np.std(tmp)
    nse = nse/(np.max(yRow-flr))
    
    #normalize the spectra so that the predicted line strengths are on the same scale as the observed spectrum
    yRow = (yRow-flr)/np.max((yRow-flr))
    template = template/np.max(template)

    #use this offset to determine window range and list of lines to use
    #base on provided (linear) dispersion solution from template, which is expected to be close enough
    #can fix range or use a different technique if solution is very non-linear
    atlasPix = (atlas[:,0]-templateSol[0])/templateSol[1] + pixOffset

    whr = np.where(atlasPix >0)
    atlas = atlas[whr[0],:]
    atlasPix = atlasPix[whr[0]]
    whr = np.where(atlasPix <len(yRow))
    atlas = atlas[whr[0],:]
    atlasPix = atlasPix[whr[0]]

    #now find lines, taking only lines with strength >= 1*noise level, based on the predicted line strength
    atlas[:,1] = atlas[:,1]/np.max(atlas[:,1])
    whr = np.where(atlas[:,1] >= 1.*nse)
    atlas = atlas[whr[0],:]
    atlasPix = atlasPix[whr[0]]

    if (plot):
        plt.ioff()
        plt.figure()
        plt.plot(yRow)
        plt.plot(np.arange(totPix)+pixOffset,template)
        plt.plot([0,len(yRow)],[nse, nse],'--')
        plt.show()

    goodFit = False
    
    #check to make sure that there are any lines to fit and if so continue with routine
    if len(whr[0] > 1):
            
        #loop through the list and fit Gaussians to regions
        #of central location

        nlines = len(atlasPix)
        centFit = []#np.zeros(nlines)
        widthFit = []#np.zeros(nlines)
        ampFit = []
        atlasFit = []

        #print(pixOffset)
        #print('Going to fit', nlines)
        #print(best)
        #print(bestPix)

        for i in range(nlines):
            try:
                pixRng = (np.arange(winRng)-winRng/2 + atlasPix[i]).astype('int') #Centre window on predicted line centre

                if (np.max(pixRng) < totPix):
                    yRng = yRow[pixRng]

                    #find location of maximum signal
                    mx = np.argmax(yRng)
                    mxPos = pixRng[mx]
                    prevMx = -1
                    
                    while (mxPos != prevMx):
                        #update the search range to centre on peak
                        prevMx = mxPos
                        pixRng = (np.arange(winRng)-winRng/2 + pixRng[mx]).astype('int')
                        yRng = yRow[pixRng]
                        mx = np.argmax(yRng)
                        mxPos = pixRng[mx]

                    #check if S/N of range is sufficient for fitting
                    #requires at least two consecutive pixels > noise criteria
                    if (yRng[mx] >= 3.*nse):
                        if (yRng[mx-1] >= 3.*nse or yRng[mx+1] >= 3.*nse):
                        #fit guassian to region to determine central location
                        #print('Trying to fit line', bestPix[i])
                            try:
                                amp,cent,wid = gaussFit(pixRng,yRng, winRng/3.,plot=plot)
                                
                                #only keep line if amplitude of fit >3*noise level
                                if (amp/nse >= 3.):
                                    centFit.append(cent)
                                    widthFit.append(wid)
                                    ampFit.append(amp)
                                    atlasFit.append(atlas[i,0])

                                if (len(centFit)>3):
                                    #update "guessed" dispersion solution to get better line centres
                                        tmpCoef = np.polyfit(centFit, atlasFit,mxorder, w=ampFit)
                                        atlasPix = (atlas[:,0]-tmpCoef[1])/tmpCoef[0]
                                    
                            except (RuntimeError):
                                pass
            except (IndexError, ValueError):
                pass

        #exclude poorly fit lines based on width of line
        widthFit = np.abs(widthFit)
        whr = np.where(widthFit <= winRng/3.) 
        ln = len(whr[0])

        if ln > mxorder:
            #if (plot == True):
            #plt.plot(centFit,atlasFit, 'bo')

            centFit = np.array(centFit)
            atlasFit = np.array(atlasFit)
            widthFit = np.array(widthFit)
            ampFit = np.array(ampFit)
                
            centFit = centFit[whr[0]]
            atlasFit = atlasFit[whr[0]]
            widthFit = widthFit[whr[0]]
            ampFit = ampFit[whr[0]]

            if (plot):
                plt.subplot(211)
                plt.plot(centFit, atlasFit, 'bo')

            if (useWeights):
                fitCoef = np.polyfit(centFit, atlasFit,mxorder, w=ampFit) # returns polynomial coefficients in reverse order
            else:
                fitCoef = np.polyfit(centFit, atlasFit,mxorder) # returns polynomial coefficients in reverse order

                #exclude poorly fit lines based on deviation from best fit
                for i in range(1):
                    poly = np.poly1d(fitCoef)
                    dev = atlasFit-poly(centFit)
                    whr = np.where(np.abs(dev) < 1.*np.std(dev))
                    if (plot):
                        print('std dev for round',i, 'is',np.std(dev))
                
                    centFit = centFit[whr[0]]
                    atlasFit = atlasFit[whr[0]]
                    widthFit = widthFit[whr[0]]
                    ampFit = ampFit[whr[0]]

                if (len(centFit) > mxorder):
                    if (useWeights):
                        fitCoef = np.polyfit(centFit, atlasFit,mxorder, w=ampFit) # returns polynomial coefficients in reverse order
                    else:
                        fitCoef = np.polyfit(centFit, atlasFit,mxorder) # returns polynomial coefficients in reverse order
                    goodFit = True
            
                    #compute RMS
                    poly = np.poly1d(fitCoef)
                    diff = atlasFit - poly(centFit)
                    rms = np.sqrt((1./centFit.shape[0])*(np.sum(diff**2.)))
                    rms /= fitCoef[0]
                    
                    #for testing purposes only
                    if (plot):
                        plt.plot(centFit, atlasFit, 'ro')
                        plt.plot(np.arange(len(yRow)), poly(np.arange(len(yRow))))
                        plt.subplot(212)
                        plt.plot(centFit, atlasFit - poly(centFit), 'ro')
                        plt.plot([0, len(atlasFit)],[0,0],'--')
                        print('final std dev:',np.std(atlasFit - poly(centFit)))
                        plt.show()
        
    
    if (goodFit):
        return(fitCoef[::-1],widthFit*2.*np.sqrt(2.*np.log(2.)), centFit, atlasFit, np.abs(rms))
    else:
        return np.repeat(np.nan,mxorder+1), [],[]
    

def getWaveSol (data, template,atlas, mxorder, prevSolution, dispAxis=1, winRng=7, mxCcor=30, weights=False, buildSol=False, ncpus=None):
    """
    Computes dispersion solution for each set of pixels along the dispersion axis in the provided image.
    Usage: output = getWaveSol(data, template, mxorder, prevSolution, dispAxis, winRng, mxCcor, weights)
    data is the input 2D image from which the new dispersion solution will be derived
    template is a template image from which a known solution is already determined. Template can instead be a 1D array that will be used as input for all vectors along the dispersion axis.
    atlas is the name of the file containing the atlas line list to use for fitting
    mxorder is the highest allowable term in the polynomial solution of the form y = Sum_i^mx a_i*x^i, where mx is the maximum polynomial order (e.g. mx = 1 -> y = a0 + a1*x).
    prevSolution is a list of containing the previously determined coefficients for each set of pixels. If a single solution is given, then it is used as input for all vectors along the dispersion axis.
    disAxis specifies the dispersion direction (0 -> along the y-axis, 1-> along the x-axis)
    winRng is a keyword to change window range for searching for line centre and fitting (default is 7 pixels)
    mxCcor is a keyword to change the the window range for finding the maximum searchable pixel shift between the template and spectrum (default is 30 pixels)
    weights is a keyword to carry out a weighted polynomial fit for determining dispersion solution (default is False) 
    Returns ???
    """

    #read in line atlas 
    bestLines = wifisIO.readTable(atlas)

    #set up input data for running with multiprocessing
    lst = []

    #rotate image if dispersion axis not aligned along the x-axis
    if (dispAxis==0):
        dTmp = data.T
        tempTemp = template.T
    else:
        dTmp = data
        tempTemp = template

    if (template.ndim == 2):
        for i in range(dTmp.shape[1]):
            lst.append([dTmp[i,:],template[i,:], bestLines, mxorder,prevSolution,winRng, mxCcor,weights, False, buildSol])
    else:
        lst = []
        for i in range(dTmp.shape[1]):
            lst.append([dTmp[i,:],template, bestLines, mxorder,prevSolution,winRng, mxCcor,weights, False, buildSol])
   
    #setup multiprocessing routines
    if (ncpus == None):
        ncpus =mp.cpu_count()
    pool = mp.Pool(ncpus)

    #run code
    result = pool.map(getSolQuick, lst)
    #close other processes
    pool.close()

    #do other things and return something different?
    
    return result
    
def buildWaveMap(img, dispSol, dispAxis=1):
    """
    Routine to build a wavelength mapping for each pixel on the provided image
    Usage: waveMap = buildWaveMap(img, dispSol, dispAxis=1)
    img is the input image onto which the mapping should be done
    dispSol is an array providing the measured dispersion solution for each pixel in the image
    dispAxis is a keyword that specifies which axis is the dispersion axis (0 - along the y-axis, 1 - along the x-axis)
    waveMap is the output image providing the wavelength at each pixel
    """

    #initialize wave map array
    waveMap = np.zeros(img.shape)

    if (dispAxis==0):
        x = np.arange(img.shape[0])
    else:
        x = np.arange(img.shape[1])
        
        #populate map with solution
    for i in range(dispSol.shape[0]):
        waveMap[i,:] = dispSol[i,0] + dispSol[i,1]*x

    if (dispAxis==0):
        waveMap = waveMap.T

    return waveMap

def trimWaveSliceAll(waveSlices, flatSlices, threshold, MP=True, ncpus=None):
    """
    Routine used to identify useful limits of wavelenth mapping 
    Usage: out = trimWaveSlice(waveSlices, flatSlices, threshold, MP=True, ncpus=None)
    waveSlices is a list of the wavelength mapping slices
    flatSlices is a list of the flatfield image slices
    threshold is a value indicating the cutoff value relative to the maximum flatfield flux in a given slice to be used to determine the limits
    MP is a boolean used to indicate whether multiprocessing should be used
    ncpus is an integer to set the maximum number of simultaneously run processes during MP mode
    out is a list containing the trimmed wavelength mapped slices. All values outside of the trim limits are set to NaN
    """

    
    if (MP):
        lst = []
        for i in range(len(waveSlices)):
            lst.append([waveSlices[i],flatSlices[i],threshold])

        if (ncpus == None):
            ncpus =mp.cpu_count()
        pool = mp.Pool(ncpus)
        out = pool.map(trimWaveSlice, lst)
        pool.close()
        
    else:
    
        out = []
        for i in range(len(waveSlices)):
            out.append(trimWaveSlice([waveSlices[i],flatSlices[i],threshold]))
            
    return out

    
def trimWaveSlice(input):
    """
    Routine used to identify useful limits of wavelenth mapping slice
    Usage: slc = trimWaveSlice(input), where input is a list containing:
    waveSlc - the wavelength mapping slice
    flatSlc - the flatfield slice image
    threshold - the cutoff value relative to the maximum flatfield flux to determine the limits
    slc is the trimmed wavelength mapped slice. All values outside of the trim limits are set to NaN
    """

    slc = np.empty((input[0].shape))
    np.copyto(slc, input[0])
    flatSlc = input[1]
    threshold = input[2]

    mx = np.max(flatSlc)
    
    #work on axis 0 first
    for i in range(slc.shape[0]):
        y = flatSlc[i,:]
        whr = np.where(y < threshold*mx)[0]
        slc[i,whr] = np.nan
        
    #work on axis 1 now
    for i in range(slc.shape[1]):
        y = flatSlc[:,i]
        whr = np.where(y < threshold*mx)[0]
        slc[whr,i] = np.nan

    return slc
