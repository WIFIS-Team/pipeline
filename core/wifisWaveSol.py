"""

Tools to assist in the processing of arc lamp and wavelength calibration data

"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import multiprocessing as mp
import wifisIO
from scipy.interpolate import interp1d
from astropy import convolution as conv
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec
from scipy.optimize import OptimizeWarning
from scipy.interpolate import spline
from scipy.interpolate import Akima1DInterpolator

import warnings
warnings.simplefilter('ignore', np.RankWarning)
warnings.simplefilter('ignore', OptimizeWarning)

def crossCor (spectrum, template,mx):
    """
    Carries out cross-correlation between input spectrum and template and returns the relative shift between the two.
    Usage: shift = crossCor(spectrum, template, mx)
    spectrum is a 1D numpy array
    template is a 1D numpy array
    template AND spectrum must have the same dimensions
    mx sets the maximum window size from which the pixel-shift is determined
    Returns the shift in pixels
    """

    if (len(spectrum) != len(template)):
        raise InputError('spectrum and template MUST have same length')

    rng = np.arange(spectrum.shape[0]*2-1)-spectrum.shape[0]+1
    lag = np.correlate(spectrum, template, mode="full")
    whr = np.where(np.abs(rng) < mx)
    rng = rng[whr[0]]
    lag = lag[whr[0]]
    mx = np.nanargmax(lag)
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

def polyGaussian(x,amp1,cen1,wid1,a0,a1,a2):
    """
    Returns a Gaussian function of the form y = A*exp(-z^2/2), where z=(x-cen)/wid
    Usage: output = gaussian(x, amp, cen, wid)
    x is the input 1D array of coordinates
    amp is the amplitude of the Gaussian
    cen is the centre of the Gaussian
    wid is the 1-sigma width of the Gaussian
    returns an array with same size as x corresponding to the Gaussian solution
    """
    
    z1 = (x-cen1)/wid1

    return amp1*np.exp(-z1**2/2.) + a0+a1*x+a2*x**2

def getWeightedCent(x,y):
    """
    Returns the weighted centre of the x-coordinate array, based on the weight values provided the y-values array
    Usage: cent = getWeightedCent(x,y)
    x is a numpy array of coordinate values
    y is a numpy array of signal values, to be used as the weights
    cent is the returned centre-of-gravity
    """

    #get the 4-pixels with the greatest flux
    srt = np.argsort(y)

    #check if the first or last pixel is >2 pixel difference from the rest
    xsrt = np.sort(srt[-4:])

    if np.abs(xsrt[0]-xsrt[1]) >2:
        xsrt = np.sort(np.append(np.asarray(srt[-5]),xsrt[1:]))
    elif np.abs(xsrt[-1]-xsrt[-2]) >2:
        xsrt = np.sort(np.append(np.asarray(srt[-5]),xsrt[:-1]))
        
    #check once again
    if np.abs(xsrt[0]-xsrt[1]) >2:
        xsrt = np.sort(np.append(np.asarray(srt[-6]),xsrt[1:]))
    elif np.abs(xsrt[-1]-xsrt[-2]) >2:
        xsrt = np.sort(np.append(np.asarray(srt[-6]),xsrt[:-1]))
        
    #compute the weigthed mean
    cent = np.sum(y[xsrt]*x[xsrt])/np.sum(y[xsrt])

    return cent

def gaussFit(x, y, guessWidth, plot=False,title=''):
    """
    Routine to fit a Gaussian to provided x and y data points and return the fitted coefficients.
    Usage: params = gaussFit(x, y, guessWidth, plot=True/False,title='')
    x is the 1D numpy array of coordinates
    y is the 1D numpy array of data values at the given coordinates
    guessWidth is the initial guess for the width of the Gaussian
    plot is a keyword to allow for plotting of the fit (for debug purposes)
    title is an optional title to provide the plot
    params is a list containing the fitted parameters in the following order: amplitude, centre, width
    """

    ytmp = y - np.min(y)
    #use scipy to do fitting
    popt, pcov = curve_fit(gaussian, x, ytmp, p0=[np.max(ytmp), np.mean(x),guessWidth])

    if (plot):
        plt.close('all')
        plt.plot(x,ytmp)
        plt.plot(x, ytmp, 'ro')
        plt.plot(x, gaussian(x, popt[0], popt[1], popt[2]))
        plt.xlabel('Pixel')
        plt.ylabel('Value')
        plt.title(title)
        plt.show()
        
    return popt

def polyGaussFit(x, y, guessWidth, plot=False,title=''):
    """
    Routine to fit a Gaussian to provided x and y data points and return the fitted coefficients.
    Usage: params = gaussFit(x, y, guessWidth, plot=True/False,title='')
    x is the 1D numpy array of coordinates
    y is the 1D numpy array of data values at the given coordinates
    guessWidth is the initial guess for the width of the Gaussian
    plot is a keyword to allow for plotting of the fit (for debug purposes)
    title is an optional title to provide the plot
    params is a list containing the fitted parameters in the following order: amplitude, centre, width
    """

    ytmp = y - np.min(y)
    #use scipy to do fitting
    popt, pcov = curve_fit(polyGaussian, x, ytmp, p0=[np.max(ytmp), np.mean(x),guessWidth, 0.,0.,0.])

    if (plot):
        plt.close('all')
        plt.plot(x,ytmp)
        plt.plot(x, ytmp, 'ro')
        plt.plot(x, polyGaussian(x, popt[0], popt[1], popt[2],popt[3],popt[4],popt[5]))

        plt.xlabel('Pixel')
        plt.ylabel('Value')
        plt.title(title)
        plt.show()

    #polyGaussian(x,amp1,cen1,wid1,a0,a1,a2):

    #determine more accurate fwhm
    #npix = x.shape[0]
    #xtmp = np.linspace(x.min()-npix,x.max()+npix,x.shape[0]*20)
    #ytmp = polyGaussian(xtmp,popt[0], popt[1], popt[2],popt[3],popt[4],popt[5])
    #whr = np.where(ytmp >= ytmp.max()/2.)[0]
    #popt[2]=xtmp[whr.max()]-xtmp[whr.min()]
    popt[2] = np.abs(popt[2])
    return popt[:3]


def splineFit(x,y, plot=False, title=None):
    """
    """

    xInt = np.linspace(x.min(),x.max(),x.shape[0]*20.)
    fInt = Akima1DInterpolator(x,y)
    yInt = fInt(xInt)
    whr = np.where(yInt >= yInt.max()/2.)[0]
    fwhm = xInt[whr.max()]-xInt[whr.min()]

    if plot:
        plt.close('all')
        plt.plot(x,y)
        plt.plot(x, y, 'ro')
        plt.vlines(xInt[np.argmax(yInt)],0,yInt.max())
        plt.xlabel('Pixel')
        plt.ylabel('Value')
        plt.title(title)
        plt.show()
        
    return [yInt.max(), xInt[np.argmax(yInt)], fwhm]


def getSolQuick(input):
    """
    Determine dispersion solution for given dataset
    Usage solution = getSolQuick(input)
    input is a list containing:
    - a numpy array providing the spectrum
    - a numpy array providing the template spectrum
    - a numpy array providing the line atlas. The line atlas is a list of line centres (in units of wavelength) and corresponding predicted strengths
    - the maximum polynomial order for fitting of the dispersion solution
    - the dispersion solution (in wavelength -> pixel mapping) associated with the template
    - the window range for line fitting
    - the maximum allowed pixel shift between the template and the observed spectrum, in pixels
    - a boolean flag that specifies if a weighted fitting of the polynomial solution should be used (weighted by the fitted line strengths)
    - a boolean flag that indicates if plotting of the fitting process should occur (for debugging purposes only)
    - a boolean flag that specifies if the dispersion solution used to find/identify lines should be built up during the current fitting process (useful for building a template solution, but can often lead to poor solutions)
    - a boolean flag that indicates if a polynomial order lower than the degree specified by mxorder can be used to fit spectra with insufficient number of lines
    - the sigma-clipping threshold to use during the dispersion solution fitting stage
    - a boolean flag that indicates if a linear dispersion solution will be forced on the spectrum, if the pixel distance between the furthest separated fitted lines is less than 1000 pixels
    - a boolean flag that specifies if the fit window can be automatically adjusted by the code in order to increase the window range to achieve a better fit to the line
    - the sigma-clipping threshold to use when searching for useable lines to fit. Lines with strengths/amplitude fits less than this value times the estimated noise level are rejected and not used to find the dispersion solution
    - a boolean flag to indicate if the window position for identifying/fitting lines is allowed to wander from its initial guess. This option can be useful if the dispersion solution is slightly off, but it can also cause problems if there are several lines that are close in proximity.
    - the number of sigma-clipping iterations to use during the dispersion solution fitting stage
    - the number of pixels to use for continuum fitting of the spectrum.
    - the maximum number of iterations allowed if the search window is allowed to wander
    Returned is a list containing the following elements:
    - the fitted polynomial coefficients (in increasing order) describing the transformation from pixel to wavelength
    - the FWHM of each fitted line used to determine the dispersion solution
    - the line centres (in pixels) of the fitted lines
    - the predicted line centres (in wavelength) corresponding to the fitted lines
    - the RMS difference between the final fitted polynomial solution and the measured line centres (in pixels)
    - the fitted polynomial coefficients (in increasing order) describing the transformation from wavelength to pixels
    """
    
    yRow = np.copy(input[0])
    template = np.copy(input[1])
    atlas = input[2]
    mxorder = input[3]
    templateSol = input[4]
    winRng = input[5]
    winRng2 = int(winRng/2)
    mxCcor = input[6]
    useWeights = input[7]
    plot = input[8]
    buildSol = input[9]
    allowLower = input[10]
    sigmaClip = input[11]
    lngthConstraint = input[12]
    adjustFitWin = input[13]
    sigmaLimit = input[14]
    allowSearch = input[15]
    sigmaClipRounds = input[16]
    nCont = input[17]
    nSearchRounds = input[18]
    totPix = len(yRow)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        ySigma = np.sqrt(yRow)

    #for testing
    useQCWeights = False
    
    #first make sure that the yRow isn't all NaNs

    if np.all(~np.isfinite(yRow)):
        return np.repeat(np.nan,mxorder+1), [],[], [],np.nan, np.repeat(np.nan,mxorder+1)
    
    #get cross-correlation correction to correct any offsets to within 1 pixel
    #to improve search window and line identification
    if (mxCcor > 0):
        yTmp = np.zeros(yRow.shape,dtype=yRow.dtype)
        yTmp[np.where(np.isfinite(yRow))] = yRow[np.isfinite(yRow)]

        tmpTmp = np.zeros(template.shape, dtype=template.dtype)
        tmpTmp[np.where(np.isfinite(template))] = template[np.isfinite(template)]

        pixOffset = crossCor(yTmp, tmpTmp, mxCcor)
    else:
        pixOffset = 0

    
    #try to subtract any offset such that the noise level is at 0
    #using a recursive sigma clipping routine
    #may be unecessary if the proper processing already deals with this

    xFlr = np.arange(yRow.shape[0])
    whr = np.where(np.isfinite(yRow))[0]

    tmp = np.copy(yRow[whr])
    ytmp = np.copy(yRow[whr])
    xtmp = np.arange(yRow.shape[0])[whr]
    xFlr = xFlr[whr]
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        flr = np.nanmedian(tmp)
        
        for i in range(10):
            whr = np.where((tmp-flr) < 3.*np.nanstd(tmp))[0]
            tmp = tmp[whr]
            xFlr = xFlr[whr]
            flr = np.nanmedian(tmp)


        if len(whr)>1:
            goodFit = True
        else:
            goodFit =False

        if goodFit:
            #get and subtract a cubic spline fitted to continuum
            xSpline = [xtmp[0]]
            ySpline = [np.nanmedian(ytmp[0:int(nCont/2)])]

            for i in range(xtmp[0],xtmp[-1]-nCont,nCont):
                rng = np.where(np.logical_and(xtmp>=i,xtmp<i+nCont))[0]
                xrng = xtmp[rng]
                yrng = ytmp[rng]
                xSpline.append(xrng.mean())
                ySpline.append(np.nanmedian(yrng))

            xSpline.append(xtmp[-1])
            ySpline.append(np.nanmedian(ytmp[-int(nCont/2):]))

            #remove any nans, in case they're still there
            whr = np.where(np.isfinite(ySpline))[0]
            if len(whr) < 3:
                goodFit = False
            else:
                xSpline = np.asarray(xSpline)[whr]
                ySpline = np.asarray(ySpline)[whr]
                fInt = interp1d(xSpline, ySpline, fill_value='extrapolate')
                tmp -= fInt(xFlr)
                #tmp -= spline(xSpline,ySpline, xFlr, order=1)
                #flrFit = spline(xSpline,ySpline, np.arange(yRow.shape[0]), order=1)
                flrFit = fInt(np.arange(yRow.shape[0]))
                
                #carry out one more round of pixel rejection after continuum subtraction
                whr = np.where(tmp< 3.*np.nanstd(tmp))
                tmp = tmp[whr[0]]

                #measure the noise level
                nse = np.nanstd(tmp)
        
                #remove continuum from input spectrum
                yRow -= flrFit
                nse = nse/(np.nanmax(yRow))
    
                #normalize the spectra so that the predicted line strengths are on the same scale as the observed spectrum
                ySigma = ySigma/np.nanmax(yRow)
                yRow = yRow/np.nanmax(yRow)

                template = template/np.nanmax(template) #for plotting purposes only

            #use this offset to determine window range and list of lines to use
            #base on provided dispersion solution from template, which is expected to be close enough

    if (np.all(np.isfinite(templateSol)) and goodFit):
        atlasPix = templateSol[0] + pixOffset

        for i in range(1,len(templateSol)):
            atlasPix += templateSol[i]*atlas[:,0]**i
    
        #atlasPix = (atlas[:,0]-templateSol[0])/templateSol[1] + pixOffset #the linear case where the f(pixels) -> lambda

        #exclude NaNs
        whrFinite = np.where(np.isfinite(yRow))[0]

        if len(whrFinite)>1:
            mnLngth = np.min(whrFinite)
            mxLngth = np.max(whrFinite)
        else:
            mxLngth = 0
            
        if (mxLngth>0):
            whr = np.where(np.logical_and(atlasPix >mnLngth,atlasPix < mxLngth))[0]
            if (len(whr)>0):
                atlas = atlas[whr,:]
                atlasPix = atlasPix[whr]
                                
                #now find lines, taking only lines with strength >= sigmaLimit*noise level, based on the predicted line strength, normalized to the strongest line in the observed spectrum
                mxObsLine = np.nanargmax(yRow)
                
                #find closest predicted line
                mxAtlasLine = np.argmin(np.abs(atlasPix-mxObsLine))
                                 
                atlas[:,1] = atlas[:,1]/atlas[mxAtlasLine,1]
                whr = np.where(atlas[:,1] >= sigmaLimit*nse)[0]
                atlas = atlas[whr,:]
                atlasPix = atlasPix[whr]
        else:
            whr=[]
    else:
        whr =[]

    if plot and goodFit:
        plt.ioff()
        plt.close('all')
        plt.figure()
        print('Pixel offset of ' + str(pixOffset))
        plt.plot(yRow)
        plt.plot(np.arange(totPix)+pixOffset,template)
        plt.plot([0,len(yRow)],[sigmaLimit*nse, sigmaLimit*nse],'--')
        for i in range(len(atlasPix)):
            plt.plot([atlasPix[i],atlasPix[i]], [0, atlas[i,1]], 'k:')
        plt.show()
        
    goodFit = False

    #check to make sure that there are any lines to fit and if so continue with routine
    if (len(whr) > 1):

        #remove all NaNs to avoid fitting problems
        yRow[~np.isfinite(yRow)] = 0.
        
        #loop through the list and fit Gaussians to regions
        #of central location

        nlines = len(atlasPix)
        centFit = []#np.zeros(nlines)
        widthFit = []#np.zeros(nlines)
        ampFit = []
        atlasFit = []
        atlasPixFit = []
        QCFit = []
        
        #print(pixOffset)
        #print('Going to fit', nlines)
        #print(best)
        #print(bestPix)

        for i in range(nlines):
            try:
                pixRng = (np.arange(winRng)-winRng2 + atlasPix[i]).astype('int') #Centre window on predicted line centre
                
                yRng = yRow[pixRng]
                sRng = ySigma[pixRng]
                
                #find location of maximum signal
                mx = np.argmax(yRng)
                mxPos = pixRng[mx]
                prevMx = -1
                if plot:
                    print('*********')
                    print('Fitting line ' +str(atlas[i,0]))
                    print('Noise level ' + str(sigmaLimit*nse))

                if allowSearch:
                    while (mxPos != prevMx):
                        #update the search range to centre on peak
                        if (plot):
                            print(pixRng)
                            plt.plot(pixRng, yRng)
                            plt.show()
                        
                        prevMx = mxPos
                        pixRng = (np.arange(winRng)-winRng2 + pixRng[mx]).astype('int')
                        yRng = yRow[pixRng]
                        sRng = ySigma[pixRng]
                        
                        mx = np.argmax(yRng)
                        mxPos = pixRng[mx]
                else:
                    for iSrch in range(nSearchRounds):
                        #update the search range to centre on peak
                        if (plot):
                            print(pixRng)
                            plt.plot(pixRng, yRng)
                            plt.show()
                        
                        prevMx = mxPos
                        pixRng = (np.arange(winRng)-winRng2 + pixRng[mx]).astype('int')
                        yRng = yRow[pixRng]
                        mx = np.argmax(yRng)
                        mxPos = pixRng[mx]
                        
                pixRng = pixRng[np.logical_and(pixRng >=0, pixRng<totPix)]
                if(plot):
                    print(pixRng)
                if len(pixRng)>winRng/2:
                    
                    #check if S/N of range is sufficient for fitting
                    #requires at least two consecutive pixels > noise criteria
                    if ((yRng[mx]-yRng.min()) >= sigmaLimit*nse):
                        if ((yRng[mx-1]-yRng.min()) >= sigmaLimit*nse or (yRng[mx+1]-yRng.min()) >= sigmaLimit*nse):
                        #fit guassian to region to determine central location
                        #print('Trying to fit line', bestPix[i])
                            try:
                                winRngTmp = winRng
                                #amp,cent,wid = splineFit(pixRng, yRng, plot=plot,title=str(atlasPix[i]))
                                #amp,cent,wid = gaussFit(pixRng,yRng, winRng/3.,plot=plot,title=str(atlasPix[i]))
                                amp,cent,wid = polyGaussFit(pixRng,yRng, winRng/3.,plot=plot,title=str(atlasPix[i]))

                                #get weighted average as well
                                #cent2 = getWeightedCent(pixRng,yRng)
 
                                #if the two centres don't agree, favour the weighted average
                                #if np.abs(cent2-cent)>0.5:
                                #    cent = cent2
                                #    if plot:
                                #        print('Using weighted centre instead')
                                
                                #only continue if cent is within fitting window
                                if (cent < pixRng[0] or cent > pixRng[-1]):
                                    if plot:
                                        print('badly fit line (centre outside of window range), excluding')
                                else:
                                    if (adjustFitWin and wid > winRng/3):
                                        if (plot):
                                            print('adjusting fit window')
                                        
                                        pixRng = int(cent) + np.arange(int(wid)*5)-int(wid*5/2)
                                        pixRng= pixRng[np.logical_and(pixRng >=0, pixRng<totPix)]
                                        yRng = yRow[pixRng]
                                        sRng = ySigma[pixRng]

                                        winRngTmp = wid*4
                                        #amp,cent,wid = splineFit(pixRng, yRng, plot=plot,title=str(atlasPix[i]))
                                        #amp,cent,wid = gaussFit(pixRng,yRng, winRngTmp/3.,plot=plot,title=str(atlasPix[i]))
                                        amp,cent,wid = polyGaussFit(pixRng,yRng, winRngTmp/3.,plot=plot,title=str(atlasPix[i]))
                                        if cent > 3000:
                                            mpl.plot(pixRng, yRng)
                                            mpl.show()

                                        #get weighted average as well
                                        #cent2 = getWeightedCent(pixRng,yRng)
 
                                        #if the two centres don't agree, favour the weighted average
                                        #if np.abs(cent2-cent)>0.5:
                                        #    cent = cent2
                                        #    if plot:
                                        #        print('Using weighted centre instead')
                                                
                                    #only keep line if amplitude of fit >2*noise level #and width of fit <1/2 of winRng and cent in expected range
                                    if (amp/nse >= sigmaLimit and np.abs(wid) < winRngTmp/2.):
                                        if plot:
                                            print('Keeping line')
                                        
                                        centFit.append(cent)
                                        widthFit.append(wid)
                                        ampFit.append(amp)
                                        atlasFit.append(atlas[i,0])
                                        atlasPixFit.append(atlasPix[i])
                                        QCFit.append(np.sum((yRng-gaussian(pixRng,amp,cent,wid))**2/sRng**2))
                                    else:
                                        if (plot):
                                            print('badly fit line, excluding')
                                        
                                if (len(centFit)>mxorder and buildSol):
                                    #update "guessed" dispersion solution to get better line centres
                                    tmpCoef = np.polyfit(centFit, atlasFit,1, w=ampFit)
                                    atlasPix = (atlas[:,0]-tmpCoef[1])/tmpCoef[0]
                                    
                            except:
                                pass
                        else:
                            if plot:
                                print('S/N too low, excluding')
                    else:
                        if plot:
                            print('S/N too low, excluding')
            except (IndexError, ValueError, RuntimeWarning):
                pass

        #exclude poorly fit lines based on width of line
        #widthFit = np.abs(widthFit)
        #whr = np.where(widthFit <= winRng) 
        
        #exclude poorly fit lines based on the deviation from prediction
        dev = np.abs(np.array(atlasPixFit)-np.array(centFit))
        #whr = np.where(dev < 1.*np.std(dev))
        ln = len(centFit)
        #ln = len(whr[0])
        
                        
        if ((ln < mxorder) and allowLower and (ln>1)):
            #find the highest order polynomial that could fit the data
            
            while ((ln <= mxorder) and (mxorder>1)):
                mxorder = mxorder - 1
                
        if (ln > mxorder):

            #iterate through all possible polynomial orders <= mxorder and select best one
            if allowLower:
                new_mx_order = np.arange(1,mxorder+1)
            else:
                new_mx_order = [mxorder]
                

            #convert lists to arrays
            centFit = np.asarray(centFit)
            atlasFit = np.asarray(atlasFit)
            widthFit = np.asarray(widthFit)
            ampFit = np.asarray(ampFit)
            QCFit = np.asarray(QCFit)

            rms_poly_fit = []

            #
            centFitTemp = np.copy(centFit)
            atlasFitTemp = np.copy(atlasFit)
            widthFitTemp = np.copy(widthFit)
            ampFitTemp = np.copy(ampFit)
            QCFitTemp = np.copy(QCFit)
                    
            if (plot):
                fig = plt.figure(figsize=(10,5))
                #ax = fig.add_subplot(211)
                gs = gridspec.GridSpec(2,2)
                gs.update(left=0.1,right = 0.98)
                ax1 = fig.add_subplot(gs[0,0])
                ax1.plot(centFitTemp, atlasFit, 'bo')
                
            #exclude poorly fit lines based on deviation from best fit
                
            if sigmaClipRounds ==0:
                sigmaClipRounds =1
                sigmaClip = 0.

            #carry out first rounds
            for i in range(sigmaClipRounds):
                try:
                    if (useWeights):
                        fitCoef = np.polyfit(centFitTemp, atlasFitTemp,mxorder, w=1./widthFit**2)# returns polynomial coefficients in reverse order
                        #fitCoef = np.polyfit(centFit, atlasFit,mxorder, w=ampFit) # returns polynomial coefficients in reverse order
                    elif useQCWeights:
                        fitCoef = np.polyfit(centFitTemp, atlasFitTemp,mxorder, w=1./np.sqrt(QCFit)) # returns polynomial coefficients in reverse order
                    else:
                        fitCoef = np.polyfit(centFitTemp, atlasFitTemp,mxorder) # returns polynomial coefficients in reverse order
                except:
                    fitCoef = np.empty(mxorder)
                    fitCoef[:] = np.nan
                        
                poly = np.poly1d(fitCoef)
                dev = atlasFitTemp-poly(centFitTemp)
                    
                if sigmaClip >0:
                    whr = np.where(np.abs(dev) < sigmaClip*np.std(dev))[0]
                else:
                    whr = np.arange(dev.shape[0])
                    
                if (plot):
                    print('std dev for round',i, 'is',np.std(dev), ' in wavelength')
                        
                centFitTemp = centFitTemp[whr]
                atlasFitTemp = atlasFitTemp[whr]
                widthFitTemp = widthFitTemp[whr]
                ampFitTemp = ampFitTemp[whr]
                QCFitTemp = QCFitTemp[whr]

            #check if any lines remaining and get final fit
            if (len(centFitTemp) > mxorder or (lngthConstraint and len(centFit)>0)):

                #constrain fit to a line if line separation is <1000
                if (lngthConstraint):
                    if ((np.nanmax(centFitTemp)-np.nanmin(centFitTemp)) < 1000):
                        mxorder=1

                        if (plot):
                            print('Forcing linear solution')

                        #carry out one more stage of rejection since we've moved to a new model
                        try:
                            if (useWeights):        
                                #fitCoef = np.polyfit(centFit, atlasFit,mxorder, w=ampFit) # returns polynomial coefficients in reverse order
                                fitCoef = np.polyfit(centFitTemp, atlasFitTemp,mxorder, w=1./widthFit**2) # returns polynomial coefficients in reverse order
                        
                            elif useQCWeights:
                                fitCoef = np.polyfit(centFitTemp, atlasFitTemp,mxorder, w=1./np.sqrt(QCFit)) # returns polynomial coefficients in reverse order
                            else:
                                fitCoef = np.polyfit(centFitTemp, atlasFitTemp,mxorder) # returns polynomial coefficients in reverse order
                        except:
                            fitCoef = np.empty(mxorder)
                            fitCoef[:]= np.nan

                        poly = np.poly1d(fitCoef)
                        #print(len(centFitTemp), len(atlasFitTemp))
                        dev = atlasFitTemp-poly(centFitTemp)
                        whr = np.where(np.abs(dev) < sigmaClip*np.std(dev))
                        
                        if (plot):
                            print('std dev for round',i, 'is',np.std(dev), ' in wavelength')
                
                        centFitTemp = centFitTemp[whr[0]]
                        atlasFitTemp = atlasFitTemp[whr[0]]
                        widthFitTemp = widthFitTemp[whr[0]]
                        ampFitTemp = ampFitTemp[whr[0]]
                        QCFitTemp = QCFitTemp[whr[0]]

                for mxOrder in new_mx_order:
                    #now get final polynomial fit
                    try:
                        if (useWeights):
                            #fitCoef = np.polyfit(centFit, atlasFit,mxorder, w=ampFit) # returns polynomial coefficients in reverse order
                            fitCoef = np.polyfit(centFitTemp, atlasFitTemp,mxorder, w=1./widthFit**2) # returns polynomial coefficients in reverse order
                            
                        else:
                            fitCoef = np.polyfit(centFitTemp, atlasFitTemp,mxorder) # returns polynomial coefficients in reverse order
                    except:
                        fitCoef = np.empty(mxorder)
                        fitCoef = np.nan
                        
                    #compute RMS, in terms of pixels
                    if (useWeights):
                        #pixCoef = np.polyfit(atlasFit, centFit, mxorder, w=ampFit)
                        pixCoef = np.polyfit(atlasFitTemp, centFitTemp, mxorder, w=1./widthFit**2)
                        
                    else:
                        pixCoef = np.polyfit(atlasFitTemp, centFitTemp, mxorder)
                        
                    poly = np.poly1d(fitCoef)
                    polyPix = np.poly1d(pixCoef)
                    diff = polyPix(atlasFitTemp) - centFitTemp # wavelength units
                    rms = np.sqrt((1./centFitTemp.shape[0])*(np.sum(diff**2.))) #wavelength units
                    rms_poly_fit.append([mxorder, rms, centFitTemp, widthFitTemp,atlasFitTemp])
                    
                    #for testing purposes only
                    if (plot):
                        ax1.set_xlim(0, yRow.shape[0])
                        ax1.plot(centFit, atlasFit, 'ro')
                        ax1.plot(centFitTemp,atlasFitTemp,'bo')
                        ax1.set_xlabel('Pixel #')
                        ax1.set_ylabel('Wavelength')
                        ax1.plot(np.arange(len(yRow)), poly(np.arange(len(yRow))))
                        ax2 = fig.add_subplot(gs[1,0])
                        ax2.set_xlim(0, yRow.shape[0])
                        ax2.plot(centFit, polyPix(atlasFit) - centFit, 'ro')
                        ax2.set_xlabel('Pixel #')
                        ax2.set_ylabel('Residuals (pixels)')
                        ax2.plot([0, len(atlasFit)],[0,0],'--')
                        print('final std dev:',np.std(atlasFit - poly(centFit)), ' in wavelength')
                        
                        ax3 =fig.add_subplot(gs[:,1])
                        ax3.set_xlim(0, yRow.shape[0])
                        ax3.plot(yRow,'k')
                        ax3.set_xlabel('Pixel')
                        ax3.set_ylabel('Normalized signal')
                        for lne in range(centFit.shape[0]):
                            ax3.plot([centFit[lne],centFit[lne]], [0,ampFit[lne]], 'r--')
                        
                        plt.show()

            #now check which solution was the best
            rms = np.asarray([i[1] for i in rms_poly_fit])

            try:
                mn_loc = np.nanargmin(rms)
                goodFit = True
                mxorder, rms, centFit, widthFit, atlasFit = rms_poly_fit[mn_loc]

            except:
                goodFit=False
                
    if (goodFit):
        #return(fitCoef[::-1],widthFit, centFit, atlasFit, np.abs(rms), pixCoef[::-1])

        #if everything good carry out one last fitting using old gaussian fit algorithm to get more representative FWHM/be compatible with older measurements
        if len(centFit)>0:
            widthFit2 = np.zeros_like(widthFit)
            for i, cent in enumerate(centFit):
                pixRng = (np.arange(winRng)-winRng2 + cent).astype('int')
                yRng = yRow[pixRng]
                try:
                    a,c,wid = gaussFit(pixRng,yRng, winRng/3.,plot=plot,title=str(atlasPix[i]))
                    widthFit2[i] = wid
                except:
                    widthFit2[i] = widthFit[i]

            
        return(fitCoef[::-1],widthFit2*2.*np.sqrt(2.*np.log(2.)), centFit, atlasFit, np.abs(rms), pixCoef[::-1])
    else:
        return np.repeat(np.nan,mxorder+1), [],[], [],np.nan, np.repeat(np.nan,mxorder+1)
    

def getWaveSol (dataSlices, templateSlices,atlas, mxorder, prevSol, winRng=7, mxCcor=30, weights=False, buildSol=False, ncpus=None, allowLower=False, sigmaClip=2., lngthConstraint=False, MP=True, adjustFitWin=False, sigmaLimit=3, allowSearch=False, sigmaClipRounds=1,nPixContFit=50,nSearchRounds=1,plot=False):
    """
    Computes dispersion solution for each row of pixels (along the dispersion axis) in the provided image slices.
    Usage: output = getWaveSol(dataSlices, templateSlices,atlas, mxorder, prevSol, winRng=7, mxCcor=30, weights=False, buildSol=False, ncpus=None, allowLower=False, sigmaClip=2., lngthConstraint=False, MP=True, adjustFitWin=False, sigmaLimit=3, allowSearch=False, sigmaClipRounds=1,nPixContFit=50,nSearchRounds=1,plot=False)
    dataSlices is a list of the input 2D images from which the new dispersion solution will be derived. The dispersion axis of each slice is oriented along the x-axis.
    templateSlices is a list of template images from which a known solution is already determined. templateSlices can also be a list of 1D spectra, with a length equal to the number of slices, which will be used as input for all vectors along the dispersion axis for each image slice.
    atlas is the name of the file containing the atlas line list to use for fitting (a 2 column file with the first column corresponding to the central wavelength of the line, and the second the line intensity/flux)
    mxorder is the highest allowable term in the polynomial solution of the form y = Sum_i^mx a_i*x^i, where mx is the maximum polynomial order (e.g. mx = 1 -> y = a0 + a1*x).
    prevSolution is a list of containing the previously determined polynomial solution coefficients for each row of each slice. If a single solution is given, then it is used as input for all vectors along the dispersion axis.
    winRng specifies the window range used for searching fitting of individual line profiles
    mxCcor specifies the maximum searchable pixel shift between the template and spectrum
    weights is a boolean flag that specifies if a weighted fitting of the polynomial solution should be used (weighted by the fitted line strengths)
    buildSol is a boolean keyword indicating if the dispersion solution (and hence finding of lines) should be determined as searching goes, or whether the previous input solution is only used
    ncpus is an integer indicating the number of simultaneously run processes to use if run in MP mode
    allowLower is a boolean flag that indicates if a polynomial order lower than the degree specified by mxorder can be used to fit spectra with insufficient number of useable lines
    sigmaClip is the sigma-clipping threshold to use during the dispersion solution fitting stage
    lngthConstraint is a boolean flag that indicates if a linear dispersion solution will be forced, if the pixel distance between the furthest separated fitted lines is less than 1000 pixels 
    MP is a boolean value that indicates if the fitting routine should be run in multiprocessing mode
    adjustFitWin is a boolean flag that specifies if the fit window can be automatically adjusted by the code in order to increase the window range to achieve a better fit to the line
    sigmaLimit is the sigma-clipping threshold to use when searching for useable lines to fit. Lines with strengths/amplitude fits less than this value times the estimated noise level are rejected and not used to find the dispersion solution
    allowSearch is a boolean flag to indicate if the window position for identifying/fitting lines is allowed to wander from its initial guess. This option can be useful if the dispersion solution is slightly off, but it can also cause problems if there are several lines that are close in proximity.
    sigmaClipRounds specifies the number of sigma-clipping iterations to use during the dispersion solution fitting stage
    nPixContFit is the number of pixels to use for identifying continuum points during the continuum fitting stage of the spectrum.
    nSearchRounds indicates the maximum number of iterations allowed if the search window is allowed to wander (allowSearch is True)
    plot is a boolean value that indicates if plotting of the fitting process should occur (useful for debugging purposes only)
    Returned is a list containing the following elements for each row in each slice:
    - the fitted polynomial coefficients (in increasing order) describing the transformation from pixel to wavelength
    - the FWHM of each fitted line used to determine the dispersion solution
    - the line centres (in pixels) of the fitted lines
    - the predicted line centres (in wavelength) corresponding to the fitted lines
    - the RMS difference between the final fitted polynomial solution and the measured line centres (in pixels)
    - the fitted polynomial coefficients (in increasing order) describing the transformation from wavelength to pixels
    """

    #read in line atlas 
    bestLines = wifisIO.readTable(atlas)

    dataLst = []
    tmpLst = []
    
    for i in range(len(dataSlices)):
        dTmp = np.zeros(dataSlices[i].shape, dtype=dataSlices[i].dtype)
        tempTemp = np.zeros(templateSlices[i].shape, dtype=templateSlices[i].dtype)
            
        np.copyto(dTmp, dataSlices[i])
        np.copyto(tempTemp, templateSlices[i])
    #    
    #    whr = np.where(np.isnan(dTmp))
    #    dTmp[whr] = 0.
    #    whr = np.where(np.isnan(tempTemp))
    #    tempTemp[whr] = 0.
        dataLst.append(dTmp)
        tmpLst.append(tempTemp)
        
    #set up input data for running with multiprocessing
    #by extracting all vectors along the dispersion axis
    lst = []

    for i in range(len(dataLst)):

        if (tmpLst[i].ndim >1):

            for j in range(dataLst[i].shape[0]):
                #check if slice sizes the same
                if(j >= len(prevSol[i])):
                    #find last good solution
                    #search for closest solution on left
                    for lowJ in range(len(prevSol[i])-1,-1,-1):
                        if (np.all(np.isfinite(prevSol[i][lowJ]))):
                            tmpSol = prevSol[i][lowJ]
                            tmpTemp = tmpLst[i][lowJ,:]
                            break

                else:
                    ##check for NaN solutions and exchange with closest non-NaN solution
                    if np.any(~np.isfinite(prevSol[i][j])):

                        #search for closest solution on left
                        for lowJ in range(j-1,-1,-1):
                            if (np.all(np.isfinite(prevSol[i][lowJ]))):
                                lowSol = prevSol[i][lowJ]
                                lowTemp = tmpLst[i][lowJ,:]
                                break

                        #search for closest solution on right
                        for highJ in range(j+1,tmpLst[i].shape[0]):
                            if (np.all(np.isfinite(prevSol[i][highJ]))):
                                highSol = prevSol[i][highJ]
                                highTemp = tmpLst[i][highJ,:]
                                break

                        #just adopt the solution that is closest
                        if 'lowSol' in locals():
                            if 'highSol' in locals():
                                closestJ = np.argmin([j-lowJ, highJ-j])
                                tmpSol = [lowSol, highSol][closestJ]
                                tmpTemp =[lowTemp, highTemp][closestJ]
                                del lowSol
                                del highSol
                            else:
                                tmpSol = lowSol
                                tmpTemp = lowTemp
                                del lowSol
                        elif 'highSol' in locals():
                            tmpSol = highSol
                            tmpTemp = highTemp
                            del highSol
                        
                        else:
                            tmpSol = prevSol[i][j]
                            tmpTemp = tmpLst[i][j,:]
                    else:
                        tmpSol = prevSol[i][j]
                        tmpTemp = tmpLst[i][j,:]

                lst.append([dataLst[i][j,:],tmpTemp, bestLines, mxorder,tmpSol,winRng, mxCcor,weights, plot, buildSol,allowLower,sigmaClip,lngthConstraint, adjustFitWin, sigmaLimit, allowSearch, int(sigmaClipRounds),nPixContFit,nSearchRounds])
                    
        else:
            for j in range(dataLst[i].shape[0]):
                lst.append([dataLst[i][j,:],tmpLst[i], bestLines, mxorder,prevSol[i],winRng, mxCcor,weights, plot, buildSol, allowLower, sigmaClip,lngthConstraint, adjustFitWin,sigmaLimit, allowSearch,int(sigmaClipRounds),nPixContFit,nSearchRounds])
                
    #MP = False
    if (MP):
        #setup multiprocessing routines
        if (ncpus == None):
            ncpus =mp.cpu_count()
        pool = mp.Pool(ncpus)

        #run code
        result = pool.map(getSolQuick, lst)
        pool.close()
    else:
        nRows = float(dataLst[0].shape[0])
        
        result = []
        for i in range(len(lst)):
            print('Working on slice ', np.floor(i/nRows), ' and column ', np.mod(i,nRows))
            result.append(getSolQuick(lst[i]))
            
    #extract results and organize them
    dispSolLst = []
    fwhmLst = []
    pixCentLst =[]
    waveCentLst =[]
    rmsLst = []
    pixSolLst = []
    
    strt=0
    for i in range(len(dataLst)):
        dsubLst =[]
        fsubLst =[]
        psubLst = []
        wsubLst = []
        rsubLst = []
        pixsubLst = []

        for j in range(dataLst[i].shape[0]):
            dsubLst.append(result[j+strt][0])
            fsubLst.append(result[j+strt][1])
            psubLst.append(result[j+strt][2])
            wsubLst.append(result[j+strt][3])
            rsubLst.append(result[j+strt][4])
            pixsubLst.append(result[j+strt][5])

        strt += dataLst[i].shape[0]
        dispSolLst.append(dsubLst)
        fwhmLst.append(fsubLst)
        pixCentLst.append(psubLst)
        waveCentLst.append(wsubLst)
        rmsLst.append(rsubLst)
        pixSolLst.append(pixsubLst)
        
    return [dispSolLst, fwhmLst, pixCentLst, waveCentLst, rmsLst, pixSolLst]
    
def buildWaveMap(dispSolLst, npts, fill_missing=True, extrapolate=False):
    """
    Routine to build a wavelength map from the provided dispersion solution list for each image slice
    Usage: waveMapLst = buildWaveMap(dispSolLst, npts, fill_missing=True, extrapolate=False)
    dispSolLst is a list of arrays providing the measured dispersion solution for each pixel in the image slice
    npts sets the length of the resulting image
    fill_missing is a boolean keyword to indicate if interpolation between fitted lines should be used to fill in regions where no information exists
    extrapolate is a boolean keyword to specify if extrapolation should be used to fill in regions outside of regions with fitted lines
    waveMapLst is the output image providing the wavelength at each pixel
    """

    waveMapLst = []
    x = np.arange(npts)

    for dispSol in dispSolLst:
        #initialize wave map array
        waveMap = np.zeros((len(dispSol),npts),dtype='float32')
        
        #populate map with solution
        for i in range(len(dispSol)):
            wave = 0.

            for j in range(dispSol[i].shape[0]):
                wave += dispSol[i][j]*x**j
                
            waveMap[i,:] = wave

        #now fill in rows with missing solutions
        if (fill_missing):
            good = []
            bad = []
            
            for i in range(waveMap.shape[0]):
                #assume that either all points are NaN or none
                if np.any(np.isfinite(waveMap[i,:])):
                    good.append(i)
                else:
                    bad.append(i)
                    
            xint = np.asarray(good)
            bad = np.asarray(bad)
            
            if (bad.shape[0]>0) :
                #get values
                if (extrapolate):
                    finter = interp1d(xint,waveMap[good,:],kind='linear', bounds_error=False,axis=0,fill_value='extrapolate')
                else:
                    finter = interp1d(xint,waveMap[good,:],kind='linear', bounds_error=False,axis=0)

                waveMap[bad,:] = finter(bad)
                                      
  
        waveMapLst.append(waveMap)

    return waveMapLst

def buildFWHMMap(pixCentLst,fwhmLst,npts):
    """
    Routine to build a wavelength map from the provided dispersion solution list for each image slice
    Usage: waveMapLst = buildWaveMap(dispSolLst, npts)
    pixCentLst provides the central pixel location for each fitted line, in each row, of each slice
    fwhmLst is a list containing the measured FWHM at each fitted line position provided in pixCentLst
    npts sets the length of the resulting image to create
    waveMapLst is the output image providing the FWHM at each pixel, interpolated using the information provided in pixCentLst and fwhmLst
    """

    fwhmMapLst = []
    xgrid = np.arange(npts)

    for i in range(len(fwhmLst)):
    
        fwhm = fwhmLst[i]
        cent = pixCentLst[i]
    
        #initialize fwhm map array
        fwhmMap = np.empty((len(fwhm),npts),dtype='float32')
        fwhmMap[:] = np.nan
    
        for j in range(len(fwhm)):
            y = fwhm[j]
            if (len(y) > 1):
                x = cent[j]
                argsort = np.argsort(x)
                #finter = interp1d(x,y, kind='linear', bounds_error=False)
                fwhmMap[j,:] = np.interp(xgrid, x[argsort], y[argsort], left=np.nan, right=np.nan)#finter(xgrid)
        
        fwhmMapLst.append(fwhmMap)
        
    return fwhmMapLst

def trimWaveSliceAll(waveSlices, flatSlices, threshold, MP=True, ncpus=None):
    """
    Routine used to identify useful limits of the wavelenth mapping 
    Usage: out = trimWaveSlice(waveSlices, flatSlices, threshold, MP=True, ncpus=None)
    waveSlices is a list of the wavelength mapping slices
    flatSlices is a list of the flatfield image slices
    threshold is a value indicating the cutoff value relative to the scaled maximum flatfield flux in a given slice to be used to determine the limits (i.e. maximum flux as scaled value of 1.0, minimum flux has scaled value of 0.0)
    MP is a boolean value used to indicate whether multiprocessing should be used
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
    Routine used to identify useful limits of wavelength mapping for a single slice
    Usage: slc = trimWaveSlice(input), where input is a list containing:
    slc - the wavelength mapping slice
    flatSlc - the flatfield slice image
    threshold - the scaled cutoff value relative to the maximum flatfield flux to determine the limits. A median-averaging routine is useed to determine the scaling values, to avoid hot pixels
    slc is the trimmed wavelength mapped slice. All values outside of the trim limits are set to NaN
    """

    slc = np.empty((input[0].shape), dtype=input[0].dtype)
    np.copyto(slc, input[0])
    flatSlc = np.empty(input[1].shape)
    np.copyto(flatSlc, input[1])
    threshold = input[2]

    #get median-averaged values for each column along the dispersion axis to avoid effects of hot pixels

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        flatMed = np.nanmedian(flatSlc, axis=0)

        #rescale so that max is 1.0 and min is 0.0
        flatMed = (flatMed-flatMed.min())/(flatMed.max()-flatMed.min())
        #mx = np.nanmax(flatMed)
    
    #get rid of problematic values
    #flatSlc[~np.isfinite(flatSlc)] = 0.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        whr = np.where(flatMed < threshold)[0]
        
    slc[:,whr] = np.nan

    return slc

def polyFitDispSolution(dispIn,plotFile=None, degree=2):
    """
    Routine to fit a polynomial to the dispersion solution coefficients, to describe the change in the dispersion solution across the spatial direction of each slice. This routine is useful for excluding badly-fit rows.
    Usage: polySol = polyFitDispSolution(dispIn,plotFile=None, degree=2)
    dispIn is a list containing the dispersion coefficients for each row of each slice
    plotFile is the name of the file to save the QC control plots illustrating the quality of fit to each coefficient
    degree is the maximum order/degree of the polynomial allowed to fit each coefficient
    Returned is a list of the same size as dispIn with the polynomial coefficients replaced by the fits
    """
    
    nTerms = 0
    #determine maximum degree of polynomial
    for d in dispIn:
        for i in range(len(d)):
            if (len(d[i])>nTerms):
                nTerms = len(d[i])
   
    polySol = []
    if (plotFile is not None):
        with PdfPages(plotFile) as pdf:
            for i in range(len(dispIn)):
                #organize the coefficients into separates arrays/lists
                c = []
                for j in range(nTerms):
                    c.append([])
            
                d = dispIn[i]
        
                for j in range(len(d)):
                    for k in range(len(d[j])):
                        c[k].append([j,d[j][k]])

                #determine polynomial fits for each coefficient
                f=[]
                for k in range(nTerms):
                    cTmp = np.asarray(c[k])
                    whr = np.where(np.isfinite(cTmp[:,1]))[0]
                    for j in range(3):
                        fc =np.polyfit(cTmp[whr,0].flatten(),cTmp[whr,1].flatten(),degree)
                        ff = np.poly1d(fc)
                        dev = np.abs(ff(cTmp[whr,0])-cTmp[whr,1])
                        whr = whr[np.where(dev <1.*np.std(dev))[0]]
                    f.append(ff)

                #create output list that matches structure of input list
                tmpSol = []
                for j in range(len(d)):
                    tmpLst = []
                    for k in range(nTerms):
                        tmpLst.append(f[k](j))
                    tmpSol.append(np.asarray(tmpLst))
        
                polySol.append(tmpSol)

                fig,ax = plt.subplots(1,nTerms, figsize=(19,6))
                for k in range(nTerms):
                    tmp = np.asarray(c[k])
                    ax[k].plot(tmp[:,0], tmp[:,1], 'o')
                    ax[k].plot(f[k](np.arange(len(c[k]))),'--')
            
                pdf.savefig(fig)
                plt.close()
    else:
        for i in range(len(dispIn)):
            #organize the coefficients into separates arrays/lists
            c = []
            for j in range(nTerms):
                c.append([])
            
            d = dispIn[i]
            
            for j in range(len(d)):
                for k in range(len(d[j])):
                    c[k].append([j,d[j][k]])
                    
            #determine polynomial fits for each coefficient
            f=[]
            for k in range(nTerms):
                cTmp = np.asarray(c[k])
                whr = np.where(np.isfinite(cTmp[:,1]))[0]
                for j in range(3):
                    fc =np.polyfit(cTmp[whr,0].flatten(),cTmp[whr,1].flatten(),degree)
                    ff = np.poly1d(fc)
                    dev = np.abs(ff(cTmp[whr,0])-cTmp[whr,1])
                    whr = whr[np.where(dev <1.*np.std(dev))[0]]
                f.append(ff)

            #create output list that matches structure of input list
            tmpSol = []
            for j in range(len(d)):
                tmpLst = []
                for k in range(nTerms):
                    tmpLst.append(f[k](j))
                tmpSol.append(np.asarray(tmpLst))
        
            polySol.append(tmpSol)
                
#    for i in range(len(dispIn)):
#        #organize the coefficients into separates arrays/lists
#        c = []
#        for j in range(nTerms):
#            c.append([])
#            
#        d = dispIn[i]
#        
#        for j in range(len(d)):
#            for k in range(len(d[j])):
#                c[k].append([j,d[j][k]])
#
#        #determine polynomial fits for each coefficient
#        f=[]
#        for k in range(nTerms):
#            cTmp = np.asarray(c[k])
#            whr = np.where(np.isfinite(cTmp[:,1]))[0]
#            for j in range(2):
#                fc =np.polyfit(cTmp[whr,0].flatten(),cTmp[whr,1].flatten(),degree)
#                ff = np.poly1d(fc)
#                dev = np.abs(ff(cTmp[whr,0])-cTmp[whr,1])
#                whr = whr[np.where(dev <1.*np.std(dev))[0]]
#            f.append(ff)
#
#        #create output list that matches structure of input list
#        tmpSol = []
#        for j in range(len(d)):
#            tmpLst = []
#            for k in range(nTerms):
#                tmpLst.append(f[k](j))
#            tmpSol.append(np.asarray(tmpLst))
#        
#        polySol.append(tmpSol)
#
#        if plotFile:
#            fig,ax = plt.subplots(1,nTerms, figsize=(19,6))
#            for k in range(nTerms):
#                tmp = np.asarray(c[k])
#                ax[k].plot(tmp[:,0], tmp[:,1], 'o')
#                ax[k].plot(f[k](np.arange(len(c[k]))),'--')
#            
#            pdf.savefig(fig)
#            plt.close
           
    return polySol

def medSmoothDispSolution(dispIn, nPix=5, plot=False):
    """
    Routine used to smooth a list of coefficients of a polynomial dispersion solutions by using median-averaging.
    Usage: smoothSol = medSmoothDispSolution(dispIn, nPix=5, plot=False)
    dispIn is a list containing the dispersion coefficients for each row of each slice
    nPix is the number of pixels to use for determining the median-averaging
    plot is a boolean keyword to indicate whether plotting should be carried out (for debugging purposes)
    Returned is the smoothed version of dispIn
    """

    nTerms = 0
    #determine maximum degree of polynomial
    for d in dispIn:
        for i in range(len(d)):
            if (len(d[i])>nTerms):
                nTerms = len(d[i])


    smoothSol = []
    for d in dispIn:
        c = []
        for j in range(nTerms):
            c.append([])

        for j in range(len(d)):
            for k in range(nTerms):
                xrng = np.arange(nPix)-int(nPix/2)+j
                xrng = xrng[np.logical_and(xrng>=0,xrng<len(d))]
                yrng = []
                
                for l in range(len(xrng)):
                    if len(d[xrng[l]])>k:
                        yrng.append(d[xrng[l]][k])
                    else:
                        yrng.append(np.nan)
                c[k].append(np.nanmedian(yrng))

        tmpSol = []
        for j in range(len(d)):
            tmpLst = []
            for k in range(nTerms):
                tmpLst.append(c[k][j])
            tmpSol.append(np.asarray(tmpLst))
        smoothSol.append(tmpSol)

        if (plot):
            fig,ax = plt.subplots(1,nTerms, figsize=(19,6))
            for k in range(nTerms):
                for j in range(len(d)):
                    if len(d[j])>k:
                        ax[k].plot(j, d[j][k],'bo')
                ax[k].plot(c[k])
            plt.show()

    return smoothSol

def gaussSmoothDispSolution(dispIn, nPix=5, plotFile=None):
    """
    Routine used to smooth a list of coefficients of a polynomial dispersion solutions by convolution with a Gaussian kernel.
    Usage: smoothSol = gaussSmoothDispSolution(dispIn, nPix=5, plotFile=None)
    dispIn is a list containing the dispersion coefficients for each row of each slice
    nPix is the 1-sigma width of the Gaussian kernel 
    plotFile is the name of the file to save the QC control plots illustrating the quality of fit to each coefficient
    Returned is the smoothed version of dispIn
    """

    gKern = conv.Gaussian1DKernel(nPix)
    
    nTerms = 0
    #determine maximum degree of polynomial
    for d in dispIn:
        for i in range(len(d)):
            if (len(d[i])>nTerms):
                nTerms = len(d[i])


    smoothSol = []
    if (plotFile is not None):
        with PdfPages(plotFile) as pdf:

            for d in dispIn:
                c = []
                smth=[]
                for j in range(nTerms):
                    c.append([])
                    smth.append([])
            
                for j in range(len(d)):
                    for k in range(nTerms):
                        if len(d[j])>k:
                            c[k].append(d[j][k])
                        else:
                            c[k].append(np.nan)

                #smoothing time!
                for k in range(nTerms):
                    smth[k] = conv.convolve(c[k],gKern,boundary='extend', normalize_kernel=True)
            
                tmpSol = []
                for j in range(len(d)):
                    tmpLst = []
                    for k in range(nTerms):
                        tmpLst.append(smth[k][j])
                    tmpSol.append(np.asarray(tmpLst))
                smoothSol.append(tmpSol)

                fig,ax = plt.subplots(1,nTerms, figsize=(19,6))
                for k in range(nTerms):
                    for j in range(len(d)):
                        if len(d[j])>k:
                            ax[k].plot(j, d[j][k],'bo')
                        ax[k].plot(smth[k])
                pdf.savefig(fig)
                plt.close()

    else:
        
        for d in dispIn:
            c = []
            smth=[]
            for j in range(nTerms):
                c.append([])
                smth.append([])
            
            for j in range(len(d)):
                for k in range(nTerms):
                    if len(d[j])>k:
                        c[k].append(d[j][k])
                    else:
                        c[k].append(np.nan)

            #smoothing time!
            for k in range(nTerms):
                smth[k] = conv.convolve(c[k],gKern,boundary='extend', normalize_kernel=True)
            
            tmpSol = []
            for j in range(len(d)):
                tmpLst = []
                for k in range(nTerms):
                    tmpLst.append(smth[k][j])
                tmpSol.append(np.asarray(tmpLst))
            smoothSol.append(tmpSol)
      
    
#    for d in dispIn:
#        c = []
#        smth=[]
#        for j in range(nTerms):
#            c.append([])
#            smth.append([])
#            
#        for j in range(len(d)):
#            for k in range(nTerms):
#                if len(d[j])>k:
#                    c[k].append(d[j][k])
#                else:
#                    c[k].append(np.nan)
#
#        #smoothing time!
#        for k in range(nTerms):
#            smth[k] = conv.convolve(c[k],gKern,boundary='extend', normalize_kernel=True)
#            
#        tmpSol = []
#        for j in range(len(d)):
#            tmpLst = []
#            for k in range(nTerms):
#                tmpLst.append(smth[k][j])
#            tmpSol.append(np.asarray(tmpLst))
#        smoothSol.append(tmpSol)
#
#        if (plotFile):
#            with PdfPages(plotFile) as pdf:
#                fig,ax = plt.subplots(1,nTerms, figsize=(19,6))
#                for k in range(nTerms):
#                    for j in range(len(d)):
#                        if len(d[j])>k:
#                            ax[k].plot(j, d[j][k],'bo')
#                    ax[k].plot(smth[k])
#                pdf.savefig(fig)
#                plt.close()

    return smoothSol

def cleanDispSol(result, plotFile=None, threshold=1.5):
    """
    Routine to identify rows with badly-fit dispersion solutions
    result is the full output returned from getWaveSol
    plotFile is the name of the file for which to save the QC image
    threshold specifies the sigma-clipping threshold used to identify badly-fit rows
    Returned is a list containing the following:
    - rmsClean - the list of RMS values for each row of each slice, with the bad solutions replaced with NaNs
    - dispSolClean - the list of coefficients describing the transformation from pixel to wavelengths for each row of each slice, with the bad solutions replaced with NaNs
    - pixSolClean - the list of coefficients describing the transformation from wavelength to pixel for each row of each slice, with the bad solution replaced with NaNs
   """

    dispSolLst = result[0]
    rms = result[4]
    pixSolLst = result[5]

    dispSolClean = []
    pixSolClean = []
    rmsClean = []

    medLst = []
    stdLst = []

    #*******************
    #CONSIDER THREADING THIS SECTION
    #*******************
    
    for i in range(len(rms)):
        r = np.asarray(rms[i])
        
        dispSlice = []
        pixSlice = []
        rmsSlice = []
                
        for k in range(2):
            m = np.nanmedian(r)
            s = np.nanstd(r)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                r[r>m+s] = np.nan

        medLst.append(m)
        stdLst.append(s)
        
        for j in range(len(r)):
            if rms[i][j] > m+threshold*s:
                dispSlice.append(np.asarray([np.nan]))
                pixSlice.append(np.asarray([np.nan]))
            else:
                dispSlice.append(dispSolLst[i][j])
                pixSlice.append(pixSolLst[i][j])
                
        dispSolClean.append(dispSlice)
        pixSolClean.append(pixSlice)
        rmsClean.append(r)
        
    #print quality control stuff
    if plotFile is not None:
        print('Plotting quality control')

        with PdfPages(plotFile) as pdf:
            for i in range(len(rms)):
                
                fig = plt.figure()
                plt.plot(rms[i], 'b')
                plt.plot([0,len(rms[i])],[medLst[i]+threshold*stdLst[i],medLst[i]+threshold*stdLst[i]], 'r--')
                plt.xlabel('Column #')
                plt.ylabel('RMS in pixels')
                plt.title('Slice: ' +str(i) + ', RMS Cutoff: '+'{:4.2f}'.format(medLst[i]+threshold*stdLst[i])+', Median RMS: ' + '{:4.2f}'.format(medLst[i]))
                pdf.savefig(fig, dpi=300)
                plt.close()
                
    return rmsClean, dispSolClean, pixSolClean

def buildWaveMap2(dispSolLst, npts, fill_missing=True, extrapolate=False, MP=True, ncpus=None,offset=0.):
    """
    Routine to build a wavelength map from the provided dispersion solution list for each image slice
    Usage: waveMapLst = buildWaveMap2(dispSolLst, npts, fill_missing=True,extrapolate=False,MP=True, ncpus=None)
    dispSolLst is a list of arrays providing the polynomial coefficients for the dispersion solution (pixel to wavelength) for each row in each slice
    npts is not used and is only arround for backwards compatability with older routines
    fill_missing is a boolean keyword to specify if rows with missing solutions should be filled by linear interpolation of the surrounding rows with good solutions
    extrapolate is a boolean flag to indicate if rows with missing solutions that are not surrounded by good solutions should be filled by linear extrapolation
    MP is a boolean flag to indicate if the routine should be run in multiprocessing mode
    ncpus sets the maximum number of processes to simultaneously run in MP mode
    waveMapLst is the list output images providing the wavelength at each pixel
    """

    x = np.arange(npts)

    inpLst = []

    if MP:
        #build input list
        for i in range(len(dispSolLst)):
            inpLst.append([dispSolLst[i], x, fill_missing, extrapolate, offset])

        #setup multiprocessing routines
        if (ncpus == None):
            ncpus =mp.cpu_count()
        pool = mp.Pool(ncpus)

        #run code
        waveMapLst = pool.map(buildWaveMap2Slice, inpLst)
        pool.close()
    else:
        waveMapLst = []

        for i in range(len(dispSolLst)):
            waveMapLst.append(buildWaveMap2Slice([dispSolLst[i], x, fill_missing, extrapolate,offset]))

    return waveMapLst

def buildWaveMap2Slice(input):
    """
    Routine to build a wavelength map from the provided dispersion solution for a single image slice
    Usage: waveMapOut = buildWaveMap2Slice(input)
    input is a list containing:
    - a list of arrays providing the polynomial coefficients for the dispersion solution (pixel to wavelength) for each row in the slice
    - a numpy array providing the pixel coordinates along the dispersion direction of the input slice
    - a boolean keyword to specify if rows with missing solutions should be filled by linear interpolation of the surrounding rows with good solutions
    - a boolean flag to indicate if rows with missing solutions that are not surrounded by good solutions should be filled by linear extrapolation
    - an optional offset in pixels to apply to the solution
    waveMapOut is the output maps providing the wavelength at each pixel
    """

    dispSol = input[0]
    x = input[1]
    npts = x.shape[0]
    fill_missing = input[2]
    extrapolate = input[3]
    offset = input[4]
    
    #initialize wave map array
    waveMap = np.zeros((len(dispSol),npts),dtype='float32')
        
    #populate map with solution
    for i in range(len(dispSol)):
        wave = 0.

        for j in range(dispSol[i].shape[0]):
            wave += dispSol[i][j]*(x-offset)**j
                
        waveMap[i,:] = wave

    #now fill in rows with missing solutions
    if (fill_missing):
        good = []
        bad = []
            
        for i in range(waveMap.shape[0]):
            #assume that either all points are NaN or none
            if np.any(np.isfinite(waveMap[i,:])):
                good.append(i)
            else:
                bad.append(i)
                    
        xint = np.asarray(good)
        bad = np.asarray(bad)
            
        if (bad.shape[0]>0) :
            finter = interp1d(xint,waveMap[good,:],kind='linear', bounds_error=False,axis=0)
            waveMap[bad,:] = finter(bad)

    if extrapolate:
        #first check if extrapolation is needed
        if np.any(~np.isfinite(waveMap)):
            whrGood = np.where(np.isfinite(waveMap[:,j]))[0]
            whrBad = np.where(~np.isfinite(waveMap[:,j]))[0]
            
            xfit = x[whrGood]
            xBad = x[whrBad]
                
            for j in range(waveMap.shape[1]):
                yfit = waveMap[whrGood,j]
                pcoef = np.polyfit(xfit, yfit, 1)
                poly = np.poly1d(pcoef)
                waveMap[whrBad,j] = poly(xBad)
                              
    return waveMap
    
def smoothWaveMapAll(waveMapLst, smth=3, MP=True, ncpus=None):
    """
    """

    inpLst = []

    gKern = conv.Gaussian1DKernel(smth)
    
    if MP:
        #build input list
        for i in range(len(waveMapLst)):
            inpLst.append([waveMapLst[i],gKern])

        #setup multiprocessing routines
        if (ncpus == None):
            ncpus =mp.cpu_count()
        pool = mp.Pool(ncpus)

        #run code
        waveMapOut = pool.map(smoothWaveMapSlice, inpLst)
                        
        pool.close()
    else:
        waveMapOut = []

        for i in range(len(waveMapLst)):
            waveMapOut.append(smoothWaveMapSlice([waveMapLst[i], gKern]))

    return waveMapOut
                               

def smoothWaveMapSlice(input):
    """
    Routine used to smooth a map of pixel to wavelength coordinates using a Gaussian convolution. The smoothing occurs along the spatial axis only.
    Usage: waveSmth = smoothWaveMapSlice(input)
    input is a list containing:
    - the original wavelength map image to be smoothed
    - the number of pixels specifying the 1-sigma width of the Gaussian kernel to be used for smoothing
    Returned is a smoothed version of the input wavelength map    
    """

    waveMap = input[0]
    gKern = input[1]
    
    waveSmth = np.empty(waveMap.shape, dtype=waveMap.dtype)
    waveSmth[:] = np.nan

    
    for i in range(waveMap.shape[1]):
        waveSmth[:,i] = conv.convolve(waveMap[:,i],gKern,boundary='extend')
    
    return waveSmth

def polyFitWaveMapSlice(input):
    """
    Routine used to replace all values of a wavelength map by a polynomial fit instead. The polynomial fit occurs along the spatial axis, for each pixel along the dispersion axis.
    Usage: waveFit = polyFitWaveMapSlice(input)
    input is a list containing:
    - the input wavemap image
    - the polynomial degree/order to use for the fitting
    Returned is an image of the same dimensions as the input image, but with all values replaced by the fits 
    """

    waveMap = input[0]
    degree = input[1]
    
    waveFit = np.empty(waveMap.shape, dtype=waveMap.dtype)
    waveFit[:] = np.nan

    x = np.arange(waveMap.shape[0])
    whr = np.where(np.isfinite(waveMap[:,0]))
    whr = whr[0]
    
    xfit = x[whr]
    
    for i in range(waveMap.shape[1]):
        pcoef = np.polyfit(xfit,waveMap[whr,i],degree)
        poly = np.poly1d(pcoef)
        waveFit[:,i] = poly(x)
    
    return waveFit

def polyFitWaveMapAll(waveMapLst, degree=3, MP=True, ncpus=None):
    """
    Routine used to replace all values of a list of wavelength maps by polynomial fits instead. The polynomial fits occur along the spatial axis, for each pixel along the dispersion axis.
    Usage: waveMapOut = polyFitWaveMapAll(waveMapLst, degree=3, MP=True, ncpus=None)
    waveMapLst is the list of wavemap images to be fit
    degree specifies the polynomial order/degree to use for fitting
    MP is boolean value that specifies if multiprocessing should be used for the routine
    ncpus specifies the number of processes to simultaneously run when in MP mode
    Returned is a list of images of the same dimensions as the input images, but with all wavemap values replaced by the fits 
    """

    inpLst = []

    if MP:
        #build input list
        for i in range(len(waveMapLst)):
            inpLst.append([waveMapLst[i],degree])

        #setup multiprocessing routines
        if (ncpus == None):
            ncpus =mp.cpu_count()
        pool = mp.Pool(ncpus)

        #run code
        waveMapOut = pool.map(polyFitWaveMapSlice, inpLst)
                        
        pool.close()
    else:
        waveMapOut = []

        for i in range(len(waveMapLst)):
            waveMapOut.append(polyFitWaveMapSlice([waveMapLst[i], degree]))

    return waveMapOut
                            
def polyCleanDispSol(result, plotFile=None, threshold=1.5, returnFit=False):
    """
    """

    dispSolLst = result[0]
    rms = result[4]
    pixSolLst = result[5]
    threshold = 1.5
    #plotFile = 'test.pdf'

    cleanSolLst = []
    polyLst = []
    for dispSol in dispSolLst:
        d = np.array(dispSol)
        cleanSol = np.empty(d.shape)
        cleanSol[:] = np.nan
        polySol = np.empty(d.shape)
        polySol[:] = np.nan
    
        for i in range(d.shape[1]):
            whrGood = np.where(np.isfinite(d[:,i]))[0]
            if whrGood.shape[0] > 0:
                for j in range(3):
                    fitCoef = np.polyfit(whrGood,d[whrGood,i],2)
                    poly = np.poly1d(fitCoef)
                    #cleanSol[:,i] = poly(np.arange(d.shape[0]))
                    dev = np.abs(poly(whrGood)-d[whrGood,i])
                    d[whrGood[dev > 2.*np.std(dev)+np.nanmedian(dev)],i] = np.nan
                    whrGood = np.where(np.isfinite(d[:,i]))[0]

                #now replace all solutions that deviate by more than threshold*deviation away from solution
                fitCoef = np.polyfit(whrGood,d[whrGood,i],2)
                poly = np.poly1d(fitCoef)
                #cleanSol[:,i] = poly(np.arange(d.shape[0]))
                dev = np.abs(poly(whrGood)-d[whrGood,i])
                d[whrGood[dev > threshold*np.nanstd(dev)+np.nanmedian(dev)],i] = np.nan
                whrBad = np.where(~np.isfinite(d[:,i]))[0]
        
            polySol[:,i] = poly(np.arange(d.shape[0]))
            cleanSol[:,i] = np.asarray(dispSol)[:,i]
            cleanSol[whrBad,i] = poly(whrBad)

        cleanSolLst.append(cleanSol)
        polyLst.append(polySol)
        
    #print quality control stuff
    if plotFile is not None:
        print('Plotting quality control')

        with PdfPages(plotFile) as pdf:
            for j in range(len(dispSolLst)):

                d = np.asarray(dispSolLst[j])
                c = cleanSolLst[j]
                p = polyLst[j]
            
                fig = plt.figure()
                #plt.title('Slice: ' +str(j))
                                
                for i in range(d.shape[1]):
                    locals()['ax'+str(i)] = fig.add_subplot(4,1, i+1)
                    plt.plot(c[:,i], 'bo',markersize=2, label='Good pixels')
                    plt.plot(d[:,i],'ro', markersize=2, label='Bad pixels')
                    plt.plot(p[:,i],'k',label='Polynomial fit')
                    plt.xticks(plt.xticks()[0], np.repeat('',plt.xticks()[0].shape[0]))
                plt.xticks(plt.xticks()[0], plt.xticks()[0])
                plt.xlabel('Slice # ' + str(j) + ', Column #')
                plt.legend()
                plt.tight_layout()
                pdf.savefig(fig, dpi=300)
                plt.close()


    
    #need to add pixSolClean
    if returnFit:
        return [np.nan,polyLst,np.nan]
    else:
        return [np.nan,cleanSolLst,np.nan]
