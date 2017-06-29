"""

Tools to help determine dispersion solution and line fitting

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

def gaussFit(x, y, guessWidth, plot=False,title=''):
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
        plt.xlabel('Pixel')
        plt.ylabel('Value')
        plt.title(title)
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
    the dispersion solution is a list containing the polynomial coefficients in increasing order (e.g. [a0, a1, ..., an]) for converting WAVELENGTH TO PIXELS, NOT PIXELS TO WAVELENGTH (i.e. it is the inverse solution, but is an output of this code)
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
    nMed = input[17]
    totPix = len(yRow)

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
    tmp = yRow[np.isfinite(yRow)]
    xFlr = xFlr[np.isfinite(yRow)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        flr = np.nanmedian(tmp)
        
        for i in range(10):
            whr = np.where((tmp-flr) < 3.*np.nanstd(tmp))
            tmp = tmp[whr[0]]
            xFlr = xFlr[whr[0]]
            flr = np.nanmedian(tmp)

        #get and subtract a cubic spline fitted to continuum
        xSpline = [0]
        ySpline = [np.nanmedian(yRow[0:int(nMed/2)])]

        for i in range(0,yRow.shape[0]-nMed,nMed):
            xSpline.append(i+nMed/2.)
            ySpline.append(np.nanmedian(yRow[i:i+nMed]))

        xSpline.append(yRow.shape[0]-1)
        ySpline.append(np.nanmedian(yRow[-int(nMed/2):]))

        tmp -= spline(xSpline,ySpline, xFlr, order=3)
        flrFit = spline(xSpline,ySpline, np.arange(yRow.shape[0]), order=3)

        #carry out one more round of pixel rejection after continuum subtraction
        whr = np.where(tmp< 3.*np.nanstd(tmp))
        tmp = tmp[whr[0]]
                
        #measure the noise level
        nse = np.nanstd(tmp)
        
        #remove continuum from input spectrum
        yRow -= flrFit
        nse = nse/(np.nanmax(yRow))
    
        #normalize the spectra so that the predicted line strengths are on the same scale as the observed spectrum
        yRow = yRow/np.nanmax(yRow)
        template = template/np.nanmax(template) #for plotting purposes only

    #use this offset to determine window range and list of lines to use
    #base on provided dispersion solution from template, which is expected to be close enough

    if (np.all(np.isfinite(templateSol))):
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

    if (plot):
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
        
        #print(pixOffset)
        #print('Going to fit', nlines)
        #print(best)
        #print(bestPix)

        for i in range(nlines):
            try:
                pixRng = (np.arange(winRng)-winRng2 + atlasPix[i]).astype('int') #Centre window on predicted line centre
                
                yRng = yRow[pixRng]

                #find location of maximum signal
                mx = np.argmax(yRng)
                mxPos = pixRng[mx]
                prevMx = -1
                if plot:
                    print('*********')
                    print('Fitting line ' +str(atlas[i,0]))
                    
                while (mxPos != prevMx):
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

                    if (not allowSearch):
                        break
                        
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
                                amp,cent,wid = gaussFit(pixRng,yRng, winRng/3.,plot=plot,title=str(atlasPix[i]))

                                if (adjustFitWin and wid > winRng/3):
                                    if (plot):
                                        print('adjusting fit window')
                                        
                                    pixRng = int(cent) + np.arange(int(wid)*5)-int(wid*5/2)
                                    pixRng= pixRng[np.logical_and(pixRng >=0, pixRng<totPix)]
                                    yRng = yRow[pixRng]

                                    winRngTmp = wid*4
                                    amp,cent,wid = gaussFit(pixRng,yRng, winRngTmp/3.,plot=plot,title=str(atlasPix[i]))
                                
                                #only keep line if amplitude of fit >2*noise level #and width of fit <1/2 of winRng
                                if (amp/nse >= sigmaLimit and np.abs(wid) < winRngTmp/2.):
                                    if plot:
                                        print('Keeping line')
                                        
                                    centFit.append(cent)
                                    widthFit.append(wid)
                                    ampFit.append(amp)
                                    atlasFit.append(atlas[i,0])
                                    atlasPixFit.append(atlasPix[i])
                                else:
                                    if (plot):
                                        print('badly fit line, excluding')
                                        
                                if (len(centFit)>mxorder and buildSol):
                                    #update "guessed" dispersion solution to get better line centres
                                    tmpCoef = np.polyfit(centFit, atlasFit,1, w=ampFit)
                                    atlasPix = (atlas[:,0]-tmpCoef[1])/tmpCoef[0]
                                    
                            except (RuntimeError):
                                pass
                        else:
                            if plot:
                                print('S/N too low, excluding')
                    else:
                        if plot:
                            print('S/N too low, excluding')
            except (IndexError, ValueError):
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
            #if (plot == True):
            #plt.plot(centFit,atlasFit, 'bo')

            centFit = np.array(centFit)
            atlasFit = np.array(atlasFit)
            widthFit = np.array(widthFit)
            ampFit = np.array(ampFit)
                
            #centFit = centFit[whr[0]]
            #atlasFit = atlasFit[whr[0]]
            #widthFit = widthFit[whr[0]]
            #ampFit = ampFit[whr[0]]

            if (plot):
                fig = plt.figure(figsize=(10,5))
                #ax = fig.add_subplot(211)
                gs = gridspec.GridSpec(2,2)
                gs.update(left=0.1,right = 0.98)
                ax1 = fig.add_subplot(gs[0,0])
                ax1.plot(centFit, atlasFit, 'bo')
                
            if (useWeights):        
                fitCoef = np.polyfit(centFit, atlasFit,mxorder, w=ampFit) # returns polynomial coefficients in reverse order
            else:
                fitCoef = np.polyfit(centFit, atlasFit,mxorder) # returns polynomial coefficients in reverse order

            #exclude poorly fit lines based on deviation from best fit
            for i in range(sigmaClipRounds):
                poly = np.poly1d(fitCoef)
                dev = atlasFit-poly(centFit)
                whr = np.where(np.abs(dev) < sigmaClip*np.std(dev))
                if (plot):
                    print('std dev for round',i, 'is',np.std(dev), ' in wavelength')
                
                centFit = centFit[whr[0]]
                atlasFit = atlasFit[whr[0]]
                widthFit = widthFit[whr[0]]
                ampFit = ampFit[whr[0]]

            if (len(centFit) > mxorder):
                #constrain fit to a line if line separation is <1000
                if (lngthConstraint):
                    if ((np.nanmax(centFit)-np.nanmin(centFit)) < 1000):
                        mxorder=1

                        if (plot):
                            print('Forcing linear solution')
                            
                        if (useWeights):        
                            fitCoef = np.polyfit(centFit, atlasFit,mxorder, w=ampFit) # returns polynomial coefficients in reverse order
                        else:
                            fitCoef = np.polyfit(centFit, atlasFit,mxorder) # returns polynomial coefficients in reverse order

                        poly = np.poly1d(fitCoef)
                        dev = atlasFit-poly(centFit)
                        whr = np.where(np.abs(dev) < sigmaClip*np.std(dev))
                        if (plot):
                            print('std dev for round',i, 'is',np.std(dev), ' in wavelength')
                
                        centFit = centFit[whr[0]]
                        atlasFit = atlasFit[whr[0]]
                        widthFit = widthFit[whr[0]]
                        ampFit = ampFit[whr[0]]
                        
                if (useWeights):
                    fitCoef = np.polyfit(centFit, atlasFit,mxorder, w=ampFit) # returns polynomial coefficients in reverse order
                else:
                    fitCoef = np.polyfit(centFit, atlasFit,mxorder) # returns polynomial coefficients in reverse order
                goodFit = True
            
                #compute RMS, in terms of pixels
                if (useWeights):
                    pixCoef = np.polyfit(atlasFit, centFit, mxorder, w=ampFit)
                else:
                    pixCoef = np.polyfit(atlasFit, centFit, mxorder)

                poly = np.poly1d(fitCoef)
                polyPix = np.poly1d(pixCoef)
                diff = polyPix(atlasFit) - centFit # wavelength units
                rms = np.sqrt((1./centFit.shape[0])*(np.sum(diff**2.))) #wavelength units
                #compute dispersion
                #dispersion =0
                #for k in range(1,mxorder):
                #    dispersion += fitCoef[k]**k
                #rms /= dispersion #fitCoef[0]
                
                #for testing purposes only
                if (plot):
                    ax1.set_xlim(0, yRow.shape[0])
                    ax1.plot(centFit, atlasFit, 'ro')
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
        
    
    if (goodFit):
        return(fitCoef[::-1],widthFit*2.*np.sqrt(2.*np.log(2.)), centFit, atlasFit, np.abs(rms), pixCoef[::-1])
    else:
        return np.repeat(np.nan,mxorder+1), [],[], [],np.nan, np.repeat(np.nan,mxorder+1)
    

def getWaveSol (dataSlices, templateSlices,atlas, mxorder, prevSol, winRng=7, mxCcor=30, weights=False, buildSol=False, ncpus=None, allowLower=False, sigmaClip=2., lngthConstraint=False, MP=True, adjustFitWin=False, sigmaLimit=3, allowSearch=False, sigmaClipRounds=1,nPixContFit=200):
    """
    Computes dispersion solution for each set of pixels along the dispersion axis in the provided image slices.
    Usage: output = getWaveSol(dataSlices, template, mxorder, prevSolution, winRng, mxCcor, weights, buildSol, ncpus, allowLower, sigmaClip, lngthConstraint)
    dataSlices is a list of the input 2D images from which the new dispersion solution will be derived. The dispersion axis of each slice is oriented along the x-axis.
    templateSlices is a list of template images from which a known solution is already determined. templateSlices can instead be a list of 1D spectra that will be used as input for all vectors along the dispersion axis for each image slice.
    atlas is the name of the file containing the atlas line list to use for fitting (a 2 column file with the first column corresponding to the central wavelength of the line, and the second the line intensity/flux)
    mxorder is the highest allowable term in the polynomial solution of the form y = Sum_i^mx a_i*x^i, where mx is the maximum polynomial order (e.g. mx = 1 -> y = a0 + a1*x).
    prevSolution is a list of containing the previously determined coefficients for each set of pixels. If a single solution is given, then it is used as input for all vectors along the dispersion axis.
    dispAxis specifies the dispersion direction (0 -> along the y-axis, 1-> along the x-axis)
    winRng is a keyword to change window range for searching for line centre and fitting (default is 7 pixels)
    mxCcor is a keyword to change the the window range for finding the maximum searchable pixel shift between the template and spectrum (default is 30 pixels)
    weights is a keyword to carry out a weighted polynomial fit for determining dispersion solution (default is False) 
    buildSol is a boolean keyword indicating if the dispersion solution (and hence finding of lines) should be determined as searching goes, or whether the previous input solution is only used
    ncpus is an integer keyword indicating the number of simultaneously run processes
    Returns the coefficients of the polynomial fits [lambda(pixel) = p(pixel)], the FWHM of each fit, the line centres of the lines that were fit (in pixels), the corresponding line centres according to the line atlas, the RMS of the polynomial solution
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

                lst.append([dataLst[i][j,:],tmpTemp, bestLines, mxorder,tmpSol,winRng, mxCcor,weights, False, buildSol,allowLower,sigmaClip,lngthConstraint, adjustFitWin, sigmaLimit, allowSearch, sigmaClipRounds,nPixContFit])
                        
        else:
            for j in range(dataLst[i].shape[0]):
                    lst.append([dataLst[i][j,:],tmpLst[i], bestLines, mxorder,prevSol[i],winRng, mxCcor,weights, False, buildSol, allowLower, sigmaClip,lngthConstraint, adjustFitWin,sigmaLimit, allowSearch,sigmaClipRounds,nPixContFit])

    if (MP):
        #setup multiprocessing routines
        if (ncpus == None):
            ncpus =mp.cpu_count()
        pool = mp.Pool(ncpus)

        #run code
        result = pool.map(getSolQuick, lst)
        pool.close()
    else:
        result = []
        for i in range(len(lst)):
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
    Usage: waveMapLst = buildWaveMap(dispSolLst, npts)
    dispSolLst is a list of arrays providing the measured dispersion solution for each pixel in the image slice
    npts sets the length of the resulting image
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
    dispSolLst is a list of arrays providing the measured dispersion solution for each pixel in the image slice
    npts sets the length of the resulting image
    waveMapLst is the output image providing the wavelength at each pixel
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
                finter = interp1d(x,y, kind='linear', bounds_error=False)
                fwhmMap[j,:] = finter(xgrid)
        
        fwhmMapLst.append(fwhmMap)
        
    return fwhmMapLst

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

    slc = np.empty((input[0].shape), dtype=input[0].dtype)
    np.copyto(slc, input[0])
    flatSlc = np.empty(input[1].shape)
    np.copyto(flatSlc, input[1])
    threshold = input[2]

    #get median-averaged values for each column along the dispersion axis to avoid effects of hot pixels

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        flatMed = np.nanmedian(flatSlc, axis=0)
        mx = np.nanmax(flatMed)
    
    #get rid of problematic values
    #flatSlc[~np.isfinite(flatSlc)] = 0.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        whr = np.where(flatMed < threshold*mx)[0]
        
    slc[:,whr] = np.nan

    return slc

def polyFitDispSolution(dispIn,plotFile=None, degree=2):
    """
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
    """

    dispSolLst = result[0]
    rms = result[4]
    pixSolLst = result[5]

    dispSolClean = []
    pixSolClean = []
    rmsClean = []

    medLst = []
    stdLst = []
    
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
                plt.title('Slice: ' +str(i) + ', RMS Cutoff: '+'{:4.2f}'.format(medLst[i]+threshold*stdLst[i]))
                pdf.savefig(fig, dpi=300)
                plt.close()
                
    return rmsClean, dispSolClean, pixSolClean

def buildWaveMap2(dispSolLst, npts, fill_missing=True, extrapolate=False, MP=True, ncpus=None):
    """
    Routine to build a wavelength map from the provided dispersion solution list for each image slice
    Usage: waveMapLst = buildWaveMap(dispSolLst, npts)
    dispSolLst is a list of arrays providing the measured dispersion solution for each pixel in the image slice
    npts sets the length of the resulting image
    waveMapLst is the output image providing the wavelength at each pixel
    """

    x = np.arange(npts)

    inpLst = []

    if MP:
        #build input list
        for i in range(len(dispSolLst)):
            inpLst.append([dispSolLst[i], x, fill_missing, extrapolate])

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
            waveMapLst.append(buildWaveMap2Slice([dispSolLst[i], x, fill_missing, extrapolate]))

    return waveMapLst

def buildWaveMap2Slice(input):
    """
    """

    dispSol = input[0]
    x = input[1]
    npts = x.shape[0]
    fill_missing = input[2]
    extrapolate = input[3]
    
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
                            
