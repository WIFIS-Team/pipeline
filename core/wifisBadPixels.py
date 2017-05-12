"""


"""

import numpy as np
import multiprocessing as mp
import wifisIO
import matplotlib.pyplot as plt


def corBadPixelsAll(data,dispAxis=0,mxRng=2,MP=True, ncpus=None, sigma=False):
    """
    """

    #run through list of all bad pixels and correct them through linear interpolation along dispersion axis

    #create copies of input data array
    #tmpData = np.empty(data.shape, dtype=data.dtype)
    #np.copyto(tmpData, data)
    #tmpBPM = np.zeros(BPM.shape, dtype='int')
    #np.copyto(tmpBPM, BPM.astype('int'))
    
    #if (dispAxis==0):
    #    tmpBPM = tmpBPM.T
    #    tmpData = tmpData.T
           
    #create list of bad pixels
    bad = np.where(~np.isfinite(data))

    if (MP == False):
        corrections=[]

        #run separate algorithm on uncertainty data
        if (sigma):
            for i in range(bad[0].shape[0]):
                if (dispAxis==0):
                    #only provide the correction routine with the disperion vector for that particular bad pixel
                    corrections.append(corBadPixelSigma([data[:,bad[1][i]],  bad[0][i], mxRng]))
                else:
                    corrections.append(corBadPixelSigma([data[bad[0][i],:], bad[1][i], mxRng]))
        else:
            for i in range(bad[0].shape[0]):
                #only provide the correction routine with the disperion vector for that particular bad pixel
                if (dispAxis==0):
                    corrections.append(corBadPixel([data[:,bad[1][i]], bad[0][i], mxRng]))
                else:
                    corrections.append(corBadPixel([data[bad[0][i],:], bad[1][i], mxRng]))
    else:
        #setup input list to feed MP routine
        lst = []
        
        for i in range(bad[0].shape[0]):
            if (dispAxis==0):
                lst.append([data[:,bad[1][i]], bad[0][i], mxRng])
            else:
                lst.append([data[bad[0][i],:], bad[1][i], mxRng])

        #run MP routine
        if (ncpus == None):
            ncpus =mp.cpu_count()
        pool = mp.Pool(ncpus)
        if (sigma):
            corrections = pool.map(corBadPixelSigma, lst)
        else:
            corrections = pool.map(corBadPixel, lst)
        pool.close()

    #create new array with corrected pixels
    outData = np.empty(data.shape, dtype=data.dtype)
    np.copyto(outData, data)
    
    for i in range(len(corrections)):
        outData[bad[0][i],bad[1][i]] = corrections[i]

   # # reorient if necessary
   # if (dispAxis==0):
   #     tmpData = tmpData.T
        
    return outData

def corBadPixel(input):
    """
    Interpolate only along dispersion axis and only if pixels available within specified
    """

    data = input[0]
    badPix = input[1]
    mxRng = input[2]

    nx = data.shape[0]
    #zero pad current vector
    ytmp = np.zeros(nx+2)
    np.copyto(ytmp[1:-1],data)

    badPix += 1

    #determine if any good pixels within range
    xa = badPix
    keepxa = False
    for xa in xrange(badPix-1,badPix-mxRng-1,-1):
        if(np.isfinite(ytmp[xa])):
            keepxa = True
            break

    xb = badPix
    keepxb = False
    for xb in xrange(badPix+1,badPix+mxRng+1,1):
        if (np.isfinite(ytmp[xb])):
            keepxb = True
            break

    #interpolate between values, if useable, else put in a NaN
    if (keepxa and keepxb):
        ia = (xb-badPix)/float(xb-xa)*ytmp[xa]*keepxa
        ib = (badPix-xa)/float(xb-xa)*ytmp[xb]*keepxa
        corr = ia + ib
    else:
        corr = np.nan

    #restore pixx to input value, in case it modifies the input
    badPix -= 1
    
    return corr
    
def corBadPixelSigma(input):
    """
    Interpolate only along dispersion axis and only if pixels available within specified, progagating uncertainties.
    """
    data = input[0]
    badPix = input[1]
    mxRng = input[2]

    print(type(data))
    print(data.shape)
    nx = data.shape[0]
    #zero pad current vector
    ytmp = np.zeros(nx+2)
    np.copyto(ytmp[1:-1],data)

    badPix += 1

    #determine if any good pixels within range
    xa = badPix
    keepxa = False
    for xa in xrange(badPix-1,badPix-mxRng-1,-1):
        if(np.isfinite(ytmp[xa])):
            keepxa = True
            break

    xb = badPix
    keepxb = False
    for xb in xrange(badPix+1,badPix+mxRng+1,1):
        if (np.isfinite(ytmp[xb])):
            keepxb = True
            break

    #interpolate between values, if useable, else put in a NaN
    if (keepxa and keepxb):
        ia = (xb-pixx)/float(xb-xa)*ytmp[xa]*keepxa
        ib = (pixx-xa)/float(xb-xa)*ytmp[xb]*keepxa
        corr = np.sqrt(ia**2 + ib**2)
    else:
        corr = np.nan

    #restore pixx to input value, in case it modifies the input
    pixx -= 1
    
    return corr
