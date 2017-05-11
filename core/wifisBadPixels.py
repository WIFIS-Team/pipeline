"""


"""

import numpy as np
import multiprocessing as mp
import wifisIO
import matplotlib.pyplot as plt


def corBadPixelsAll(data, BPM, dispAxis=0,mxRng=2,MP=True, ncpus=None, sigma=False):
    """
    """

    #run through list of all bad pixels and correct them through linear interpolation along dispersion axis

    #create copies of input data array
    tmpData = np.empty(data.shape, dtype=data.dtype)
    np.copyto(tmpData, data)
    tmpBPM = np.zeros(BPM.shape, dtype='int')
    np.copyto(tmpBPM, BPM.astype('int'))
    
    if (dispAxis==0):
        tmpBPM = tmpBPM.T
        tmpData = tmpData.T
           
    #create list of bad pixels
    bad = np.where(tmpBPM != 1)

    if (MP == False):
        corrections=[]

        #run separate algorithm on uncertainty data
        if (sigma):
            for i in range(bad[0].shape[0]):
                corrections.append(corBadPixelSigma([tmpData, tmpBPM, bad[0][i],bad[1][i], mxRng]))
        else:
            for i in range(bad[0].shape[0]):
                corrections.append(corBadPixel([tmpData, tmpBPM, bad[0][i],bad[1][i], mxRng]))
    else:
        #setup input list to feed MP routine
        lst = []
        
        for i in range(bad[0].shape[0]):
            lst.append([tmpData, tmpBPM, bad[0][i],bad[1][i], mxRng])

        #run MP routine
        if (ncpus == None):
            ncpus =mp.cpu_count()
        pool = mp.Pool(ncpus)
        if (sigma):
            corrections = pool.map(corBadPixelSigma, lst)
        else:
            corrections = pool.map(corBadPixel, lst)
        pool.close()

    #apply corrections to input array in place
    for i in range(len(corrections)):
        tmpData[bad[0][i],bad[1][i]] = corrections[i]

    # reorient if necessary
    if (dispAxis==0):
        tmpData = tmpData.T
        
    return tmpData

def corBadPixel(input):
    """
    Interpolate only along dispersion axis and only if pixels available within specified
    """

    data = input[0]
    BPM = input[1]
    pixy = input[2]
    pixx = input[3]
    mxRng = input[4]

    nx = BPM.shape[1]
    ny = BPM.shape[0]
    
    #zero pad current row
    ytmp = np.zeros(nx+2)
    np.copyto(ytmp[1:-1],data[pixy,:])
    bpmTmp = np.zeros((nx+2), dtype='int')
    np.copyto(bpmTmp[1:-1],BPM[pixy,:])

    pixx += 1

    #determine if any good pixels within range
    xa = pixx
    keepxa = False
    for xa in xrange(pixx-1,pixx-mxRng-1,-1):
        if(bpmTmp[xa] == 0):
            keepxa = True
            break

    xb = pixx
    keepxb = False
    for xb in xrange(pixx+1,pixx+mxRng+1,1):
        if (bpmTmp[xb] == 0):
            keepxb = True
            break

    #interpolate between values, if useable, else put in a NaN
    if (keepxa and keepxb):
        ia = (xb-pixx)/float(xb-xa)*ytmp[xa]*keepxa
        ib = (pixx-xa)/float(xb-xa)*ytmp[xb]*keepxa
        corr = ia + ib
    else:
        corr = np.nan

    #restore pixx to input value, in case it modifies the input
    pixx -= 1
    
    return corr
    
def corBadPixelSigma(input):
    """
    Interpolate only along dispersion axis and only if pixels available within specified, progagating uncertainties.
    """

    data = input[0]
    BPM = input[1]
    pixy = input[2]
    pixx = input[3]
    mxRng = input[4]

    nx = BPM.shape[1]
    ny = BPM.shape[0]
    
    #zero pad current row
    ytmp = np.zeros(nx+2)
    np.copyto(ytmp[1:-1],data[pixy,:])
    bpmTmp = np.zeros((nx+2), dtype='int')
    np.copyto(bpmTmp[1:-1],BPM[pixy,:])

    pixx += 1

    #determine if any good pixels within range
    xa = pixx
    keepxa = False
    for xa in xrange(pixx-1,pixx-mxRng-1,-1):
        if(bpmTmp[xa] == 0):
            keepxa = True
            break

    xb = pixx
    keepxb = False
    for xb in xrange(pixx+1,pixx+mxRng+1,1):
        if (bpmTmp[xb] == 0):
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
