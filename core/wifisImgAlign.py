"""

Tools used to align images

"""

import numpy as np
from scipy import interpolate
from scipy.ndimage.interpolation import shift
from astropy.modeling import models,fitting
import time
import numpy as n
import scipy
from scipy import signal
import astropy.convolution as convolution

def findPixelShift(img1, img2):
    """
    """

    dim1 = img1.shape
    dim2 = img2.shape

    n1 = np.max([dim1[0],dim2[0]])
    n2 = np.max([dim1[1],dim2[1]])

    #create new padded images and copy old images into new images
    inp1 = np.zeros((n1,n2))
    inp2 = np.zeros((n1,n2))

    inp1[:dim1[0],:dim1[1]] = img1
    inp2[:dim2[0],:dim2[1]] = img2

    #now get FFT of images and accompanying translation function
    fft1 = np.fft.fft2(inp1)
    fft2 = np.fft.fft2(inp2)

    den = np.abs(fft1)*np.abs(fft2)
    num = fft1*fft2.conjugate()
    r = num/den
    m = np.fft.ifft2(r)

    #pixel with maximum signal corresponds to translation
    t0, t1 = np.unravel_index(np.nanargmax(m), m.shape)
    if t0 > m.shape[0] // 2.:
        t0 -= m.shape[0]
    if t1 > m.shape[1] // 2.:
        t1 -= m.shape[1]

    return t0, t1

def findSubPixelShift(img1, img2, oversample=100.):
    """
    """

    dim1 = img1.shape
    dim2 = img2.shape

    n1 = np.max([dim1[0],dim2[0]])
    n2 = np.max([dim1[1],dim2[1]])

    #create new padded images and copy old images into new images
    inp1 = np.zeros((n1,n2))
    inp2 = np.zeros((n1,n2))

    inp1[:dim1[0],:dim1[1]] = img1
    inp2[:dim2[0],:dim2[1]] = img2

    #now get FFT of images and accompanying translation function
    fft1 = np.fft.fft2(inp1)
    fft2 = np.fft.fft2(inp2)

    den = np.abs(fft1)*np.abs(fft2)
    num = fft1*fft2.conjugate()
    r = num/den
    m = np.abs(np.fft.ifft2(r))

    #now interpolate onto higher resolution grid and rearrange so that  centre corresponds to 0 frequency
    m2 = np.zeros((11,11))
    m2[0:5,0:5]= m[m.shape[1]-5:,m.shape[0]-5:]
    m2[0:5,5:] = m[m.shape[0]-5:,:6]
    m2[5:,5:] = m[:6,:6]
    m2[5:,0:5] = m[:6,m.shape[0]-5:]

    x=np.arange(m2.shape[0])
    y=np.arange(m2.shape[1])
    interpSpline = interpolate.RectBivariateSpline(y,x,m2,kx=2,ky=2)
    x = np.linspace(0,m2.shape[0]-1,m2.shape[0]*oversample)
    y = np.linspace(0,m2.shape[1]-1,m2.shape[1]*oversample)
    m2I = interpSpline(y,x, m2)

    y,x=np.mgrid[:m2I.shape[0], :m2I.shape[1]]

    #use a simple Gaussian function to find centre
    gInit = models.Gaussian2D(x_stddev=1,y_stddev=1, x_mean=m2I.shape[1]/2.,y_mean=m2I.shape[0]/2.)
    gFit = fitting.LevMarLSQFitter()
    g = gFit(gInit,x,y,m2I)
    t1 =(g.x_mean-m2I.shape[0]/2.)/oversample
    t0 = (g.y_mean-m2I.shape[1]/2.)/oversample
    
    return t0,t1

def shiftImg(img, shiftVals):
    """
    Returns the image corrected by given offset
    """
    #shift interpolated image
    imS = shift(img,[shiftVals[0],shiftVals[1]])

    out = imS
        
    return out
    
