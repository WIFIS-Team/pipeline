import wifisIO
import matplotlib.pyplot as plt
import numpy as np
import astropy.convolution as convolution
from scipy.interpolate import spline
import wifisWaveSol
from scipy.interpolate import interp1d

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

def crossCor(x, y1, y2, regions=None, oversample=100):
    """
    Determine the velocity difference between input spectra y1 and y2 on the same wavelength grid/coordinate system x.
    Usage: 
    """

    #go through regions list and construct new arrays from subsets
    xtmp = []
    y1tmp = []
    y2tmp = []
    for reg in regions:
        whr = np.where(np.logical_and(x>reg[0],x<reg[1]))[0]
        xtmp.append(x[whr])
        y1tmp.append(y1[whr])
        y2tmp.append(y2[whr])

    xtmp = np.concatenate(xtmp)
    y1tmp = np.concatenate(y1tmp)
    y2tmp = np.concatenate(y2tmp)
    
    
    #now compute and interpolate onto constant velocity grid, and converting to line intensity (i.e. 1 - continuum)
    v = (xtmp-xtmp[int(xtmp.shape[0]/2)])/xtmp[int(xtmp.shape[0]/2)]*2.99792458e5 # in km/s
    vconst = np.linspace(v[0],v[-1],num=v.shape[0]*oversample)
    finter = interp1d(v,y1tmp, kind='linear', bounds_error=False, fill_value=0.)
    y1Const = 1. - finter(vconst)
    finter = interp1d(v, y2tmp, kind='linear', bounds_error=False, fill_value=0.)
    y2Const = 1. - finter(vconst)
    dv = vconst[1]-vconst[0]
    
    #Use numpy correlate to determine cross-correlation between input spectra
    yCC = np.correlate(y1Const, y2Const, mode="full")
    xCC = (np.arange(y1Const.shape[0]*2-1)-y1Const.shape[0]) #computes the velocity shift for each oversampled pixel in the cross-correlation output

    #now determine the RV offset, from the maximum of the cross-correlation function
    rvOut = xCC[np.nanargmax(yCC)]*dv

    return rvOut
    
