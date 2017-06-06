import wifisIO
import matplotlib.pyplot as plt
import numpy as np
import wifisSlices as slices
import wifisCreateCube as createCube
import wifisCombineData as combData
import wifisHeaders as headers
import wifisGetSatInfo as satInfo
from astropy import wcs 
import os
from astropy.modeling import models, fitting

os.environ['PYOPENCL_CTX'] = '1' # Used to specify which OpenCL device to target. Should be uncommented and pointed to correct device to avoid future interactive requests

#******************************************************************************
#required user input

rampFolder = '20170510232750' #must point to location of folder containing the ramp
flatFolder = '20170510233851' #must point to location flat field folder associated with observation

#(mostly) static input
distMapFile = '/home/jason/wifis/static_processed/ronchi_map_polyfit.fits' #must point to location of distortion map file
distLimitsFile = '/home/jason/wifis/static_processed/master_flat_limits.fits' #must point to the location of the flat-field associated with the Ronchi mask image
satFile = '/data/WIFIS/H2RG-G17084-ASIC-08-319/UpTheRamp/20170504201819/processed/master_detLin_satCounts.fits' #must point to location of saturation limits file, from detector linearity measurements
spatGridProps = wifisIO.readTable('/home/jason/wifis/static_processed/spatGridProps.dat')
bpmFile = 'bpm.fits'
#******************************************************************************

wifisIO.createDir('quick_reduction')

#read in data
print('Processing ramp')
data, inttime, hdr = wifisIO.readRampFromFolder(rampFolder)

#find if any pixels are saturated and avoid usage
satCounts = wifisIO.readImgsFromFile(satFile)[0]
satFrame = satInfo.getSatFrameCL(data, satCounts,32)

#get processed ramp
fluxImg = combData.upTheRampCL(inttime, data, satFrame, 32)[0]

#remove bad pixels
if os.path.exists(bpmFile):
    print('Removing bad pixels')
    bpm = wifisIO.readImgsFromFile(bpmFile)[0]
    fluxImg[bpm.astype(bool)] = np.nan
    fluxImg[fluxImg < 0] = np.nan
    fluxImg[satFrame < 2] = np.nan
    
fluxImg = fluxImg[4:2044, 4:2044]

#extract slices
#first check if limits already exists
if os.path.exists('quick_reduction/'+flatFolder+'_limits.fits'):
    limitsFile = 'quick_reduction/'+flatFolder+'_limits.fits'
    limits = wifisIO.readImgsFromFile(limitsFile)[0]
else:
    print('Processing flat')
    #read in and process flat to find limits
    flatData, inttime, hdr = wifisIO.readRampFromFolder(rampFolder)
    satFrame = satInfo.getSatFrameCL(flatData, satCounts,32)
    flat= combData.upTheRampCL(inttime, flatData, satFrame, 32)[0]
    
    if os.path.exists(bpmFile):
        print('removing bad pixels')
        flat[bpm.astype(bool)] = np.nan
        flat[satFrame < 2] = np.nan
        #flatCor = np.empty(flat.shape, dtype=flat.dtype)
        #flatCor[4:2044,4:2044] = badPixels.corBadPixelsAll(flat[4:2044,4:2044], mxRng=20)
        
        # else:
        #     flatCor = flat[4:2044, 4:2044]

    print('Getting slice limits')
    limits = slices.findLimits(flat, dispAxis=0, rmRef=True)
    wifisIO.writeFits(limits, 'quick_reduction/'+flatFolder+'_limits.fits')
    
distLimits = wifisIO.readImgsFromFile(distLimitsFile)[0]

#determine shift
shft = np.median(limits[1:-1, :] - distLimits[1:-1,:])

print('Extracting slices')
dataSlices = slices.extSlices(fluxImg, distLimits, dispAxis=0, shft=shft)

#place on uniform spatial grid
print('Distortion correcting')
distMap = wifisIO.readImgsFromFile(distMapFile)[0]
dataGrid = createCube.distCorAll(dataSlices, distMap, spatGridProps=spatGridProps)

#create cube
print('Creating image')
dataCube = createCube.mkCube(dataGrid, ndiv=1)

#create output image
dataImg = createCube.collapseCube(dataCube)

#set the pixel scale
xScale = 0.532021532706
yScale = -0.545667026386 #valid for npix=1, i.e. 35 pixels spanning the 18 slices. This value needs to be updated in agreement with the choice of interpolation

print('Computing FWHM')
#fit 2D Gaussian to image to determine FWHM of star's image
y,x = np.mgrid[:dataImg.shape[0],:dataImg.shape[1]]
cent = np.unravel_index(np.nanargmax(dataImg),dataImg.shape)
gInit = models.Gaussian2D(np.nanmax(dataImg), cent[1],cent[0])
fitG = fitting.LevMarLSQFitter()
gFit = fitG(gInit, x,y,dataImg)

#y,x = np.mgrid[0:dataImg.shape[0]:0.1,0:dataImg.shape[1]:0.1]
#gMod = gFit(x,y)
#
#r = np.sqrt((x-cent[1])**2+(y-cent[0])**2)
#gMod /= np.max(gMod)
#whr = np.where(gMod < 0.5)
#fwhm2 = np.min(r[whr[0],whr[1]])

#get average FWHM
sigPix = (gFit.x_stddev+gFit.y_stddev)/2.
fwhmPix = 2.*np.sqrt(2.* np.log(2))*sigPix

sigDeg = (np.abs(gFit.x_stddev*xScale)+np.abs(gFit.y_stddev*yScale))/2.
fwhmDeg = 2.*np.sqrt(2.* np.log(2))*sigDeg

#fill in header info
obsinfoFile = rampFolder+'/obsinfo.dat'
headers.addTelInfo(hdr, obsinfoFile)

headers.getWCSImg(dataImg, hdr, xScale, yScale)

#save image
wifisIO.writeFits(dataImg, 'quick_reduction/'+rampFolder+'_quickRedImg.fits', hdr=hdr, ask=False)

#plot the data
WCS = wcs.WCS(hdr)

print('Plotting data')
fig = plt.figure()
fig.add_subplot(111, projection=WCS)
plt.imshow(dataImg, origin='lower', cmap='jet')
r = np.arange(360)*np.pi/180.
x = fwhmPix*np.cos(r) + gFit.x_mean
y = fwhmPix*np.sin(r) + gFit.y_mean
plt.plot(x,y, 'r--')
plt.title('Average FWHM of object is '+str(fwhmDeg)+' arcsec')
plt.show()
