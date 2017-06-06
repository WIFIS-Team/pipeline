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
import wifisBadPixels as badPixels

os.environ['PYOPENCL_CTX'] = '1' # Used to specify which OpenCL device to target. Should be uncommented and pointed to correct device to avoid future interactive requests

#******************************************************************************
#required user input

rampFolder = '20170512072416' #must point to location of folder containing the ramp
flatFolder = '20170512073634' #must point to location flat field folder associated with observation

#(mostly) static input
distMapFile = '/home/jason/wifis/static_processed/ronchi_map_polyfit.fits' #must point to location of distortion map file
distLimitsFile = '/home/jason/wifis/static_processed/master_flat_limits.fits' #must point to the location of the flat-field associated with the Ronchi mask image
satFile = '/data/WIFIS/H2RG-G17084-ASIC-08-319/UpTheRamp/20170504201819/processed/master_detLin_satCounts.fits' #must point to location of saturation limits file, from detector linearity measurements
spatGridProps = wifisIO.readTable('/home/jason/wifis/static_processed/spatGridProps.dat')
bpmFile = 'bpm.fits'
#******************************************************************************

wifisIO.createDir('quick_reduction')

#read in data
data, inttime, hdr = wifisIO.readRampFromFolder(rampFolder)

#find if any pixels are saturated and avoid usage
satCounts = wifisIO.readImgsFromFile(satFile)[0]
satFrame = satInfo.getSatFrameCL(data, satCounts,32)

#get processed ramp
fluxImg = combData.upTheRampCL(inttime, data, satFrame, 32)[0]

#remove bad pixels
if os.path.exists(bpmFile):
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
    #read in and process flat to find limits
    flatData, inttime, hdr = wifisIO.readRampFromFolder(rampFolder)
    satFrame = satInfo.getSatFrameCL(flatData, satCounts,32)
    flat= combData.upTheRampCL(inttime, flatData, satFrame, 32)[0]
    
    if os.path.exists(bpmFile):
        flat[bpm.astype(bool)] = np.nan
        flat[flat < 0] = np.nan
        flat[satFrame < 2] = np.nan
        flatCor = np.empty(flat.shape, dtype=flat.dtype)
        flatCor[4:2044,4:2044] = badPixels.corBadPixelsAll(flat[4:2044,4:2044], mxRng=20)

    limits = slices.findLimits(flatCor, dispAxis=0, rmRef=True)
    wifisIO.writeFits(limits, 'quick_reduction/'+flatFolder+'_limits.fits')
    
distLimits = wifisIO.readImgsFromFile(distLimitsFile)[0]

#determine shift
shft = np.median(limits[1:-1, :] - distLimits[1:-1,:])

dataSlices = slices.extSlices(fluxImg, distLimits, dispAxis=0, shft=shft)

#place on uniform spatial grid
distMap = wifisIO.readImgsFromFile(distMapFile)[0]
dataGrid = createCube.distCorAll(dataSlices, distMap, spatGridProps=spatGridProps)

#create cube
dataCube = createCube.mkCube(dataGrid, ndiv=1)

#create output image
dataImg = createCube.collapseCube(dataCube)

#fill in header info
obsinfoFile = rampFolder+'/obsinfo.dat'
headers.addTelInfo(hdr, obsinfoFile)

#set the pixel scale
xScale = 0.532021532706
yScale = -0.545667026386 #valid for npix=1, i.e. 35 pixels spanning the 18 slices. This value needs to be updated in agreement with the choice of interpolation

headers.getWCSImg(dataImg, hdr, xScale, yScale)

#save image
wifisIO.writeFits(dataImg, 'quick_reduction/'+rampFolder+'_quickRedImg.fits', hdr=hdr, ask=False)

#plot the data
WCS = wcs.WCS(hdr)

fig = plt.figure()
fig.add_subplot(111, projection=WCS)
plt.imshow(dataImg, origin='lower', cmap='viridis')
plt.show()
