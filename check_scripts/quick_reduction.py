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

os.environ['PYOPENCL_CTX'] = '1' # Used to specify which OpenCL device to target. Should be uncommented and pointed to correct device to avoid future interactive requests

#******************************************************************************
#required user input
rampFolder = '20170511005013' #must point to location of folder containing the ramp
distMapFile = '/home/jason/wifis/static_processed/ronchi_map_polyfit.fits' #must point to location of distortion map file
distLimitsFile = '/home/jason/wifis/static_processed/master_flat_limits.fits' #must point to the location of the flat-field associated with the Ronchi mask image
limitsFile = 'processed/20170510233851_flat_limits.fits' #must point to location of limits file, corresponding to this observation
satFile = '/data/WIFIS/H2RG-G17084-ASIC-08-319/UpTheRamp/20170504201819/processed/master_detLin_satCounts.fits' #must point to location of saturation limits file, from detector linearity measurements
spatGridProps = wifisIO.readTable('/home/jason/wifis/static_processed/spatGridProps.dat')
#******************************************************************************

#read in data
data, inttime, hdr = wifisIO.readRampFromFolder(rampFolder)

#find if any pixels are saturated and avoid usage
satCounts = wifisIO.readImgsFromFile(satFile)[0]
satFrame = satInfo.getSatFrameCL(data, satCounts,32)

#get processed ramp
fluxImg = combData.upTheRampCL(inttime, data, satFrame, 32)[0]

#extract slices
limits = wifisIO.readImgsFromFile(limitsFile)[0]
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
wifisIO.writeFits(dataImg, rampFolder+'_quickRedImg.fits', hdr=hdr, ask=False)

#plot the data
WCS = wcs.WCS(hdr)

fig = plt.figure()
fig.add_subplot(111, projection=WCS)
plt.imshow(dataImg, origin='lower', cmap='viridis')
plt.show()
