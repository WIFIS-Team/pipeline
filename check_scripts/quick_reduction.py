import matplotlib
#*******************************************************************
matplotlib.use('gtkagg') #default is agg, but using gtkagg can speed up window creation on some systems
#*******************************************************************

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
import glob
import warnings
from wifisIO import sorted_nicely

os.environ['PYOPENCL_CTX'] = '1' # Used to specify which OpenCL device to target. Should be uncommented and pointed to correct device to avoid future interactive requests

#******************************************************************************
#required user input
rootFolder = '/data/WIFIS/H2RG-G17084-ASIC-08-319/'
pipelineFolder = '/data/pipeline/'
noProc = False

#change here
rampFolderFile = 'obsQuick.inp' # ascii file containing the name of the folder containing the target ramp
flatFolderFile = 'flatQuick.inp' # ascii containing the name of the flat ramp folder associated with the target ramp

#optional
skyFolderFile = 'skyQuick.inp' # ascii containing the name of the sky ramp folder associated with the target ramp

if os.path.exists(rampFolderFile):
    rampFolder = wifisIO.readAsciiList(rampFolderFile).tostring()  
else:
    raise SystemExit('*** '+ rampFolderFile + ' does not exist ***')

if os.path.exists(flatFolderFile):
    flatFolder = wifisIO.readAsciiList(flatFolderFile).tostring()
else:
    raise SystemExit('*** ' + flatFolderFile + ' does not exist ***')

if os.path.exists(skyFolderFile):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        skyFolder = wifisIO.readAsciiList(skyFolderFile).tostring()

    if skyFolder == '':
        skyFolder = None
else:
    skyFolder = None

#(mostly) static input

distMapFile = pipelineFolder + 'external_data/distMap.fits' #must point to location of distortion map file
distLimitsFile = pipelineFolder + 'external_data/distMap_limits.fits' #must point to the location of the flat-field associated with the Ronchi mask image
satFile = pipelineFolder + 'external_data/master_detLin_satCounts.fits' #must point to location of saturation limits file, from detector linearity measurements
spatGridProps = wifisIO.readTable(pipelineFolder+'external_data/distMap_spatGridProps.dat')
bpmFile = pipelineFolder+'external_data/bpm.fits'

#set the pixel scale
xScale = 0.532021532706
yScale = -0.545667026386 #valid for npix=1, i.e. 35 pixels spanning the 18 slices. This value needs to be updated in agreement with the choice of interpolation
#******************************************************************************

satCounts = wifisIO.readImgsFromFile(satFile)[0]

wifisIO.createDir('quick_reduction')

#read in data
print('Processing object ramp')

#check file type
#CDS
if os.path.exists(rootFolder+'/CDSReference/'+rampFolder):
    folderType = '/CDSReference/'
    fluxImg, hdr = wifisIO.readImgsFromFile(rootFolder + folderType + rampFolder+'/Result/CDSResult.fits')
    UTR = False
#Fowler
elif os.path.exists(rootFolder+'/FSRamp/'+rampFolder):
    folderType = '/FSRamp/'
    UTR =  False
    if os.path.exists(rootFolder + folderType + rampFolder+'/Result/CDSResult.fits'):
        fluxImg, hdr = wifisIO.readImgsFromFile(rootFolder + folderType + rampFolder+'/Result/CDSResult.fits')
    else:
        data, inttime, hdr = wifisIO.readRampFromFolder(rootFolder + folderType + rampFolder)

        #find if any pixels are saturated and avoid usage
        satFrame = satInfo.getSatFrameCL(data, satCounts,32)

        #get processed ramp
        fluxImg = combData.FowlerSamplingCL(inttime, data, satFrame, 32)[0]  
elif os.path.exists(rootFolder + '/UpTheRamp/'+rampFolder):
    UTR = True
    folderType = '/UpTheRamp/'

    if noProc:
        lst = glob.glob(rootFolder+folderType+rampFolder + '/H2*fits')
        lst = sorted_nicely(lst)

        img1,hdr1 = wifisIO.readImgsFromFile(lst[-1])
        img0,hdr0 = wifisIO.readImgsFromFile(lst[0])
        t0 = hdr0['INTTIME']
        t1 = hdr1['INTTIME']
        
        fluxImg = ((img1-img0)/(t1-t0)).astype('float32')
        hdr = hdr1
   
    else:
        data, inttime, hdr = wifisIO.readRampFromFolder(rootFolder + folderType + rampFolder)

        #find if any pixels are saturated and avoid usage
        satFrame = satInfo.getSatFrameCL(data, satCounts,32)

        #get processed ramp
        fluxImg = combData.upTheRampCL(inttime, data, satFrame, 32)[0]
else:
    raise SystemExit('*** Ramp folder ' + rampFolder + ' does not exist ***')


obsinfoFile = rootFolder + folderType + rampFolder+'/obsinfo.dat'

#now process sky, if it exists

#read in data
if (skyFolder is not None):
    print('Processing and subtracting sky ramp')

    #check file type
    #CDS
    if os.path.exists(rootFolder+'/CDSReference/'+skyFolder):
        folderType = '/CDSReference/'
        skyImg, hdr = wifisIO.readImgsFromFile(rootFolder + folderType + skyFolder+'/Result/CDSResult.fits')
        UTR = False
    #Fowler
    elif os.path.exists(rootFolder+'/FSRamp/'+skyFolder):
        folderType = '/FSRamp/'
        UTR =  False
        if os.path.exists(rootFolder + folderType + skyFolder+'/Result/CDSResult.fits'):
            skyImg, hdr = wifisIO.readImgsFromFile(rootFolder + folderType + skyFolder+'/Result/CDSResult.fits')
        else:
            data, inttime, hdr = wifisIO.readRampFromFolder(rootFolder + folderType + skyFolder)

            #find if any pixels are saturated and avoid usage
            satFrame = satInfo.getSatFrameCL(data, satCounts,32)

            #get processed ramp
            skyImg = combData.FowlerSamplingCL(inttime, data, satFrame, 32)[0]  
    elif os.path.exists(rootFolder + '/UpTheRamp/'+skyFolder):
        UTR = True
        folderType = '/UpTheRamp/'

        if noProc:
            lst = glob.glob(rootFolder+folderType+skyFolder + '/H2*fits')
            lst = sorted_nicely(lst)
            
            img1,hdr1 = wifisIO.readImgsFromFile(lst[-1])
            img0,hdr0 = wifisIO.readImgsFromFile(lst[0])
            t0 = hdr0['INTTIME']
            t1 = hdr1['INTTIME']
        
            skyImg = ((img1-img0)/(t1-t0)).astype('float32')
        else:
            data, inttime, hdrSky = wifisIO.readRampFromFolder(rootFolder + folderType + skyFolder)
            
            #find if any pixels are saturated and avoid usage
            satFrame = satInfo.getSatFrameCL(data, satCounts,32)
        
            #get processed ramp
            skyImg = combData.upTheRampCL(inttime, data, satFrame, 32)[0]

    else:
        raise SystemExit('*** sky Ramp folder ' + skyFolder + ' does not exist ***')

    #subtract
    fluxImg -= skyImg

#remove bad pixels
if os.path.exists(bpmFile):
    print('Removing bad pixels')
    bpm = wifisIO.readImgsFromFile(bpmFile)[0]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        fluxImg[bpm.astype(bool)] = np.nan
        if UTR and not noProc:
            fluxImg[fluxImg < 0] = np.nan
            fluxImg[satFrame < 2] = np.nan
            
fluxImg = fluxImg[4:-4, 4:-4]

#first check if limits already exists
if os.path.exists('quick_reduction/'+flatFolder+'_limits.fits'):
    limitsFile = 'quick_reduction/'+flatFolder+'_limits.fits'
    limits = wifisIO.readImgsFromFile(limitsFile)[0]
else:
    print('Processing flat')

    #check file type
    #CDS
    if os.path.exists(rootFolder+'/CDSReference/'+flatFolder):
        folderType = '/CDSReference/'
        flat = wifisIO.readImgsFromFile(rootFolder + folderType + flatFolder+'/Result/CDSResult.fits')[0]
        UTR = False
    
    #Fowler
    elif os.path.exists(rootFolder+'/FSRamp/'+ flatFolder):
        folderType = '/FSRamp/'
        UTR =  False
        if os.path.exists(rootFolder + folderType + flatFolder+'/Result/CDSResult.fits'):
            flat, hdr = wifisIO.readImgsFromFile(rootFolder + folderType + rampFolder+'/Result/CDSResult.fits')
        else:
            data, inttime, hdr = wifisIO.readRampFromFolder(rootFolder + folderType + flatFolder)

            #find if any pixels are saturated and avoid usage
            satFrame = satInfo.getSatFrameCL(data, satCounts,32)

            #get processed ramp
            flat = combData.FowlerSamplingCL(inttime, data, satFrame, 32)[0]  

    elif os.path.exists(rootFolder + '/UpTheRamp/'+flatFolder):
        folderType = '/UpTheRamp/'
        UTR = True

        if noProc:
            lst = glob.glob(rootFolder+folderType+flatFolder + '/H2*fits')
            lst = sorted_nicely(lst)
            
            img1,hdr1 = wifisIO.readImgsFromFile(lst[-1])
            img0,hdr0 = wifisIO.readImgsFromFile(lst[0])
            t0 = hdr0['INTTIME']
            t1 = hdr1['INTTIME']
        
            flat = ((img1-img0)/(t1-t0)).astype('float32')
        else:
            data, inttime, hdrFlat = wifisIO.readRampFromFolder(rootFolder + folderType + flatFolder)
            
            #find if any pixels are saturated and avoid usage
            satFrame = satInfo.getSatFrameCL(data, satCounts,32)
            
            #get processed ramp
            flat = combData.upTheRampCL(inttime, data, satFrame, 32)[0]

    else:
        raise SystemExit('*** flat folder ' + flatFolder + ' does not exist ***')
    
    
    if os.path.exists(bpmFile):
        print('removing bad pixels')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            flat[bpm.astype(bool)] = np.nan
            if UTR and not noProc:
                flat[satFrame < 2] = np.nan
                flat[flat<0] = np.nan

        #flatCor = np.empty(flat.shape, dtype=flat.dtype)
        #flatCor[4:2044,4:2044] = badPixels.corBadPixelsAll(flat[4:2044,4:2044], mxRng=20)
        
        # else:
        #     flatCor = flat[4:2044, 4:2044]

    print('Getting slice limits')
    limits = slices.findLimits(flat, dispAxis=0, rmRef=True)
    wifisIO.writeFits(limits, 'quick_reduction/'+flatFolder+'_limits.fits')

#get ronchi slice limits
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

print('Computing FWHM')
#fit 2D Gaussian to image to determine FWHM of star's image
y,x = np.mgrid[:dataImg.shape[0],:dataImg.shape[1]]
cent = np.unravel_index(np.nanargmax(dataImg),dataImg.shape)
gInit = models.Gaussian2D(np.nanmax(dataImg), cent[1],cent[0])
fitG = fitting.LevMarLSQFitter()
gFit = fitG(gInit, x,y,dataImg)

#fill in header info
headers.addTelInfo(hdr, obsinfoFile)

#save distortion corrected slices
wifisIO.writeFits(dataGrid, 'quick_reduction/'+rampFolder+'_quickRed_slices_grid.fits', hdr=hdr, ask=False)

headers.getWCSImg(dataImg, hdr, xScale, yScale)

#save image
wifisIO.writeFits(dataImg, 'quick_reduction/'+rampFolder+'_quickRedImg.fits', hdr=hdr, ask=False)

#plot the data
WCS = wcs.WCS(hdr)

print('Plotting data')
fig = plt.figure()
ax = fig.add_subplot(111, projection=WCS)
plt.imshow(dataImg, origin='lower', cmap='jet')
r = np.arange(360)*np.pi/180.
fwhmX = np.abs(2.3548*gFit.x_stddev*xScale)
fwhmY = np.abs(2.3548*gFit.y_stddev*yScale)
x = fwhmX*np.cos(r) + gFit.x_mean
y = fwhmY*np.sin(r) + gFit.y_mean
plt.plot(x,y, 'r--')
ax.set_ylim([0, dataImg.shape[0]-1])
ax.set_xlim([0, dataImg.shape[1]-1])

plt.title(hdr['Object'] + ': FWHM of object is: '+'{:4.2f}'.format(fwhmX)+' in x and ' + '{:4.2f}'.format(fwhmY)+' in y, in arcsec')
plt.savefig('quick_reduction/'+rampFolder+'_quickRedImg.png', dpi=300)
plt.show()
plt.close('all')
