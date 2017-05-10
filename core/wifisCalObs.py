"""

Calibrate a set of (science) images

Produces:
- image cube


"""

import numpy as np
import time
import matplotlib.pyplot as plt
import wifisGetSatInfo as satInfo
import wifisNLCor as NLCor
import wifisRefCor as refCor
import os
import wifisIO 
import wifisCombineData as combData
import wifisSlices as slices
import wifisUncertainties
import wifisBadPixels as badPixels
import wifisCreateCube as createCube

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1' # Used to show compile errors for debugging, can be removed
os.environ['PYOPENCL_CTX'] = ':1' # Used to specify which OpenCL device to target. Should be uncommented and pointed to correct device to avoid future interactive requests

t0 = time.time()

#*****************************************************************************
#************************** Required input ***********************************
fileList = 'list'
nlFile = 'processed/master_detLin_NLCoeff.fits'        
satFile = 'processed/master_detLin_satCounts.fits'
bpmFile = 'processed/bad_pixel_mask.fits'
distTrimFile = ''
waveTrimFile = ''
distMapFile = ''
waveMapFile = ''
limitsFile = ''
spatGridPropsFile = ''
waveGridPropsFile = ''


#*****************************************************************************

#first check if required input exists
if not (os.path.exists(nlFile) and os.path.exists(satFile) and os.path.exists(limitsFile) and os.path.exists(ronchiMapFile) and os.path.exists(waveMapFile)):
    if not (os.path.exists(satFile)):
        print ('*** ERROR: Cannot continue, file ' + satFile + ' does not exist. Please process a detector linearity calibration sequence or provide the necessary file ***')
    if not (os.path.exists(nlFile)):
        print ('*** ERROR: Cannot continue, file ' + nlFile + ' does not exist. Please process a detector linearity calibration sequence or provide the necessary file ***')
    if not (os.path.exists(limitsFile)):
        print ('*** ERROR: Cannot continue, file ' + limitsFile + ' does not exist. Please process flat field calibration sequence or provide the necessary file ***')
    if not (os.path.exists(distMapFile)):
        print ('*** ERROR: Cannot continue, file ' + distMapFile + ' does not exist. Please process a ronchi calibration sequence or provide the necessary file ***')
    if not (os.path.exists(waveMapFile)):
        print ('*** ERROR: Cannot continue, file ' + waveMapFile + ' does not exist. Please process an arc lamp calibration sequence or provide the necessary file ***')
    
    raise SystemExit('*** Missing required calibration files, exiting ***')

#create processed directory, in case it doesn't exist
wifisIO.createDir('processed')

#open calibration files needed for future processing
distSlices = wifisIO.readImgsFromFile(distMapFile)[0]
waveSlices = wifisIO.readImgsFromFile(waveMapfile)[0]
limits = wifisIO.readImgsFromFile(limitsFile)[0]

#check if grid properties are known, if not, create them
if ((not os.path.exists(spatGridPropsFile)) or (not os.path.exists(waveGridPropsFile)) or (not os.path.exists(waveTrimFile)) or (not os.path.exists(distTrimFile))):

    #open master flat field slices to determine trim limits of distortion corrected image
    if (os.path.exists('processed/master_flat_slices.fits')):
        flatSlices, sigmaFlat = wifisIO.readImgsFromFile('processed/master_flat_slices.fits')[0]

        if (os.path.exists('processed/master_flat_slices_distCor.fits')):
            flatCor = wifisIO.readImgsFromFile('processed/master_flat_slices_distCor.fits')
        else:
            #distortion correct the calibration slices
            flatCor = createCube.distCorAll(flatSlices, distSlices)
            wifisIO.writeFits('processed/master_flat_slices_distCor.fits')
            
        #get trim limits of distortion corrected image
        trimLims = slices.getTrimLimsAll(flatCor,0.75, plot=False)

        #get correct then trimmed calibration files and compute grid properties
        if (not (os.path.exists(spatGridPropsFile)) or (not os.path.exists(distTrimFile))):
            if (os.path.exists(distTrimFile)):
                distTrim = wifisIO.readImgsFromFile(distTrimFile)[0]
            else:
                distCor = createCube.distCorAll(distSlices, distSlices)
                distTrim = slices.trimSliceAll(distCor, trimLims)
                wifisIO.writeFits('processed/distMap_distCorTrim.fits')

            if (os.path.exists(spatGridPropsFile)):
                spatGridProps = wifisIO.readTable(spatGridPropsFile)
            else:
                spatGridProps = createCube.compSpatGrid(distTrim)
                #save grid properties to files
                np.savetxt('processed/spatGridProps.dat', spatGridProps)
        else:
            spatGridProps = wifisIO.readTable(spatGridPropsFile)
            distTrim = wifisIO.readImgsFromFile(distTrimFilee)[0]
            
        if (not (os.path.exists(waveGridPropsFile)) or (not os.path.exists(waveTrimFile))):
            if (os.path.exists(waveTrimFile)):
                waveTrim = wifisIO.readImgsFromFile(waveTrimFile)[0]
            else:
                waveCor = createCube.distCorAll(waveSlices, distSlices)
                waveTrim = slices.trimSliceAll(wavecor, trimLims)
                wifisIO.writeFits('processed/waveMap_distCorTrim.fits')

            if (os.path.exists(waveGridPropsFile)):
                waveGridProps = wifisIO.readTable(waveGridPropsFile)
            else:
                waveGridProps = createCube.compWaveGrid(waveTrim)
                #save grid properties to files
                np.savetxt('processed/waveGridProps.dat', waveGridProps)
        else:
            waveGridProps = wifisIO.readTable(waveGridPropsFile)
            waveTrim = wifisIO.readImgsFromFile(waveTrimFile)
    else:
        print ('*** ERROR: Cannot continue, file master flat field slices file does not exist. Please process a flat field calibration sequence or provide the necessary file ***')
        raise SystemExit('*** Missing required calibration files, exiting ***')

else:
    spatGridProps = wifisIO.readTable(spatGridPropsFile)
    waveGridProps = wifisIO.readTable(waveGridPropsFile)
    distTrim = wifisIO.readImgsFromFile(distTrimFilee)[0]
    waveTrim = wifisIO.readImgsFromFile(waveTrimFile)

#read file list
lst= wifisIO.readAsciiList(fileList)

if lst.ndim == 0:
    lst = [lst]

for folder in lst:

    folder = folder.tostring()
    savename = 'processed/'+folder
    
    #first check if image cube already exists
    if(os.path.exists(savename+'_calObs_cube.fits')):
        cont = wifisIO.userInput('Image cube already exists for folder' + folder +', do you want to continue processing (y/n)?')

        if (cont.lower() == 'y'):
            contProc = True
        else:
            contProc = False
    else:
        contProc = True
    
    if (contProc):
        #check if processed ramp already exists
        if (os.path.exists(savename+'_calObs.fits')):
            cont = wifisIO.userInput('Processed image already exists for folder' + folder +', do you want to continue processing (y/n)?')

            if (cont.lower() == 'y'):
                contProc2 = True
            else:
                contProc2 = False
        else:
            contProc2 = True

        if (contProc2):
            #Read in data
            ta = time.time()
            data, inttime, hdr = wifisIO.readRampFromFolder(folder)
            print("time to read all files took", time.time()-ta, " seconds")
            
            nFrames = inttime.shape[0]
            nx = data.shape[1]
            ny = data.shape[0]
            
            #******************************************************************************
            #Correct data for reference pixels
            ta = time.time()
            print("Subtracting reference pixel channel bias")
            refCor.channelCL(data, 32)
            print("Subtracting reference pixel row bias")
            refCor.rowCL(data, nFrames, 4,1)
            print("time to apply reference pixel corrections ", time.time()-ta, " seconds")
            
            #******************************************************************************
            #find if any pixels are saturated to avoid use in future calculations

            satCounts = wifisIO.readImgFromFile(satFile)[0]
            satFrame = satInfo.getSatFrameCL(data, satCounts,32)
            
            #******************************************************************************
            #apply non-linearity correction
            ta = time.time()
            print("Correcting for non-linearity")
            nlCoeff = wifisIO.readImgFromFile(nlFile)[0]
            NLCor.applyNLCorCL(data, nlCoeff, 32)
            print("time to apply non-linearity corrections ", time.time()-ta, " seconds")
        
            #******************************************************************************
            #Combine data cube into single image
            fluxImg = combData.upTheRampCL(inttime, data, satFrame, 32)[0]
            data = 0

            #get uncertainties for each pixel
            sigma = wifisUncertainties.getUTR(inttime, fluxImg, satFrame)
            
            #write image to a file
            
            #add additional header information here

            #save initial ramp
            wifisIO.writeFits([fluxImg, sigma, satFrame], savename+'_calObs.fits', hdr=hdr)
 
        else:
            #read processed ramp from file
            fluxImg, sigma, satFrame = wifisIO.readImgsFromFile(savename+'_calObs.fits')[0]
            
        #Correct for dark current
        #Identify appropriate dark image for subtraction
        iTime = inttime[-1]-inttime[0]
        darkName = 'processed/master_dark_I'+str(iTime)+'.fits'
        if (os.path.exists(darkName)):
            darkImg,darkSig = wifisIO.readImgsFromFile(darkName)[0][0,1] #get the first two extensions
            fluxImg -= darkImg
            sigmaImg = np.sqrt(sigmaImg**2 + darkSig**2)

            #add additional header information here
            #modify header to include comment about dark subtraction
                
            #save dark subtracted image
            wifisIO.writeFits([fluxImg, sigma, satFrame], savename+'_calObs_darkCor.fits', hdr=hdr)

        else:
            cont = wifisIO.userInput('No corresponding master dark image could be found, do you want to proceed without dark subtraction (y/n)?')
            if (cont.lower() == 'n'):
                break
            
        #extract individual slices
        dataSlices = slices.extSlices(fluxImg, limits, dispAxis=0)
        sigmaSlices = slices.extSlices(sigmaImg, limits, dispAxis=0)
        satSlices = slices.extSlices(satFrame, limits, dispAxis=0)
            
        #flat-field correct the slices
        if not 'flatSlices' in locals():
            if (os.path.exists('processed/master_flat_slices.fits')):
                flatSlices, sigmaFlat, satFlat = wifisIO.readImgsFromFile()[0][0,1] #get the first two extensions
            else:
                cont = wifisIO.userInput('No corresponding master dark image could be found, do you want to proceed without dark subtraction (y/n)?')
            if (cont.lower() == 'n'):
                break

        if 'flatSlices' in locals():

            dataCor = slices.ffCorrectAll(dataSlices, flatSlices)
            sigmaCor = []
            for i in range(len(sigmaSlices)):
                sigmaCor.append(dataCor[i]*np.sqrt((sigmaSlices[i]/dataSlices[i])**2 + (sigmaFlat[i]/flatSlices[i])**2))

            dataSlices = dataCor
            del dataCor

            sigmaSlices = sigmaCor
            del sigmaCor
                                
        #add additional header information here
        #modify header to include comment about flat-field correction

        #save slices
        wifisIO.writeFits([dataSlices, sigmaSlices, satSlices], savename+'_calObs_slices.fits', hdr=hdr)
        
        #check if gridded slices already exists, if not, create them
        if (os.path.exists(savename+'_calObs_grid_slices.fits')):
            cont = wifisIO.userInput('Gridded image slices already exists for folder' + folder +', do you want to continue processing (y/n)?')

            if (cont.lower() == 'y'):
                contProc2 = True
            else:
                contProc2 = False
        else:
            contProc2 = True

        if (contProc2):
            #get gridded data

            dataGrid = createCube.mkWaveSpatGridAll(dataSlices,waveTrim,distTrim, waveGridProps, spatGridProps)
            
            #save gridded data image
            wifisIO.writeFits(dataGrid, savename+'_calObs_grid_slices.fits')
        else:
            dataGrid = wifisIO.readImgsFromFile(savename+'_calObs_grid_slices.fits')[0]
            
        #create data cube!

        dataCube = createCube.mkCube(dataGrid, ndiv=1)

        #save data cube
        wifisIO.writeFits(dataCube, savename+'_calObs_cube.fits')

                
