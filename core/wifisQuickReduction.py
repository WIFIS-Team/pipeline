import matplotlib
#*******************************************************************
matplotlib.use('gtkagg') #default is tkagg, but using gtkagg can speed up window creation on some systems
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
import wifisWaveSol as waveSol
import wifisProcessRamp as processRamp
from matplotlib.backends.backend_pdf import PdfPages
import wifisSpatialCor as spatialCor

#******************************************************************************

def initPaths(hband=False):
    """
    """

    #create global variables
    global rootFolder
    global pipelineFolder
    global distMapFile
    global distMapLimitsFile
    global satFile
    global spatGridPropsFile
    global bpmFile
    global xScale
    global yScale
    global templateFile
    global prevResultsFile
    global atlasFile
    global nlFile

    #set paths here

    rootFolder = '/data/WIFIS/H2RG-G17084-ASIC-08-319/'

    pipelineFolder = '/data/pipeline/'

    if hband:
        #needs updating
        templateFile = '/data/pipeline/external_data/hband_template.fits'
        prevResultsFile = '/data/pipeline/external_data/hband_template.pkl'
        distMapFile = pipelineFolder + '/external_data/distMap.fits'
        spatGridPropsFile = pipelineFolder + '/external_data/distMap_spatGridProps.dat'
        distMapLimitsFile = pipelineFolder+'/external_data/distMap_limits.fits'

    else:
        #TB band
        templateFile = pipelineFolder+'/external_data/waveTemplate.fits'
        prevResultsFile = pipelineFolder+'/external_data/waveTemplateFittingResults.pkl'
        distMapFile = pipelineFolder + '/external_data/distMap.fits'
        spatGridPropsFile = pipelineFolder + '/external_data/distMap_spatGridProps.dat'
        distMapLimitsFile = pipelineFolder+'/external_data/distMap_limits.fits'

        distMapFile = '/home/jason/wifis/data/ronchi_map_july/tb/processed/20170707175840_ronchi_distMap.fits'
        spatGridPropsFile ='/home/jason/wifis/data/ronchi_map_july/tb/processed/20170707175840_ronchi_spatGridProps.dat'
        distMapLimitsFile ='/home/jason/wifis/data/ronchi_map_july/tb/processed/20170707180443_flat_limits.fits'
        
        #should be (mostly) static
        atlasFile = pipelineFolder + '/external_data/best_lines2.dat'
        satFile = pipelineFolder + '/external_data/master_detLin_satCounts.fits'
        bpmFile = pipelineFolder+'external_data/bpm.fits'
        nlFile = pipelineFolder + 'external_data/master_detLin_NLCoeff.fits' 

        #set the pixel scale
        xScale = 0.532021532706
        yScale = -0.545667026386 #valid for npix=1, i.e. 35 pixels spanning the 18 slices. This value needs to be updated in agreement with the choice of interpolation

    return
#*******************************************************************************


def procScienceData(rampFolder='', flatFolder='', noProc=False, skyFolder=None, pixRange=None):
    """
    """
    
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
            data = 0
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
            data = 0
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
                data = 0
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
                data = 0
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
                flat, hdr = wifisIO.readImgsFromFile(rootFolder + folderType + flatFolder+'/Result/CDSResult.fits')
            else:
                data, inttime, hdr = wifisIO.readRampFromFolder(rootFolder + folderType + flatFolder)

                #find if any pixels are saturated and avoid usage
                satFrame = satInfo.getSatFrameCL(data, satCounts,32)
                
                #get processed ramp
                flat = combData.FowlerSamplingCL(inttime, data, satFrame, 32)[0]  
                data = 0
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
                data = 0 
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


        print('Getting slice limits')
        limits = slices.findLimits(flat, dispAxis=0, rmRef=True)
        wifisIO.writeFits(limits, 'quick_reduction/'+flatFolder+'_limits.fits', ask=False)

    #get ronchi slice limits
    distLimits = wifisIO.readImgsFromFile(distMapLimitsFile)[0]

    #determine shift
    shft = np.median(limits[1:-1, :] - distLimits[1:-1,:])

    print('Extracting slices')
    dataSlices = slices.extSlices(fluxImg, distLimits, dispAxis=0, shft=shft)

    #place on uniform spatial grid
    print('Distortion correcting')
    distMap = wifisIO.readImgsFromFile(distMapFile)[0]
    spatGridProps = wifisIO.readTable(spatGridPropsFile)
    dataGrid = createCube.distCorAll(dataSlices, distMap, spatGridProps=spatGridProps)

    #create cube
    print('Creating image')
    dataCube = createCube.mkCube(dataGrid, ndiv=1)

    #create output image
    if pixRange is not None:
        dataImg = np.nansum(dataCube[:,:,pixRange[0]:pixRange[1]],axis=2)
    else:
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
    ax.set_ylim([-0.5, dataImg.shape[0]-0.5])
    ax.set_xlim([-0.5, dataImg.shape[1]-0.5])

    rotAng = hdr['CRPA']
    rotMat = np.asarray([[np.cos(rotAng*np.pi/180.),np.sin(rotAng*np.pi/180.)],[-np.sin(rotAng*np.pi/180.),np.cos(rotAng*np.pi/180.)]])

    raAx = np.dot([5,0], rotMat)
    decAx = np.dot([0,-5], rotMat)

    cent = np.asarray([10,25])
    ax.arrow(cent[0],cent[1], raAx[0],raAx[1], head_width=1, head_length=1, fc='w', ec='w')
    ax.arrow(cent[0],cent[1], decAx[0],decAx[1], head_width=1, head_length=1, fc='w', ec='w')

    ax.text((cent+decAx)[0]+1, (cent+decAx)[1]+1,"N",ha="left", va="top", rotation=rotAng, color='w')
    ax.text((cent+raAx)[0]+1, (cent+raAx)[1]+1,"E",ha="left", va="bottom", rotation=rotAng, color='w')

    plt.title(hdr['Object'] + ': FWHM of object is: '+'{:4.2f}'.format(fwhmX)+' in x and ' + '{:4.2f}'.format(fwhmY)+' in y, in arcsec')
    plt.savefig('quick_reduction/'+rampFolder+'_quickRedImg.png', dpi=300)
    plt.show()
    plt.close('all')
    return

def procArcData(waveFolder, flatFolder, hband=False, colorbarLims = None):
    """
    """

    sigmaClipRounds=1 #number of iterations when sigma-clipping of dispersion solution
    sigmaClip = 3 #sigma-clip cutoff when sigma-clipping dispersion solution
    sigmaLimit= 3 #relative noise limit (x * noise level) for which to reject lines
    mxOrder = 3

    
    wifisIO.createDir('quick_reduction')

    #read in previous results and template
    template = wifisIO.readImgsFromFile(templateFile)[0]
    prevResults = wifisIO.readPickle(prevResultsFile)
    prevSol = prevResults[5]
    distMap = wifisIO.readImgsFromFile(distMapFile)[0]
    spatGridProps = wifisIO.readTable(spatGridPropsFile)
    
    satCounts = wifisIO.readImgsFromFile(satFile)[0]

    if (os.path.exists('quick_reduction/'+waveFolder+'_wave_fwhm_map.png') and os.path.exists('quick_reduction/'+waveFolder+'_wave_fwhm_map.fits') and os.path.exists('quick_reduction/'+waveFolder+'_wave_wavelength_map.fits')):
        print('*** ' + waveFolder + ' arc/wave data already processed, skipping ***')
    else:
        print('Processing arc file '+ waveFolder)
        #check the type of raw data, only assumes CDS or up-the-ramp
        if (os.path.exists(rootFolder + '/CDSReference/'+waveFolder+'/Result/CDSResult.fits')):
            #CDS image
            wave = wifisIO.readImgsFromFile(rootFolder + '/CDSReference/'+waveFolder+'/Result/CDSResult.fits')[0]
            wave = wave[4:-4, 4:-4] #trim off reference pixels
        elif os.path.exists(rootFolder+'/FSRamp/'+waveFolder):
            folderType = '/FSRamp/'
            UTR =  False
            if os.path.exists(rootFolder + folderType + waveFolder+'/Result/CDSResult.fits'):
                wave, hdr = wifisIO.readImgsFromFile(rootFolder + folderType + waveFolder+'/Result/CDSResult.fits')
            else:
                data, inttime, hdr = wifisIO.readRampFromFolder(rootFolder + folderType + waveFolder)

                #find if any pixels are saturated and avoid usage
                satFrame = satInfo.getSatFrameCL(data, satCounts,32)

                #get processed ramp
                wave = combData.FowlerSamplingCL(inttime, data, satFrame, 32)[0]
                data = 0
        elif os.path.exists(rootFolder+'/UpTheRamp/'+waveFolder):
            #assume up-the-ramp
            data, inttime, hdr = wifisIO.readRampFromFolder(rootFolder + '/UpTheRamp/'+waveFolder)
            satFrame = satInfo.getSatFrameCL(data, satCounts,32)
            
            #get processed ramp
            wave = combData.upTheRampCL(inttime, data, satFrame, 32)[0]
            wave = wave[4:-4,4:-4]
            data = 0
        else:
            raise SystemExit('*** Wave folder ' + waveFolder + ' does not exist ***')

        if (os.path.exists('quick_reduction/'+flatFolder+'_flat_limits.fits') and os.path.exists('quick_reduction/'+flatFolder+'_flat_slices.fits')):
            limits, limitsHdr = wifisIO.readImgsFromFile('quick_reduction/'+flatFolder+'_flat_limits.fits')
            flatSlices = wifisIO.readImgsFromFile('quick_reduction/'+flatFolder+'_flat_slices.fits')[0]
            shft = limitsHdr['LIMSHIFT']
        else:
            print('Processing flat file')
            #check the type of raw data, only assumes CDS or up-the-ramp
            if (os.path.exists(rootFolder + '/CDSReference/'+flatFolder+'/Result/CDSResult.fits')):
                #CDS image
                flat, flatHdr = wifisIO.readImgsFromFile(rootFolder + '/CDSReference/'+flatFolder+'/Result/CDSResult.fits')
            elif os.path.exists(rootfolder+'/FSRamp/'+flatFolder):
                folderType = '/FSRamp/'
                UTR =  False
                if os.path.exists(rootFolder + folderType + flatFolder+'/Result/CDSResult.fits'):
                    flat, hdr = wifisIO.readImgsFromFile(rootFolder + folderType + flatFolder+'/Result/CDSResult.fits')
                else:
                    data, inttime, hdr = wifisIO.readRampFromFolder(rootFolder + folderType + flatFolder)
                    
                    #find if any pixels are saturated and avoid usage
                    satFrame = satInfo.getSatFrameCL(data, satCounts,32)

                    #get processed ramp
                    flat = combData.FowlerSamplingCL(inttime, data, satFrame, 32)[0]  
                    data = 0
            elif os.path.exists(rootFolder + '/UpTheRamp/'+flatFolder):
                data, inttime, flatHdr = wifisIO.readRampFromFolder(rootFolder + '/UpTheRamp/'+flatFolder)
                satCounts = wifisIO.readImgsFromFile(satFile)[0]
                satFrame = satInfo.getSatFrameCL(data, satCounts,32)
                #get processed ramp
                flat = combData.upTheRampCL(inttime, data, satFrame, 32)[0]
                data = 0
            else:
                raise SystemExit('*** Flat folder ' + flatFolder + ' does not exist ***')

            print('Finding flat limits')
            limits = slices.findLimits(flat, dispAxis=0, winRng=51, imgSmth=5, limSmth=20,rmRef=True)
            distMapLimits = wifisIO.readImgsFromFile(distMapLimitsFile)[0]
            shft = int(np.nanmedian(limits[1:-1,:] - distMapLimits[1:-1,:]))
            limits = distMapLimits
            flatSlices = slices.extSlices(flat[4:2044,4:2044], limits)

            flatHdr.set('LIMSHIFT',shft, 'Limits shift relative to Ronchi slices')
            wifisIO.writeFits(limits,'quick_reduction/'+flatFolder+'_flat_limits.fits',hdr=flatHdr, ask=False)
            wifisIO.writeFits(flatSlices,'quick_reduction/'+flatFolder+'_flat_slices.fits', ask=False)
                
        print('extracting wave slices')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", "RuntimeWarning")
            waveSlices = slices.extSlices(wave, limits, dispAxis=0, shft=shft)

        print('getting normalized wave slices')
        if hband:
            flatNorm = slices.getResponseAll(flatSlices, 0, 0.6)
        else:
            flatNorm = slices.getResponseAll(flatSlices, 0, 0.1)

        waveNorm = slices.ffCorrectAll(waveSlices, flatNorm)

        print ('getting distortion corrected slices')
        waveCor = createCube.distCorAll(waveSlices, distMap, spatGridProps=spatGridProps)

        #save data
        wifisIO.writeFits(waveCor, 'quick_reduction/'+waveFolder+'_wave_slices_distCor.fits', ask=False)
        print('Getting dispersion solution')

        result = waveSol.getWaveSol(waveCor, template, atlasFile, 3, prevSol, winRng=9, mxCcor=150, weights=False, buildSol=False, sigmaClip=sigmaClip, allowLower=False, lngthConstraint=True, MP=True, adjustFitWin=True, sigmaLimit=sigmaLimit, allowSearch=False, sigmaClipRounds=sigmaClipRounds)        
       
        print('Extracting solution results')
        dispSolLst = result[0]
        fwhmLst = result[1]
        pixCentLst = result[2]
        waveCentLst = result[3]
        rmsLst = result[4]
        pixSolLst = result[5]

        print('Building maps of results')
        npts = waveSlices[0].shape[1]
        waveMapLst = waveSol.buildWaveMap(dispSolLst,npts)

        for fwhm in fwhmLst:
            for i in range(len(fwhm)):
                fwhm[i] = np.abs(fwhm[i])
        
        fwhmMapLst = waveSol.buildFWHMMap(pixCentLst, fwhmLst, npts)
        #get max and min starting wavelength based on median of central slice (slice 8)

        if hband:
            trimSlc = waveSol.trimWaveSlice([waveMapLst[8], flatSlices[8], 0.5])
            waveMin = np.nanmin(trimSlc)
            waveMax = np.nanmax(trimSlc)
        else:
            trimSlc = waveMapLst[8]
            waveMax = np.nanmedian(trimSlc[:,0])
            waveMin = np.nanmedian(trimSlc[:,-1])
 
        print('*******************************************************')
        print('*** Minimum median wavelength for slice 8 is ' + str(waveMin)+ ' ***')
        print('*** Maximum median wavelength for slice 8 is ' + str(waveMax)+ ' ***')
        print('*******************************************************')

        
        #determine length along spatial direction
        ntot = 0
        for j in range(len(rmsLst)):
            ntot += len(rmsLst[j])

        #get median FWHM
        fwhmAll = []
        for f in fwhmLst:
            for i in range(len(f)):
                for j in range(len(f[i])):
                    fwhmAll.append(f[i][j])
            
        fwhmMed = np.nanmedian(fwhmAll)
        print('**************************************')
        print('*** MEDIAN FWHM IS '+ str(fwhmMed) + ' ***')
        print('**************************************')

        #build "detector" map images
        #wavelength solution
        waveMap = np.empty((npts,ntot),dtype='float32')
        strt=0
        for m in waveMapLst:
            waveMap[:,strt:strt+m.shape[0]] = m.T
            strt += m.shape[0]

        #fwhm map
        fwhmMap = np.empty((npts,ntot),dtype='float32')
        strt=0
        for f in fwhmMapLst:
            fwhmMap[:,strt:strt+f.shape[0]] = f.T
            strt += f.shape[0]

        #save results
        wifisIO.writeFits(waveMap, 'quick_reduction/'+waveFolder+'_wave_wavelength_map.fits', ask=False)
        wifisIO.writeFits(fwhmMap, 'quick_reduction/'+waveFolder+'_wave_fwhm_map.fits', ask=False)

        print('plotting results')
        fig = plt.figure()

        plt.imshow(fwhmMap, aspect='auto', cmap='jet', clim=colorbarLims, origin='lower')

        plt.colorbar()
        plt.title('Median FWHM is '+'{:3.1f}'.format(fwhmMed) +', min wave is '+'{:6.1f}'.format(waveMin)+', max wave is '+'{:6.1f}'.format(waveMax))
        plt.savefig('quick_reduction/'+waveFolder+'_wave_fwhm_map.png', dpi=300)
        plt.show()
        plt.close()

    return

def procRonchiData(ronchiFolder, flatFolder, hband=False, colorbarLims=None, mxWidth=4):
    """
    """

    
    #read in calibration data
    satCounts = wifisIO.readImgsFromFile(satFile)[0]
    nlCoeff = wifisIO.readImgsFromFile(nlFile)[0]
    BPM = wifisIO.readImgsFromFile(bpmFile)[0]

    #create processed directory, in case it doesn't exist
    wifisIO.createDir('quick_reduction')

    #process the flat
    
    if (os.path.exists('quick_reduction/'+flatFolder+'_flat.fits') and os.path.exists('quick_reduction/'+flatFolder+'_flat_limits.fits') and os.path.exists('quick_reduction/'+flatFolder+'_flat_slices.fits') and os.path.exists('quick_reduction/'+flatFolder+'_flat_slices_norm.fits')):
        limits = wifisIO.readImgsFromFile('quick_reduction/'+flatFolder+'_flat_limits.fits')[0]
        flatSlices = wifisIO.readImgsFromFile('quick_reduction/'+flatFolder+'_flat_slices.fits')[0][0:18]
        flatNorm = wifisIO.readImgsFromFile('quick_reduction/'+flatFolder+'_flat_slices_norm.fits')[0][0:18]
    else:
        print('processing flat ' + flatFolder)

        flat, sigmaImg, satFrame, hdr = processRamp.auto(flatFolder, rootFolder,'quick_reduction/'+flatFolder+'_flat.fits', satCounts, nlCoeff, BPM,nChannel=32, rowSplit=1, satSplit=32, nlSplit=32, combSplit=32, crReject=False, bpmCorRng=20, saveAll=False)
    
        #find limits
        print('finding limits and extracting flat slices')
        limits = slices.findLimits(flat, dispAxis=0, limSmth=5, rmRef=True)

        polyLims = slices.polyFitLimits(limits, degree=2, sigmaClipRounds=1)

        with PdfPages('quick_reduction/'+flatFolder+'_flat_slices_traces.pdf') as pdf:
            fig = plt.figure()
            med1= np.nanmedian(flat)
            
            plt.imshow(flat[4:-4,4:-4], aspect='auto', cmap='jet', clim=[0,2.*med1], origin='lower')
            plt.xlim=(0,2040)
            plt.colorbar()
            for l in range(limits.shape[0]):
                plt.plot(limits[l], np.arange(limits.shape[1]),'k', linewidth=1) #drawn limits
                plt.plot(np.clip(polyLims[l],0, flat[4:-4,4:-4].shape[0]-1), np.arange(limits.shape[1]),'r--', linewidth=1) #shifted ronchi limits, if provided, or polynomial fit
            pdf.savefig()
            plt.close(fig)
           
        flat = flat[4:-4,4:-4]
        flatSlices = slices.extSlices(flat, polyLims)
        flatNorm = slices.getResponseAll(flatSlices, 0, 0.1)

        wifisIO.writeFits(polyLims, 'quick_reduction/'+flatFolder+'_flat_limits.fits', ask=False)
        wifisIO.writeFits(flatSlices, 'quick_reduction/'+flatFolder+'_flat_slices.fits', ask=False)
        wifisIO.writeFits(flatNorm, 'quick_reduction/'+flatFolder+'_flat_slices_norm.fits', ask=False)
        limits = polyLims

    if not os.path.exists('quick_reduction/'+ronchiFolder+'_ronchi_amp_map.png'):

        print('Processing ronchi')
        
        #now process the Ronchi ramp
        #check the type of raw data, only assumes CDS or up-the-ramp
        ronchi, sigmaImg, satFrame, hdr = processRamp.auto(ronchiFolder, rootFolder,'quick_reduction/'+ronchiFolder+'_ronchi.fits', satCounts, nlCoeff, BPM,nChannel=32, rowSplit=1, satSplit=32, nlSplit=32, combSplit=32, crReject=False, bpmCorRng=20, saveAll=False)

        print('extracting ronchi slices')
        ronchi=ronchi[4:-4, 4:-4]
        ronchiSlices = slices.extSlices(ronchi, limits, dispAxis=0)
        ronchiFlat = slices.ffCorrectAll(ronchiSlices, flatNorm)
        wifisIO.writeFits(ronchiSlices, 'quick_reduction/'+ronchiFolder+'_ronchi_slices.fits',ask=False)

        print('Getting traces')
        ronchiTraces = []
        ronchiAmps =[]
        for i in range(len(ronchiSlices)):
            ronchiTraces.append([])
            ronchiAmps.append([])

        #easy traces
        
        inp = [ronchiSlices[0],ronchiSlices[1],ronchiSlices[2],ronchiSlices[3],ronchiSlices[4],ronchiSlices[5],ronchiSlices[6],ronchiSlices[7],ronchiSlices[8],ronchiSlices[9],ronchiSlices[10],ronchiSlices[11],ronchiSlices[14]]
        out = spatialCor.traceRonchiAll(inp, nbin=2, winRng=7, mxWidth=mxWidth,smth=20, bright=False, flatSlices=None, MP=True)

        ronchiTraces[:12] = out[0][:12]
        ronchiTraces[14] = out[0][12]

        ronchiAmps[:12] = out[1][:12]
        ronchiAmps[14] = out[1][12]

        #difficult slices
        out = spatialCor.traceRonchiAll([ronchiFlat[12], ronchiFlat[13],ronchiFlat[15], ronchiFlat[16], ronchiFlat[17]], nbin=1, winRng=7, mxWidth=mxWidth,smth=20, bright=False, flatSlices=[flatSlices[12],flatSlices[13],flatSlices[15],flatSlices[16], flatSlices[17]], MP=True, threshold=0.75)
        
        [ronchiTraces[12], ronchiTraces[13], ronchiTraces[15], ronchiTraces[16], ronchiTraces[17]] = out[0]
        [ronchiAmps[12], ronchiAmps[13], ronchiAmps[15], ronchiAmps[16], ronchiAmps[17]] = out[1]

        #get rid of bad traces
        for i in range(len(ronchiTraces)):
            r = ronchiTraces[i]

            whr = np.where(np.logical_or(r<0, r>=ronchiSlices[i].shape[0]))
            ronchiTraces[i][whr] = np.nan
            ronchiAmps[i][whr] = np.nan

            
        with PdfPages('quick_reduction/'+ronchiFolder+'_ronchi_slices_traces.pdf') as pdf:
            for i in range(len(ronchiSlices)):
                fig = plt.figure()
                m = np.nanmedian(ronchiSlices[i])
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    s = np.nanstd(ronchiSlices[i][np.logical_and(ronchiSlices[i] < 5*m, ronchiSlices[i]>0)])
            
                plt.imshow(ronchiSlices[i], aspect='auto', clim=[0,m+1.5*s], origin='lower')

                for j in range(len(ronchiTraces[i])):
                    plt.plot(ronchiTraces[i][j,:], 'r--')
                plt.title('Slices #'+str(i))
                pdf.savefig(dpi=300)
                plt.close()

        print('plotting amp map')
        #build resolution map
        ampMapLst = spatialCor.buildAmpMap(ronchiTraces, ronchiAmps, ronchiSlices)

        #get median FWHM
        ampAll = []
        for f in ronchiAmps:
            for i in range(len(f)):
                for j in range(len(f[i])):
                    ampAll.append(f[i][j])
            
        ampMed = np.nanmedian(ampAll)


        print('**************************************')
        print('*** MEDIAN amplitude IS '+ str(ampMed) + ' ***')
        print('**************************************')
        
        ntot = 0
        for r in ronchiSlices:
            ntot += r.shape[0]
    
        ampMap = np.empty((r.shape[1],ntot),dtype='float32')
    
        strt=0
        for a in ampMapLst:
            ampMap[:,strt:strt+a.shape[0]] = a.T
            strt += a.shape[0]

        fig = plt.figure()
        m = np.nanmedian(ampMap)
        s = np.nanstd(ampMap)

        if colorbarLims is None:
            clim = [m-3*s, m+3*s]
        else:
            clim=colorbarLims
        
        plt.imshow(ampMap, origin='lower', aspect='auto', clim=clim,cmap='jet')
        plt.title('Ronchi amplitude map - Med amp ' + '{:4.2f}'.format(ampMed))
        
        plt.colorbar()
        plt.savefig('quick_reduction/'+ronchiFolder+'_ronchi_amp_map.png',dpi=300)
        plt.close()
        
        
        print('saving results')
        #write results!
        wifisIO.writeFits(ronchiTraces, 'quick_reduction/'+ronchiFolder+'_ronchi_traces.fits', ask=False)
        wifisIO.writeFits(ampMap, 'quick_reduction/'+ronchiFolder+'_ronchi_amp_map.fits', ask=False)
    else:
        print('Ronchi ' + ronchiFolder + ' already processed')
    return
