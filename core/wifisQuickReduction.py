"""

set of functions used to run a quick reduction of the WIFIS pipeline

"""

import matplotlib
#*******************************************************************
#matplotlib.use('gtkagg') #default is tkagg, but using gtkagg can speed up window creation on some systems
#*******************************************************************

import wifisIO
import matplotlib.pyplot as plt
import numpy as np
import wifisSlices as slices
import wifisCreateCube as createCube
import wifisHeaders as headers
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
from astropy.visualization import ZScaleInterval
from scipy.optimize import curve_fit

import colorama
colorama.init()

#******************************************************************************

def initPaths(hband=False):
    """
    THIS FUNCTION IS DEPRECATED. IT IS NO LONGER NEEDED AS ALL VARIABLES ARE INITIALIZED FROM A CONFIGURATION FILE WITHIN THE INDIVIDUAL ROUTINES
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
    global obsCoords

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

        #july
        distMapFile = '/home/jason/wifis/data/ronchi_map_july/tb/processed/20170707175840_ronchi_distMap.fits'
        spatGridPropsFile ='/home/jason/wifis/data/ronchi_map_july/tb/processed/20170707175840_ronchi_spatGridProps.dat'
        distMapLimitsFile ='/home/jason/wifis/data/ronchi_map_july/tb/processed/20170707180443_flat_limits.fits'

        #august
        distMapFile = '/home/jason/wifis/data/ronchi_map_august/tb/processed/20170831211259_ronchi_distMap.fits'
        distMapLimitsFile = '/home/jason/wifis/data/ronchi_map_august/tb/processed/20170831210255_flat_limits.fits'
        spatGridPropsFile = '/home/jason/wifis/data/ronchi_map_august/tb/processed/20170831211259_ronchi_spatGridProps.dat'
        
    #should be (mostly) static
    atlasFile = pipelineFolder + '/external_data/best_lines2.dat'
    satFile = pipelineFolder + '/external_data/master_detLin_satCounts.fits'
    bpmFile = pipelineFolder+'external_data/bpm.fits'
    nlFile = pipelineFolder + 'external_data/master_detLin_NLCoeff.fits' 

    #set the pixel scale
    xScale = 0.532021532706
    yScale = -0.545667026386 #valid for npix=1, i.e. 35 pixels spanning the 18 slices. This value needs to be updated in agreement with the choice of interpolation
    obsCoords = [-111.600444444,31.9629166667,2071]


    return
#*******************************************************************************

def procRamp(rampFolder, noProc=False, satCounts=None, bpm=None,nlCoef=None, saveName='', varFile='', bpmCorRng=0):
    """
    """

    #initialize variables using configuration file
    varInp = wifisIO.readInputVariables(varFile)
    for var in varInp:
        globals()[var[0]]=var[1]    

    #execute pyOpenCL section here
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = pyCLCompOut

    if len(pyCLCTX)>0:
        os.environ['PYOPENCL_CTX'] = pyCLCTX 

    #process ramp
    if noProc:
        #check file type
        #CDS
        if os.path.exists(rootFolder+'/CDSReference/'+rampFolder):
            folderType = '/CDSReference/'
            fluxImg, hdr = wifisIO.readImgsFromFile(rootFolder + folderType + rampFolder+'/Result/CDSResult.fits')
            UTR = False
            noProc = True
        #Fowler
        elif os.path.exists(rootFolder+'/FSRamp/'+rampFolder):
            folderType = '/FSRamp/'
            UTR =  False
            if os.path.exists(rootFolder + folderType + rampFolder+'/Result/CDSResult.fits'):
                fluxImg, hdr = wifisIO.readImgsFromFile(rootFolder + folderType + rampFolder+'/Result/CDSResult.fits')
                noProc = True
            else:
                noProc = False

        elif os.path.exists(rootFolder + '/UpTheRamp/'+rampFolder):
            UTR = True
            folderType = '/UpTheRamp/'

            lst = glob.glob(rootFolder+folderType+rampFolder + '/H2*fits')
            lst = sorted_nicely(lst)

            img1,hdr1 = wifisIO.readImgsFromFile(lst[-1])
            img0,hdr0 = wifisIO.readImgsFromFile(lst[0])
            t0 = hdr0['INTTIME']
            t1 = hdr1['INTTIME']
        
            fluxImg = ((img1-img0)/(t1-t0)).astype('float32')
            hdr = hdr1
            noProc=True
        else:
            raise Warning('*** Ramp folder ' + rampFolder + ' does not exist ***')
    else:
        noProc =False

    if not noProc:
        print('Processing ramp')
        fluxImg,sigImg,satFrame, hdr =processRamp.auto(rampFolder, rootFolder, saveName, satCounts, nlCoef, bpm, nRows=0,bpmCorRng=bpmCorRng,saveAll=False,ignoreBPM=True,avgAll=True, nChannel=0)

    if noProc:
        #get obsinfo file to fill in hdr info
        obsinfoFile = rootFolder + folderType + rampFolder+'/obsinfo.dat'
        wifisIO.writeFits(fluxImg.astype('float32'),saveName,hdr=hdr,ask=False)
    else:
        obsinfoFile = None
        
    return fluxImg, hdr, obsinfoFile

def procScienceData(rampFolder='', flatFolder='', noProc=False, skyFolder=None, pixRange=None, varFile='',scaling='zscale', colorbar=False):
    """
    Routine to quickly process the raw data from a science ramp and plot a 2D collapse of the final image cube
    Usage procScienceData(rampFolder='', flatFolder='', noProc=False, skyFolder=None, pixRange=None, varFile='',scaling='zscale')
    rampFolder is the folder name of the observation to process and plot
    flatFolder is the corresponding flat field observation associated with rampFolder
    noProc is a boolean keyword used to indicate if the ramp image should be computed using the pipeline (but skipping non-linearity and reference corrections) (False) or if the ramp image should be found in a simple manner using the first and last images (True). The latter option does not handle saturation well, which is handled with the usual pipeline method if False.
    skyFolder is an optional keyword to specify the name of an associated sky ramp folder
    pixRange is an optional list containing the first and last pixel in a range (along the dispersion axis) of pixels to use for creating the ramp image.
    varFile is the name of the configuration file
    scaling is a keyword that specifies the type of image scaling to use for plotting the final image. If set to "zscale", z-scaling is used. Anything else will set the scale to min-max scaling.
    """

    #initialize variables using configuration file
    varInp = wifisIO.readInputVariables(varFile)
    for var in varInp:
        globals()[var[0]]=var[1]    

    #execute pyOpenCL section here
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = pyCLCompOut

    if len(pyCLCTX)>0:
        os.environ['PYOPENCL_CTX'] = pyCLCTX 

    if os.path.exists(satFile):
        satCounts = wifisIO.readImgsFromFile(satFile)[0]
    else:
        satCounts = None

    if os.path.exists(bpmFile):
        bpm = wifisIO.readImgsFromFile(bpmFile)[0]
    else:
        bpm = None
        
    wifisIO.createDir('quick_reduction')

    #process science data
    if noProc:
        print('Attempting to process science and sky (if exists) ramps without usual processing')

    fluxImg, hdr, obsinfoFile = procRamp(rampFolder, noProc=noProc, satCounts=satCounts, bpm=bpm, saveName='quick_reduction/'+rampFolder+'_obs.fits',varFile=varFile)

    #now process sky, if it exists
       
    if (skyFolder is not None):
        skyImg, hdrSky, skyobsinfo = procRamp(skyFolder,noProc=noProc, satCounts=satCounts, bpm=bpm, saveName='quick_reduction/'+skyFolder+'_sky.fits',varFile=varFile)
        
        #subtract
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            fluxImg -= skyImg

    fluxImg = fluxImg[4:-4, 4:-4]

    #first check if limits already exists
    if os.path.exists('quick_reduction/'+flatFolder+'_limits.fits'):
        limitsFile = 'quick_reduction/'+flatFolder+'_limits.fits'
        limits = wifisIO.readImgsFromFile(limitsFile)[0]
    else:
        flat, hdrFlat, hdrobsinfo = procRamp(flatFolder, noProc=False, satCounts=satCounts, bpm=bpm, saveName='quick_reduction/'+flatFolder+'_flat.fits',varFile=varFile)
        
        print('Getting slice limits')
        limits = slices.findLimits(flat, dispAxis=0, rmRef=True,centGuess=centGuess)
        wifisIO.writeFits(limits, 'quick_reduction/'+flatFolder+'_limits.fits', ask=False)

    if os.path.exists(distMapLimitsFile):
        #get ronchi slice limits
        distLimits = wifisIO.readImgsFromFile(distMapLimitsFile)[0]

        #determine shift
        shft = np.median(limits[1:-1, :] - distLimits[1:-1,:])
    else:
        print(colorama.Fore.RED+'*** WARNING: NO DISTORTION MAP LIMITS PROVIDED. LIMITS ARE DETERMINED ENTIRELY FROM THE FLAT FIELD DATA ***'+colorama.Style.RESET_ALL)
        shft = 0
        distLimits = limits
        
    print('Extracting slices')
    dataSlices = slices.extSlices(fluxImg, distLimits, dispAxis=0, shft=shft)

    #place on uniform spatial grid
    print('Distortion correcting')
    if not os.path.exists(distMapLimitsFile) or not os.path.exists(distMapFile) or not os.path.exists(spatGridPropsFile):
        print(colorama.Fore.RED+'*** WARNING: NO DISTORTION MAP PROVIDED, ESTIMATING DISTORTION MAP FROM SLICE SHAPE ***'+colorama.Style.RESET_ALL)
        #read flat image from file and extract slices
        flat,flatHdr = wifisIO.readImgsFromFile('quick_reduction/'+flatFolder+'_flat.fits')
        flatSlices = slices.extSlices(flat[4:-4,4:-4], limits)
        #get fake distMap and grid properties
        distMap, spatGridProps = makeFakeDistMap(flatSlices)
    else:
        distMap = wifisIO.readImgsFromFile(distMapFile)[0]
        spatGridProps = wifisIO.readTable(spatGridPropsFile)

    dataGrid = createCube.distCorAll(dataSlices, distMap, spatGridProps=spatGridProps)

    #create cube
    print('Creating image')
    dataCube = createCube.mkCube(dataGrid, ndiv=1, missing_left=missing_left_slice, missing_right=missing_right_slice)

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

    if noProc:
        #fill in header info
        headers.addTelInfo(hdr, obsinfoFile, obsCoords=obsCoords)

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
    if scaling=='zscale':
        interval=ZScaleInterval()
        lims=interval.get_limits(dataImg)
    else:
        lims=[dataImg.min(),dataImg.max()]
    plt.imshow(dataImg, origin='lower', cmap='jet', clim=lims)

    if colorbar:
        plt.colorbar()
        
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
    plt.tight_layout()
    plt.savefig('quick_reduction/'+rampFolder+'_quickRedImg.png', dpi=300)
    plt.show()
    plt.close('all')
    return

def procArcData(waveFolder, flatFolder, hband=False, colorbarLims = None, varFile='', noPlot=False):
    """
    Routine to quickly process the raw data from an arc lamp/wavelength correction ramp and plot the resulting FWHM map across each slice
    Usage: procArcData(waveFolder, flatFolder, hband=False, colorbarLims = None, varFile='')
    waveFolder is the ramp folder to be processed
    flatFolder is the flat field ramp folder associated with waveFolder
    hband is a boolean keyword to specify if the ramp used the h-band filter (and thus does not span the entire detector)
    colorbarLims is a keyword that allows one to specify the limits to use when plotting the FWHM map. If set to None, the default method uses z-scaling
    varFile is the name of the input configuration file.
    """

    #initialize variables using configuration file
    varInp = wifisIO.readInputVariables(varFile)
    for var in varInp:
        globals()[var[0]]=var[1]    

    #execute pyOpenCL section here
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = pyCLCompOut

    if len(pyCLCTX)>0:
        os.environ['PYOPENCL_CTX'] = pyCLCTX 

    wifisIO.createDir('quick_reduction')

    #read in previous results and template
    template = wifisIO.readImgsFromFile(waveTempFile)[0]
    prevResults = wifisIO.readPickle(waveTempResultsFile)
    prevSol = prevResults[5]

    if os.path.exists(distMapFile) and os.path.exists(spatGridPropsFile):
        distMap = wifisIO.readImgsFromFile(distMapFile)[0]
        spatGridProps = wifisIO.readTable(spatGridPropsFile)
    else:
        distMap = None
        spatGridProps = None
        
    if os.path.exists(satFile):
        satCounts = wifisIO.readImgsFromFile(satFile)[0]
    else:
        satCounts = None
    if os.path.exists(bpmFile):
        bpm = wifisIO.readImgsFromFile(bpmFile)[0]
    else:
        bpm = None
        
    if (os.path.exists('quick_reduction/'+waveFolder+'_wave_fwhm_map.png') and os.path.exists('quick_reduction/'+waveFolder+'_wave_fwhm_map.fits') and os.path.exists('quick_reduction/'+waveFolder+'_wave_wavelength_map.fits')):
        print('*** ' + waveFolder + ' arc/wave data already processed, skipping ***')
    else:
        wave, hdr,obsinfo = procRamp(waveFolder, satCounts=satCounts, bpm=bpm, saveName='quick_reduction/'+waveFolder+'_wave.fits',varFile=varFile)
    
        if (os.path.exists('quick_reduction/'+flatFolder+'_flat_limits.fits') and os.path.exists('quick_reduction/'+flatFolder+'_flat_slices.fits')):
            limits, limitsHdr = wifisIO.readImgsFromFile('quick_reduction/'+flatFolder+'_flat_limits.fits')
            flatSlices,flatHdr = wifisIO.readImgsFromFile('quick_reduction/'+flatFolder+'_flat_slices.fits')
            shft = limitsHdr['LIMSHIFT']
        else:
            print('Processing flat file')
            flat, flatHdr, obsinfo = procRamp(flatFolder, satCounts=satCounts, bpm=bpm, saveName='quick_reduction/'+flatFolder+'_flat.fits',varFile=varFile)

            print('Finding flat limits')
            limits = slices.findLimits(flat, dispAxis=0, winRng=51, imgSmth=5, limSmth=20,rmRef=True, centGuess=centGuess)
            if os.path.exists(distMapLimitsFile):
                distMapLimits = wifisIO.readImgsFromFile(distMapLimitsFile)[0]
                shft = int(np.nanmedian(limits[1:-1,:] - distMapLimits[1:-1,:]))
                limits = distMapLimits
            else:
                print(colorama.Fore.RED+'*** WARNING: NO LIMITS FILE ASSOCIATED WITH THE DISTORTION MAP PROVIDED, USING LIMITS DETERMINED FROM FLATS ONLY ***'+colorama.Style.RESET_ALL)
                shft = 0

            flatSlices = slices.extSlices(flat[4:-4, 4:-4], limits, shft=shft)
            flatHdr.set('LIMSHIFT',shft, 'Limits shift relative to Ronchi slices')
            wifisIO.writeFits(limits,'quick_reduction/'+flatFolder+'_flat_limits.fits',hdr=flatHdr, ask=False)
            wifisIO.writeFits(flatSlices,'quick_reduction/'+flatFolder+'_flat_slices.fits', ask=False)
                
        print('Extracting wave slices')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", "RuntimeWarning")
            waveSlices = slices.extSlices(wave[4:-4,4:-4], limits, dispAxis=0, shft=shft)

        print('Getting normalized wave slices')
        if hband:
            flatNorm = slices.getResponseAll(flatSlices, 0, 0.6)
        else:
            flatNorm = slices.getResponseAll(flatSlices, 0, 0.1)

        if distMap is None:
            print(colorama.Fore.RED+'*** WARNING: NO DISTORTION MAP PROVIDED, ESTIMATING FROM FLAT FIELD SLICES ***'+colorama.Style.RESET_ALL)

            flatSlices = wifisIO.readImgsFromFile('quick_reduction/'+flatFolder+'_flat_slices.fits')[0]
            distMap, spatGridProps = makeFakeDistMap(flatSlices)

        print ('Getting distortion corrected slices')
        waveCor = createCube.distCorAll(waveSlices, distMap, spatGridProps=spatGridProps)

        #save data
        wifisIO.writeFits(waveCor, 'quick_reduction/'+waveFolder+'_wave_slices_distCor.fits', ask=False)
        print('Getting dispersion solution')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            
            result = waveSol.getWaveSol(waveCor, template, atlasFile,mxOrder, prevSol, winRng=waveWinRng, mxCcor=waveMxCcor, weights=False, buildSol=False, sigmaClip=sigmaClip, allowLower=False, lngthConstraint=False, MP=True, adjustFitWin=True, sigmaLimit=sigmaLimit, allowSearch=False, sigmaClipRounds=sigmaClipRounds)        
       
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
        hdr.set('QC_WMIN',waveMin,'Minimum median wavelength for middle slice')
        hdr.set('QC_WMAX',waveMax,'Maximum median wavelength for middle slice')
        hdr.set('QC_WFWHM', fwhmMed, 'Median FWHM of all slices')

        wifisIO.writeFits(waveMap, 'quick_reduction/'+waveFolder+'_wave_wavelength_map.fits', ask=False,hdr=hdr)
        wifisIO.writeFits(fwhmMap, 'quick_reduction/'+waveFolder+'_wave_fwhm_map.fits', ask=False,hdr=hdr)

        print('plotting results')
        fig = plt.figure()

        if colorbarLims is None:
            interval=ZScaleInterval()
            lims=interval.get_limits(fwhmMap)
        else:
            lims = colorbarLims

        plt.imshow(fwhmMap, aspect='auto', cmap='jet', clim=lims, origin='lower')
        plt.colorbar()
        plt.title('Median FWHM is '+'{:3.1f}'.format(fwhmMed) +', min wave is '+'{:6.1f}'.format(waveMin)+', max wave is '+'{:6.1f}'.format(waveMax))
        plt.tight_layout()
        plt.savefig('quick_reduction/'+waveFolder+'_wave_fwhm_map.png', dpi=300)
        if not noPlot:
            plt.show()
        plt.close()

    return

def procRonchiData(ronchiFolder, flatFolder, hband=False, colorbarLims=None, varFile='',noPlot=False,noFlat=False):
    """
    Routine to quickly process a ramp containing a Ronchi mask observation and plot the map of the measured amplitudes
    Usage: procRonchiData(ronchiFolder, flatFolder, hband=False, colorbarLims=None, mxWidth=4, varFile='',noPlot=False)
    ronchiFolder is the ramp folder of the Ronchi mask observation to be processed
    flatFolder is the flat field ramp folder associated with the ronchiFolder
    hband is a boolean flag to indicate whether the data was obtained with the H-band filter
    colorbarLims is a keyword that allows one to set the colour bar limits of the amplitude map plot. If None given, the code uses z-scaling
    varFile is the name of the configuration file to read
    noPlot is a boolean keyword used to specify if no map plotting should be carried out (the plots are still saved to files)
    """

    #initialize variables using configuration file
    varInp = wifisIO.readInputVariables(varFile)
    for var in varInp:
        globals()[var[0]]=var[1]    

    #execute pyOpenCL section here
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = pyCLCompOut

    if len(pyCLCTX)>0:
        os.environ['PYOPENCL_CTX'] = pyCLCTX 
    
    #read in calibration data
    if os.path.exists(satFile):
        satCounts = wifisIO.readImgsFromFile(satFile)[0]
    else:
        satCounts = None
        
    #nlCoeff = wifisIO.readImgsFromFile(nlFile)[0]
    if os.path.exists(bpmFile):
        BPM = wifisIO.readImgsFromFile(bpmFile)[0]
    else:
        BPM = None
        
    #create processed directory, in case it doesn't exist
    wifisIO.createDir('quick_reduction')

    #process the flat
    
    if (os.path.exists('quick_reduction/'+flatFolder+'_flat.fits') and os.path.exists('quick_reduction/'+flatFolder+'_flat_limits.fits') and os.path.exists('quick_reduction/'+flatFolder+'_flat_slices.fits') and os.path.exists('quick_reduction/'+flatFolder+'_flat_slices_norm.fits')):
        limits = wifisIO.readImgsFromFile('quick_reduction/'+flatFolder+'_flat_limits.fits')[0]
        flatSlices = wifisIO.readImgsFromFile('quick_reduction/'+flatFolder+'_flat_slices.fits')[0]
        flatLst = wifisIO.readImgsFromFile('quick_reduction/'+flatFolder+'_flat_slices_norm.fits')
        flatNorm = flatLst[0]
        flatHdr = flatLst[1]
    else:
        print('processing flat ' + flatFolder)

        flat, sigmaImg, satFrame, flatHdr = processRamp.auto(flatFolder, rootFolder,'quick_reduction/'+flatFolder+'_flat.fits', satCounts, None, BPM,nChannel=0, rowSplit=nRowSplit, satSplit=nSatSplit, nlSplit=nlSplit, combSplit=nCombSplit, crReject=False, bpmCorRng=flatbpmCorRng, saveAll=False,nRows=0)
    
        #find limits
        print('finding limits and extracting flat slices')
        limits = slices.findLimits(flat, dispAxis=dispAxis, limSmth=flatLimSmth, rmRef=True, centGuess=centGuess)

        polyLims = slices.polyFitLimits(limits, degree=limitsPolyFitDegree, sigmaClipRounds=1)

        interval = ZScaleInterval()
        
        with PdfPages('quick_reduction/'+flatFolder+'_flat_slices_traces.pdf') as pdf:
            fig = plt.figure()
            med1= np.nanmedian(flat)

            lims = interval.get_limits(flat[4:-4,4:-4])
            plt.imshow(flat[4:-4,4:-4], aspect='auto', cmap='jet', clim=lims, origin='lower')
            plt.xlim=(0,2040)
            plt.colorbar()
            for l in range(limits.shape[0]):
                plt.plot(limits[l], np.arange(limits.shape[1]),'k', linewidth=1) #drawn limits
                plt.plot(np.clip(polyLims[l],0, flat[4:-4,4:-4].shape[0]-1), np.arange(limits.shape[1]),'r--', linewidth=1) #shifted ronchi limits, if provided, or polynomial fit
            plt.tight_layout()
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
        ronchi, sigmaImg, satFrame, hdr = processRamp.auto(ronchiFolder, rootFolder,'quick_reduction/'+ronchiFolder+'_ronchi.fits', satCounts, None, BPM,nChannel=0, rowSplit=nRowSplit, satSplit=nSatSplit, nlSplit=nlSplit, combSplit=nCombSplit, crReject=False, bpmCorRng=20, saveAll=False,nRows=0)

        print('extracting ronchi slices')
        ronchi=ronchi[4:-4, 4:-4]
        ronchiSlices = slices.extSlices(ronchi, limits, dispAxis=0)
        ronchiFlat = slices.ffCorrectAll(ronchiSlices, flatNorm)
        wifisIO.writeFits(ronchiSlices, 'quick_reduction/'+ronchiFolder+'_ronchi_slices.fits',ask=False)

        print('Getting traces')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            if noFlat:
                ronchiTraces, ronchiAmps = spatialCor.traceRonchiAll(ronchiSlices, nbin=ronchiNbin, winRng=ronchiWinRng, mxWidth=ronchiMxWidth,smth=ronchiSmth, bright=ronchiBright, flatSlices=None, MP=True)
            else:
                ronchiTraces, ronchiAmps = spatialCor.traceRonchiAll(ronchiFlat, nbin=ronchiNbin, winRng=ronchiWinRng, mxWidth=ronchiMxWidth,smth=ronchiSmth, bright=ronchiBright, flatSlices=flatSlices, MP=True)

        #get rid of bad traces
        for i in range(len(ronchiTraces)):
            r = ronchiTraces[i]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                whr = np.where(np.logical_or(r<0, r>=ronchiSlices[i].shape[0]))
            ronchiTraces[i][whr] = np.nan
            ronchiAmps[i][whr] = np.nan

        interval=ZScaleInterval()
        
        with PdfPages('quick_reduction/'+ronchiFolder+'_ronchi_slices_traces.pdf') as pdf:
            for i in range(len(ronchiSlices)):
                fig = plt.figure()
                m = np.nanmedian(ronchiSlices[i])
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    s = np.nanstd(ronchiSlices[i][np.logical_and(ronchiSlices[i] < 5*m, ronchiSlices[i]>0)])

                lims = interval.get_limits(ronchiSlices[i])
                plt.imshow(ronchiSlices[i], aspect='auto', clim=lims, origin='lower')
                plt.colorbar()
                for j in range(len(ronchiTraces[i])):
                    plt.plot(ronchiTraces[i][j,:], 'r--')
                plt.title('Slices #'+str(i))
                plt.tight_layout()
                pdf.savefig(dpi=300)
                plt.close()

        print('plotting amp map')
        #build resolution map
        ampMapLst = spatialCor.buildAmpMap(ronchiTraces, ronchiAmps, ronchiSlices)


        #get median amplitude/contrast measurement
        ampAll = []

        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            
            for f in ronchiAmps:

                #remove points from different traces that share same coordinates, to derive a more accurate average

                for j in range(f.shape[0]-1):
                    for k in range(j+1,f.shape[0]):
                        whr = np.where(np.abs(f[j,:]-f[k,:])<0.5)[0]
                        if len(whr)>0:
                            f[k,whr]=np.nan
                            
                for i in range(f.shape[0]):
                    for j in range(f.shape[1]):
                        ampAll.append(f[i][j])
            
        ampMed = np.nanmedian(ampAll)


        print('**************************************')
        print('*** MEDIAN AMPLITUDE IS '+ str(ampMed) + ' ***')
        print('**************************************')
        
        ntot = 0
        for r in ronchiSlices:
            ntot += r.shape[0]
    
        ampMap = np.empty((r.shape[1],ntot),dtype='float32')
    
        strt=0
        for a in ampMapLst:
            ampMap[:,strt:strt+a.shape[0]] = a.T
            strt += a.shape[0]

        m = np.nanmedian(ampMap)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            s = np.nanstd(ampMap[ampMap < 10*m])
            
        if colorbarLims is None:
            interval = ZScaleInterval()
            clim=interval.get_limits(ampMap)
        else:
            clim=colorbarLims

        fig = plt.figure()
        plt.imshow(ampMap, origin='lower', aspect='auto', clim=clim,cmap='jet')
        plt.title('Ronchi amplitude map - Med amp ' + '{:4.2f}'.format(ampMed))
        plt.colorbar()
        plt.tight_layout()

        plt.savefig('quick_reduction/'+ronchiFolder+'_ronchi_amp_map.png',dpi=300)
        if not noPlot:
            plt.show()
        plt.close()        
        
        print('saving results')
        #write results!
        hdr.set('QC_AMP',ampMed,'Median Ronchi amplitude of all slices')
        wifisIO.writeFits(ronchiTraces, 'quick_reduction/'+ronchiFolder+'_ronchi_traces.fits', ask=False,hdr=hdr)
        wifisIO.writeFits(ampMap, 'quick_reduction/'+ronchiFolder+'_ronchi_amp_map.fits', ask=False,hdr=hdr)
    else:
        print('Ronchi ' + ronchiFolder + ' already processed')
    return

def makeFakeDistMap(flatSlices):
    """
    """

    #get list of traces of central slice position
    trace = spatialCor.traceCentreFlatAll(flatSlices,cutoff=0.7, limSmth=20,MP=True, plot=False)

    #create fake ronchi maps
    distMap = []
    #create ronchi maps
    for i in range(len(flatSlices)):
        #fit a simple polynomial to trace to smooth things out
        x = np.arange(trace[i].shape[0])
        polyCoeff = np.polyfit(x,trace[i],3)
        poly = np.poly1d(polyCoeff)
        polyTrace = poly(x)
        
        x,y = np.mgrid[:flatSlices[i].shape[0],:flatSlices[i].shape[1]]
        distMap.append(x-polyTrace)

    #now check if either of the first or last slice are significantly cutoff
    #by comparing length relative to median length of inner slices
    lngth = []
    for i in range(len(flatSlices)):
        lngth.append(flatSlices[i].shape[0])

    medLngth = np.median(lngth[1:-1])
    stdLngth = np.std(lngth[1:-1])

    #if first or last slice is cutoff, modify map accordingly
    if medLngth-lngth[0] > stdLngth:
        distMap[0] += (medLngth-lngth[0])/4.
    if medLngth-lngth[-1] > stdLngth:
        distMap[-1] += (lngth[-1]-medLngth)/4.
        
    #determine grid properties
    spatGridProps =  createCube.compSpatGrid(distMap)
    
    return distMap, spatGridProps

def gaussian(x,amp,cen,wid):
    """
    Returns a Gaussian function of the form y = A*exp(-z^2/2), where z=(x-cen)/wid
    Usage: output = gaussian(x, amp, cen, wid)
    x is the input 1D array of coordinates
    amp is the amplitude of the Gaussian
    cen is the centre of the Gaussian
    wid is the 1-sigma width of the Gaussian
    returns an array with same size as x corresponding to the Gaussian solution
    """
    
    z = (x-cen)/wid    
    return amp*np.exp(-z**2/2.)

def gaussFit(x, y, guessWidth):
    """
    Routine to fit a Gaussian to provided x and y data points and return the fitted coefficients.
    Usage: params = gaussFit(x, y, guessWidth, plot=True/False,title='')
    x is the 1D numpy array of coordinates
    y is the 1D numpy array of data values at the given coordinates
    guessWidth is the initial guess for the width of the Gaussian
    plot is a keyword to allow for plotting of the fit (for debug purposes)
    title is an optional title to provide the plot
    params is a list containing the fitted parameters in the following order: amplitude, centre, width
    """

    ytmp = y-y.min()
    #use scipy to do fitting
    popt, pcov = curve_fit(gaussian, x, ytmp, p0=[ytmp.max(), np.mean(x),guessWidth])

    return popt

def getPSFMap(rampFolder='',skyFolder=None,varFile='', guessWidth=8, threshold=2., nBest=5,dispAxis=0, noPlot=False, noProc=False, pixRng=None,nSlices=18, fracLev=0.25, nlCor=False):
    """
    Routine to estimate PSF along the dispersion axis of a continuum point source.
    Usage: getPSFMap(rampFolder='',skyFolder=None,varFile='', guessWidth=8, sliceWidth=150, threshold=2., dispAxis=0)
    rampFolder is the folder number/name of the science observation to derive the PSF map from
    skyFolder is optional folder name/number corresponding to the associated sky data of the science target
    guessWidth is the estimated FWHM of the data
    sliceWidth is the approximate number of pixels corresponding to each slice
    threshold is the threshold limit at which to search for continuum sources in the collapsed 1D spectrum (i.e. regions are identified with flux above threshold times noise level)
    dispAxis corresponds to the dispersion direction of the data
    """

    #initialize variables using configuration file
    varInp = wifisIO.readInputVariables(varFile)
    for var in varInp:
        globals()[var[0]]=var[1]    

    #execute pyOpenCL section here
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = pyCLCompOut

    if len(pyCLCTX)>0:
        os.environ['PYOPENCL_CTX'] = pyCLCTX 

    if os.path.exists(satFile):
        satCounts = wifisIO.readImgsFromFile(satFile)[0]
    else:
        satCounts = None

    if os.path.exists(bpmFile):
        bpm = wifisIO.readImgsFromFile(bpmFile)[0]
    else:
        bpm = None

    if os.path.exists(nlFile) and nlCor:
        nlCoef = wifisIO.readImgsFromFile(nlFile)[0]
    else:
        nlCoef = None

    wifisIO.createDir('quick_reduction')

    #process science target data
    fluxImg, hdr, obsinfo = procRamp(rampFolder, satCounts=satCounts, bpm=bpm, saveName='quick_reduction/'+rampFolder+'_obs.fits',varFile=varFile, noProc=noProc,nlCoef=nlCoef)
    
    if (skyFolder is not None):
        skyImg, skyHdr, obsinfo = procRamp(skyFolder, satCounts=satCounts, bpm=bpm, saveName='quick_reduction/'+rampFolder+'_sky.fits',varFile=varFile, noProc=noProc,nlCoef=nlCoef)

        #subtract
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            fluxImg -= skyImg

    obs = fluxImg[4:-4, 4:-4]

    print('Finding locations of continuum sources')
    #get collapse of data to identify continuum regions
    if dispAxis==0:
        y = np.nansum(obs,axis=0).astype('float64')
    else:
        y = np.nansum(obs,axis=1).astype('float64')

    if not pixRng is None:
        y = y[pixRng[0]:pixRng[1]]
        
    #get a quick median averaging over 5-pixels to avoid bad/hot pixels
    yClean = np.empty(y.shape)

    for i in range(y.shape[0]):
        xRng = np.arange(7)-3 + i
        xRng = xRng[np.logical_and(xRng >=0, xRng <y.shape[0])]
        yClean[i] = np.nanmedian(y[xRng])
        
    #find noise level
    tmp = np.copy(yClean)

    flr = np.nanmedian(tmp)
    xFlr = np.arange(y.shape[0])

    for i in range(10):
        whr = np.where((tmp-flr) < 3.*np.nanstd(tmp))
        tmp = tmp[whr[0]]
        xFlr = xFlr[whr[0]]
        flr = np.nanmedian(tmp)

    #now find regions with flux > threshold times noise level
    whr = np.where(yClean>=threshold*flr)[0]

    #check to see if any part of image satisfies condition
    #if not, quit
    if len(whr) <1:
        return np.nan
        
    #now step through and find the different continuum regions
    regs = []
    x = np.arange(y.shape[0])

    #go through pixels that are above threshold and find local maxima
    pos = x[whr[0]]

    sliceWidth = y.shape[0]/nSlices
    posNew =0
    
    while pos < whr[-1]:
        #find peak as location where pix value > 5 pixels before and 5 pixels after
        xRng = np.arange(2*nBest+1)-nBest + pos
        #xRng = xRng[np.logical_and(xRng>=0,xRng<y.shape[0])]
        yRng = yClean[xRng]

        #plt.plot(xRng, yRng, 'o')
        #check if current
        #print(xRng, np.all(yRng[nBest] >= yRng), yRng[nBest]>=yRng)
        
        if np.all(yRng[nBest] >= yRng):
            #carry out fitting

            xFit = np.arange(sliceWidth)+pos- sliceWidth/2
            xFit = xFit[xFit>=0]
            xFit = xFit[xFit<yClean.shape[0]]
            if len(regs)>0:
                xFit = xFit[xFit > np.max(regs)]
            xFit = xFit[np.isfinite(yClean[xFit])]
            yFit = yClean[xFit]
            
            try:
                gFit = gaussFit(xFit,yFit,guessWidth)
                
                lim1 = gFit[1]-np.abs(gFit[2])*3.
                if lim1 < 0:
                    lim1 = 0
                    
                #make sure there is no overlap
                if len(regs)>0:
                    if lim1 < np.max(regs):
                        lim1 = np.max(regs)
                    
                lim2 = gFit[1] +np.abs(gFit[2])*3.
                if lim2 > y.shape[0]:
                    lim2 = y.shape[0]

                #print('lim1, lim2', lim1, lim2)
                #make sure that separation makes sense
                #should be less than slice width. but allow for some leeway
                if lim2-lim1 > sliceWidth*1.5 or lim2-lim1 < nBest:
                    lim2 = xFit[-1]
                else:
                    regs.append([int(lim1),int(lim2)])
                    #plt.vlines(lim1, 0, y.max())
                    #plt.vlines(lim2, 0, y.max())
            except(RuntimeError):
                lim2 = xFit[-1]

            posNew = x[whr[np.where(x[whr] > lim2+nBest)]]
            if len(posNew)>0:
                pos = posNew[0]
            else:
                break
        else:
            #change position to pixel with maximum value in given range
            posNew = xRng[np.nanargmax(yRng)]

            if posNew <= pos:
                posNew = x[whr[[np.where(x[whr] >= xRng[-1]+1)]]][0]
                if len(posNew)>0:
                    pos = posNew[0]
                else:
                    break
            else:
                pos = posNew
                
    #now go through and get FWHM along each continuum region
    fwhmLst = []
    centLst = []

    if dispAxis ==0:
        npts = obs.shape[0]
    else:
        npts = obs.shape[1]
        
    for reg in regs:
        centLst.append([])
        fwhmLst.append([])

        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            for i in range(npts):
                xReg = np.arange(reg[0],reg[1])
                if dispAxis == 0:
                    #check to make sure the region is not all NaNs
                    if np.all(~np.isfinite(obs[i,xReg])):
                        fwhm = np.nan
                    else:
                        centLst[-1].append(xReg[np.nanargmax(obs[i,xReg])])
                        mx = np.nanmax(obs[i,xReg])
                        whr = np.where(obs[i,xReg]>=fracLev*mx)[0]
                        fwhm = whr.shape[0]
                else:
                    if np.all(~np.isfinite(obs[i,xReg])):
                        fwhm = np.nan
                    else:
                        centLst[-1].append(xReg[np.nanargmax(obs[xReg,i])])
                        mx = np.nanmax(obs[xReg,i])
                        whr = np.where(obs[xReg,i]>=fracLev*mx)[0]
                        fwhm = whr.shape[0]
                fwhmLst[-1].append(fwhm)

    #now build map
    mapOut = np.empty(obs.shape,dtype='float32')
    mapOut[:] = np.nan

    for cents, fwhm in zip(centLst,fwhmLst):
        for i in range(len(cents)):
            if dispAxis ==0:
                if np.isfinite(cents[i]) and np.isfinite(fwhm[i]):
                    mapOut[i, int(cents[i]-2*(1/(1-fracLev))*fwhm[i]):int(cents[i]+2*(1/(1-fracLev))*fwhm[i])+1] = fwhm[i]
            else:
                if np.isfinite(cents[i]) and np.isfinite(fwhm[i]):
                    mapOut[int(cents[i]-2*(1/(1-fracLev))*fwhm[i]):int(cents[i]+2*(1/(1-fracLev))*fwhm[i])+1,i] = fwhm[i]

    #now plot and save data
    print('Saving results')
    interval=ZScaleInterval()
    clim = interval.get_limits(fwhmLst)

    fig = plt.figure()
    plt.imshow(mapOut, aspect='auto', origin='lower', clim=clim)
    plt.colorbar()
    medPSF = np.nanmedian(fwhmLst)
    plt.title('Median FWHM of PSF is: '+str(medPSF))
    plt.tight_layout()
    plt.savefig('quick_reduction/'+rampFolder+'_obs_psf_map.png')
    if not noPlot:
        plt.show()
    plt.close()
    hdr.set('PSF_LEV',fracLev,'Fractional level at which PSF width determined')
    hdr.set('MED_PSF',medPSF,'Median width of PSF')
    wifisIO.writeFits(mapOut, 'quick_reduction/'+rampFolder+'_obs_psf_map.fits',hdr=hdr)

    return medPSF

def getFocusMeas(rampFolder='', flatFolder='', noProc=False, skyFolder=None, pixRange=None, varFile='',scaling='zscale', colorbar=False, limLevel=0.05):
    """
    
    """

    #initialize variables using configuration file
    varInp = wifisIO.readInputVariables(varFile)
    for var in varInp:
        globals()[var[0]]=var[1]    

    #execute pyOpenCL section here
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = pyCLCompOut

    if len(pyCLCTX)>0:
        os.environ['PYOPENCL_CTX'] = pyCLCTX 

    if os.path.exists(satFile):
        satCounts = wifisIO.readImgsFromFile(satFile)[0]
    else:
        satCounts = None

    if os.path.exists(bpmFile):
        bpm = wifisIO.readImgsFromFile(bpmFile)[0]
    else:
        bpm = None
        
    wifisIO.createDir('quick_reduction')

    #process science data
    if noProc:
        print('Attempting to process science and sky (if exists) ramps without usual processing')

    fluxImg, hdr, obsinfoFile = procRamp(rampFolder, noProc=noProc, satCounts=satCounts, bpm=bpm, saveName='quick_reduction/'+rampFolder+'_obs.fits',varFile=varFile)

    #now process sky, if it exists
       
    if (skyFolder is not None):
        skyImg, hdrSky, skyobsinfo = procRamp(skyFolder,noProc=noProc, satCounts=satCounts, bpm=bpm, saveName='quick_reduction/'+skyFolder+'_sky.fits',varFile=varFile)
        
        #subtract
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            fluxImg -= skyImg

    fluxImg = fluxImg[4:-4, 4:-4]

    #first check if limits already exists
    if os.path.exists('quick_reduction/'+flatFolder+'_limits.fits'):
        limitsFile = 'quick_reduction/'+flatFolder+'_limits.fits'
        limits = wifisIO.readImgsFromFile(limitsFile)[0]
    else:
        flat, hdrFlat, hdrobsinfo = procRamp(flatFolder, noProc=False, satCounts=satCounts, bpm=bpm, saveName='quick_reduction/'+flatFolder+'_flat.fits',varFile=varFile)
        
        print('Getting slice limits')
        limits = slices.findLimits(flat, dispAxis=0, rmRef=True,centGuess=centGuess)
        wifisIO.writeFits(limits, 'quick_reduction/'+flatFolder+'_limits.fits', ask=False)

    if os.path.exists(distMapLimitsFile):
        #get ronchi slice limits
        distLimits = wifisIO.readImgsFromFile(distMapLimitsFile)[0]

        #determine shift
        shft = np.median(limits[1:-1, :] - distLimits[1:-1,:])
    else:
        print(colorama.Fore.RED+'*** WARNING: NO DISTORTION MAP LIMITS PROVIDED. LIMITS ARE DETERMINED ENTIRELY FROM THE FLAT FIELD DATA ***'+colorama.Style.RESET_ALL)
        shft = 0
        distLimits = limits
        
    print('Extracting slices')
    dataSlices = slices.extSlices(fluxImg, distLimits, dispAxis=0, shft=shft)

    print('Finding locations of continuum source')
    #get collapse of data to identify continuum regions
    #and slice with maximum flux
    
    yCol = []
    mx  = []
    for slc in dataSlices:
        yCol.append(np.nansum(slc,axis=1).astype('float64'))
        mx.append(np.nanmax(yCol[-1]))
        
    #get slice with maximum flux
    mxSlc = dataSlices[np.nanargmax(mx)]
    
    #compute average line profile across the spatial axis at 5 different locations
    pixLoc = [100,500,900, 1300,1800]

    widthList = []
    limsList = []
    yList = []
    centList = []
    
    for cent in pixLoc:
        #compute average profile over small window about centre
        y = np.nanmedian(mxSlc[:,cent-3:cent+4], axis=1)
        
        #find noise level
        tmp = np.copy(y)

        flr = np.nanmedian(tmp)
        xFlr = np.arange(y.shape[0])

        for i in range(10):
            whr = np.where((tmp-flr) < 3.*np.nanstd(tmp))
            tmp = tmp[whr[0]]
            xFlr = xFlr[whr[0]]
            flr = np.nanmedian(tmp)

        #now subtract noise level from profile
        y-= flr

        #now find limits where flux reaches limLevel% of maximum flux
        y/=np.nanmax(y)
        centList.append(np.nanargmax(y))

        lims = np.where(y>=limLevel)[0]
        limsList.append([lims[0], lims[-1]])
        widthList.append(lims[-1]-lims[0])
        yList.append(y)

    #now plot and print results
    fig, (ax1, ax2, ax3, ax4,ax5) = plt.subplots(1,5, sharey=True, figsize=(15,5))

    for i in range(len(centList)):
        locals()['ax'+str(i+1)].plot(np.arange(-20,21),yList[i][centList[i]-20:centList[i]+21])
        locals()['ax'+str(i+1)].vlines(np.array(limsList[i])-centList[i], 0,1, 'r', linestyle='--')
        locals()['ax'+str(i+1)].set_title('Pixel '+ str(pixLoc[i])+ ', width '+str(widthList[i]))
        #plt.title('Width = ' + str(widthList[i]))
        
    plt.tight_layout()
    plt.savefig('quick_reduction/PSF_cross_section.png')
    plt.show()
        
        
    return
