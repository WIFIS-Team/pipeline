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
from astropy.visualization import ZScaleInterval
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


def procScienceData(rampFolder='', flatFolder='', noProc=False, skyFolder=None, pixRange=None, varFile='',scaling='zscale'):
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

    #read in data

    if noProc:
        print('Attempting to extracting ramp image without processing')
        #check file type
        #CDS
        if os.path.exists(rootFolder+'/CDSReference/'+rampFolder):
            folderType = '/CDSReference/'
            fluxImg, hdr = wifisIO.readImgsFromFile(rootFolder + folderType + rampFolder+'/Result/CDSResult.fits')
            UTR = False
            noProcObs = True
        #Fowler
        elif os.path.exists(rootFolder+'/FSRamp/'+rampFolder):
            folderType = '/FSRamp/'
            UTR =  False
            if os.path.exists(rootFolder + folderType + rampFolder+'/Result/CDSResult.fits'):
                fluxImg, hdr = wifisIO.readImgsFromFile(rootFolder + folderType + rampFolder+'/Result/CDSResult.fits')
                noProcObs = True
            else:
                noProcObs = False

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
            noProcObs=True
        else:
            raise Warning('*** Ramp folder ' + rampFolder + ' does not exist ***')
    else:
        noProcObs =False

    if not noProcObs:
        print('Processing ramp')

        fluxImg,sigImg,satFrame, hdr =processRamp.auto(rampFolder, rootFolder, 'quick_reduction/'+rampFolder+'_obs.fits', satCounts, None, bpm, nRows=0, rowSplit=nRowSplit, satSplit=nSatSplit,nlSplit=nlSplit, combSplit=nCombSplit, bpmCorRng=bpmCorRng,saveAll=False,ignoreBPM=True,avgAll=True, bpmFile=bpmFile, satFile=satFile,nChannel=0)

    if noProcObs:
        #get obsinfo file to fill in hdr info
        obsinfoFile = rootFolder + folderType + rampFolder+'/obsinfo.dat'

    #now process sky, if it exists
       
    if (skyFolder is not None):
        if noProc:
            print('Attempting to extract sky image without processing')
            
            #check file type
            #CDS
            if os.path.exists(rootFolder+'/CDSReference/'+skyFolder):
                folderType = '/CDSReference/'
                skyImg, hdr = wifisIO.readImgsFromFile(rootFolder + folderType + skyFolder+'/Result/CDSResult.fits')
                UTR = False
                noProcSky = True
                #Fowler
            elif os.path.exists(rootFolder+'/FSRamp/'+skyFolder):
                folderType = '/FSRamp/'
                UTR =  False
                if os.path.exists(rootFolder + folderType + skyFolder+'/Result/CDSResult.fits'):
                    skyImg, hdr = wifisIO.readImgsFromFile(rootFolder + folderType + skyFolder+'/Result/CDSResult.fits')
                    noProcSky=True
                else:
                    noProcSky = False
            elif os.path.exists(rootFolder + '/UpTheRamp/'+skyFolder):
                UTR = True
                folderType = '/UpTheRamp/'

                lst = glob.glob(rootFolder+folderType+skyFolder + '/H2*fits')
                lst = sorted_nicely(lst)
            
                img1,hdr1 = wifisIO.readImgsFromFile(lst[-1])
                img0,hdr0 = wifisIO.readImgsFromFile(lst[0])
                t0 = hdr0['INTTIME']
                t1 = hdr1['INTTIME']
        
                skyImg = ((img1-img0)/(t1-t0)).astype('float32')
                noProcSky = True
            else:
                raise Warning('*** sky Ramp folder ' + skyFolder + ' does not exist ***')
        else:
            noProcSky = False

        if not noProcSky:
            'Processing sky ramp'
            skyImg, skySig, skySat, hdrSky =processRamp.auto(skyFolder, rootFolder, 'quick_reduction/'+skyFolder+'_sky.fits', satCounts, None, bpm, nRows=nRowsAvg, rowSplit=nRowSplit, satSplit=nSatSplit,nlSplit=nlSplit, combSplit=nCombSplit, bpmCorRng=bpmCorRng,saveAll=False,ignoreBPM=True,avgAll=True, bpmFile=bpmFile,satFile=satFile)

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
        if noProc:
            print('Attempting to extract flatfield image without processing')
            #check file type
            #CDS
            if os.path.exists(rootFolder+'/CDSReference/'+flatFolder):
                folderType = '/CDSReference/'
                flat = wifisIO.readImgsFromFile(rootFolder + folderType + flatFolder+'/Result/CDSResult.fits')[0]
                UTR = False
                noProcFlat = True
            #Fowler
            elif os.path.exists(rootFolder+'/FSRamp/'+ flatFolder):
                folderType = '/FSRamp/'
                UTR =  False
                if os.path.exists(rootFolder + folderType + flatFolder+'/Result/CDSResult.fits'):
                    flat, hdr = wifisIO.readImgsFromFile(rootFolder + folderType + flatFolder+'/Result/CDSResult.fits')
                else:
                    noProcFlat = False
            elif os.path.exists(rootFolder + '/UpTheRamp/'+flatFolder):
                folderType = '/UpTheRamp/'
                UTR = True

                lst = glob.glob(rootFolder+folderType+flatFolder + '/H2*fits')
                lst = sorted_nicely(lst)
            
                img1,hdr1 = wifisIO.readImgsFromFile(lst[-1])
                img0,hdr0 = wifisIO.readImgsFromFile(lst[0])
                t0 = hdr0['INTTIME']
                t1 = hdr1['INTTIME']
        
                flat = ((img1-img0)/(t1-t0)).astype('float32')
                noProcFlat = True
            else:
                raise Warning('*** flat folder ' + flatFolder + ' does not exist ***')
        else:
            noProcFlat = False

        if not noProcFlat:
            'Processing flatfield ramp'
            flat, flatSig, flatSat, hdrFlat =processRamp.auto(flatFolder, rootFolder, 'quick_reduction/'+flatFolder+'_flat.fits', satCounts, None, bpm, nRows=0, rowSplit=nRowSplit, satSplit=nSatSplit,nlSplit=nlSplit, combSplit=nCombSplit, bpmCorRng=flatbpmCorRng,saveAll=False,ignoreBPM=True,avgAll=True, bpmFile=bpmFile,satFile=satFile,nChannel=0)
            

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

    if noProcObs:
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

def procArcData(waveFolder, flatFolder, hband=False, colorbarLims = None, varFile=''):
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

    wifisIO.createDir('quick_reduction')

    #read in previous results and template
    template = wifisIO.readImgsFromFile(waveTempFile)[0]
    prevResults = wifisIO.readPickle(waveTempResultsFile)
    prevSol = prevResults[5]
    distMap = wifisIO.readImgsFromFile(distMapFile)[0]
    spatGridProps = wifisIO.readTable(spatGridPropsFile)
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
        print('Processing arc file '+ waveFolder)
        #check the type of ramp
        if (os.path.exists(rootFolder + '/CDSReference/'+waveFolder+'/Result/CDSResult.fits')):
            #CDS image
            wave = wifisIO.readImgsFromFile(rootFolder + '/CDSReference/'+waveFolder+'/Result/CDSResult.fits')[0]
            wave = wave #trim off reference pixels
        elif os.path.exists(rootFolder+'/FSRamp/'+waveFolder):
            folderType = '/FSRamp/'
            UTR =  False
            if os.path.exists(rootFolder + folderType + waveFolder+'/Result/CDSResult.fits'):
                wave, hdr = wifisIO.readImgsFromFile(rootFolder + folderType + waveFolder+'/Result/CDSResult.fits')
            else:
                wave, waveSig, waveSat,hdr = processRamp.auto(waveFolder, rootFolder, 'quick_reduction/'+waveFolder+'_wave.fits', satCounts, None, bpm, nRows=0, rowSplit=nRowSplit, satSplit=nSatSplit,nlSplit=nlSplit, combSplit=nCombSplit, bpmCorRng=bpmCorRng,saveAll=False,ignoreBPM=True,avgAll=True, bpmFile=bpmFile,satFile=satFile,nChannel=0)

        elif os.path.exists(rootFolder+'/UpTheRamp/'+waveFolder):
            wave, waveSig, waveSat,hdr = processRamp.auto(waveFolder, rootFolder, 'quick_reduction/'+waveFolder+'_wave.fits', satCounts, None, bpm, nRows=0, rowSplit=nRowSplit, satSplit=nSatSplit,nlSplit=nlSplit, combSplit=nCombSplit, bpmCorRng=bpmCorRng,saveAll=False,ignoreBPM=True,avgAll=True, bpmFile=bpmFile,satFile=satFile,nChannel=0)
        else:
            raise Warning('*** Wave folder ' + waveFolder + ' does not exist ***')

    
        if (os.path.exists('quick_reduction/'+flatFolder+'_flat_limits.fits') and os.path.exists('quick_reduction/'+flatFolder+'_flat_slices.fits')):
            limits, limitsHdr = wifisIO.readImgsFromFile('quick_reduction/'+flatFolder+'_flat_limits.fits')
            flatSlices,flatHdr = wifisIO.readImgsFromFile('quick_reduction/'+flatFolder+'_flat_slices.fits')
            flatSlices=flatSlices[:18]
            shft = limitsHdr['LIMSHIFT']
        else:
            print('Processing flat file')
            #check the type of raw data, only assumes CDS or up-the-ramp
            if (os.path.exists(rootFolder + '/CDSReference/'+flatFolder+'/Result/CDSResult.fits')):
                #CDS image
                flat, flatHdr = wifisIO.readImgsFromFile(rootFolder + '/CDSReference/'+flatFolder+'/Result/CDSResult.fits')
            elif os.path.exists(rootFolder+'/FSRamp/'+flatFolder):
                folderType = '/FSRamp/'
                UTR =  False
                if os.path.exists(rootFolder + folderType + flatFolder+'/Result/CDSResult.fits'):
                    flat, flatHdr = wifisIO.readImgsFromFile(rootFolder + folderType + flatFolder+'/Result/CDSResult.fits')
                else:
                    flat, flatSig, flatSat,flatHdr = processRamp.auto(flatFolder, rootFolder, 'quick_reduction/'+flatFolder+'_flat.fits', satCounts, None, bpm, nRows=0, rowSplit=nRowSplit, satSplit=nSatSplit,nlSplit=nlSplit, combSplit=nCombSplit, bpmCorRng=bpmCorRng,saveAll=False,ignoreBPM=True,avgAll=True, bpmFile=bpmFile,satFile=satFile,nChannel=0)

            elif os.path.exists(rootFolder + '/UpTheRamp/'+flatFolder):
                flat, flatSig, flatSat,flatHdr = processRamp.auto(flatFolder, rootFolder, 'quick_reduction/'+flatFolder+'_flat.fits', satCounts, None, bpm, nRows=0, rowSplit=nRowSplit, satSplit=nSatSplit,nlSplit=nlSplit, combSplit=nCombSplit, bpmCorRng=bpmCorRng,saveAll=False,ignoreBPM=True,avgAll=True, bpmFile=bpmFile,satFile=satFile,nChannel=0)
            else:
                raise Warning('*** Flat folder ' + flatFolder + ' does not exist ***')

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
            waveSlices = slices.extSlices(wave[4:-4,4:-4], limits, dispAxis=0, shft=shft)

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

        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            
            result = waveSol.getWaveSol(waveCor, template, atlasFile,mxOrder, prevSol, winRng=waveWinRng, mxCcor=waveMxCcor, weights=False, buildSol=False, sigmaClip=sigmaClip, allowLower=False, lngthConstraint=True, MP=True, adjustFitWin=True, sigmaLimit=sigmaLimit, allowSearch=False, sigmaClipRounds=sigmaClipRounds)        
       
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
        plt.show()
        plt.close()

    return

def procRonchiData(ronchiFolder, flatFolder, hband=False, colorbarLims=None, mxWidth=4, varFile='',noPlot=False):
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
        flatSlices = wifisIO.readImgsFromFile('quick_reduction/'+flatFolder+'_flat_slices.fits')[0][:18]
        flatLst = wifisIO.readImgsFromFile('quick_reduction/'+flatFolder+'_flat_slices_norm.fits')
        flatNorm = flatLst[0][:18]
        flatHdr = flatLst[1]
    else:
        print('processing flat ' + flatFolder)

        flat, sigmaImg, satFrame, flatHdr = processRamp.auto(flatFolder, rootFolder,'quick_reduction/'+flatFolder+'_flat.fits', satCounts, None, BPM,nChannel=0, rowSplit=nRowSplit, satSplit=nSatSplit, nlSplit=nlSplit, combSplit=nCombSplit, crReject=False, bpmCorRng=flatbpmCorRng, saveAll=False,nRows=0)
    
        #find limits
        print('finding limits and extracting flat slices')
        limits = slices.findLimits(flat, dispAxis=dispAxis, limSmth=flatLimSmth, rmRef=True)

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

                for j in range(len(ronchiTraces[i])):
                    plt.plot(ronchiTraces[i][j,:], 'r--')
                plt.title('Slices #'+str(i))
                plt.tight_layout()
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

        if not noPlot:
            plt.show()
        plt.savefig('quick_reduction/'+ronchiFolder+'_ronchi_amp_map.png',dpi=300)
        plt.close()        
        
        print('saving results')
        #write results!
        wifisIO.writeFits(ronchiTraces, 'quick_reduction/'+ronchiFolder+'_ronchi_traces.fits', ask=False)
        wifisIO.writeFits(ampMap, 'quick_reduction/'+ronchiFolder+'_ronchi_amp_map.fits', ask=False)
    else:
        print('Ronchi ' + ronchiFolder + ' already processed')
    return
