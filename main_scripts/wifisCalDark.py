"""

Calibrate dark images

Input: 
- BPM from linearity measurents (optional, will be merged) *** NOT IMPLEMENTED YET ***

Requires:
- specifying the ascii list containing folder names to process and from which to create a master dark frame

Produces: 
- master dark image
- bad pixel map (merged) *** STILL TO DO ***


TODO: ADD LOGFILE WRITING
"""

import matplotlib
matplotlib.use('gtkagg')
import numpy as np
import time
import matplotlib.pyplot as plt
import wifisGetSatInfo as satInfo
import wifisNLCor as NLCor
import wifisRefCor as refCor
import os
import wifisIO 
import wifisCombineData as combData
import warnings
import wifisUncertainties
import astropy.io.fits as fits
import wifisBadPixels as badPixels
from astropy.visualization import ZScaleInterval
import copy

try:
    import colorama
    colorama.init()
    colorFlag =True
except:
    colorFlag = False
    
t0 = time.time()

#*****************************************************************************
#************************** user input ******************************
varFile = 'wifisConfig.inp'

#initialize all variables here.
#DO NOT CHANGE VALUES HERE, EDIT THE 'variables.inp' FILE, WHICH OVERWRITES VALUES HERE
darkLstFile = 'dark.lst'
rootFolder = '/data/WIFIS/H2RG-G17084-ASIC-08-319'
pipelineFolder = '/data/pipeline/'
nlFile = '/home/jason/wifis/data/june_cals/processed/master_detLin_NLCoeff.fits' # the non-linearity correction coefficients file        
satFile = '/home/jason/wifis/data/non-linearity/july/processed/master_detLin_satCounts.fits' # the saturation limits file
bpmFile = '//home/jason/wifis/data/non-linearity/july/processed/master_detLin_BPM.fits'
nRowsAvg =  4
nChannel = 32
nCombSplit=32
nSatSplit=32
getBadPixDark = True
getBadPixRON = True

print('Reading input variables from file ' + varFile)
#logfile.write('Reading input variables from file ' + varFile)
varInp = wifisIO.readInputVariables(varFile)
for var in varInp:
    locals()[var[0]]=var[1]

#VARIABLE OVERRIDES - USEFUL FOR STICKING WITH SAME INPUT SCRIPT, BUT FOR DIFFERENT USES
nRowSplit=5
#*****************************************************************************

#execute pyOpenCL section here
os.environ['PYOPENCL_COMPILER_OUTPUT'] = pyCLCompOut 
os.environ['PYOPENCL_CTX'] = pyCLCTX 

interval=ZScaleInterval()

#first check if required input exists
if not os.path.exists(nlFile):
    if colorFlag:
        print(colorama.Fore.RED+'*** WARNING: No non-linearity corrections provided. Skipping non-linearity corrections ***'+colorama.Style.RESET_ALL)
    else:
        print('*** WARNING: No non-linearity corrections provided. Skipping non-linearity corrections ***')

if not os.path.exists(satFile):
        print(colorama.Fore.RED+'*** WARNING: No saturation info provided. Using all frames ***'+colorama.Style.RESET_ALL)

#create processed directory, in case it doesn't exist
wifisIO.createDir('processed')
wifisIO.createDir('quality_control')

#read file list
lst= wifisIO.readAsciiList(darkLstFile)

if lst.ndim == 0:
    lst = np.asarray([lst])

masterSave = 'processed/master_dark'

#first check if master dark exists
if(os.path.exists(masterSave+'.fits')):
    cont = wifisIO.userInput('Master dark file already exists, do you want to replace (y/n)?')
    if (cont.lower() == 'y'):
        contProc = True
    else:
        contProc = False
else:
    contProc = True
    
if (contProc):
    
    procDark = []
    procSigma = []
    procSatFrame = []
    procVarImg = []
    nRampsLst = []
    #go through list and process each file individually

    for folder in lst:

        folder = folder

        #check if folder contains multiple ramps
        nRamps = wifisIO.getNumRamps(folder,rootFolder=rootFolder)
        nRampsLst.append(nRamps)
        
        for rampNum in range(1,nRamps+1):
            if nRamps > 1:
                savename = 'processed/'+folder+'_R'+'{:02d}'.format(rampNum)
            else:
                savename = 'processed/'+folder
                
            if(os.path.exists(savename+'_dark.fits')):
                cont = 'n'
                cont = wifisIO.userInput('Processed dark file already exists for ' +folder+', do you want to continue processing (y/n)?')
                if (not cont.lower() == 'y'):
                    print('Reading image '+savename+'_dark.fits instead')
                    contProc2 = False
                    #read in file instead
                    fluxImg, sigma, satFrame = wifisIO.readImgsFromFile(savename+'_dark.fits')[0]
                    ron = wifisIO.readImgsFromFile(savename+'_dark_RON.fits')[0]
                    varImg = ron**2
                else:
                    contProc2 = True
            else:
                contProc2 = True
        
            if (contProc2):
                if nRamps >1:
                    print('*** Working on folder '+ folder+ ', ramp '+ '{:02d}'.format(rampNum)+' ***')
                else:
                    print('*** Working on folder '+ folder+ ' ***')
 
                print('Reading data into cube')

                #Read in data
                if os.path.exists(rootFolder + '/UpTheRamp/'+folder):
                    folderType = '/UpTheRamp/'
                    UTR = True
                    data, inttime, hdr = wifisIO.readRampFromFolder(rootFolder+folderType + folder, rampNum=rampNum)
                elif os.path.exists(rootFolder + '/CDSReference/' + folder):
                    folderType = '/CDSReference/'
                    UTR = False
                    data, inttime, hdr = wifisIO.readRampFromFolder(rootFolder+folderType + folder, rampNum=rampNum)
                elif os.path.exists(rootFolder + '/FSRamp/' + folder):
                    folderType = '/FSRamp/'
                    UTR = False
                    data, inttime, hdr = wifisIO.readRampFromFolder(rootFolder+folderType+folder, rampNum=rampNum)
                else:
                    raise Warning('*** Ramp folder ' + folder + ' does not exist ***')


                data = data.astype('float32')*1.
                hdr.add_history('Processed using WIFIS pyPline')
                hdr.add_history('on '+time.strftime("%c"))
                
                #******************************************************************************
                #Correct data for reference pixels
                print("Subtracting reference pixel channel bias")
                refCor.channelCL(data, nChannel)
                hdr.add_history('Channel reference pixel corrections applied using '+ str(nChannel) +' channels')
                
                print("Subtracting reference pixel row bias")
                refCor.rowCL(data, nRowsAvg,nRowSplit) 
                hdr.add_history('Row reference pixel corrections applied using '+ str(int(nRowsAvg+1))+ ' pixels')
                
                #******************************************************************************
                #find if any pixels are saturated to avoid use in future calculations

                if os.path.exists(satFile):
                    satCounts = wifisIO.readImgsFromFile(satFile)[0]
                    satFrame = satInfo.getSatFrameCL(data, satCounts,nSatSplit, ignoreRefPix=True)
                    hdr.add_history('Saturation levels determined from file:')
                    hdr.add_history(satFile)

                else:
                    satFrame = np.empty((data.shape[0],data.shape[1]),dtype='uint16')
                    satFrame[:] = data.shape[2]
                    hdr.add_history('No saturation levels determined. Using all ramp frames')

                #******************************************************************************
                #apply non-linearity correction
                print("Correcting for non-linearity")

                if os.path.exists(nlFile):
                    nlCoeff = wifisIO.readImgsFromFile(nlFile)[0]
                    NLCor.applyNLCorCL(data, nlCoeff, nlSplit)
                    hdr.add_history('Non-linearity corrections applied using file:')
                    hdr.add_history(nlFile)
                    
                #******************************************************************************
                #Combine data cube into single image
                fluxImg, zpntImg, varImg = combData.upTheRampCL(inttime, data, satFrame, nCombSplit)
                #fluxImg = combData.upTheRampCRRejectCL(inttime, data, satFrame, nCombSplit)
                hdr.add_history('Flux determined from linear regression')

                #reset cube to reduce memory impact 
                data = 0
 
                #get uncertainties for each pixel
                if UTR:
                    sigma = wifisUncertainties.compUTR(inttime, fluxImg, satFrame)
                else:
                    sigma = wifisUncertainties.compFowler(inttime, fluxImg, satFrame)

                #plot and determine quality control stuff here
                medDark = np.nanmedian(fluxImg)
                stdDark = np.nanstd(fluxImg)

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    for k in range(4):
                        stdDark = np.nanstd(fluxImg[np.logical_and(fluxImg >= medDark-stdDark, fluxImg <= medDark+stdDark)])
                    lims = interval.get_limits(fluxImg.flatten())
                    fig = plt.figure()
                    plt.hist(fluxImg.flatten(), bins=100, range=lims)#[medDark-5*stdDark, medDark+5*stdDark])
                    plt.xlabel('Dark current')
                    plt.ylabel('# of pixels')
                    plt.title('Median dark current of ' + '{:10.2e}'.format(medDark))
                    if nRamps > 1:
                        plt.savefig('quality_control/'+folder+'_R'+'{:02d}'.format(rampNum)+'_dark_hist.png',dpi=300)
                    else:
                        plt.savefig('quality_control/'+folder+'_dark_hist.png',dpi=300)
                
                    plt.close()
                
                #compute read-out-noise
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    ron = np.sqrt(varImg)
                    medRon = np.nanmedian(ron)
                    stdRon = np.nanstd(ron)

                    for k in range(4):
                        stdRon = np.nanstd(ron[np.logical_and(ron >= medRon-stdRon, ron <= medRon+stdRon)])

                
                fig = plt.figure()
                lims = interval.get_limits(ron.flatten())
                plt.hist(ron.flatten(),range=lims,bins=100)#np.round([medRon-5.*stdRon,medRon+5.*stdRon]),bins=100)
                plt.title('Median RON of ' + '{: e}'.format(medRon))
                plt.xlabel('counts')
                plt.ylabel('# of pixels')
                if nRamps>1:
                    plt.savefig('quality_control/'+folder+'_R{:02d}'.format(rampNum)+'_dark_ron_hist.png',dpi=300)
                else:
                    plt.savefig('quality_control/'+folder+'_dark_ron_hist.png',dpi=300)

                plt.close()
                hdr.set('QC_RON', medRon, 'Median readout noise, in counts')
                hdr.set('QC_STDRN',stdRon, 'Standard deviation of readout noise, in counts')
                hdr.add_comment('File contains the estimated readout noise per pixel')
                wifisIO.writeFits(ron.astype('float32'), savename+'_dark_RON.fits',ask=False, hdr=hdr)

                hdrTmp = hdr[::-1]
                hdrTmp.remove('COMMENT')
                hdr = hdrTmp[::-1]
                
                #add additional header information here
                hdr.set('QC_DARK',medDark,'Median dark current of entire array, in counts')
                hdr.set('QC_STDDK',stdDark, 'Standard deviation of dark current, in counts')
                hdr.add_comment('File contains the dark, uncertainty, and saturation frames as multi-extensions')
                wifisIO.writeFits([fluxImg,sigma, satFrame],savename+'_dark.fits',hdr=hdr)

            procDark.append(fluxImg)
            procSigma.append(sigma)
            procSatFrame.append(satFrame)
            procVarImg.append(varImg)

    print('Getting master images')
    #logfile.write('Getting master images\n')
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        #now combine all dark images into master dark, propagating uncertainties as needed
        masterDark, masterSigma  = wifisUncertainties.compMedian(np.array(procDark),np.array(procSigma), axis=0)
    
        #combine satFrame
        masterSatFrame = np.median(np.array(procSatFrame), axis=0).astype('int')

        #combine variance frame
        masterVarImg = np.nanmedian(np.array(procVarImg), axis=0)
    
    #******************************************************************************
    #******************************************************************************

    #add/modify header information here
    hdr = fits.Header()
    hdr.add_history('Processed using WIFIS pyPline')
    hdr.add_history('on '+time.strftime("%c"))
    hdr.add_history('This file was constructed from the median average of the following:')

    for i in range(len(lst)):
        hdr.add_history(lst[i] + ' with ' +  str(nRampsLst[i]) + ' ramps;')

    print('plotting quality control figures')
    #logfile.write('plotting quality control figures\n')
    
    #plot quality control stuff here
    medDark = np.nanmedian(masterDark)
    stdDark = np.nanstd(masterDark)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        for k in range(4):
            medDark = np.nanmedian(masterDark[np.logical_and(masterDark >= medDark-stdDark, masterDark <= medDark+stdDark)])
            stdDark = np.nanstd(masterDark[np.logical_and(masterDark >= medDark-stdDark, masterDark <= medDark+stdDark)])

    fig = plt.figure()
    lims = interval.get_limits(fluxImg.flatten())
    plt.hist(fluxImg.flatten(), bins=100, range=lims)#[medDark-5*stdDark, medDark+5*stdDark])
    plt.xlabel('Dark current')
    plt.ylabel('# of pixels')
    plt.title('Median dark current of ' + '{:10.2e}'.format(medDark))
    plt.savefig('quality_control/master_dark_hist.png',dpi=300)
    plt.close()
            
    #compute read-out-noise
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        ron = np.sqrt(masterVarImg)
        medRon = np.nanmedian(ron)
        stdRon = np.nanstd(ron)

        for k in range(4):
            medRon = np.nanmedian(ron[np.logical_and(ron >= medRon-stdRon, ron <= medRon+stdRon)])
            stdRon = np.nanstd(ron[np.logical_and(ron >= medRon-stdRon, ron <= medRon+stdRon)])

    fig = plt.figure()
    lims = interval.get_limits(ron.flatten())
    plt.hist(ron.flatten(),range=lims,bins=100)#[medRon-5.*stdRon,medRon+5.*stdRon],bins=100)
    plt.title('Median RON of ' + '{: e}'.format(medRon))
    plt.xlabel('Counts')
    plt.ylabel('# of pixels')
    plt.savefig('quality_control/'+folder+'_ron_hist.png',dpi=300)
    plt.close()

    hdr.set('QC_RON', medRon, 'Median readout noise, in counts')
    hdr.set('QC_STDRN',stdRon, 'Standard deviation of readout noise, in counts')
    hdr.add_comment('File contains the estimated readout noise per pixel')
    wifisIO.writeFits(ron, masterSave +'_RON.fits',ask=False,hdr=hdr)

    hdrTmp = hdr[::-1]
    hdrTmp.remove('COMMENT')
    hdr = hdrTmp[::-1]

    hdr.add_comment('File contains the dark, uncertainty, and saturation frames as multi-extensions')

    #add additional header information here
    hdr.set('QC_DARK',medDark,'Median dark current of entire array, in counts')
    hdr.set('QC_STDDK',stdDark, 'Standard deviation of dark current, in counts')
               
    #save file
    wifisIO.writeFits([masterDark.astype('float32'), masterSigma.astype('float32'), masterSatFrame.astype('float32')], masterSave+'.fits', ask=False,hdr=hdr)

    hdrTmp = hdr[::-1]
    hdrTmp.remove('COMMENT')
    hdrTmp.remove('COMMENT')
    hdr = hdrTmp[::-1]

    #determine bad pixels from master dark and master RON
    if getBadPixDark:
        if bpmFile is not None and os.path.exists(bpmFile):
                cont = 'n'
                cont = wifisIO.userInput('Bad pixel mask ' + bpmFile + ' already exists, do you want to update to reflect bad dark current levels and high readout noise (y/n)?')

                if (cont.lower() == 'y'):
                        print('Updating bad pixel mask')
                        bpm, bpmHdr = wifisIO.readImgsFromFile(bpmFile)
                        with warnings.catch_warnings():
                                warnings.simplefilter('ignore', RuntimeWarning)
                                bpm, bpmHdr = badPixels.getBadPixelsFromDark(masterDark,bpmHdr, saveFile='quality_control/master_dark',darkFile=masterSave+'.fits',BPM=bpm)

                                if getBadPixRON:
                                    bpm, bpmHdr = badPixels.getBadPixelsFromRON(ron, bpmHdr, saveFile='quality_control/master_ron',ronFile=masterSave+'_ron.fits',BPM=bpm)
                                
                        wifisIO.writeFits(bpm,masterSave+'_BPM.fits',hdr=bpmHdr,ask=False)
        else:
                print('Creating bad pixel mask')
                bpm,hdr = badPixels.getBadPixelsFromDark(masterDark,hdr,saveFile='quality_control/master_dark', darkFile=masterSave+'.fits',BPM=None)
                if getBadPixRON:
                    bpm, bpmHdr = badPixels.getBadPixelsFromRON(ron, bpmHdr, saveFile='quality_control/master_ron',ronFile=masterSave+'_ron.fits',BPM=bpm)
                hdr.add_comment('File contains the bad pixel mask')
                wifisIO.writeFits(bpm,masterSave+'_BPM.fits' ,hdr=hdr,ask=False)


else:
     print('No processing necessary')   

print ("Total time to run entire script: ",time.time()-t0)
