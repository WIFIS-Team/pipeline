"""

Calibrate dark images

Input: 
- BPM from linearity measurents (optional, will be merged) *** NOT IMPLEMENTED YET ***

Requires:
- specifying the ascii list containing folder names to process and from which to create a master dark frame

Produces: 
- master dark image
- bad pixel map (merged) *** STILL TO DO ***

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

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0' # Used to show compile errors for debugging, can be removed
os.environ['PYOPENCL_CTX'] = '1' # Used to specify which OpenCL device to target. Should be uncommented and pointed to correct device to avoid future interactive requests

t0 = time.time()

#*****************************************************************************
#************************** Required user input ******************************
fileList = 'dark.lst'
rootFolder = '/data/WIFIS/H2RG-G17084-ASIC-08-319'
pipelineFolder = '/data/pipeline/'

nlFile = pipelineFolder+'external_data/master_detLin_NLCoeff.fits' # the non-linearity correction coefficients file        
satFile = pipelineFolder+'external_data/master_detLin_satCounts.fits' # the saturation limits file
bpmFile = pipelineFolder+'external_data/bpm.fits'

nRowSplit = 5
nRowAvg =  4
#*****************************************************************************

#first check if required input exists
if not os.path.exists(nlFile):
        warnings.warn('*** No non-linearity corrections provided. Skipping non-linearity corrections ***')
if not os.path.exists(satFile):
        warnings.warn('*** No saturation info provided. Using all frames ***')

#create processed directory, in case it doesn't exist
wifisIO.createDir('processed')
wifisIO.createDir('quality_control')

#read file list
lst= wifisIO.readAsciiList(fileList)

if lst.ndim == 0:
    lst = [lst]

masterSave = 'processed/master_dark.fits'

#first check if master dark exists
if(os.path.exists(masterSave)):
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

    #go through list and process each file individually

    for folder in lst:

        folder = folder.tostring()

        #check if folder contains multiple ramps
        nRamps = wifisIO.getNumRamps(folder,rootFolder=rootFolder)

        for rampNum in range(1,nRamps+1):
            if nRamps > 1:
                savename = 'processed/'+folder+'_R'+'{:02d}'.format(rampNum)
            else:
                savename = 'processed/'+folder
                
            if(os.path.exists(savename+'_dark.fits')):
                cont = 'n'
                cont = wifisIO.userInput('Processed dark file already exists for ' +folder+', do you want to continue processing (y/n)?')
                if (not cont.lower() == 'y'):
                    print('Reading image'+savename+'_dark.fits instead')
                    contProc2 = False
                    #read in file instead
                    fluxImg, sigma, satFrame = wifisIO.readImgsFromFile(savename+'_dark.fits')[0]
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
                #******************************************************************************
                #Correct data for reference pixels
                print("Subtracting reference pixel channel bias")
                refCor.channelCL(data, 32)
                print("Subtracting reference pixel row bias")
                refCor.rowCL(data, nRowAvg,nRowSplit) 
        
                #******************************************************************************
                #find if any pixels are saturated to avoid use in future calculations

                if os.path.exists(satFile):
                    satCounts = wifisIO.readImgsFromFile(satFile)[0]
                    satFrame = satInfo.getSatFrameCL(data, satCounts,32, ignoreRefPix=True)
                else:
                    satFrame = np.empty((data.shape[0],data.shape[1]),dtype='int')
                    satFrame[:] = data.shape[2]
                    
                #******************************************************************************
                #apply non-linearity correction
                print("Correcting for non-linearity")

                if os.path.exists(nlFile):
                    nlCoeff = wifisIO.readImgsFromFile(nlFile)[0]
                    NLCor.applyNLCorCL(data, nlCoeff, 32)
                
                #******************************************************************************
                #Combine data cube into single image
                fluxImg, zpntImg, varImg = combData.upTheRampCL(inttime, data, satFrame, 32)
                #fluxImg = combData.upTheRampCRRejectCL(inttime, data, satFrame, 32)
                
                #reset cube to reduce memory impact 
                data = 0
 
                #get uncertainties for each pixel
                if UTR:
                    sigma = wifisUncertainties.compUTR(inttime, fluxImg, satFrame)
                else:
                    sigma = wifisUncertainties.compFowler(inttime, fluxImg, satFrame)

                #add additional header information here

                wifisIO.writeFits([fluxImg, sigma, satFrame], savename+'_dark.fits', hdr=hdr)

                #plot quality control stuff here
                medDark = np.nanmedian(fluxImg)
                stdDark = np.nanstd(fluxImg)

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    for k in range(4):
                        stdDark = np.nanstd(fluxImg[np.logical_and(fluxImg >= medDark-stdDark, fluxImg <= medDark+stdDark)])

                    fig = plt.figure()
                    plt.hist(fluxImg.flatten(), bins=100, range=[medDark-5*stdDark, medDark+5*stdDark])
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
                    plt.hist(ron.flatten(),range=[medRon-5.*stdRon,medRon+5.*stdRon],bins=100)
                    plt.title('Median RON of ' + '{: e}'.format(medRon))
                    plt.xlabel('counts')
                    plt.ylabel('# of pixels')
                    if nRamps>1:
                        plt.savefig('quality_control/'+folder+'_R{:02d}'.format(rampNum)+'_dark_ron_hist.png',dpi=300)
                    else:
                        plt.savefig('quality_control/'+folder+'_dark_ron_hist.png',dpi=300)

                    plt.close()
                    wifisIO.writeFits(ron, savename+'_dark_RON.fits',ask=False)

            
            procDark.append(fluxImg)
            procSigma.append(sigma)
            procSatFrame.append(satFrame)
            procVarImg.append(varImg)
        
    #now combine all dark images into master dark, propagating uncertainties as needed
    masterDark, masterSigma  = wifisUncertainties.compMedian(np.array(procDark),np.array(procSigma), axis=0)
    
    #combine satFrame
    masterSatFrame = np.median(np.array(procSatFrame), axis=0).astype('int')

    #combine variance frame
    masterVarImg = np.nansum(np.array(procVarImg), axis=0)
    
    #******************************************************************************
    #******************************************************************************

    #add/modify header information here

    #save file
    wifisIO.writeFits([masterDark, masterSigma, masterSatFrame], masterSave, ask=False)

    #plot quality control stuff here
    medDark = np.nanmedian(masterDark)
    stdDark = np.nanstd(masterDark)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        for k in range(4):
            medDark = np.nanmedian(masterDark[np.logical_and(masterDark >= medDark-stdDark, masterDark <= medDark+stdDark)])
            stdDark = np.nanstd(masterDark[np.logical_and(masterDark >= medDark-stdDark, masterDark <= medDark+stdDark)])

    fig = plt.figure()
    plt.hist(fluxImg.flatten(), bins=100, range=[medDark-5*stdDark, medDark+5*stdDark])
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
    plt.hist(ron.flatten(),range=[medRon-5.*stdRon,medRon+5.*stdRon],bins=100)
    plt.title('Median RON of ' + '{: e}'.format(medRon))
    plt.xlabel('Counts')
    plt.ylabel('# of pixels')
    plt.savefig('quality_control/'+folder+'_dark_ron_hist.png',dpi=300)
    plt.close()
    wifisIO.writeFits(ron, masterSave +'_dark_RON.fits',ask=False)
    
else:
     print('No processing necessary')   

print ("Total time to run entire script: ",time.time()-t0)
