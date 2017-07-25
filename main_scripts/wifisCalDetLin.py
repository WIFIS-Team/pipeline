"""

Fully calibrates set of images used to measure the non-linearity behaviour of the detector

Requires:
- specifying the folder where these files can be found

Produces:
- map of saturation level
- map of per-pixel non-linearity corrections
- bad pixel mask for pixels with very bad non-linearity (*** STILL TO DO ***)

"""

import matplotlib
matplotlib.use('gtkagg')
import numpy as np
import astropy.io.fits as fits
import astropy.io.ascii as ascii
import time
import matplotlib.pyplot as plt
import wifisGetSatInfo as satInfo
import wifisNLCor as NLCor
import wifisRefCor as refCor
import os
import wifisIO
import wifisHeaders as headers
import warnings
import wifisBadPixels as badPixels

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0' # Used to show compile errors for debugging, can be removed
os.environ['PYOPENCL_CTX'] = '1' # Used to specify which OpenCL device to target, should uncomment and set to preferred device to avoid interactively selecting each time

t0 = time.time()

#*****************************************************************************
#*************************** Required input **********************************

#set file list
fileList = 'det.lst'

rootFolder = '/data/WIFIS/H2RG-G17084-ASIC-08-319'

#assume that if rootFolder is blank, that the folders to read from are local and folderType is not needed
if rootFolder == '':
    folderType = ''
else:
    folderType = '/UpTheRamp/'

#if exists and to be updated
#bpmFile = 'processed/bad_pixel_mask.fits'

nChannels = 32
nRowAvg = 4
nRowSplit = 1
nSatSplit = 1
nReadSkip = 2
satLimit=0.97
bpmCutoff = 1e-5
#*****************************************************************************

#read file list
lst= wifisIO.readAsciiList(fileList)

if lst.ndim == 0:
    lst = np.asarray([lst])

#create processed directory
wifisIO.createDir('processed')

#check if processing needs to be done
if(os.path.exists('processed/master_detLin_NLCoeff.fits') and os.path.exists('processed/master_detLin_satCounts.fits') and os.path.exists('processed/master_detLin_BPM.fits')):
    cont = 'n'
    cont = wifisIO.userInput('Master non-linearity processed files already exists, do you want to continue processing (y/n)?')

    if (cont.lower() == 'y'):
        contProc = True
    else:
        contProc = False
else:
    contProc = True
    
if (contProc):

    satCountsLst = []
    nlCoeffLst = []
    zpntLst = []
    rampLst = []
    
    for folder in lst:

        #check if folder contains multiple ramps
        nRamps = wifisIO.getNumRamps(folder, rootFolder=rootFolder)

        for rampNum in range(1, nRamps+1):
            if nRamps > 1:
                savename = 'processed/'+folder+'_R'+'{:02d}'.format(rampNum)
            else:
                savename = 'processed/'+folder

            if(os.path.exists(savename+'_detLin_satCounts.fits') and os.path.exists(savename+'_detLin_NLCoeff.fits')):
                cont = 'n'
                if nRamps > 1:
                    cont = wifisIO.userInput('Non-linearity processed files already exists for ' + folder +' ramp ' + '{:02d}'.format(rampNum) +', do you want to continue processing (y/n)?')
                else:
                    cont = wifisIO.userInput('Non-linearity processed files already exists for ' + folder +', do you want to continue processing (y/n)?')

                if (cont.lower() == 'y'):
                    contProc2 = True
                else:
                    contProc2 = False
            else:
                contProc2 = True
           
            if (contProc2):
                print('Processing '+folder + ' ramp '+str(rampNum))
        
                #**********************************************************************
                #**********************************************************************

                #Read in data
                ta = time.time()
                print('Reading data into cube')
                #adjust accordingly depending on data source
                data, inttime, hdr = wifisIO.readRampFromFolder(rootFolder + folderType + folder, rampNum=rampNum, nSkip=2)
                hdr.add_history('Processed using WIFIS pyPline')
                hdr.add_history('on '+time.strftime("%c"))
                #data, inttime, hdr = wifisIO.readRampFromAsciiList(filename)
                #data, inttime, hdr = wifisIO.readRampFromFile(filename)
         
                data = data.astype('float32')*1.
                #**********************************************************************
                #******************************************************************************
                #Correct data for reference pixels
                #ta = time.time()
                print("Subtracting reference pixel channel bias")
                refCor.channelCL(data, nChannels)
                
                hdr.add_history('Channel reference pixel corrections applied using '+ str(nChannels) +' channels')
                
                print("Subtracting reference pixel row bias")
                refCor.rowCL(data, nRowAvg,nRowSplit)
                hdr.add_history('Row reference pixel corrections applied using '+ str(int(nRowAvg+1))+ ' pixels')
   
                #print("time to apply reference pixel corrections ", time.time()-ta, " seconds")
                #**********************************************************************

                #Get saturation info
                if(os.path.exists(savename+'_detLin_satCounts.fits')):
                    cont ='n'
                    if nRamps > 1:
                        cont = wifisIO.userInput('satCounts file already exists for ' +folder+', ramp '+ '{:02d}'.format(rampNum) +' do you want to replace (y/n)?')
                    else:
                        cont = wifisIO.userInput('satCounts file already exists for ' +folder+', do you want to replace (y/n)?')
                        
                else:
                    cont = 'y'

                if (cont.lower() == 'y'):
                    print('Getting saturation info')
                    ta = time.time()
                    satCounts = satInfo.getSatCountsCL(data,satLimit, nSatSplit)
                    hdr.add_history('Saturation level defined as '+ str(satLimit)+ ' saturation limit')
                    hdr.add_comment('File contains saturation level for each pixel')
                    
                    #save file
                    wifisIO.writeFits(satCounts, savename+'_detLin_satCounts.fits',hdr=hdr,ask=False)

                    #now remove last history and comment line
                    hdrTmp = hdr[::-1]
                    hdrTmp.remove('HISTORY')
                    hdrTmp.remove('COMMENT')
                    hdr = hdrTmp[::-1]
                else:
                    print('Reading saturation info from file')
                    satCounts = wifisIO.readImgsFromFile(savename+'_detLin_satCounts.fits')[0]

                satCountsLst.append(satCounts)
        
                #find the first saturated frames
                satFrame = satInfo.getSatFrameCL(data,satCounts,32, ignoreRefPix=True)
                
                #**********************************************************************
                #**********************************************************************

                # Get the non-linearity correction coefficients
                if(os.path.exists(savename+'_detLin_NLCoeff.fits')):
                    cont='n'
                    if nRamps > 1:
                        cont = wifisIO.userInput('NLCoeff file already exists for ' +folder+', ramp ' + '{:02d}'.format(rampNum) + ' do you want to replace (y/n)?')
                    else:
                        cont = wifisIO.userInput('NLCoeff file already exists for ' +folder+', do you want to replace (y/n)?')
  
                else:
                    cont = 'y'

                if (cont.lower() == 'y'):
                    print('Determining non-linearity corrections')
                    ta = time.time()
                    nlCoeff, zpntImg, rampImg = NLCor.getNLCorCL(data,satFrame,32)

                    #hard code the NLCoeff for reference pixels
                    #reset the values of the reference pixels so that all frames are used
                    refFrame = np.ones(satFrame.shape, dtype=bool)
                    refFrame[4:-4,4:-4] = False

                    nlCoeff[refFrame,0] = 1.
                    nlCoeff[refFrame,1:] = 0.

                    hdr.add_comment('This cube contains the non-linearity correction coefficients')
                    #save file
                    wifisIO.writeFits(nlCoeff, savename+'_detLin_NLCoeff.fits',ask=False,hdr=hdr)

                    hdrTmp = hdr[::-1]
                    hdrTmp.remove('COMMENT')
                    hdr = hdrTmp[::-1]

                    hdr.add_comment('File contains the bias and ramp levels as multi-extensions')
                    wifisIO.writeFits([zpntImg, rampImg], savename+'_detLin_polyCoeff.fits', ask=False,hdr=hdr)

                    hdrTmp = hdr[::-1]
                    hdrTmp.remove('COMMENT')
                    hdr = hdrTmp[::-1]

                else:
                    print('Reading non-linearity coefficient file')
                    nlCoeff = wifisIO.readImgsFromFile(savename+'_detLin_NLCoeff.fits')[0]
                    zpntImg, rampImg = wifisIO.readImgsFromFile(savename+'_detLin_polyCoeff.fits')[0]
                
                nlCoeffLst.append(nlCoeff)
                zpntLst.append(zpntImg)
                rampLst.append(rampImg)
            
                #**********************************************************************
                #**********************************************************************
        
                print('Done processing, determining saturation info, and non-linearity coefficients')
                data = 0 #clean up data cube to reduce memory usage for next iteration
            else:
                #read data from files
                print('Reading information from files instead')
                satCountsLst.append(wifisIO.readImgsFromFile(savename+'_detLin_satCounts.fits')[0])
                nlCoeffLst.append(wifisIO.readImgsFromFile(savename+'_detLin_NLCoeff.fits')[0])
    
    #create and write master files
    print('Creating master files')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        masterSatCounts = np.nanmedian(np.array(satCountsLst),axis=0)
        masterNLCoeff = np.nanmedian(np.array(nlCoeffLst),axis=0)
        masterZpnt = np.nanmedian(np.array(zpntLst), axis=0)
        masterRamp = np.nanmedian(np.array(rampLst),axis=0)
    

    #create fits header
    hdr = fits.Header()
    hdr.add_history('Processed using WIFIS pyPline')
    hdr.add_history('on '+time.strftime("%c"))
    hdr.add_history('This file was constructed from the median average of the following:')

    for s in lst:
        hdr.add_history(s + ' with ' +  str(wifisIO.getNumRamps(s, rootFolder=rootFolder)) + ' ramps;')
  
    #write files, if necessary
    hdr.add_comment('Master saturation info file defining limits for each pixel')
    wifisIO.writeFits(masterSatCounts.astype('float32'),'processed/master_detLin_satCounts.fits',hdr=hdr)

    hdrTmp = hdr[::-1]
    hdrTmp.remove('COMMENT')
    hdr = hdrTmp[::-1]

    hdr.add_comment('Master non-linearity correction coefficients cube for each pixel')
    wifisIO.writeFits(masterNLCoeff.astype('float32'),'processed/master_detLin_NLCoeff.fits', hdr=hdr)

    hdrTmp = hdr[::-1]
    hdrTmp.remove('COMMENT')
    hdr = hdrTmp[::-1]

    hdr.add_comment('Master bias and ramp values for each pixel, as multi-extensions')
    wifisIO.writeFits([masterZpnt.astype('float32'), masterRamp.astype('float32')], 'processed/master_detLin_polyCoeff.fits',hdr=hdr)

    hdrTmp = hdr[::-1]
    hdrTmp.remove('COMMENT')
    hdr = hdrTmp[::-1]

    print('Determining bad pixels from master NL coefficients')
    hdr.add_comment('Master bad pixel mask')

    bpm,hdr =  badPixels.getBadPixelsFromNLCoeff(masterNLCoeff,hdr,saveFile='quality_control/master_detLin_NLCoeff',cutoff=bpmCutoff)

    wifisIO.writeFits(bpm, 'processed/master_detLin_BPM.fits',hdr=hdr)
    
else:
    print('No processing necessary')


print ("Total time to run entire script: "+str(time.time()-t0))+" seconds"
