"""

Main script to process a set of raw data to measure the non-linearity behaviour of the detector. Data is expected to consist of an up-the-ramp exposure of long enough duration that each pixel saturates.

Input:
- All variables are read from file given by variable varFile
- Including, at minimum, an ASCII file containing the folder name(s) of the raw observations

Produces:
- individual (for each ramp) and master:
  - map of saturation level (XXX_satCounts.fits)
  - map of non-linearity corrections coefficients (XXX_detLin_NLCoeff.fits)
  - map of bias and ramp coefficients corresponding to the polynomial fitting of the linearized ramp (XXX_detLin_polyCoeff.fits)
- master bad pixel mask for pixels based on outliers from the first two non-linearity coefficients (master_detLin_BPM.fits)

"""

#change the next two lines as needed
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

t0 = time.time()

#*****************************************************************************
#*************************** Required input **********************************
#INPUT VARIABLE FILE NAME
varFile = 'wifisConfig.inp'

#*****************************************************************************

#open log file to record all processing steps
logfile = open('wifis_reduction_log.txt','a')

logfile.write('********************\n')
logfile.write(time.strftime("%c")+'\n')
logfile.write('Processing detector non-linearity ramps with WIFIS pyPline\n')

print('Reading input variables from file ' + varFile)
logfile.write('Reading input variables from file ' + varFile+'\n')
varInp = wifisIO.readInputVariables(varFile)
for var in varInp:
    locals()[var[0]]=var[1]
    
#execute pyOpenCL section here
os.environ['PYOPENCL_COMPILER_OUTPUT'] = pyCLCompOut
if len(pyCLCTX) >0:
    os.environ['PYOPENCL_CTX'] = pyCLCTX    

logfile.write('Root folder containing raw data: ' + str(rootFolder)+'\n')
logfile.write('Detector gain of' + str(gain)+ ' in electrons/counts\n')
#logfile.write('Detector readout noise of' + str(ron) +' electrons per frame\n')

#read file list
lst= wifisIO.readAsciiList(detLstFile)

if lst.ndim == 0:
    lst = np.asarray([lst])

#create processed directory
wifisIO.createDir('processed')
wifisIO.createDir('quality_control')

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

        #first get ramp type
        folderType = wifisIO.getRampType(folder,rootFolder=rootFolder)
        
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
                logfile.write('Processing ' + folder + ' ramp ' + str(rampNum)+'\n')
                
                #**********************************************************************
                #**********************************************************************

                #Read in data
                ta = time.time()
                print('Reading data into cube')
                #adjust accordingly depending on data source
                data, inttime, hdr = wifisIO.readRampFromFolder(rootFolder + folderType + folder, rampNum=rampNum, nSkip=nReadSkip)
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
                refCor.channelCL(data, nChannel)
                
                hdr.add_history('Channel reference pixel corrections applied using '+ str(nChannel) +' channels')
                logfile.write('Subtracted reference pixel channel bias using ' + str(nChannel) + ' channels\n')

                if nRowsAvg > 0:
                    print("Subtracting reference pixel row bias")
                    refCor.rowCL(data, nRowsAvg,nRowSplit)
                    hdr.add_history('Row reference pixel corrections applied using '+ str(int(nRowsAvg+1))+ ' pixels')

                    logfile.write('Subtraced row reference pixel bias using moving average of ' + str(int(nRowsAvg)+1) + ' rows\n')
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
                    satCounts = satInfo.getSatCountsCL(data,satLimit, nSatSplit, satThresh=satThresh)
                    hdr.add_history('Saturation level defined as '+ str(satLimit)+ ' saturation limit')
                    hdr.add_comment('File contains saturation level for each pixel')
                    logfile.write('Determined saturation limit as ' + str(satLimit) + ' times the mean of all pixels with flux greater than ' + str(satThresh)+'\n')
                    
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
                satFrame = satInfo.getSatFrameCL(data,satCounts,nSatSplit, ignoreRefPix=True)

                logfile.write('Determined first saturated frame based on saturation limits\n')
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
                    nlCoeff, zpntImg, rampImg = NLCor.getNLCorCL(data,satFrame,nlSplit)

                    #hard code the NLCoeff for reference pixels
                    #reset the values of the reference pixels so that all frames are used
                    refFrame = np.ones(satFrame.shape, dtype=bool)
                    refFrame[4:-4,4:-4] = False

                    nlCoeff[refFrame,0] = 1.
                    nlCoeff[refFrame,1:] = 0.

                    hdr.add_comment('This cube contains the non-linearity correction coefficients')
                    #save file
                    wifisIO.writeFits(nlCoeff, savename+'_detLin_NLCoeff.fits',ask=False,hdr=hdr)
                    logfile.write('Determined non-linearity correction coefficients\n')
                    
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
    
    logfile.write('\n')
    logfile.write('Created master saturation counts, non-linearity correction coefficients, bias, and ramp/flux files as the median of all processed ramps\n')
    
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
    hdr.add_comment('File contains the master bad pixel mask')

    bpm,hdr =  badPixels.getBadPixelsFromNLCoeff(masterNLCoeff,hdr,saveFile='quality_control/master_detLin_NLCoeff',cutoff=nlBpmCutoff)

    logfile.write('Determined bad pixel mask based on the master non-linearity coefficients,\n')
    logfile.write('excluding all pixels with non-linearity corrections with a probability density of less than ' + str(nlBpmCutoff) + ', or greater than ' + str(1.-nlBpmCutoff)+'\n')
    
    wifisIO.writeFits(bpm, 'processed/master_detLin_BPM.fits',hdr=hdr)
    logfile.write('\n')
    
else:
    print('No processing necessary')


print ("Total time to run entire script: "+str(time.time()-t0))+" seconds"
logfile.write('Total time to run entire script: '+str(time.time()-t0)+' seconds\n')
logfile.write('********************\n')
logfile.write('\n')

logfile.close()
