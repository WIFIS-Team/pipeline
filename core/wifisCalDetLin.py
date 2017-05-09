"""

Fully calibrates set of images used to measure the non-linearity behaviour of the detector

Requires:
- specifying the folder where these files can be found

Produces:
- map of saturation level
- map of per-pixel non-linearity corrections
- bad pixel mask for pixels with very bad non-linearity (*** STILL TO DO ***)

"""

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

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0' # Used to show compile errors for debugging, can be removed
os.environ['PYOPENCL_CTX'] = '1' # Used to specify which OpenCL device to target, should uncomment and set to preferred device to avoid interactively selecting each time

t0 = time.time()

#*****************************************************************************
#*************************** Required input **********************************

#set file list
fileList = 'det.lst'

#if exists and to be updated
bpmFile = 'processed/bad_pixel_mask.fits'
#*****************************************************************************

#read file list
lst= wifisIO.readAsciiList(fileList)

#create processed directory
wifisIO.createDir('processed')

#check if processing needs to be done
if(os.path.exists('processed/master_detLin_NLCoeff.fits') and os.path.exists('processed/master_detLin_satCounts.fits')):
    
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
    
    for filename in lst:
        filename = filename.tostring()
        savename = 'processed/'+filename.replace('.fits','')
        savename = 'processed/'+filename.replace('.gz','')

        if(os.path.exists(savename+'_satCounts.fits') and os.path.exists(savename+'_NLCoeff.fits')):
            cont = wifisIO.userInput('Non-linearity processed files already exists for ' + filename +', do you want to continue processing (y/n)?')
           
            if (cont.lower() == 'y'):
                contProc2 = True
            else:
                contProc2 = False
        else:
            contProc2 = True
           
        if (contProc2):
            print('Processing '+filename)
        
            #**********************************************************************
            #**********************************************************************

            #Read in data
            ta = time.time()

            #adjust accordingly depending on data source
            #data, inttime, hdr = wifisIO.readRampFromFolder(filename)
            data, inttime, hdr = wifisIO.readRampFromAsciiList(filename)

            #data, inttime, hdr = wifisIO.readRampFromFile(filename)
            
            print("time to read all files took "+str(time.time()-ta) + " seconds")

            #**********************************************************************
            #**********************************************************************

            #Get saturation info
            if(os.path.exists(savename+'_satCounts.fits')):
                cont = wifisIO.userInput('satCounts file already exists for ' +filename+', do you want to replace (y/n)?')
            else:
                cont = 'y'

            if (cont.lower() == 'y'):
                print('Getting saturation info')
                ta = time.time()
                satCounts = satInfo.getSatCountsCL(data,0.95, 1)
                print ("saturation code took "+str(time.time()-ta)+ " seconds")

                #save file
                wifisIO.writeFits(satCounts, savename+'_satCounts.fits',ask=False)
                
            else:
                print('Reading saturation info from file')
                satCounts = wifisIO.readImgsFromFile(savename+'_satCounts.fits')[0]

            satCountsLst.append(satCounts)
        
            #find the first saturated frames
            satFrame = satInfo.getSatFrameCL(data,satCounts,1)

            #**********************************************************************
            #**********************************************************************

            # Get the non-linearity correction coefficients
            if(os.path.exists(savename+'_NLCoeff.fits')):
                cont = wifisIO.userInput('NLCoeff file already exists for ' +filename+', do you want to replace (y/n)?')
            else:
                cont = 'y'

            if (cont.lower() == 'y'):
                print('Determining non-linearity corrections')
                ta = time.time()
                nlCoeff, zpntImg, rampImg = NLCor.getNLCorCL(data,satFrame,32)
                print ("non-linearity code took" +str(time.time()-ta)+ " seconds")

                #save file
                wifisIO.writeFits(nlCoeff, savename+'_NLCoeff.fits',ask=False)
                wifisIO.writeFits([zpntImg, rampImg], savename+'_polyCoeff.fits', ask=False)
            else:
                print('Reading non-linearity coefficient file')
                nlCoeff = wifisIO.readImgsFromFile(savename+'_NLCoeff.fits')[0]
                zpntImg, rampImg = wifisIO.readImgsFromFile(savename+'_polyCoeff.fits')[0]
                
            nlCoeffLst.append(nlCoeff)
            zpntLst.append(zpntImg)
            rampLst.append(rampImg)
            
            #**********************************************************************
            #**********************************************************************
        
            print('Done processing and determining saturation info and non-linearity coefficients')
            data = 0 #clean up data cube to reduce memory usage for next iteration
        else:
            #read data from files
            print('Reading information from files instead')
            satCountsLst.append(wifisIO.readImgsFromFile(savename+'_satCounts.fits')[0])
            nlCoeffLst.append(wifisIO.readImgsFromFile(savename+'_NLCoeff.fits')[0])
    
    #create and write master files
    print('Creating master files')
    masterSatCounts = np.nanmedian(np.array(satCountsLst),axis=0)
    masterNLCoeff = np.nanmedian(np.array(nlCoeffLst),axis=0)
    masterZpnt = np.nanmedian(np.array(zpntLst), axis=0)
    masterRamp = np.nanmedian(np.array(rampLst),axis=0)
    
    #write files, if necessary
    wifisIO.writeFits(masterSatCounts,'processed/master_detLin_satCounts.fits')
    wifisIO.writeFits(masterNLCoeff,'processed/master_detLin_NLCoeff.fits')
    wifisIO.writeFits([masterZpnt, masterRamp], 'processed/master_detLin_polyCoeff.fits')
else:
    print('No processing necessary')



print('EXITING SCRIPT, BAD PIXEL IDENTIFICATION CURRENTLY A WORK IN PROGRESS')
exit()

#check if analysis of NL coefficients needs to be done
if(os.path.exists('processed/bad_pixel_mask.fits')):
    cont = wifisIO.userInput('bad pixel mask already exists, do you want to update, replace, skip? ("update"/"replace"/anything else to skip)')
    
    if (cont.lower() == 'update' or cont.lower() == 'replace'):
        contAnalysis = True
    else:
        contAnalysis = False
else:
    contAnalysis = True
    cont = 'replace'




if (contAnalysis):
    print('Determining bad pixels from non-linearity coefficients')
    
    if (~contProc):
        #read in nlCoeff instead
        print('Reading non-linearity coefficient file')
        nlCoeff = wifisIO.readImgsFromFile('processed/master_detLin_NLCoeff.fits')[0]
    
    if (cont == 'update'):
        BPM = wifisIO.readImgsFromFile(bpmFile)[0]
    else:
        BPM = np.zeros(nlCoeff[:,:,0].shape, dtype='int8')

    print('Analyzing non-linearity coefficients and determining outliers')

    #NLCoeff 0, ignoring reference pixels
    n = nlCoeff[4:2044,4:2044,0]
    
    for i in range(5):
        med = np.median(n)
        std = np.std(n)
        whr = np.where(np.abs(n-med) < 5.*std)
        n = n[whr]
        
    plt.close('all')
    plt.hist(nlCoeff[4:2044,4:2044,0].flatten(), range=[med-1.5*std, med+1.5*std],bins=100)
    hist = np.histogram(nlCoeff[4:2044,4:2044,0].flatten(), range=[med-1.5*std, med+1.5*std], bins=100)
   
    std = np.std(n)
    med = np.median(n)
    
    plt.plot([med-5.*std,med-5.*std], [np.min(hist[0]),np.max(hist[0])], 'r--')
    plt.plot([med+5.*std,med+5.*std], [np.min(hist[0]),np.max(hist[0])], 'r--')
    plt.xlabel('value')
    plt.ylabel('count')
    plt.title('NL correction coefficient 0')
    plt.savefig('processed/nlcoeff_0_hist.eps', bbox_inches='tight')
    #plt.show()
    plt.close('all')
    
    whr = np.where(np.abs(nlCoeff[4:2044,4:2044,0]-med) > 5.*std)
    BPM[whr] = 1
    
    #NLCoeff 1
    n = nlCoeff[4:2044,4:2044,1]
    
    for i in range(6):
        med = np.median(n)
        std = np.std(n-med)
        whr = np.where(np.abs(n-med) < 7.*std)
        n = n[whr]
    
    plt.hist(nlCoeff[4:2044,4:2044,1].flatten(), range=[med-std, med+std],bins=100)
    hist = np.histogram(nlCoeff[4:2044,4:2044,1].flatten(), range=[med-std, med+std], bins=100)
    
    med = np.median(n)
    std = np.std(n)

    plt.plot([med-5.*std,med-5.*std], [np.min(hist[0]),np.max(hist[0])], 'r--')
    plt.plot([med+5.*std,med+5.*std], [np.min(hist[0]),np.max(hist[0])], 'r--')
    
    plt.xlabel('value')
    plt.ylabel('count')
    plt.title('NL correction coefficient 1')
    plt.savefig('processed/nlcoeff_1_hist.eps')
    #plt.show()
    plt.close('all')
    
    whr = np.where(np.abs(nlCoeff[:,:,1]-med) > 5.*std)
    #BPM[whr] = 1
    
    #NLCoeff 2
    n = nlCoeff[4:2044,4:2044,2]
    
    for i in range(8):
        med = np.median(n)
        std = np.std(n)
        whr = np.where(np.abs(n-med) < 7.*std)
        n = n[whr]
        
    plt.hist(nlCoeff[4:2044,4:2044,2].flatten(), range=[med-std, med+std],bins=100)
    hist = np.histogram(nlCoeff[4:2044,4:2044,2].flatten(), range=[med-std, med+std], bins=100)
    
    med = np.median(n)
    std = np.std(n)
    
    plt.plot([med-5.*std,med-5.*std], [np.min(hist[0]),np.max(hist[0])], 'r--')
    plt.plot([med+5.*std,med+5.*std], [np.min(hist[0]),np.max(hist[0])], 'r--')
    plt.xlabel('value')
    plt.ylabel('count')
    plt.title('NL correction coefficient 2')
    plt.savefig('processed/nlcoeff_2_hist.eps')
    #plt.show()
    plt.close('all')
    
    whr = np.where(np.abs(nlCoeff[:,:,2]-med) > 5.*std)
    #BPM[whr] = 1
    
    #NLCoeff 3
    n = nlCoeff[4:2044,4:2044,3]
    
    for i in range(9):
        med = np.median(n)
        std = np.std(n)
        whr = np.where(np.abs(n-med) < 5.*std)
        n = n[whr]
    
    plt.hist(nlCoeff[4:2044,4:2044,3].flatten(), range=[med-std, med+std],bins=100)
    hist = np.histogram(nlCoeff[4:2044,4:2044,3].flatten(), range=[med-std, med+std], bins=100)
        
    med = np.median(n)
    std = np.std(n)
    
    plt.plot([med-5.*std,med-5.*std], [np.min(hist[0]),np.max(hist[0])], 'r--')
    plt.plot([med+5.*std,med+5.*std], [np.min(hist[0]),np.max(hist[0])], 'r--')
    plt.xlabel('value')
    plt.ylabel('count')
    plt.title('NL correction coefficient 3')
    plt.savefig('processed/nlcoeff_3_hist.eps')
    #plt.show()
    plt.close('all')

    whr = np.where(np.abs(nlCoeff[:,:,3]-med) > 5.*std)
    #BPM[whr] = 1
    
    print('Saving bad pixel mask')
    wifisIO.writeFits(BPM.astype('int'),'processed/bad_pixel_mask.fits',ask=False)

print ("Total time to run entire script: "+str(time.time()-t0))+" seconds"
