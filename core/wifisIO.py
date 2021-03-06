"""

Tools to handle various input and output tasks

"""

import numpy as np
import astropy.io.fits as fits
import glob
import re 
import os
import sys
import astropy
import cPickle
import numpy as np
import ast

def readRampFromFile(filename):
    """
    Read a set of images contained as multi extensions from the FITS file provided. Assumes that each extensions is at most a 2D image.
    Usage: outImg, outTime, hdr = readRampFromFile(filename)
    filename is name of the FITS file from which to read
    outImg is the returned data cube
    outTime is the returned array containing the integration times for each image in the list
    hdr is the header of the last FITS file
    """

    #open first file to get image dimensions
    tmp = fits.open(filename)
    ny = tmp[0].header['NAXIS2']
    nx = tmp[0].header['NAXIS1']
    listLen = len(tmp)
    
    outImg = np.zeros((ny,nx,listLen), dtype=tmp[0].data.dtype)
    outTime = np.zeros((listLen), dtype='float32')

    #now populate the array

    for i in range(0,listLen):
        outTime[i] = tmp[i].header['ACTEXP']
        np.copyto(outImg[:,:,i], tmp[i].data)
        
    #get header of last image only
    hdr = tmp[-1].header
    tmp.close()
    del tmp
    
    return outImg, outTime, hdr

def readImgsFromFile(file):
    """
    Reads a FITS file or cube from the specified file. File can contain multiple extensions, image cubes, or other various formats. Each extension must contain a 2D image at minimum or a 3D image at maximum.
    Usage: out,hdr = readImgsFromFile(file)
    file is a string corresponding to the file name
    out is the returned image (or list)
    hdr is the return astropy image header (or list of headers, if multi-extensions)
    """

    #get FITS file information
    tmp = fits.open(file)

    #determine the number of extensions
    nexts = len(tmp)

    #if multiple extensions
    if (nexts > 1):

        #start by creating output image and header lists
        out = []
        hdr = []

        #go through each extension
        for i in range(nexts):
            #assume that each extension must contain at a minimum a 2D image
            ny = tmp[i].header['NAXIS2']
            nx = tmp[i].header['NAXIS1']

            #check if a third axis exists
            try:
                nz = tmp[i].header['NAXIS3']
            except(KeyError):
                nz = 1

            #data is read in as [naxis3, naxis2, naxis1], swap
            #so that it becomes [naxis2, naxis1, naxsi3]
            
            if (nz > 1):
                outImg = np.zeros((ny,nx, nz), dtype=tmp[i].data.dtype)
        
                for j in range(nz):
                    np.copyto(outImg[:,:,j],tmp[i].data[j,:,:])
            else:
                outImg = tmp[i].data

            out.append(outImg)
            hdr.append(tmp[i].header)
    else:
        ny = tmp[0].header['NAXIS2']
        nx = tmp[0].header['NAXIS1']
        
        try:
            nz = tmp[0].header['NAXIS3']
        except(KeyError):
            nz = 1

        if (nz > 1):
            out = np.zeros((ny,nx, nz), dtype=tmp[0].data.dtype)
            
            for j in range(nz):
               np.copyto(out[:,:,j],tmp[0].data[j,:,:])
        else:
            out = tmp[0].data
        hdr = tmp[0].header

    tmp.close()
    del tmp
    return out, hdr

def readImgExtFromFile(file):
    """
    Reads a FITS image with multiple extensions of possibly different dimensions from the specified file
    Usage: outImg, hdr = readImgExtFromFile(file)
    file is a text input corresponding to the file name
    outImg is the returned image list
    hdr is the returned astropy header object (of the last entry only)
    """

    #get FITS file information
    tmp = fits.open(file)

    #check if there are multiple HDUs
    nz = len(tmp)

    outImg = []
    if (nz > 1):
        for i in range(nz):
            outImg.append(tmp[i].data)

    #get header
    hdr = tmp[-1].header
    
    tmp.close()
    del tmp
    return outImg, hdr

def readRampFromList(list):
    """
    Read a set of images from the list provided
    Usage: outImg, outTime, hdr = readRampFromList(list)
    list is an array or python list containing the names of each file from which to read
    outImg is the returned data cube
    outTime is the returned array containing the integration times for each image in the list
    hdr is the header of the last FITS file returned as an astropy header object
    """

    #open first file to get image dimensions
    tmp = fits.open(list[0])
    nx = tmp[0].header['NAXIS2']
    ny = tmp[0].header['NAXIS1']
    listLen = len(list)
    
    outImg = np.zeros((nx,ny,listLen), dtype=tmp[0].data.dtype)
    outTime = np.zeros((listLen), dtype='float32')

    #now populate the array
    for i in range(0,listLen):
        tmp = fits.open(list[i])
        outTime[i] = tmp[0].header['INTTIME']
        np.copyto(outImg[:,:,i], tmp[0].data)
        tmp.close()
        
    #get header of last image only
    hdr = fits.getheader(list[-1])
    del tmp
    return outImg, outTime, hdr


def readRampFromAsciiList(filename, sort=True):
    """
    Read a set of images from an ASCII file containing a list of strings (filenames) 
    Usage: output, outtime, hdr = readRampFromAsciiList(filename)
    filename is the name of the ascii file containing the names of each file from which to read
    outImg is the returned data cube
    outTime is the returned array containing the integration times for each image in the list
    hdr is the header of the last FITS file, returned as an astropy header object
    sort is an optional boolean keyword to specify is the input list should first be sorted alpha-numerically
    """
    
    list = np.loadtxt(filename, dtype=np.str_)
    if (sort):
        list = sorted_nicely(list)

    output, outtime, hdr = readRampFromList(list)

    return output, outtime, hdr

def writeFits_old(data, filename, hdr=None):
    """
    *** THIS ROUTINE IS DEPRECATED AND REPLACED BY writeFits ***
    Write a data file to a FITS file. 
    Usage writeFits(data, filename)
    data is the input data (can be more than 2 dimensions). Assumes data is stored as follows: [y-coordinates,x-coordinates,lamba-coordinates]
    filename is the name of the file to write the FITS file to.
    """

    if (data.ndim > 2):
        #convert data from [y,x,lambda] -> [lambda, y, x]

        out = np.zeros((data.shape[2], data.shape[0], data.shape[1]), dtype=data.dtype)
    
        for i in range(data.shape[2]):
            out[i,:,:] = data[:,:,i]
    else:
        out = data
        
    if (hdr is not None):
        #add additional FITS keywords
        prihdr = fits.Header()
        prihdr = hdr
    else:
        prihdr = None

    hdu = fits.PrimaryHDU(out, header=prihdr)

    if str(astropy.version.version)<'1.3':
        hdu.writeto(filename,clobber=True)
    else:
        hdu.writeto(filename,overwrite=True)

    return
        
def readRampFromFolder(folderName, rampNum=None,nSkip=0):
    """
    Routine to read all H2RG files from specified folder.
    Usage: output, outtime, hdr = readRampFromFolder(folderName)
    folderName is the name of the folder from which to read files
    rampNum is an integer keyword to specify which ramp to read, if multiple ramps are present in a given folder. If None is specified and multiple ramps exist, a warning is issued.
    nSkip is an integer keyword to specify the number of reads to skip when reading the ramp data.
    output is the data cube generated from the files contained in the specified folder.    
    outtime is the returned array containing the integration times for each image in the list
    hdr is the returned astropy header object corresponding to the last image in the ramp
    """

    if rampNum is None:
        #Make sure that only one set of ramps is present in folder
        lst = glob.glob(folderName+'/*N01.fits')
        #if the list is empty, search for gzipped files
        if len(lst)==0:
            lst = glob.glob(folderName+'/*N01.fits.gz')
            
        if len(lst)>1:
            raise Warning('*** More than one set of ramps present in folder ' + folderName + '. You must specify which ramp to use. ***')

        else:
            lst = glob.glob(folderName+'/H2*fits*')
            if len(lst)==0:
                lst = glob.glob(folderName+'/H2*fits.gz')

            lst = sorted_nicely(lst)
    else:
        lst = glob.glob(folderName+'/H2*'+'R'+'{:02d}'.format(rampNum)+'*.fits')
        if len(lst)==0:
            lst = glob.glob(folderName+'/H2*'+'R'+'{:02d}'.format(rampNum)+'*.fits.gz')

        lst = sorted_nicely(lst)

    if nSkip>0:
        lst = lst[nSkip:]
        
    output,outtime,hdr = readRampFromList(lst)    
    
    return output,outtime,hdr

def readTable (filename):
    """
    Routine to read a table from an ASCII file
    Usage: out = readTable(filename)
    filename is the name of the file from which to read the table
    out is the numpy array created from the table
    The table is expected to be space or tab separated and a # to denote comment lines (which are ignored)
    """
    
    out = np.loadtxt(filename, dtype='float32',comments='#')

    return out

def readAsciiList (filename):
    """
    Read a list of strings from an ASCII file
    Usage: out = readAsciiList(filename)
    filename is the name of the file from which to read the list
    out is the numpy array created from the list
    The list is expected to be separated on different lines
    """
    out = np.loadtxt(filename, dtype=np.str_)
    
    return out

def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect."""

    #based on code from https://gist.github.com/limed/473a498641bbc7761a20
    
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def createDir(name):
    """
    Creates directory name, if directory does not exist, otherwise does nothing
    Usage: createDir(name)
    name is the name of the folder/directory
    """

    if os.path.exists(name):
        pass
    else:
        os.mkdir(name)

    return

def userInput(strng):
    """
    Handles user input requests
    usage cont = userInput(strng)
    strng is the prompt text to appear to the user
    cont is the returned input of the user
    """

    #check python version first
    if sys.version_info >= (3,0):
        cont = input(strng+' ')
    else:
        cont = raw_input(strng+' ')

    return cont

def writeTable (data,filename):
    """
    Routine to write a data array into a tabular format to an ASCII file
    Usage: writeTable(data, filename)
    data is the array to be written
    filename is the name of the file to write the table
    """

    np.savetxt(filename, data)

    return 

def writeImgSlices(data, extSlices,filename, hdr=None):
    """
    Write a data file to a FITS image including the extracted slices    
    Usage writeFits(data, extSlices, filename)
    data is the input data (can contain multiple HDUs, i.e. more than 2 dimensions)
    filename is the name of the file to write the FITS file to.
    """

    if (hdr is not None):
        #add additional FITS keywords
        prihdr = fits.Header()
        prihdr = hdr
    else:
        prihdr = None

    list = [fits.PrimaryHDU(data, header=prihdr)]
    
    nSlices = len(extSlices)
        
    for s in extSlices:
        list.append(fits.ImageHDU(s))

    hdu = fits.HDUList(list)

    if str(astropy.version.version) <'1.3':
        hdu.writeto(filename,clobber=True)
    else:
        hdu.writeto(filename,overwrite=True)

    return

def writeFits(data, filename, hdr=None, ask=True):
    """
    Write a data file to a FITS file.
    Usage writeFits(data, filename, hdr=None, ask=True)
    data is the input data. It can be a list of images or just a single image itself. If data is 3D, assumes data is stored as follows: [y-coordinates,x-coordinates,lamba-coordinates]
    filename is the name of the file to write the FITS file to.
    hdr is the astropy header object to write
    ask is a boolean keyword to specify if the user should be prompted to overwrite the filename, in the event the filename already exists
    """

    if (ask):
        if(os.path.exists(filename)):
            cont = userInput(filename + ' already exists, do you want to replace (y/anything else for no)?')
            if (cont.lower() == 'y'):
                contWrite = True
            else:
                contWrite = False
        else:
            contWrite = True
        
        if (not contWrite):
            return
    
    #check if a header, or list of headers, is provided
    if (hdr is not None):
        if (type(hdr) is list):
            prihdr = hdr[0]
        else:
            prihdr = hdr
    else:
        prihdr = fits.Header()

    #check if data is a list of images
    if (type(data) is list):

        d = data[0]
        if (d.ndim > 2):
            #convert data from [y,x,lambda] -> [lambda, y, x]
            #swap first and last
            d = d.swapaxes(0,2)
            #now swap last two
            d = d.swapaxes(1,2)
            
        allHDU = [fits.PrimaryHDU(d, header=prihdr)]
        
        for i in range(1,len(data)):
            d = data[i]
            
            if (d.ndim > 2):
                d = d.swapaxes(0,2)
                d = d.swapaxes(1,2)

            #check if multiple headers are provided
            if (hdr is not None):
                if (type(hdr) is list):
                    allHDU.append(fits.ImageHDU(d, header=hdr[i]))
                else:
                    allHDU.append(fits.ImageHDU(d))
            else:
                allHDU.append(fits.ImageHDU(d))
    else:
        if (data.ndim > 2):
            d = data.swapaxes(0,2)
            d = d.swapaxes(1,2)
        else:
            d = data

        allHDU = fits.PrimaryHDU(d, header=prihdr)

    hdu = fits.HDUList(allHDU)
         
    if str(astropy.version.version)  <'1.3':
        hdu.writeto(filename,clobber=True)
    else:
        hdu.writeto(filename,overwrite=True)

    hdu.close()
    return

def writePickle(objct, filename, protocol=-1):
    """
    Writes a complex object to a file using cPickle.
    Usage: writePickle(objct, filename, protocol=-1)
    objct is the object to save to a file,
    filename is the name of the file which to save
    protocol is the protocol to use for pickling (default is the highest version).
    """

    fileout = open(filename, 'w')

    cPickle.dump(objct, fileout,protocol)
    fileout.close()
    
    return

def readPickle(filename):
    """
    Reads a pickled object from a file.
    Usage: objct = readPickle(filename)
    filename is the name of the file from which to read the pickled object.
    Returns the unpickled object.
    """

    filein = open(filename,'r')
    objct = cPickle.load(filein)
    filein.close()

    return objct

def getNumRamps(folder, rootFolder=''):
    """
    Routine to determine the number of ramps present in a given folder, relative to the root folder location.
    Usage: nRamps = getNumRamps(folder, rootFolder='')
    folder is the name of the input folder
    rootFolder specifies the root folder location
    nRamps is the returned value indicating the number of ramps present
    """

    #CDS
    if os.path.exists(rootFolder+'/CDSReference/'+folder):
        rampLst = glob.glob(rootFolder+'/CDSReference/'+folder+'/*N01.fits')
        nRamps = len(rampLst)

        #check if list is empty and search for gzipped files instead
        if nRamps == 0:
            rampLst = glob.glob(rootFolder+'/CDSReference/'+folder+'/*N01.fits.gz')
            nRamps = len(rampLst)

 
    #Fowler
    elif os.path.exists(rootFolder+'/FSRamp/'+folder):
        rampLst = glob.glob(rootFolder+'/FSRamp/'+folder+'/*N01.fits')
        nRamps = len(rampLst)
        if nRamps == 0:
            rampLst = glob.glob(rootFolder+'/FSRamp/'+folder+'/*N01.fits.gz')
            nRamps = len(rampLst)
         
    elif os.path.exists(rootFolder + '/UpTheRamp/'+folder):
        rampLst = glob.glob(rootFolder+'/UpTheRamp/'+folder+'/*N01.fits')
        nRamps = len(rampLst)

        if nRamps == 0:
            rampLst = glob.glob(rootFolder+'/UpTheRamp/'+folder+'/*N01.fits.gz')
            nRamps = len(rampLst)

    else:
        raise Warning('*** Folder ' + folder +' does not exist ***')

    return nRamps

def getPath(folder, rootFolder=''):
    """
    Routine to determine the full path to a given folder name
    Usage: path = getPath(folder, rootFolder='')
    folder is the folder to search for
    rootFolder is the location of the root folder
    path is the full location to the folder
    """
    
    #CDS
    if os.path.exists(rootFolder+'/CDSReference/'+folder):
        path = rootFolder+'/CDSReference/'+folder
    #Fowler
    elif os.path.exists(rootFolder+'/FSRamp/'+folder):
       path = rootFolder+'/FSRamp/'+folder
    elif os.path.exists(rootFolder + '/UpTheRamp/'+folder):
        path = rootFolder+'/UpTheRamp/'+folder
    else:
        raise Warning('*** Folder ' + folder +' does not exist ***')
    return path

def readInputVariables(fileName='wifisConfig.inp'):
    """
    Routine to read all variables from an input file
    Usage: inpVars = readInputVariables(fileName='wifisConfig.inp')
    fileName is the name/path of the input file
    inpVars is the returned list of all input variables and their values
    """

    #read all file contents into list
    with open(fileName) as fle:
        inpLst = list(fle)

    inpVars = []
    for line in inpLst:
        tmp = line.strip('\n')
        tmp = tmp.lstrip()
        if len(tmp)>0:
            if tmp[0][0] !='#': #ignore comment lines
                #remove comment sections from line
                tmp = tmp.split('#')[0].rstrip(' ') # only keep first part of line
                tmp = tmp.split('=')
                tmp[0] = tmp[0].rstrip()
                tmp[1] = tmp[1].lstrip().rstrip()
                tmp[1] = ast.literal_eval(tmp[1])
                inpVars.append(tmp)


    return inpVars

def getRampType(folder, rootFolder=''):
    """
    Routine to determine the type of observation (CDS, FS, UTR) for the given folder
    Usage: rampType = getRampType(folder, rootFolder='')
    folder is the folder name
    rootFolder is the location of the root folder
    rampType is the returned string which provides the path (relative to the root folder) to the provided folder
    """
    
    #CDS
    if os.path.exists(rootFolder+'/CDSReference/'+folder):
        rampType = '/CDSReference/'
    #Fowler
    elif os.path.exists(rootFolder+'/FSRamp/'+folder):
        rampType = '/FSRamp/'
    elif os.path.exists(rootFolder + '/UpTheRamp/'+folder):
        rampType = '/UpTheRamp/'
    else:
        raise Warning('*** Folder ' + folder +' does not exist ***')
    return rampType
