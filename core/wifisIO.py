"""

Tools to handle reading a writing from files and images

"""

import numpy as np
import astropy.io.fits as fits
import glob
import re 
import os
import sys
import astropy
import cPickle

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
    print('Reading FITS images into data cube')

    for i in range(0,listLen):
        outTime[i] = tmp[i].header['ACTEXP']
        np.copyto(outImg[:,:,i], tmp[i].data)
        
    print('Done reading')

    #get header of last image only
    hdr = tmp[-1].header
    tmp.close()
    del tmp
    
    return outImg, outTime, hdr

def readImgsFromFile(file):
    """
    Reads a FITS image or cube from the specified file. File can contain multiple extensions as well.
    Usage: out,hdr = readImgsFromFile(file)
    file is a string corresponding to the file name
    out,hdr is the returned image array
    """

    #get FITS file information
    tmp = fits.open(file)
    
    nexts = len(tmp)
    
    if (nexts > 1):
        
        out = []
        hdr = []

        for i in range(nexts):
            ny = tmp[i].header['NAXIS2']
            nx = tmp[i].header['NAXIS1']
              
            try:
                nz = tmp[i].header['NAXIS3']
            except(KeyError):
                nz = 1

            #data is read in as [naxis3, naxis2, naxis1], swap
            #so that it becomes [naxis2, naxis1, naxsi3]
            
            if (nz > 1):
                print('more than 2 axes')
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

def readImgExtFromFile(file): # **** redundant now ****
    """
    Reads a FITS image with multiple extensions of possibly different dimensions from the specified file
    Usage: outImg = readImgExtFromFile(file)
    file is a text input corresponding to the file name
    outImg is the returned image array
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
    hdr is the header of the last FITS file
    """

    #open first file to get image dimensions
    tmp = fits.open(list[0])
    nx = tmp[0].header['NAXIS2']
    ny = tmp[0].header['NAXIS1']
    listLen = len(list)
    
    outImg = np.zeros((nx,ny,listLen), dtype=tmp[0].data.dtype)
    outTime = np.zeros((listLen), dtype='float32')

    #now populate the array
    print('Reading FITS images into data cube')

    for i in range(0,listLen):
        tmp = fits.open(list[i])
        outTime[i] = tmp[0].header['INTTIME']
        np.copyto(outImg[:,:,i], tmp[0].data)
        tmp.close()
        
    print('Done reading')

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
    hdr is the header of the last FITS file
    """
    
    list = np.loadtxt(filename, dtype=np.str_)
    if (sort):
        list = sorted_nicely(list)

    output, outtime, hdr = readRampFromList(list)

    return output, outtime, hdr

def writeFits_old(data, filename, hdr=None):
    """
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

    if (astropy.version.major >=1 and astropy.version.minor >=3):
        hdu.writeto(filename,overwrite=True)
    else:
        hdu.writeto(filename,clobber=True)

def readRampFromFolder(folderName):
    """
    read all H2RG files from specified folder.
    Usage: output = readRampFromFolder(folderName)
    folderName is the name of the folder from which to read files
    output is the data cube generated from the files contained in the specified folder.    
    """
    list = glob.glob(folderName+'/H2*fits')
    list = sorted_nicely(list)
    
    output,outtime,hdr = readRampFromList(list)    
    
    return output,outtime,hdr

def readTable (filename):
    """
    Read a table from an ASCII file
    Usage: out = readTable(filename)
    filename is the name of the file from which to read the table
    out is the array created from the table
    The table is expected to be space or tab separated
    """
    
    out = np.loadtxt(filename, dtype='float32',comments='#')

    return out

def readAsciiList (filename):
    """
    Read a list of strings from an ASCII file
    Usage: out = readAsciiList(filename)
    filename is the name of the file from which to read the list
    out is the array created from the list
    The list is expected to be separated on different lines
    """
    out = np.loadtxt(filename, dtype=np.str_)
    
    return out

def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def createDir(name):
    """
    Creates directory name, if directory does not exist
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
    """

    #check python version first
    if sys.version_info >= (3,0):
        cont = input(strng+' ')
    else:
        cont = raw_input(strng+' ')

    return cont

def writeTable (data,filename):
    """
    Write a data array into a tabular format to an ASCII file
    Usage: writeTable(filename, data)
    filename is the name of the file to write the table
    data is the array to be written
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

    if (astropy.version.major >=1 and astropy.version.minor >=3):
        hdu.writeto(filename,overwrite=True)
    else:
        hdu.writeto(filename,clobber=True)

    return

def writeFits(data, filename, hdr=None, ask=True):
    """
    Write a data file to a FITS file.
    Usage writeFits(data, filename)
    data is the input data (can be more than 2 dimensions). Assumes data is stored as follows: [y-coordinates,x-coordinates,lamba-coordinates]
    filename is the name of the file to write the FITS file to.
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
         
    if (astropy.version.major >=1 and astropy.version.minor >=3):
        hdu.writeto(filename,overwrite=True)
    else:
        hdu.writeto(filename,clobber=True)

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
