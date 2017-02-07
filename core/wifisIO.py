"""

Tools to handle reading a writing from files and images

"""

import numpy as np
import astropy.io.fits as fits
import glob
import re 
import os
import sys

def readImgFromFile(file):
    """
    Reads a FITS image from the specified file
    Usage: outImg = readImgFromFile(file)
    file is a text input corresponding to the file name
    outImg is the returned image array
    """

    #get FITS file information
    tmp = fits.open(file)
    nx = tmp[0].header['NAXIS2']
    ny = tmp[0].header['NAXIS1']

    #check if there are multiple HDUs
    nz = len(tmp)
    
    if (nz > 1):
        outImg = np.zeros((ny,nx, nz), dtype='float32')
        for i in range(nz):
            np.copyto(outImg[:,:,i], tmp[i].data)
    else:
        outImg = np.zeros((ny,nx), dtype='float32')
        np.copyto(outImg, tmp[0].data)

    #get header
    hdr = tmp[-1].header
    
    tmp.close()
    return outImg, hdr

def readImgsFromList(list):
    """
    Read a set of images from the list provided
    Usage: outImg, outTime = readImgsFromList(list)
    list is an array or python list containing the names of each file from which to read
    outImg is the returned data cube
    outTime is the returned array containing the integration times for each image in the list
    """

    #open first file to get image dimensions
    tmp = fits.open(list[0])
    nx = tmp[0].header['NAXIS2']
    ny = tmp[0].header['NAXIS1']
    listLen = len(list)
    
    outImg = np.zeros((ny,nx,listLen), dtype='float32')
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
    hdr = tmp[-1].header

    return outImg, outTime, hdr


def readImgsFromAsciiList(filename):
    """
    Read ASCII file containing a list of strings (filenames) into a python list.
    Usage: output = readAsciiList(filename)
    filename is the name of the input ASCII file containing the list
    output is the returned python array containing the list of strings
    """
    
    list = np.loadtxt(filename, dtype=np.str_)
    output, outime, hdr = readImgsFromList(list)

    return output, outtime, hdr

def writeFits(data, filename, hdr=None):
    """
    Write a data file to a FITS image.
    Usage writeFits(data, filename)
    data is the input data (can contain multiple HDUs, i.e. more than 2 dimensions)
    filename is the name of the file to write the FITS file to.

    """

    if (hdr != None):
        #add additional FITS keywords
        prihdr = fits.Header()
        prihdr = hdr
    else:
        prihdr = None

    #find out if the image requires multiple HDUs
    if (data.ndim > 2):
        nFrames = data.shape[2]

        list = [fits.PrimaryHDU(data[:,:,0], header=prihdr)]

        for i in range(1,nFrames):
            list.append(fits.ImageHDU(data[:,:,i]))

        hdu = fits.HDUList(list)
    else:
        hdu = fits.PrimaryHDU(data, header=prihdr)


    hdu.writeto(filename,clobber=True)

def readImgsFromFolder(folderName):
    """
    read all H2RG files from specified folder.
    Usage: output = readImgsFromFolder(folderName)
    folderName is the name of the folder from which to read files
    output is the data cube generated from the files contained in the specified folder.    
    """
    list = glob.glob(folderName+'/H2*fits')
    list = sorted_nicely(list)
    
    output,outtime,hdr = readImgsFromList(list)    
    
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

    if os.path.exists('processed'):
        pass
    else:
        os.mkdir('processed')

    return

def userInput(strng):
    """
    Handles user input requests
    """

    #check python version first
    if sys.version_info >= (3,0):
        cont = input(strng)
    else:
        cont = raw_input(strng)

    return cont

def writeTable (filename, data):
    """
    Write a data array into a tabular format to an ASCII file
    Usage: writeTable(filename, data)
    filename is the name of the file to write the table
    data is the array to be written
    """

    np.savetxt(filename, data)

    return 
