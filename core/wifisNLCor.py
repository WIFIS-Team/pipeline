"""

Tools to carry out the measurement and correction for non-linearity behaviour of detector

"""

import numpy as np
import pyopencl as cl
import os

#define paths to be used for OpenCL code
path = os.path.dirname(__file__)
clCodePath = path+'/opencl_code'

def getNLCorNumpy(data, satFrame):
    """
    Routine to determine non-linearity correction coefficients using built-in python methods
    Usage: nlCoef = getNLCorNumpy(data, satFrame)
    data is the input data cube
    satFrame is an array specifying the frame number of the first saturated frame for each pixel
    nlCoeff is output that contains the non-linearity corrections for each pixel as an (ny x nx x 4) size array
    """

    #get input array dimensions
    ny = data.shape[0]
    nx = data.shape[1]

    #initialize the array
    nlCoeff = np.zeros((ny,nx,4), dtype='float32') # holds the non-linearity coefficients

    # cycle through each pixel in the data array and determine non-linearity corrections
    for x in range(ny):
        for y in range(nx):

            xtmp = np.arange(0, data.shape[2])
            ytmp = data[x,y,:] # temporary array to hold counts as function of frame
            sat = satFrame[x,y] # get first saturated frame for given pixel
            goodPixs = np.arange(0,sat) # select only the non-saturated pixels
            yGood = ytmp[goodPixs]

            if sat > 3 : # make sure number of good pixels is greater than 3 for proper fitting
            
                fitCoef = np.polyfit(goodPixs,yGood,3) # returns coefficient, in decreasing order
                polyFit = np.poly1d(fitCoef) # returns polynomial
            
                yfit = polyFit(goodPixs)
                ylin = fitCoef[2]*goodPixs + fitCoef[3]
            
                #now determine non-linear correction
                
                linRat = ylin[goodPixs]/yfit[goodPixs]
                tmpCoef = np.polyfit(yfit[goodPixs],linRat,3)
                         
                np.copyto(nlCoeff[x,y,:],tmpCoef[::-1]) # save reverse coefficients (i.e. in increasing order)
                
    return nlCoeff #need to add output of polynomial fit (a0 and a1)

def getNLCorCL(data, satFrame, nSplit):
    """
    Routine to determine non-linearity correction coefficients using OpenCL code
    Usage: nlCoef = getNLCorNumpy(data, satFrame)
    data is the input data cube
    satFrame is an array specifying the frame number of the first saturated frame for each pixel
    nSplit is a value indicating the number of temporary chunks to split the input data cube into for processing (should be an integer value of the array dimension along the x-coordinate (i.e. 2nd dimension)). This is usefull for reducing the memory consumption during operation in exchange for slightly longer processing time
    # nlCoeff is output that contains the non-linearity corrections for each pixel as an (ny x nx x 4) size array
    """

    #get input array dimensions and set working size of temporary chunks
    ny = data.shape[0]
    nx = int(data.shape[1]/nSplit)
    ntime = data.shape[2]

    #initialize the output arrays
    nlCoeff = np.zeros((ny,data.shape[1],4), dtype='float32') # holds the non-linearity coefficients
    zpntImg = np.zeros((ny,data.shape[1]),dtype='float32')
    rampImg = np.zeros((ny,data.shape[1]),dtype='float32')
    
    #get OpenCL context object
    ctx = cl.create_some_context(interactive=True)
    queue = cl.CommandQueue(ctx)

    #read in OpenCL code
    filename = clCodePath+'/getNLCoeff.cl'
    f = open(filename, 'r')
    fstr = "".join(f.readlines())
    
    #build opencl program and get memory flags
    program = cl.Program(ctx, fstr).build()
    mf = cl.mem_flags   

    #now split input array into multiple chunks, if specified
    if (nSplit > 1):
        for n in range(nSplit):
            #set temporary arrays
            dTmp = np.array(data[:, n*nx:(n+1)*nx,:].astype('float32'))
            sTmp = np.array(satFrame[:,n*nx:(n+1)*nx].astype('int32'))

            zpnt_array =np.zeros((ny,nx), dtype='float32')
            ramp_array = np.zeros((ny,nx), dtype='float32')
            a0_array = np.zeros((ny,nx), dtype='float32')
            a1_array = np.zeros((ny,nx), dtype='float32')
            a2_array = np.zeros((ny,nx), dtype='float32')
            a3_array = np.zeros((ny,nx), dtype='float32')

            #create OpenCL buffers
            data_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dTmp)
            sat_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sTmp)
            a0_buf = cl.Buffer(ctx, mf.WRITE_ONLY, a0_array.nbytes)
            a1_buf = cl.Buffer(ctx, mf.WRITE_ONLY, a1_array.nbytes)
            a2_buf = cl.Buffer(ctx, mf.WRITE_ONLY, a2_array.nbytes)
            a3_buf = cl.Buffer(ctx, mf.WRITE_ONLY, a3_array.nbytes)
            zpnt_buf = cl.Buffer(ctx, mf.WRITE_ONLY, zpnt_array.nbytes)
            ramp_buf = cl.Buffer(ctx, mf.WRITE_ONLY, ramp_array.nbytes)

            #run the opencl code
            program.lsfit.set_scalar_arg_dtypes([np.uint32, np.uint32, None, None,None,None, None, None, None, None])
            program.lsfit(queue,(ny,nx),None,np.uint32(nx), np.uint32(ntime),data_buf,a0_buf,a1_buf,a2_buf,a3_buf,sat_buf, zpnt_buf, ramp_buf)
            cl.enqueue_copy(queue, a0_buf, a0_array).wait()
            cl.enqueue_copy(queue, a1_buf, a1_array).wait()
            cl.enqueue_copy(queue, a2_buf, a2_array).wait()
            cl.enqueue_copy(queue, a3_buf, a3_array).wait()
            cl.enqueue_copy(queue, zpnt_buf, zpnt_array).wait()
            cl.enqueue_copy(queue, ramp_buf, ramp_array).wait()

            #copy output from OpenCL code into output array
            np.copyto(nlCoeff[:,n*nx:(n+1)*nx,0],a0_array)
            np.copyto(nlCoeff[:,n*nx:(n+1)*nx,1],a1_array)
            np.copyto(nlCoeff[:,n*nx:(n+1)*nx,2],a2_array)
            np.copyto(nlCoeff[:,n*nx:(n+1)*nx,3],a3_array)
            np.copyto(zpntImg[:,n*nx:(n+1)*nx],zpnt_array)
            np.copyto(rampImg[:,n*nx:(n+1)*nx],ramp_array)            
    else:
        #set temporary arrays
        
        a0_array = np.zeros((ny,nx), dtype='float32')
        a1_array = np.zeros((ny,nx), dtype='float32')
        a2_array = np.zeros((ny,nx), dtype='float32')
        a3_array = np.zeros((ny,nx), dtype='float32')
           
        #create OpenCL buffers
        data_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data.astype('float32'))
        sat_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=satFrame.astype('int32'))
        a0_buf = cl.Buffer(ctx, mf.WRITE_ONLY, a0_array.nbytes)
        a1_buf = cl.Buffer(ctx, mf.WRITE_ONLY, a1_array.nbytes)
        a2_buf = cl.Buffer(ctx, mf.WRITE_ONLY, a2_array.nbytes)
        a3_buf = cl.Buffer(ctx, mf.WRITE_ONLY, a3_array.nbytes)
        zpnt_buf = cl.Buffer(ctx, mf.WRITE_ONLY, zpntImg.nbytes)
        ramp_buf = cl.Buffer(ctx, mf.WRITE_ONLY, rampImg.nbytes)

        #run the opencl code
        program.lsfit.set_scalar_arg_dtypes([np.uint32, np.uint32, None, None,None,None, None, None, None, None])
        program.lsfit(queue,(ny,nx),None,np.uint32(nx), np.uint32(ntime),data_buf,a0_buf,a1_buf,a2_buf,a3_buf,sat_buf, zpnt_buf, ramp_buf)
        cl.enqueue_copy(queue, a0_buf, a0_array).wait()
        cl.enqueue_copy(queue, a1_buf, a1_array).wait()
        cl.enqueue_copy(queue, a2_buf, a2_array).wait()
        cl.enqueue_copy(queue, a3_buf, a3_array).wait()
        cl.enqueue_copy(queue, zpnt_buf, zpntImg).wait()
        cl.enqueue_copy(queue, ramp_buf, rampImg).wait()

        #copy output from OpenCL code into output array
        np.copyto(nlCoeff[:,:,0],a0_array)
        np.copyto(nlCoeff[:,:,1],a1_array)
        np.copyto(nlCoeff[:,:,2],a2_array)
        np.copyto(nlCoeff[:,:,3],a3_array)

    #modify variables to reduce memory consumption
    dTmp = 0
    data_buf = 0
    
    return nlCoeff, zpntImg, rampImg

def applyNLCorCL(data, nlCoeff, nSplit):
    """
    Routine to apply non-linearity correction coefficients using OpenCL code to input data (can be a data cube)
    Usage: applyNLCorCL(data, nlCoeff, nSplit)
    data is the input data (image or cube)
    nlCoeff is an array specifying the non-linearity correction coefficients to use for correcting the input data
    nSplit is a value indicating the number of temporary chunks to split the input data cube into for processing (should be an integer value of the array dimension along the x-coordinate (i.e. 2nd dimension)). This is usefull for reducing the memory consumption during operation in exchange for slightly longer processing time
    The data array is corrected in place
    """

    #get input data array dimensions
    ny = data.shape[0]
    nx = int(data.shape[1]/nSplit)
    ntime = data.shape[2]

    #get OpenCL context object, can set to fixed value if wanted
    ctx = cl.create_some_context(interactive=True)
    queue = cl.CommandQueue(ctx)

    #read in OpenCL code
    filename = clCodePath+'/applyNLCor.cl'
    f = open(filename, 'r')
    fstr = "".join(f.readlines())

    #build opencl program and get memory flags
    program = cl.Program(ctx, fstr).build()
    mf = cl.mem_flags   

    #if specified, split input data into separate chunks for processing
    if (nSplit > 1):
        for n in range(nSplit):
            #set temporary arrays
            dTmp = np.array(data[:, n*nx:(n+1)*nx,:].astype('float32'))
            
            a0_array = np.zeros((ny,nx), dtype='float32')
            a1_array = np.zeros((ny,nx), dtype='float32')
            a2_array = np.zeros((ny,nx), dtype='float32')
            a3_array = np.zeros((ny,nx), dtype='float32')

            np.copyto(a0_array, nlCoeff[:,n*nx:(n+1)*nx,0])
            np.copyto(a1_array, nlCoeff[:,n*nx:(n+1)*nx,1])
            np.copyto(a2_array, nlCoeff[:,n*nx:(n+1)*nx,2])
            np.copyto(a3_array, nlCoeff[:,n*nx:(n+1)*nx,3])

            #create OpenCL buffers
            data_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=dTmp)
            a0_buf = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a0_array)
            a1_buf = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a1_array)
            a2_buf = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a2_array)
            a3_buf = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a3_array)
            
            #run the opencl code
            program.nlcor.set_scalar_arg_dtypes([np.uint32,np.uint32, None, None,None,None, None])
            program.nlcor(queue,(ny,nx, ntime),None,np.uint32(nx), np.uint32(ntime),data_buf,a0_buf,a1_buf,a2_buf,a3_buf)
            cl.enqueue_copy(queue, data_buf, dTmp).wait()

            #replace input data with non-linearity corrected values
            np.copyto(data[:,n*nx:(n+1)*nx,:],dTmp)            
    else:
        #set temporary arrays
        
        a0_array = np.zeros((ny,nx), dtype='float32')
        a1_array = np.zeros((ny,nx), dtype='float32')
        a2_array = np.zeros((ny,nx), dtype='float32')
        a3_array = np.zeros((ny,nx), dtype='float32')

        np.copyto(a0_array, nlCoeff[:,:,0])
        np.copyto(a1_array, nlCoeff[:,:,1])
        np.copyto(a2_array, nlCoeff[:,:,2])
        np.copyto(a3_array, nlCoeff[:,:,3])

        #create OpenCL buffers
        data_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=data.astype('float32'))
        a0_buf = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a0_array)
        a1_buf = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a1_array)
        a2_buf = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a2_array)
        a3_buf = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a3_array)
       
        #run the opencl code
        program.nlcor.set_scalar_arg_dtypes([int,int, None, None,None,None, None])
        program.nlcor(queue,(ny,nx, ntime),None,np.uint32(nx), np.uint32(ntime),data_buf,a0_buf,a1_buf,a2_buf,a3_buf)
        cl.enqueue_copy(queue, data_buf, data).wait()

    #modify variables to reduce memory consumption
    dTmp = 0
    data_buf = 0

    return
