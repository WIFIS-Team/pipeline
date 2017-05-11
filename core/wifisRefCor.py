"""

Tools to correct images for channel and row bias 

"""

import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl
import os

path = os.path.dirname(__file__)
clCodePath = path+'/opencl_code'

# Channel Correct
def channel(data,nChannel):
    """
    Use reference pixels to correct for channel bias using built-in python functions
    Usage: channel(data, nChannel)
    data is the input data (single image or cube)
    nChannel is an integer specifying the number of channels used when obtaining the data
    The input data is modified in place
    """

    #get array dimensions
    nx = data.shape[1]
    ny = data.shape[0]

    #get number of dimensions
    nDims = len(data.shape)
    if (nDims > 2):
        nFrames = data.shape[2]
    else:
        nFrames = 1
        
    csize = int(nx/nChannel)

    for n in range(nFrames):
        if (nFrames > 1):
            dTmp = data[:,:, n]
        else:
            dTmp = data

        for i in range(nChannel):
            corrfactor = np.mean(np.concatenate([dTmp[0:4,i*csize:i*csize+csize],dTmp[-4:,i*csize:i*csize+csize]]))
            dTmp[:,i*csize:(i+1)*csize] -= corrfactor

    return

def row(data,winsize):
    """
    Use reference pixels to correct for row bias using a moving average
    Usage: row(data, winsize)
    data is the input data (single image or cube)
    winsize is an integer specifying the number of row to use when obtaining the average correction factor. The code will use the +winsize/2 and -winsize/2 rows surrounding the current row (when possible) for determining the correction.
    The input data is modified in place
    """

    #get input data dimensions
    nx = data.shape[1]
    ny = data.shape[0]

    #get number of dimensions
    nDims = len(data.shape)
    if (nDims > 2):
        nFrames = data.shape[2]
    else:
        nFrames = 1

    if nFrames > 1:
        for n in range(nFrames):
            dTmp = data[:,:,n]

            for i in range(0, winsize/2):
                corrfactor = np.mean(np.concatenate([dTmp[0:i+winsize/2+1,0:4],dTmp[0:i + winsize/2+1,-4:]]))
                dTmp[i,4:-4] -= corrfactor
                
            for i in range(winsize/2,nx-winsize/2-1):
                corrfactor = np.mean(np.concatenate([dTmp[i-winsize/2:i+winsize/2+1,0:4],dTmp[i-winsize/2:i + winsize/2+1,-4:]]))
                dTmp[i,4:-4] -= corrfactor

            for i in range(nx-winsize/2-1,nx):
                corrfactor = np.mean(np.concatenate([dTmp[i-winsize/2:nx,0:4],dTmp[i-winsize/2:nx,-4:]]))
                dTmp[i,4:-4] -= corrfactor
    else:
        for i in range(0, winsize/2):
            corrfactor = np.mean(np.concatenate([data[0:i+winsize/2+1,0:4],data[0:i + winsize/2+1,-4:]]))
            data[i,4:-4] -= corrfactor
            
        for i in range(winsize/2,nx-winsize/2-1):
            corrfactor = np.mean(np.concatenate([data[i-winsize/2:i+winsize/2+1,0:4],data[i-winsize/2:i + winsize/2+1,-4:]]))
            data[i,4:-4] -= corrfactor
            
        for i in range(nx-winsize/2-1,nx):
            corrfactor = np.mean(np.concatenate([data[i-winsize/2:nx,0:4],data[i-winsize/2:nx,-4:]]))
            data[i,4:-4] -= corrfactor
                    
    return

def channelCL(data,nChannel):
    """
    Use reference pixels to correct for channel bias using OpenCL code
    Usage: channel(data, nChannel)
    data is the input data (single image or cube)
    nChannel is an integer specifying the number of channels used when obtaining the data
    The input data is modified in place
    """

    if (np.issubdtype(data.dtype,np.integer)):
        raise TypeError('**** INPUT DATA TYPE CANNOT BE INTEGER. CONVERT TO FLOAT ****')

    
    #get OpenCL context object, can set to fixed value if wanted
    ctx = cl.create_some_context(interactive=True)
    queue = cl.CommandQueue(ctx)

    #get input array dimensions
    ny = data.shape[0]
    nx = int(data.shape[1]/nChannel)

    #get number of dimensions
    nDims = len(data.shape)
    if (nDims > 2):
        nFrames = data.shape[2]
    else:
        nFrames = 1

    filename = clCodePath+'/refcorrect.cl'
    f = open(filename, 'r')
    fstr = "".join(f.readlines())

    #build opencl program and get memory flags
    program = cl.Program(ctx, fstr).build()
    mf = cl.mem_flags   

    if (nChannel > 1):
            
        for n in range(nChannel):

            #set temporary arrays
            if (nFrames>1):
                dTmp = np.array(data[:, n*nx:(n+1)*nx,:].astype('float32'))
            else:
                dTmp = np.array(data[:, n*nx:(n+1)*nx].astype('float32'))

            #create OpenCL buffers
            data_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=dTmp)

            #run the opencl code
            program.channel.set_scalar_arg_dtypes([np.uintc,np.uintc,np.uintc,None])
            program.channel(queue,(nFrames,),None,np.uintc(ny), np.uintc(nx),np.uintc(nFrames),data_buf)
            cl.enqueue_read_buffer(queue, data_buf, dTmp).wait()

            #replace the input data with output from OpenCL
            np.copyto(data[:,n*nx:(n+1)*nx,:],dTmp)            
    else:
        #set temporary arrays
        
        #create OpenCL buffers
        
        data_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=data.astype('float32'))
        dTmp = np.empty(data.shape, dtype='float32')
        
        #run the opencl code
        program.channel.set_scalar_arg_dtypes([np.uint32,np.uint32, np.uint32, None])
        program.channel(queue,(nFrames,),None,np.uint32(ny), np.uint32(nx),np.uint32(nFrames),data_buf)
        cl.enqueue_read_buffer(queue, data_buf, dTmp).wait()
        data[:] = dTmp[:]

    #modify variables to reduce memory consumption
    dTmp = 0
    data_buf = 0
    
    return

def rowCL(data,winSize,nSplit):
    """
    Use reference pixels to correct for row bias using a moving average and OpenCL code
    Usage: row(data, winsize, nSplit)
    data is the input data (single image or cube)
    winsize is an integer specifying the number of rows to use when obtaining the average correction factor. The code will use the +winsize/2 and -winsize/2 rows surrounding the current row (when possible) for determining the correction (i.e. total number of rows will be at most winsize+1).
    nSplit is an integer specifying the number of separate OpenCL calls to split the workload into. MUST be an integer multiple of the number of frames in input cube. Fewer is faster, but uses more ram, so can potentially be very slow if SWAP space is used or fail if buffer size is too large for OpenCL device.
    The input data is modified in place
    """
   
    if (np.issubdtype(data.dtype,np.integer)):
        raise TypeError('**** INPUT DATA TYPE CANNOT BE INTEGER. CONVERT TO FLOAT ****')
    
    #get OpenCL context object, can set to fixed value if wanted
    ctx = cl.create_some_context(interactive=True)
    queue = cl.CommandQueue(ctx)

    #get input array dimensions
    ny = data.shape[0]
    nx = data.shape[1]

    #get number of dimensions
    nDims = len(data.shape)
    if (nDims > 2):
        nFrames = data.shape[2]
    else:
        nFrames = 1

    filename = clCodePath+'/refcorrect.cl'
    f = open(filename, 'r')
    fstr = "".join(f.readlines())

    #build opencl program and get memory flags
    program = cl.Program(ctx, fstr).build()
    mf = cl.mem_flags   

    if(nFrames >1 and nSplit>1):

        if (nFrames % nSplit > 0):
            print("Error! nFrames MUST be an integer multiple of nSplit")
            return
        else:
            nt = int(nFrames/nSplit)

            for n in range(0,nSplit):
                strt = n*nt
                end = (n+1)*nt
                size = nt

                #set temporary arrays
                dTmp = np.zeros((data.shape[0], data.shape[1],size),dtype='float32')
                dTmp = data[:,:,strt:end].astype('float32')
            
                #create OpenCL buffers
                data_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=dTmp)

                #run the opencl code
                program.row.set_scalar_arg_dtypes([np.uintc,np.uintc,np.uintc,np.uintc,None])
                program.row(queue,(ny,size),None,np.uintc(ny), np.uintc(nx),np.uintc(size),np.uintc(winSize),data_buf)
                cl.enqueue_read_buffer(queue, data_buf, dTmp).wait()
                
                #replace the input data with output from OpenCL
                np.copyto(data[:,:,strt:end],dTmp)
    else:
        #set temporary arrays
        
        #create OpenCL buffers
        data_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=data.astype('float32'))
        
        #run the opencl code
        program.row.set_scalar_arg_dtypes([np.uintc,np.uintc, np.uintc, np.uintc,None])
        program.row(queue,(ny,nFrames),None,np.uintc(ny), np.uintc(nx),np.uintc(nFrames),np.uintc(winSize),data_buf)
        cl.enqueue_read_buffer(queue, data_buf, data).wait()


     #modify variables to reduce memory consumption
    dTmp = 0
    data_buf = 0

    return

