"""

Tools to correct images for channel and row bias 

"""

import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl

# Channel Correct
def channel(data,nFrames,nChannel):
    """
    Use reference pixels to correct for channel bias using built-in python functions
    Usage: channel(data, nFrames, nChannel)
    data is the input data (single image or cube)
    nFrames is an integer specifying the number of frames in the input data cube
    nChannel is an integer specifying the number of channels used when obtaining the data
    The input data is modified in place
    """

    #get array dimensions
    nx = data.shape[1]
    ny = data.shape[0]
    
    csize = nx/nChannel

    for n in range(nFrames):
        if (nFrames > 1):
            dTmp = data[:,:, n]
        else:
            dTmp = data

        for i in range(nChannel):
            corrfactor = np.mean(np.concatenate([dTmp[0:4,i*csize:i*csize+csize],dTmp[-4:,i*csize:i*csize+csize]]))
            dTmp[:,i*csize:(i+1)*csize] -= corrfactor

    return

def row(data,nFrames,winsize):
    """
    Use reference pixels to correct for row bias using a moving average
    Usage: row(data, nFrames, winsize)
    data is the input data (single image or cube)
    nFrames is an integer specifying the number of frames in the input data cube
    winsize is an integer specifying the number of row to use when obtaining the average correction factor. The code will use the +winsize/2 and -winsize/2 rows surrounding the current row (when possible) for determining the correction.
    The input data is modified in place
    """

    #get input data dimensions
    nx = data.shape[1]
    ny = data.shape[0]

    if nFrames > 1:
        for n in range(nFrames):
            dTmp = data[:,:,n]

            for i in range(0, winsize/2):
                corrfactor = np.mean(np.concatenate([dTmp[0:i+winsize/2,0:4],dTmp[0:i + winsize/2,-4:]]))
                dTmp[i,4:-4] -= corrfactor
                
                for i in range(winsize/2,nx):
                    corrfactor = np.mean(np.concatenate([dTmp[i-winsize/2:i+winsize/2,0:4],dTmp[i-winsize/2:i + winsize/2,-4:]]))
                    dTmp[i,4:-4] -= corrfactor
    else:
        for i in range(0, winsize/2):
            corrfactor = np.mean(np.concatenate([data[0:i+winsize/2,0:4],data[0:i + winsize/2,-4:]]))
            data[i,4:-4] -= corrfactor
                
            for i in range(winsize/2,nx):
                corrfactor = np.mean(np.concatenate([data[i-winsize/2:i+winsize/2,0:4],data[i-winsize/2:i + winsize/2,-4:]]))
                data[i,4:-4] -= corrfactor
                    
    return

def channelCL(data,nFrames,nChannel):
    """
    Use reference pixels to correct for channel bias using OpenCL code
    Usage: channel(data, nFrames, nChannel)
    data is the input data (single image or cube)
    nFrames is an integer specifying the number of frames in the input data cube
    nChannel is an integer specifying the number of channels used when obtaining the data
    The input data is modified in place
    """
    
    #get OpenCL context object, can set to fixed value if wanted
    ctx = cl.create_some_context(interactive=True)
    queue = cl.CommandQueue(ctx)

    #get input array dimensions
    ny = data.shape[0]
    nx = data.shape[1]/nChannel
        
    filename = 'opencl_code/refcorrect.cl'
    f = open(filename, 'r')
    fstr = "".join(f.readlines())

    #build opencl program and get memory flags
    program = cl.Program(ctx, fstr).build()
    mf = cl.mem_flags   

    if (nChannel > 1):
            
        for n in range(nChannel):

            #set temporary arrays
            if (nFrames>1):
                dTmp = np.array(data[:, n*nx:(n+1)*nx,:])
            else:
                dTmp = np.array(data[:, n*nx:(n+1)*nx])

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
        data_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=data)

        #run the opencl code
        program.channel.set_scalar_arg_dtypes([np.uint32,np.uint32, np.uint32, None])
        program.channel(queue,(nFrames,),None,np.uint32(ny), np.uint32(nx),np.uint32(nFrames),data_buf)
        cl.enqueue_read_buffer(queue, data_buf, data).wait()
        
    return

