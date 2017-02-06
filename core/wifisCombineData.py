"""

Tools to process a sequence of exposures to create a single image

"""

import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl

def upTheRampCL(intTime, data, satFrame, nSplit):
    """Routine to process a sequence of up-the-ramp exposures using OpenCL code
    Usage: output = upTheRampCL(intTime,data, satFram, nSplit)
    intTime is input array indicating the integration times of each frame the data cube in increasing order
    data is the input data cube
    satFrame is an input image indicating the frame number of the first saturated frame in the sequence per pixel
    nSplit is input variable to allow for splitting up the processing of the workload, which is needed to reduce memory to reduce memory errors/consumption for large datasets
    output is a 2D image, where each pixel represents the flux (slope/time) per pixel
    *** NOTE: Currently openCL code outputs both offset and slope of the fit. Can remove offset as not needed ***
    """

    #get dimenions of input images
    ny = data.shape[0]
    nx = data.shape[1]/nSplit
    ntime = data.shape[2]

    #create tempory array to hold the output image
    outImg = np.zeros((ny,data.shape[1]), dtype='float32')

    #start OpenCL portion

    #get OpenCL context object, can set to fixed value if wanted
    ctx = cl.create_some_context(interactive=True)
    queue = cl.CommandQueue(ctx)
    
    filename = 'opencl_code/getlinfit.cl'
    f = open(filename, 'r')
    fstr = "".join(f.readlines())
    program = cl.Program(ctx, fstr).build()
    mf = cl.mem_flags   

    #create OpenCL buffers
    time_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=intTime)

    #only create temporary arrays if needed to avoid excess RAM usage
    if (nSplit > 1):
        for n in range(0, nSplit):
            #create temporary arrays to hold portions of data cube
            dTmp = np.array(data[:,n*nx:(n+1)*nx,:])
            sTmp = np.array(satFrame[:,n*nx:(n+1)*nx])
            oTmp = np.zeros((ny,nx),dtype='float32')
            a0_array = np.zeros((ny,nx), dtype='float32')
        
            #create OpenCL buffers
            data_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dTmp)
            sat_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sTmp)
            a0_buf = cl.Buffer(ctx, mf.WRITE_ONLY, a0_array.nbytes)
            a1_buf = cl.Buffer(ctx, mf.WRITE_ONLY, oTmp.nbytes)

            #set openCL arguments and run openCL code
            program.lsfit.set_scalar_arg_dtypes([np.uint32, np.uint32, None, None,None,None, None])
            program.lsfit(queue,(ny,nx),None,np.uint32(nx), np.uint32(ntime),time_buf, data_buf,a0_buf,a1_buf,sat_buf)
            
            cl.enqueue_read_buffer(queue, a0_buf, a0_array).wait()
            cl.enqueue_read_buffer(queue, a1_buf, oTmp).wait()

            #copy to output array
            np.copyto(outImg[:,n*nx:(n+1)*nx],oTmp)
    else:
        
        a0_array = np.zeros((ny,nx), dtype='float32')

        #create OpenCL buffers
        data_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
        sat_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=satFrame)
        a0_buf = cl.Buffer(ctx, mf.WRITE_ONLY, a0_array.nbytes)
        a1_buf = cl.Buffer(ctx, mf.WRITE_ONLY, outImg.nbytes)
        
        program.lsfit.set_scalar_arg_dtypes([np.uint32, np.uint32, None, None,None,None, None])
        program.lsfit(queue,(ny,nx),None,np.uint32(nx), np.uint32(ntime),time_buf, data_buf,a0_buf,a1_buf,sat_buf)
        
        cl.enqueue_read_buffer(queue, a0_buf, a0_array).wait()
        cl.enqueue_read_buffer(queue, a1_buf, outImg).wait()
    return outImg

def createMaster(data):
    """Creates a collapsed image from a cube of images as a simple median of the images.
    Usage output = createMaster(data)
    data is the input data cube containing a series of images
    output is a 2D image created from the series of input images
    """

    out = np.median(data,axis=2)
    
    return out

def fowlerSampling(intTime, data, satFrame):
    
    # Routine to process a sequence of Fowler exposures
    # intTime is input array indicating the integration times of each frame in the data cube
    # data is the input data cube
    # satFrame is an input image providing the first saturated frame in the sequence per pixel
    # only use pairs without saturated partner -- still do implement
    
    nx = data.shape[1]
    ny = data.shape[0]
    nFrames = data.shape[2]

    #go through each pixel and only include non-saturated pairs in averaging

    #for i in range(ny):
    #    for j in range(nx):
            #do nothing for the moment
            
    nFowler = nFrames/2
    pairs = np.zeros((ny,nx, nFowler), dtype='float32')
    pairInts = np.zeros(nFowler)
    
    #carry out pairwise subtraction
    #and determine pairwise integration time
    for i in range(nFowler):
        np.copyto(pairs[:,:,i],data[:,:,i+nFowler]-data[:,:,i])
        pairInts[i] = intTime[i+nFowler] - intTime[i]
        
    #now get median of this array
    outImg = np.median(pairs, axis=2)/np.mean(pairInts)

    return outImg
    
    
