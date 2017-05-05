"""

Tools to process a sequence of exposures to create a single image

"""

import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl

def upTheRampCRRejectCL(intTime, data, satFrame, nSplit):
    """Routine to process a sequence of up-the-ramp exposures using OpenCL code with the ability to reject influence of non-saturated cosmic ray hits
    Usage: output = upTheRampCRRejectCL(intTime,data, satFrame, nSplit)
    intTime is input array indicating the integration times of each frame in the data cube in increasing order
    data is the input data cube
    satFrame is an input image indicating the frame number of the first saturated frame in the sequence per pixel
    nSplit is input variable to allow for splitting up the processing of the workload, which is needed to reduce memory to reduce memory errors/consumption for large datasets
    output is a 2D image, where each pixel represents the flux (slope/time)
    *** NOTE: Currently openCL code is hard-coded to handle a maximum of 1000 frames. Adjust code if needed. ***
    """

    #get dimenions of input images
    ny = data.shape[0]
    nx = int(data.shape[1]/nSplit)
    ntime = data.shape[2]

    #create tempory array to hold the output image
    outImg = np.zeros((ny,data.shape[1]), dtype='float32')

    #start OpenCL portion

    #get OpenCL context object, can set to fixed value if wanted
    ctx = cl.create_some_context(interactive=True)
    queue = cl.CommandQueue(ctx)
    
    filename = 'opencl_code/compgradient.cl'
    f = open(filename, 'r')
    fstr = "".join(f.readlines())
    program = cl.Program(ctx, fstr).build()
    mf = cl.mem_flags   

    #only create temporary arrays if needed to avoid excess RAM usage
    if (nSplit > 1):
        for n in range(0, nSplit):
            #create temporary arrays to hold portions of data cube
            dTmp = np.array(data[:,n*nx:(n+1)*nx,:].astype('float32'))
            sTmp = np.array(satFrame[:,n*nx:(n+1)*nx].astype('int32'))
            oTmp = np.zeros((ny,nx),dtype='float32')
        
            #create OpenCL buffers
            data_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dTmp)
            sat_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sTmp)
            oTmp_buf = cl.Buffer(ctx, mf.WRITE_ONLY, oTmp.nbytes)

            #set openCL arguments and run openCL code
            program.compmeangrad.set_scalar_arg_dtypes([np.uint32, np.uint32, None, None,None])
            program.compmeangrad(queue,(ny,nx),None,np.uint32(nx), np.uint32(ntime),data_buf,sat_buf,oTmp_buf)
            
            cl.enqueue_read_buffer(queue, oTmp_buf, oTmp).wait()

            #copy to output array
            np.copyto(outImg[:,n*nx:(n+1)*nx],oTmp)
    else:
        
        #create OpenCL buffers
        data_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data.astype('float32'))
        sat_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=satFrame.astype('int32'))
        out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, outImg.nbytes)
        
        program.compmeangrad.set_scalar_arg_dtypes([np.uint32, np.uint32, None, None,None])
        program.compmeangrad(queue,(ny,nx),None,np.uint32(nx), np.uint32(ntime), data_buf,sat_buf,out_buf)
        
        cl.enqueue_read_buffer(queue, out_buf, outImg).wait()

    outImg/=(np.mean(np.gradient(intTime)))

    #modify variables to reduce memory consumption
    dTmp = 0
    data_buf = 0
    
    return outImg


def upTheRampCL(intTime, data, satFrame, nSplit):
    """Routine to process a sequence of up-the-ramp exposures using OpenCL code
    Usage: output = upTheRampCL(intTime,data, satFram, nSplit)
    intTime is input array indicating the integration times of each frame in the data cube in increasing order
    data is the input data cube
    satFrame is an input image indicating the frame number of the first saturated frame in the sequence per pixel
    nSplit is input variable to allow for splitting up the processing of the workload, which is needed to reduce memory to reduce memory errors/consumption for large datasets
    output is a 2D image, where each pixel represents the flux (slope/time)
    *** NOTE: Currently openCL code outputs both offset and slope of the fit. Can remove offset as not needed ***
    """

    #get dimenions of input images
    ny = data.shape[0]
    nx = int(data.shape[1]/nSplit)
    ntime = data.shape[2]

    #create tempory array to hold the output image
    outImg = np.zeros((ny,data.shape[1]), dtype='float32')
    varImg = np.zeros((ny,data.shape[1]),dtype='float32')
    zpntImg = np.zeros((ny,data.shape[1]),dtype='float32')
    
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
    time_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=intTime.astype('float32'))

    #only create temporary arrays if needed to avoid excess RAM usage
    if (nSplit > 1):
        for n in range(0, nSplit):
            #create temporary arrays to hold portions of data cube
            dTmp = np.array(data[:,n*nx:(n+1)*nx,:].astype('float32'))
            sTmp = np.array(satFrame[:,n*nx:(n+1)*nx].astype('int32'))
            oTmp = np.zeros((ny,nx),dtype='float32')
            a0_array = np.zeros((ny,nx), dtype='float32')
            var_array = np.zeros((ny,nx),dtype='float32')
        
            #create OpenCL buffers
            data_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dTmp)
            sat_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sTmp)
            a0_buf = cl.Buffer(ctx, mf.WRITE_ONLY, a0_array.nbytes)
            a1_buf = cl.Buffer(ctx, mf.WRITE_ONLY, oTmp.nbytes)
            var_buf = cl.Buffer(ctx, mf.WRITE_ONLY, var_array.nbytes)
            
            #set openCL arguments and run openCL code
            program.lsfit.set_scalar_arg_dtypes([np.uint32, np.uint32, None, None,None,None, None, None])
            program.lsfit(queue,(ny,nx),None,np.uint32(nx), np.uint32(ntime),time_buf, data_buf,a0_buf,a1_buf,sat_buf, var_buf)
            
            cl.enqueue_read_buffer(queue, a0_buf, a0_array).wait()
            cl.enqueue_read_buffer(queue, a1_buf, oTmp).wait()
            cl.enqueue_read_buffer(queue, a1_buf, var_array).wait()

            #copy to output array
            np.copyto(outImg[:,n*nx:(n+1)*nx],oTmp)
            np.copyto(varImg[:,n*nx:(n+1)*nx],var_array)
            np.copyto(zpntImg[:,n*nx:(n+1)*nx],a0_array)
    else:
        
        #create OpenCL buffers
        data_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data.astype('float32'))
        sat_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=satFrame.astype('int32'))
        a0_buf = cl.Buffer(ctx, mf.WRITE_ONLY, zpntImg.nbytes)
        a1_buf = cl.Buffer(ctx, mf.WRITE_ONLY, outImg.nbytes)
        var_buf = cl.Buffer(ctx, mf.WRITE_ONLY, varImg.nbytes)

        program.lsfit.set_scalar_arg_dtypes([np.uint32, np.uint32, None, None,None,None, None,None])
        program.lsfit(queue,(ny,nx),None,np.uint32(nx), np.uint32(ntime),time_buf, data_buf,a0_buf,a1_buf,sat_buf, var_buf)
        
        cl.enqueue_read_buffer(queue, a0_buf, zpntImg).wait()
        cl.enqueue_read_buffer(queue, a1_buf, outImg).wait()
        cl.enqueue_read_buffer(queue, var_buf, varImg).wait()
                
    #modify variables to reduce memory consumption
    dTmp = 0
    data_buf = 0
    a0_array=0
    
    return outImg, zpntImg, varImg

def createMaster(data):
    """Creates a collapsed image from a cube of images as a simple median of the images.
    Usage output = createMaster(data)
    data is the input data cube containing a series of images
    output is a 2D image created from the series of input images
    """

    out = np.median(data,axis=2)
    
    return out

def fowlerSampling(intTime, data, satFrame):
    """
    Routine to process a sequence of Fowler sampled exposures
    Usage: outImg = fowlerSampling(intTime, data, satFrame)
    intTime is input array indicating the integration times of each frame in the data cube in increasing order
    data is the input data cube
    satFrame is an input image indicating the frame number of the first saturated frame in the sequence per pixel
    output is a 2D image, where each pixel represents the flux (slope/time)
    """
    
    nx = data.shape[1]
    ny = data.shape[0]
    nFrames = data.shape[2]

    outImg = np.zeros((ny,nx), dtype='float32')
    nFowler = nFrames/2

    #go through each pixel and only include non-saturated pairs in averaging
    for i in range(ny):
        for j in range(nx):
            sat2 = satFrame[i,j]

            if (sat2 >nFowler):
                sat1 = sat2 - nFowler
                tmp1 = data[i,j,0:sat1]
                tmp2 = data[i,j,nFowler:sat2]
                intTmp = intTime[nFowler:sat2]-intTime[0:sat1]
                outImg[i,j] = np.mean(tmp2-tmp1)/np.mean(intTmp)
                                
    return outImg
    
    
def fowlerSamplingCL(intTime, data, satFrame, nSplit):
    """
    Routine to process a sequence of Fowler sampled exposures
    Usage: outImg = fowlerSampling(intTime, data, satFrame)
    intTime is input array indicating the integration times of each frame in the data cube in increasing order
    data is the input data cube
    satFrame is an input image indicating the frame number of the first saturated frame in the sequence per pixel
    nSplit is input variable to allow for splitting up the processing of the workload, which is needed to reduce memory to reduce memory errors/consumption for large datasets
    output is a 2D image, where each pixel represents the flux (slope/time)
    """

    #get dimenions of input images
    ny = data.shape[0]
    nx = int(data.shape[1]/nSplit)
    ntime = data.shape[2]

    #create tempory array to hold the output image
    outImg = np.zeros((ny,data.shape[1]), dtype='float32')

    #start OpenCL portion

    #get OpenCL context object, can set to fixed value if wanted
    ctx = cl.create_some_context(interactive=True)
    queue = cl.CommandQueue(ctx)
    
    filename = 'opencl_code/getfowler.cl'
    f = open(filename, 'r')
    fstr = "".join(f.readlines())
    program = cl.Program(ctx, fstr).build()
    mf = cl.mem_flags   

    #create OpenCL buffers
    time_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=intTime.astype('float32'))

    #only create temporary arrays if needed to avoid excess RAM usage
    if (nSplit > 1):
        for n in range(0, nSplit):
            #create temporary arrays to hold portions of data cube
            dTmp = np.array(data[:,n*nx:(n+1)*nx,:].astype('float32'))
            sTmp = np.array(satFrame[:,n*nx:(n+1)*nx].astype('int32'))
            oTmp = np.zeros((ny,nx),dtype='float32')
           
            #create OpenCL buffers
            data_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dTmp)
            sat_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sTmp)
            flux_buf = cl.Buffer(ctx, mf.WRITE_ONLY, oTmp.nbytes)

            #set openCL arguments and run openCL code
            program.fowler.set_scalar_arg_dtypes([np.uint32, np.uint32, None, None,None,None])
            program.fowler(queue,(ny,nx),None,np.uint32(nx), np.uint32(ntime),time_buf, data_buf,sat_buf,flux_buf)
            
            cl.enqueue_read_buffer(queue, flux_buf, oTmp).wait()

            #copy to output array
            np.copyto(outImg[:,n*nx:(n+1)*nx],oTmp)
    else:
        
        #create OpenCL buffers
        data_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data.astype('float32'))
        sat_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=satFrame.astype('int32'))
        flux_buf = cl.Buffer(ctx, mf.WRITE_ONLY, outImg.nbytes)
        
        program.fowler.set_scalar_arg_dtypes([np.uint32, np.uint32, None, None,None,None])
        program.fowler(queue,(ny,nx),None,np.uint32(nx), np.uint32(ntime),time_buf, data_buf,sat_buf,flux_buf)
        
        cl.enqueue_read_buffer(queue, flux_buf, outImg).wait()

    #modify variables to reduce memory consumption
    dTmp = 0
    data_buf = 0
    
    return outImg

