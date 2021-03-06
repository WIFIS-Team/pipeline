"""

Tools to determine saturation information of a given data cube

"""

import numpy as np
import pyopencl as cl
import os

path = os.path.dirname(__file__)
clCodePath = path+'/opencl_code'

def getSatInfo(data, thresh, satThresh=0.97):
    """
    Determine saturation counts for each pixel using built-in python routines
    Usage: satCounts, satFrame = getSatInfo(data,thresh, satThres=0.97)
    data is the input data cube
    thresh is the threshold level to use for determining which pixels to use for determining the saturation level
    satThresh is the relative level to report as the final saturation level (i.e. satThresh * saturation level)
    satCounts is the output that will contain the saturation level for each pixel
    satFrame is the output array that will contain the number of the first saturated frame
    """

    #get input dimensions
    nx = data.shape[1]
    ny = data.shape[0]
    nt = data.shape[2]

    #initialize output arrays
    satFrame = np.repeat(nt-1,(ny*nx)).reshape(ny,nx).astype('uint32') # initialize saturation frame
    satCounts = np.zeros((ny,nx),dtype='float32') # specifies saturation frame

    #determine saturation info
    for y in xrange(data.shape[0]):
        for x in xrange(data.shape[1]):
            ytmp = data[y,x,:]

            #determine saturation level

            #assume last pixel in series is part of saturation regime
            mx = ytmp[-1]
            maxPixs = np.where(ytmp >= thresh*mx)
            satVal = satThresh*np.mean(ytmp[maxPixs])
            satCounts[y,x] = satVal
            satFrame = ((np.where(ytmp >= satVal))[0])[0] #returns first saturated frame
    return satCounts, satFrame

def getSatCounts(data, thresh, satThresh=0.97):
    """
    Determine saturation level for each pixel using built-in python routines
    Usage: satCounts = getSatCounts(data,thresh)
    data is the input data cube
    thresh is the threshold level to use for determining the pixels to use for determining the saturation level
    satThresh is the relative level to report as the final saturation level (i.e. satThresh * saturation level)
    satCounts is the output that will contain the saturation level for each pixel
    """

    #initialize output array
    satCounts = np.zeros((ny,nx),dtype='float32') # specifies saturation frame

    #determine saturation info
    for y in xrange(data.shape[0]):
        for x in xrange(data.shape[1]):
            ytmp = data[y,x,:]

            #determine saturation level
            mx = ytmp[-1] # assumes a well behaved ramp, with the last pixel having the highest counts
            maxPixs = np.where(ytmp >= thresh*mx)
            satVal = np.mean(ytmp[maxPixs])
            satCounts[y,x] = satVal*satThresh #set useful range as satThresh times the saturation value
    return satCounts

def getSatFrame(data,satCounts, ignoreRefPix=True):
    """
    Determine frame number of first saturated frame for each pixel using built-in python routines
    Usage: satFrame =getSatFrame(data,satCounts, ignoreRefPix=True)
    data is the input data cube
    satCounts is the input array containing the saturation level for each pixel
    ignoreRefPix is the boolean keyword to specify if the reference pixels should be ignored when determining the saturation frames. Reference pixels are taken to be the 4 outside pixels around the entire detector.
    satFrame is the output array that will contain the number of the first saturated frame
    """

    #get input dimensions
    nx = data.shape[1]
    ny = data.shape[0]
    nt = data.shape[2]

    #initialize output array
    satFrame = np.repeat(nt-1,(ny*nx)).reshape(ny,nx).astype('uint32') # initialize saturation frame

    #deterimine saturation info
    for y in xrange(ny):
        for x in xrange(nx):
            ytmp = data[y,x,:]
            
            satVal = satCounts[y,x]
            satFrame = ((np.where(ytmp >= satVal))[0])[0]

    if ignoreRefPix:
        #reset the values of the reference pixels so that all frames are used
        refFrame = np.ones(satFrame.shape, dtype=bool)
        refFrame[4:-4,4:-4] = False
        satFrame[refFrame] = nt

    return satFrame    

def getSatCountsCL(data, thresh, nSplit,satThresh=0.97):
    """
    Determine saturation level for each pixel using OpenCL code
    Usage: satCounts = getSatCounts(data,thresh)
    data is the input data cube
    thresh is the threshold level to use for determining the pixels to use for determining the saturation level
    nSplit is a value indicating the number of temporary arrays to split the input data array into for processing. this value should be an integer multiple of the array dimension (dimenions 1)
    satThresh is the relative level to report as the final saturation level (i.e. satThresh * saturation level)
    satCounts is the output that will contain the saturation level for each pixel
    """

    
    #get OpenCL context object, can set to fixed value if wanted
    ctx = cl.create_some_context(interactive=True)
    queue = cl.CommandQueue(ctx)

    #get data array dimensions
    ny = data.shape[0]
    nx = int(data.shape[1]/nSplit)
    nt = data.shape[2]

    #initialize output array
    satCounts = np.zeros((ny,data.shape[1]), dtype='float32') # specifies saturation level

    #read OpenCL kernel code
    filename = clCodePath+'/getsatlev.cl'
    f = open(filename, 'r')
    fstr = "".join(f.readlines())

    #Compile OpenCL code
    program = cl.Program(ctx, fstr).build()

    #Get memory flags
    mf = cl.mem_flags

    #Indicate which arrays are scalars
    program.getmaxval.set_scalar_arg_dtypes([np.uint32, np.uint32, None, None])

    if (nSplit > 1):

        #only create temporary arrays if needed to avoid excess RAM usage
        for n in range(nSplit):
            #create temporary array to hold information
            #mxCounts = np.zeros((ny,nx),dtype='float32')
            mxCounts = np.array(data[:,n*nx:(n+1)*nx,-1]).astype('float32') # assume max counts occurs at last frame
            
            dTmp = np.array(data[:, n*nx:(n+1)*nx,:].astype('float32'))
            sTmp = np.zeros((ny,nx),dtype='float32')
            
            #create OpenCL buffers
            data_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dTmp)
            #mxCounts_buf = cl.Buffer(ctx, mf.WRITE_ONLY, mxCounts.nbytes)
            
            #Run OpenCL code to and put data back into variables
            #program.getmaxval(queue,(ny,nx),None,np.uint32(nx), np.uint32(nt), data_buf, mxCounts_buf)
            #cl.enqueue_read_buffer(queue, mxCounts_buf, mxCounts).wait()
        
            #Now run code to determine saturation level (mean counts for all pixels >=thresh*max count)
            mxCounts_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mxCounts)
            satCounts_buf = cl.Buffer(ctx, mf.WRITE_ONLY, sTmp.nbytes)
    
            program.getsatlev.set_scalar_arg_dtypes([np.uint32, np.uint32, np.float32, None, None, None])
            program.getsatlev(queue,(ny,nx),None,np.uint32(nx), np.uint32(nt), np.float32(thresh),data_buf, mxCounts_buf, satCounts_buf)
            cl.enqueue_read_buffer(queue, satCounts_buf, sTmp).wait()

            np.copyto(satCounts[:,n*nx:(n+1)*nx],sTmp)
    else:
        #create OpenCL buffers
        #mxCounts = np.zeros((ny,nx),dtype='float32')
        mxCounts = np.array(data[:,:,-1]).astype('float32')

        data_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data.astype('float32'))
        #mxCounts_buf = cl.Buffer(ctx, mf.WRITE_ONLY, mxCounts.nbytes)
        
        #Run OpenCL code to and put data back into variables
        #program.getmaxval(queue,(ny,nx),None,np.uint32(nx), np.uint32(nt), data_buf, mxCounts_buf)
        #cl.enqueue_read_buffer(queue, mxCounts_buf, mxCounts).wait()
        
        #Now run code to determine saturation level (mean counts for all pixels >=thresh*max count)
        mxCounts_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mxCounts)
        satCounts_buf = cl.Buffer(ctx, mf.WRITE_ONLY, satCounts.nbytes)
    
        program.getsatlev.set_scalar_arg_dtypes([np.uint32, np.uint32, np.float32, None, None, None])
        program.getsatlev(queue,(ny,nx),None,np.uint32(nx), np.uint32(nt), np.float32(thresh),data_buf, mxCounts_buf, satCounts_buf)
        cl.enqueue_read_buffer(queue, satCounts_buf, satCounts).wait()

    satCounts *= satThresh #set useful range as satThresh  times the saturation value

    #modify variables to reduce memory consumption
    dTmp = 0
    data_buf = 0
    
    return satCounts

def getSatFrameCL(data,satCounts, nSplit, ignoreRefPix=True):
    """
    Determine frame number of first saturated frame for each pixel using OpenCL routines for each pixel
    Usage: satFrame =getSatFrame(data,satCounts,nSplit)
    data is the input data cube
    satCounts is the input array containing the saturation level for each pixel
    nSplit is a value indicating the number of temporary arrays to split the input data array into for processing. this value should be an integer multiple of the array dimension (dimenions 1)
    ignoreRefPix is the boolean keyword to specify if the reference pixels should be ignored when determining the saturation frames. Reference pixels are taken to be the 4 outside pixels around the entire detector.
    satFrame is the output array that will contain the number of the first saturated frame for each pixel
    """
    
    #get OpenCL context object, can set to fixed value if wanted
    ctx = cl.create_some_context(interactive=True)
    queue = cl.CommandQueue(ctx)

    #get data array dimensions
    ny = data.shape[0]
    nx = data.shape[1]
    nt = data.shape[2]

    satFrame = np.zeros((ny,data.shape[1]),dtype='uint32') # specifies saturation frame
    
    nx = int(nx/nSplit)
    
    #First run code to determine maximum counts
    #read OpenCL kernel code
    filename = clCodePath+'/getsatlev.cl'
    f = open(filename, 'r')
    fstr = "".join(f.readlines())

    #Compile OpenCL code
    program = cl.Program(ctx, fstr).build()

    #Get memory flags
    mf = cl.mem_flags

    if (nSplit > 1):
        for n in range(0, nSplit):
            #create temporary arrays to hold information
            dTmp = np.array(data[:, n*nx:(n+1)*nx,:].astype('float32'))
            sCountsTmp = np.array(satCounts[:,n*nx:(n+1)*nx].astype('float32'))
            sFrameTmp = np.zeros((ny, nx), dtype='uint32')
        
            #create OpenCL buffers
            data_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dTmp)
            satFrame_buf = cl.Buffer(ctx, mf.WRITE_ONLY, sFrameTmp.nbytes)
            satCounts_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sCountsTmp)
            
            #run code
            program.getsatframe.set_scalar_arg_dtypes([np.uint32, np.uint32, None, None, None])
            program.getsatframe(queue,(ny,nx),None,np.uint32(nx), np.uint32(nt),data_buf, satCounts_buf, satFrame_buf)
            cl.enqueue_read_buffer(queue, satFrame_buf, sFrameTmp).wait()
            
            np.copyto(satFrame[:,n*nx:(n+1)*nx],sFrameTmp)
    else:
        sTmp = np.array(satCounts)
        data_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data.astype('float32'))
        satFrame_buf = cl.Buffer(ctx, mf.WRITE_ONLY, satFrame.nbytes)
        satCounts_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sTmp.astype('float32'))
                    
        #run code
        program.getsatframe.set_scalar_arg_dtypes([np.uint32, np.uint32, None, None, None])
        program.getsatframe(queue,(ny,nx),None,np.uint32(nx), np.uint32(nt),data_buf, satCounts_buf, satFrame_buf)
        cl.enqueue_read_buffer(queue, satFrame_buf, satFrame).wait()


    #modify variables to reduce memory consumption
    dTmp = 0
    data_buf = 0  

    if ignoreRefPix:
        #reset the values of the reference pixels so that all frames are used
        refFrame = np.ones(satFrame.shape, dtype=bool)
        refFrame[4:-4,4:-4] = False
        satFrame[refFrame] = nt
        
    return satFrame
