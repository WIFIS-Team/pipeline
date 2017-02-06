"""

tools to carry out the measurement and correction for non-linearity behaviour of detector

"""

import numpy as np
import pyopencl as cl

def getNLCorNumpy(data, satFrame):
    """
    Routine to determine non-linearity correction coefficients using built-in python methods
    Usage: nlCoef = getNLCorNumpy(data, satFrame)
    data is the input data cube
    satFrame is an array specifying the frame number of the first saturated frame for each pixel
    # nlCoeff is output that contains the non-linearity corrections for each pixel as an (ny x nx x 4) size array
    """

    #get input array dimensions
    ny = data.shape[0]
    nx = data.shape[1]

    #initialize the array
    nlCoeff = np.zeros((ny,nx,4)) # holds the non-linearity coefficients

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
                
    return nlCoeff

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
    nx = data.shape[1]/nSplit
    ntime = data.shape[2]

    #initialize the array
    nlCoeff = np.zeros((ny,data.shape[1],4), dtype='float32') # holds the non-linearity coefficients
    
    #get OpenCL context object
    ctx = cl.create_some_context(interactive=True)
    queue = cl.CommandQueue(ctx)

    #read in OpenCL code
    filename = 'opencl_code/getNLCoeff.cl'
    f = open(filename, 'r')
    fstr = "".join(f.readlines())
    
    #build opencl program and get memory flags
    program = cl.Program(ctx, fstr).build()
    mf = cl.mem_flags   

    #now split input array into multiple chunks, if specified
    if (nSplit > 1):
        for n in range(nSplit):
            #set temporary arrays
            dTmp = np.array(data[:, n*nx:(n+1)*nx,:])
            sTmp = np.array(satFrame[:,n*nx:(n+1)*nx])
            
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
                        
            #run the opencl code
            program.lsfit.set_scalar_arg_dtypes([np.uint32, np.uint32, None, None,None,None, None, None])
            program.lsfit(queue,(ny,nx),None,np.uint32(nx), np.uint32(ntime),data_buf,a0_buf,a1_buf,a2_buf,a3_buf,sat_buf)
            cl.enqueue_read_buffer(queue, a0_buf, a0_array).wait()
            cl.enqueue_read_buffer(queue, a1_buf, a1_array).wait()
            cl.enqueue_read_buffer(queue, a2_buf, a2_array).wait()
            cl.enqueue_read_buffer(queue, a3_buf, a3_array).wait()

            #copy output from OpenCL code into output array
            np.copyto(nlCoeff[:,n*nx:(n+1)*nx,0],a0_array)
            np.copyto(nlCoeff[:,n*nx:(n+1)*nx,1],a1_array)
            np.copyto(nlCoeff[:,n*nx:(n+1)*nx,2],a2_array)
            np.copyto(nlCoeff[:,n*nx:(n+1)*nx,3],a3_array)
    else:
        #set temporary arrays
        
        a0_array = np.zeros((ny,nx), dtype='float32')
        a1_array = np.zeros((ny,nx), dtype='float32')
        a2_array = np.zeros((ny,nx), dtype='float32')
        a3_array = np.zeros((ny,nx), dtype='float32')
                    
        #create OpenCL buffers
        data_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
        sat_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=satFrame)
        a0_buf = cl.Buffer(ctx, mf.WRITE_ONLY, a0_array.nbytes)
        a1_buf = cl.Buffer(ctx, mf.WRITE_ONLY, a1_array.nbytes)
        a2_buf = cl.Buffer(ctx, mf.WRITE_ONLY, a2_array.nbytes)
        a3_buf = cl.Buffer(ctx, mf.WRITE_ONLY, a3_array.nbytes)
       
        #run the opencl code
        program.lsfit.set_scalar_arg_dtypes([np.uint32, np.uint32, None, None,None,None, None, None])
        program.lsfit(queue,(ny,nx),None,np.uint32(nx), np.uint32(ntime),data_buf,a0_buf,a1_buf,a2_buf,a3_buf,sat_buf)
        cl.enqueue_read_buffer(queue, a0_buf, a0_array).wait()
        cl.enqueue_read_buffer(queue, a1_buf, a1_array).wait()
        cl.enqueue_read_buffer(queue, a2_buf, a2_array).wait()
        cl.enqueue_read_buffer(queue, a3_buf, a3_array).wait()

        #copy output from OpenCL code into output array
        np.copyto(nlCoeff[:,:,0],a0_array)
        np.copyto(nlCoeff[:,:,1],a1_array)
        np.copyto(nlCoeff[:,:,2],a2_array)
        np.copyto(nlCoeff[:,:,3],a3_array)
        
    return nlCoeff

def getNLCorTest(data, satframe):
    """for algorithm testing only"""

    nlCoeff = np.zeros((4),dtype='float32')
    a11 =0
    a12 =0
    a13 =0
    a14 =0
    a21 =0
    a22 =0
    a23 =0
    a24 =0
    a31 =0
    a32 =0
    a33 =0
    a34 =0
    a41 =0
    a42 =0
    a43 =0
    a44 =0
    
    x = 0
    d = 0
    detA = 0
    invdetA = 0
   
    b11 =0
    b12 =0
    b13 =0
    b14 =0
    b21 =0
    b22 =0
    b23 =0
    b24 =0
    b31 =0
    b32 =0
    b33 =0
    b34 =0
    b41 =0
    b42 =0
    b43 =0
    b44 =0
   
    term1 =0
    term2 =0
    term3 =0
    term4 =0

    pcoef0 = 0
    pcoef1 = 0
    pcoef2 = 0
    pcoef3 = 0

    a0_tmp = 0
    a1_tmp = 0
    a2_tmp = 0
    a3_tmp = 0
    
    for k in range(satframe):
        
        d = k
        a11 = a11 + 1
        a12 = a12 + d
        a13 = a13 + d*d
        a14 = a14 + d*d*d
        
        a21 = a21 + d
        a22 = a22 + d*d
        a23 = a23 + d*d*d
        a24 = a24 + d*d*d*d
        
        a31 = a31 + d*d
        a32 = a32 + d*d*d
        a33 = a33 + d*d*d*d
        a34 = a34 + d*d*d*d*d
        
        a41 = a41 + d*d*d
        a42 = a42 + d*d*d*d
        a43 = a43 + d*d*d*d*d
        a44 = a44 + d*d*d*d*d*d
    
    detA =        a11*(a22*(a33*a44-a43*a34) - a23*(a32*a44-a42*a34) + a24*(a32*a43-a42*a33))
    detA = detA - a12*(a21*(a33*a44-a43*a34) - a23*(a31*a44-a41*a34) + a24*(a31*a43-a41*a33))
    detA = detA + a13*(a21*(a32*a44-a42*a34) - a22*(a31*a44-a41*a34) + a24*(a31*a42-a41*a32))
    detA = detA - a14*(a21*(a32*a43-a42*a33) - a22*(a31*a43-a41*a33) + a23*(a31*a42-a41*a32))
     
    if (detA != 0):
        invdetA = 1./detA
       
        b11 = a22*a33*a44 + a23*a34*a42 + a24*a32*a43 - a22*a34*a43 - a23*a32*a44 - a24*a33*a42
        b12 = a12*a34*a43 + a13*a32*a44 + a14*a33*a42 - a12*a33*a44 - a13*a34*a42 - a14*a32*a43
        b13 = a12*a23*a44 + a13*a24*a42 + a14*a22*a43 - a12*a24*a43 - a13*a22*a44 - a14*a23*a42
        b14 = a12*a24*a33 + a13*a22*a34 + a14*a23*a32 - a12*a23*a34 - a13*a24*a32 - a14*a22*a33
        b21 = a21*a34*a43 + a23*a31*a44 + a24*a33*a41 - a21*a33*a44 - a23*a34*a41 - a24*a31*a43
        b22 = a11*a33*a44 + a13*a34*a41 + a14*a31*a43 - a11*a34*a43 - a13*a31*a44 - a14*a33*a41
        b23 = a11*a24*a43 + a13*a21*a44 + a14*a23*a41 - a11*a23*a44 - a13*a24*a41 - a14*a21*a43
        b24 = a11*a23*a34 + a13*a24*a31 + a14*a21*a33 - a11*a24*a33 - a13*a21*a34 - a14*a23*a31
        b31 = a21*a32*a44 + a22*a34*a41 + a24*a31*a42 - a21*a34*a42 - a22*a31*a44 - a24*a32*a41
        b32 = a11*a34*a42 + a12*a31*a44 + a14*a32*a41 - a11*a32*a44 - a12*a34*a41 - a14*a31*a42
        b33 = a11*a22*a44 + a12*a24*a41 + a14*a21*a42 - a11*a24*a42 - a12*a21*a44 - a14*a22*a41
        b34 = a11*a24*a32 + a12*a21*a34 + a14*a22*a31 - a11*a22*a34 - a12*a24*a31 - a14*a21*a32
        b41 = a21*a33*a42 + a22*a31*a43 + a23*a32*a41 - a21*a32*a43 - a22*a33*a41 - a23*a31*a42
        b42 = a11*a32*a43 + a12*a33*a41 + a13*a31*a42 - a11*a33*a42 - a12*a31*a43 - a13*a32*a41
        b43 = a11*a23*a42 + a12*a21*a43 + a13*a22*a41 - a11*a22*a43 - a12*a23*a41 - a13*a21*a42
        b44 = a11*a22*a33 + a12*a23*a31 + a13*a21*a32 - a11*a23*a32 - a12*a21*a33 - a13*a22*a31
       

        b11 = b11 *invdetA
        b12 = b12 *invdetA
        b13 = b13 *invdetA
        b14 = b14 *invdetA
        b21 = b21 *invdetA
        b22 = b22 *invdetA
        b23 = b23 *invdetA
        b24 = b24 *invdetA
        b31 = b31 *invdetA
        b32 = b32 *invdetA
        b33 = b33 *invdetA
        b34 = b34 *invdetA
        b41 = b41 *invdetA
        b42 = b42 *invdetA
        b43 = b43 *invdetA
        b44 = b44 *invdetA
     
     
     
    for k in range(satframe):
       d = data[k]
       x = k
       
       term1 = term1 + d
       term2 = term2 + d*x
       term3 = term3 + d*x*x
       term4 = term4 + d*x*x*x
     
    pcoef0 = pcoef0 + b11*term1 + b12*term2 + b13*term3 + b14*term4
    pcoef1 = pcoef1 + b21*term1 + b22*term2 + b23*term3 + b24*term4
    pcoef2 = pcoef2 + b31*term1 + b32*term2 + b33*term3 + b34*term4
    pcoef3 = pcoef3 + b41*term1 + b42*term2 + b43*term3 + b44*term4
   
    term1 =0
    term2 =0
    term3 =0
    term4 =0
     
    ylin =0
    ypoly=0
    yterm=0
     
    d =0
     
    a11 = 0
    a12 = 0
    a13 = 0
    a14 = 0
    a21 = 0
    a22 = 0
    a23 = 0
    a24 = 0
    a31 = 0
    a32 = 0
    a33 = 0
    a34 = 0
    a41 = 0
    a42 = 0
    a43 = 0
    a44 = 0
    
    detA = 0
    invdetA = 0
    
    b11 = 0
    b12 = 0
    b13 = 0
    b14 = 0
    b21 = 0
    b22 = 0
    b23 = 0
    b24 = 0
    b31 = 0
    b32 = 0
    b33 = 0
    b34 = 0
    b41 = 0
    b42 = 0
    b43 = 0
    b44 = 0
    
    term1 =0
    term2 =0
    term3 =0
    term4 =0
    
    for k in range (satframe):
        d = data[k]
          
        a11 = a11 + 1
        a12 = a12 + d
        a13 = a13 + d*d
        a14 = a14 + d*d*d
        
        a21 = a21 + d
        a22 = a22 + d*d
        a23 = a23 + d*d*d
        a24 = a24 + d*d*d*d
        
        a31 = a31 + d*d
        a32 = a32 + d*d*d
        a33 = a33 + d*d*d*d
        a34 = a34 + d*d*d*d*d
        
        a41 = a41 + d*d*d
        a42 = a42 + d*d*d*d
        a43 = a43 + d*d*d*d*d
        a44 = a44 + d*d*d*d*d*d
        
    detA =        a11*(a22*(a33*a44-a43*a34) - a23*(a32*a44-a42*a34) + a24*(a32*a43-a42*a33))
    detA = detA - a12*(a21*(a33*a44-a43*a34) - a23*(a31*a44-a41*a34) + a24*(a31*a43-a41*a33))
    detA = detA + a13*(a21*(a32*a44-a42*a34) - a22*(a31*a44-a41*a34) + a24*(a31*a42-a41*a32))
    detA = detA - a14*(a21*(a32*a43-a42*a33) - a22*(a31*a43-a41*a33) + a23*(a31*a42-a41*a32))
     
    if (detA != 0):
        invdetA = 1./detA
       
        b11 = a22*a33*a44 + a23*a34*a42 + a24*a32*a43 - a22*a34*a43 - a23*a32*a44 - a24*a33*a42
        b12 = a12*a34*a43 + a13*a32*a44 + a14*a33*a42 - a12*a33*a44 - a13*a34*a42 - a14*a32*a43
        b13 = a12*a23*a44 + a13*a24*a42 + a14*a22*a43 - a12*a24*a43 - a13*a22*a44 - a14*a23*a42
        b14 = a12*a24*a33 + a13*a22*a34 + a14*a23*a32 - a12*a23*a34 - a13*a24*a32 - a14*a22*a33
        b21 = a21*a34*a43 + a23*a31*a44 + a24*a33*a41 - a21*a33*a44 - a23*a34*a41 - a24*a31*a43
        b22 = a11*a33*a44 + a13*a34*a41 + a14*a31*a43 - a11*a34*a43 - a13*a31*a44 - a14*a33*a41
        b23 = a11*a24*a43 + a13*a21*a44 + a14*a23*a41 - a11*a23*a44 - a13*a24*a41 - a14*a21*a43
        b24 = a11*a23*a34 + a13*a24*a31 + a14*a21*a33 - a11*a24*a33 - a13*a21*a34 - a14*a23*a31
        b31 = a21*a32*a44 + a22*a34*a41 + a24*a31*a42 - a21*a34*a42 - a22*a31*a44 - a24*a32*a41
        b32 = a11*a34*a42 + a12*a31*a44 + a14*a32*a41 - a11*a32*a44 - a12*a34*a41 - a14*a31*a42
        b33 = a11*a22*a44 + a12*a24*a41 + a14*a21*a42 - a11*a24*a42 - a12*a21*a44 - a14*a22*a41
        b34 = a11*a24*a32 + a12*a21*a34 + a14*a22*a31 - a11*a22*a34 - a12*a24*a31 - a14*a21*a32
        b41 = a21*a33*a42 + a22*a31*a43 + a23*a32*a41 - a21*a32*a43 - a22*a33*a41 - a23*a31*a42
        b42 = a11*a32*a43 + a12*a33*a41 + a13*a31*a42 - a11*a33*a42 - a12*a31*a43 - a13*a32*a41
        b43 = a11*a23*a42 + a12*a21*a43 + a13*a22*a41 - a11*a22*a43 - a12*a23*a41 - a13*a21*a42
        b44 = a11*a22*a33 + a12*a23*a31 + a13*a21*a32 - a11*a23*a32 - a12*a21*a33 - a13*a22*a31
        
        b11 = b11 *invdetA
        b12 = b12 *invdetA
        b13 = b13 *invdetA
        b14 = b14 *invdetA
        b21 = b21 *invdetA
        b22 = b22 *invdetA
        b23 = b23 *invdetA
        b24 = b24 *invdetA
        b31 = b31 *invdetA
        b32 = b32 *invdetA
        b33 = b33 *invdetA
        b34 = b34 *invdetA
        b41 = b41 *invdetA
        b42 = b42 *invdetA
        b43 = b43 *invdetA
        b44 = b44 *invdetA
     
        
        for k in range(satframe): 
            d = data[k]
            x = k
       
            ylin = pcoef0 + pcoef1*x
            ypoly = ylin + pcoef2*x*x + pcoef3*x*x*x
            yterm = ylin/ypoly
       
            term1 = term1 + yterm
            term2 = term2 + yterm*d
            term3 = term3 + yterm*d*d
            term4 = term4 + yterm*d*d*d
          
        a0_tmp = b11*term1 + b12*term2 + b13*term3 + b14*term4
        a1_tmp = b21*term1 + b22*term2 + b23*term3 + b24*term4
        a2_tmp = b31*term1 + b32*term2 + b33*term3 + b34*term4
        a3_tmp = b41*term1 + b42*term2 + b43*term3 + b44*term4

        nlCoeff[0] = a0_tmp
        nlCoeff[1] = a1_tmp
        nlCoeff[2] = a2_tmp
        nlCoeff[3] = a3_tmp

    return

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
    nx = data.shape[1]/nSplit
    ntime = data.shape[2]

    #get OpenCL context object, can set to fixed value if wanted
    ctx = cl.create_some_context(interactive=True)
    queue = cl.CommandQueue(ctx)

    #read in OpenCL code
    filename = 'opencl_code/applyNLCor.cl'
    f = open(filename, 'r')
    fstr = "".join(f.readlines())

    #build opencl program and get memory flags
    program = cl.Program(ctx, fstr).build()
    mf = cl.mem_flags   

    #if specified, split input data into separate chunks for processing
    if (nSplit > 1):
        for n in range(nSplit):
            #set temporary arrays
            dTmp = np.array(data[:, n*nx:(n+1)*nx,:])
            
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
            cl.enqueue_read_buffer(queue, data_buf, dTmp).wait()

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

        print(nlCoeff[1940,160,:])
        print(a0_array[1940,160], a1_array[1940,160], a2_array[1940,160],a3_array[1940,160])
        
        #create OpenCL buffers
        data_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=data)
        a0_buf = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a0_array)
        a1_buf = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a1_array)
        a2_buf = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a2_array)
        a3_buf = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a3_array)
       
        #run the opencl code
        program.nlcor.set_scalar_arg_dtypes([int,int, None, None,None,None, None])
        program.nlcor(queue,(ny,nx, ntime),None,np.uint32(nx), np.uint32(ntime),data_buf,a0_buf,a1_buf,a2_buf,a3_buf)
        cl.enqueue_read_buffer(queue, data_buf, data).wait()
        
    return
