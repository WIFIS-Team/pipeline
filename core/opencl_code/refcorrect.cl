#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void channel(const unsigned int ny, const unsigned int nx, const unsigned int nt, __global float* data)
{
  float corr = 0;
  unsigned int n = get_global_id(0);
  unsigned long pos3d = 0;
  
  unsigned int i = 0;
  unsigned int j = 0;

  for (j=0; j<nx; j++){
    
    // go through the bottom pixels
    for (i=0; i< 4; i++){
      pos3d = i*(nx*nt)+j*nt + n;
      corr += data[pos3d];
    }
    
    //go through the top pixels
    for (i=ny-4; i< ny; i++){
      pos3d = i*(nx*nt)+j*nt + n;
      corr += data[pos3d];
    }
  }
  
  corr = corr/(float)(nx*8); // get the mean
    
  //now apply the correction to all non-reference pixels
    
  for (j=0; j<nx; j++){
      
    // go through the light-sensitive pixels
    for (i=0; i< ny; i++){
      pos3d = i*(nx*nt)+j*nt + n;
      data[pos3d] -= corr;
    }
  }
}

__kernel void row(const signed int ny, const unsigned int nx, const unsigned int nt, const unsigned int winsize, __global float* data)
{
  unsigned int posy = get_global_id(0); // current row position
  unsigned int post = get_global_id(1); //current time position
  unsigned long pos3dR = 0;
  unsigned long pos3dL = 0;
  signed int i = 0; //counter for first coordinate
  unsigned int j = 0; // counter for second coordinate
  float cnt = 0.;
  float corr = 0.;
  signed int rng1 = (int)(posy - winsize/2);
  signed int rng2 = (int)(posy + winsize/2)+1;
  
  //unsigned int rng (int)(winsize/2);
  
  //compute the mean signal from all pixels within +/- winsize/2
  //i.e. all references pixels in range:
  //[i-winsize/2, i-winsize/2+1,... i, ..., i+winsize/2]

  for (i=rng1;i<rng2;i++){
    if (i>=0 && i<ny){
      cnt += 1.;
      
      for (j=0;j<4;j++){
	//go through pixels on the left
     	pos3dL = i*(nx*nt)+j*nt + post;
	corr += data[pos3dL];

	//go through pixels on the right
    	pos3dR = i*(nx*nt)+(nx-j-1)*nt + post;
	corr += data[pos3dR];
      }
    }
  }
  
  corr /= (cnt*8.);///= (double)(cnt*8.);

  // now apply correction to each pixel in row
  
  for (j=4;j<nx-4;j++){
    pos3dL = posy*(nx*nt)+j*nt + post;
    
    data[pos3dL] -= corr;
  }
}
  
__kernel void channelEO(const unsigned int ny, const unsigned int nx, const unsigned int nt, __global float* data)
{
  // corrects odd/even columns in a given channel separately
  float corrE = 0;
  float corrO = 0;
  unsigned int n = get_global_id(0);
  unsigned long pos3dEt = 0;
  unsigned long pos3dOt = 0;
  unsigned long pos3dEb = 0;
  unsigned long pos3dOb = 0;
  unsigned int i = 0;
  unsigned int j = 0;

  for (j=0; j<nx-1; j+=2){
    for (i=0; i< 4; i++){
      // top bottom pixels
      pos3dEb = i*(nx*nt)+j*nt + n;
      pos3dOb = i*(nx*nt)+(j+1)*nt + n;
      corrE += data[pos3dEb];
      corrO += data[pos3dOb];

      //bottom pixels
      pos3dEt = (ny-i-1)*(nx*nt)+j*nt + n;
      pos3dOt = (ny-i-1)*(nx*nt)+(j+1)*nt + n;
      corrE += data[pos3dEt];
      corrO += data[pos3dOt];
    }
  }

  corrE = corrE/(float)(nx*4); // get the mean
  corrO = corrO/(float)(nx*4); // get the mean
    
  //now apply the correction to all non-reference pixels
    
  for (j=0; j<nx-1; j+=2){
    // go through the light-sensitive pixels
    for (i=0; i< ny; i++){
      pos3dEt = i*(nx*nt)+j*nt + n;
      pos3dOt = i*(nx*nt)+(j+1)*nt + n;
      data[pos3dEt] -= corrE;
      data[pos3dOt] -= corrO;
    }
  }
}
