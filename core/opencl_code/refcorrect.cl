__kernel void channel(const unsigned int ny, const unsigned int nx, const unsigned int nt, __global float* data)
{
  float corr = 0;
  unsigned int n = get_global_id(0);
  unsigned long pos2d = 0;
  unsigned long pos3d = 0;
  unsigned int k = 0;
  
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
    for (i=4; i< ny-4; i++){
      pos3d = i*(nx*nt)+j*nt + n;
      data[pos3d] -= corr;
    }
  }
}

__kernel void row(const unsigned int ny, const unsigned int nx, const unsigned int nt, const unsigned int winsize, __global float* data)
{
  float corr = 0;
  unsigned int n = get_global_id(0);
  unsigned long pos2d = 0;
  unsigned long pos3d = 0;
  unsigned int k = 0;
  
  unsigned int i = 0;
  unsigned int j = 0;

  for (i=0; i<ny; i++){
    
    // go through the left pixels
    for (i=0; i< 4; i++){
      pos3d = i*(nx*nt)+j*nt + n;
      corr += data[pos3d];
    }
    
    //go through the right pixels
    for (i=ny-4; i< ny; i++){
      pos3d = i*(nx*nt)+j*nt + n;
      corr += data[pos3d];
    }
  }
  
  corr = corr/(float)(nx*8); // get the mean
    
  //now apply the correction to all non-reference pixels
    
  for (j=0; j<nx; j++){
      
    // go through the light-sensitive pixels
    for (i=4; i< ny-4; i++){
      pos3d = i*(nx*nt)+j*nt + n;
      data[pos3d] -= corr;
    }
  }
}
