#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void lsfit(const unsigned int nx, const unsigned int nt, __global float* inttime, __global float* data, __global float* a0,__global float* a1, __global unsigned int* sat, __global float* variance)
{

// nx is size of 2nd dimension
// nt is size of 3rd dimension
// inttime is an array inttegration times for each image in the input data cube
// data is input array (3D)

// a0, a1 are the output values of the fitted coefficients (2D)
// sat is input array specifying the first frame that is saturated

// i is the coordinate of the first index
// j is the coordinate of the 2nd index
// k is used as a counter
// pos3d is the coordinate in 3D array
// pos2d is the coordinate in the 2D array
  
  unsigned int i = get_global_id(0);
  unsigned int j = get_global_id(1);
  unsigned int k = 0;
  
  unsigned long pos3d = 0;
  unsigned long pos2d = i*nx+j;
  satframe = sat[pos2d];
  
  double covar = 0;
  double varx = 0;
  double d = 0;
  double meanx = 0;
  double meany = 0;
  double meanxy = 0;
  double meanx2 = 0;
  double t = 0;
  
  double a0_tmp = 0;
  double a1_tmp = 0;

  double vari = 0;
  double diff = 0;
  
// goal is to solve least squares problem where
// m = a + b*x
// b = covariance(x,y)/variance(x)
// and a = mean(y) - b*mean(x)

// first find the mean values

  if (satframe > 1){
    
    for (k=0; k<satframe; k++){
      pos3d = i*(nx*nt) + j*nt + k;
      d = (double) data[pos3d];
      t = (double) inttime[k];
      meanx = meanx + t;
      meany = meany + d;
      meanxy = meanxy + t*d;
      meanx2 = meanx2 + t*t;
    }
    
    meanx = meanx/((double)satframe);
    meany = meany/((double)satframe);
    meanxy = meanxy/((double)satframe);
    meanx2 = meanx2/((double)satframe);
    
    covar = meanxy - meanx*meany;
    varx = meanx2 - meanx*meanx;
    
    a1_tmp = covar/varx;
    a0_tmp = meany - a1_tmp*meanx;
    
    a0[pos2d] = (float)a0_tmp;
    a1[pos2d] = (float)a1_tmp;

    //now compute variance
    for (k=0; k<satframe;k++){
      pos3d = i*(nx*nt) + j*nt + k;
      d = (double) data[pos3d];
      t = (double) inttime[k];

      diff = a0_tmp + a1_tmp*t -d;
      vari = vari + diff*diff;
    }
    vari = vari/(double)satframe;
    
    variance[pos2d] = (float)vari;
    
  }
}
