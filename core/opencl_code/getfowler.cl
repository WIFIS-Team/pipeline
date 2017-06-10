#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void fowler(const unsigned int nx, const unsigned int nt, __global float* inttime, __global float* data, __global unsigned int* sat, __global float* flux)
{

// nx is size of 2nd dimension
// nt is size of 3rd dimension
// inttime is an array of inttegration times for each image in the input data cube
// data is input array (3D)

// flux is the output values of the computed  Fowler sampled flux
// sat is input array specifying the first frame that is saturated

// i is the coordinate of the first index
// j is the coordinate of the 2nd index
// k is used as a counter
// pos3d1 and pos3d2 is the coordinate in 3D array
// pos2d is the coordinate in the 2D array
// satframe is the first saturated frame at the current 2D position
// nFowler is the number of Fowler pairs  
// nGood is the number of non-saturated pairs
// meanCnts is the mean counts
// meanTime is the mean time

  unsigned int i = get_global_id(0);
  unsigned int j = get_global_id(1);
  unsigned int k = 0;
  unsigned long pos3d1 = 0;
  unsigned long pos3d2 = 0;
  unsigned long pos2d = i*nx+j;
  unsigned int satframe = (int)sat[pos2d];
  unsigned int nFowler;
  signed int nGood;;
  double meanCnts = 0;
  double meanTime = 0;
  
  // first find the mean values

  nFowler=nt/2;
  nGood = satframe-nFowler;

  if (nGood > 0){
    for (k=0; k<nGood; k++){
      pos3d1 = i*(nx*nt) + j*nt + k;
      pos3d2 = i*(nx*nt) + j*nt + k+nFowler;
      meanCnts += (data[pos3d2]-data[pos3d1]);
      meanTime += (inttime[k+nFowler]-inttime[k]);
    }
    
    meanCnts /= nGood;
    meanTime /= nGood;
    
    flux[pos2d] = meanCnts/meanTime;
  }
  else{
    flux[pos2d] = NAN;
  }
}
