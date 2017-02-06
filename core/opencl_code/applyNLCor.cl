__kernel void nlcor(const unsigned int nx, const unsigned int nt, __global float* data, __global float* a0,__global float* a1, __global float* a2, __global float* a3)
{
  // nx is size of 2nd dimension
  // nt is size of 3rd dimension
  // data is input data
  
  // a0, a1, a2 a3 are the output values of the fitted coefficients (2D)
  // sat is input array specifying the first saturated frame
  
  // i is the coordinate of the first index
  // j is the coordinate of the 2nd index
  // k is the coordinate of the 3rd index
  // pos3d is the coordinate in the 3D array space
  // pos2d is the coordinate in the 2D array
  
  unsigned int i = get_global_id(0);
  unsigned int j = get_global_id(1);
  unsigned int k = get_global_id(2);
  
  unsigned long pos3d = i*(nx*nt) + j*nt + k;
  unsigned long pos2d = i*nx+j;
  
  float nlCor = 0;
  
  nlCor = a0[pos2d] + a1[pos2d]*data[pos3d] + a2[pos2d]*data[pos3d]*data[pos3d] + a3[pos2d]*data[pos3d]*data[pos3d]*data[pos3d];

  data[pos3d] = data[pos3d]*nlCor;
}
