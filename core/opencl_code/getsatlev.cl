__kernel void getmaxval(const unsigned int nx, const unsigned int nt, __global float* data, __global float* mxcount)
{
// nx is size of 2nd dimension
// nt is size of 3rd dimension
// data is input data array (3D)
// mxcount is output array (2D)

// i is the coordinate of first index
// j is coordinate of second index
// k is counter to run through last index
// pos3d is coordinate in 3D array
// pos2d is coordinate in 2D array

  unsigned int i = get_global_id(0);
  unsigned int j = get_global_id(1);
  unsigned int k = 0;
  unsigned long pos3d = 0;
  unsigned long pos2d = i*nx+j;
  
  float mxval = 0;
  
  for (k=0; k < nt; k++) {
    pos3d = i*(nx*nt) + j*nt + k;
    if (data[pos3d] > mxval){
      mxval = data[pos3d];
    }
  }
  mxcount[pos2d] = mxval;
}

__kernel void getsatlev(const unsigned int nx, const unsigned int nt, const float levl, __global float* data, __global float* mxval, __global float* satval)
{
// nx is size of 2nd dimension
// nt is size of 3rd dimension
// data is input data array (3D)
// mxval is input array of maximum values (2D)
// satval is output array of calculated saturation level

// i is the coordinate of first index
// j is coordinate of second index
// k is counter to run through last index
// pos3d is coordinate in 3D array
// pos2d is coordinate in 2D array

  unsigned int i = get_global_id(0);
  unsigned int j = get_global_id(1);
  unsigned int k = 0;
  unsigned long pos3d = 0;
  unsigned long pos2d = i*nx+j;
  
  float mx = levl*mxval[pos2d];
  float meansat = 0;
  float n = 0;
  
  for (k=0; k < nt; k++) {
    pos3d = i*(nx*nt) + j*nt + k;
    if (data[pos3d] >= mx){
      meansat = meansat + data[pos3d];
      n++;
    } 
  }
  meansat = meansat/n;
  satval[pos2d] = meansat;
}

__kernel void getsatframe(const unsigned int nx, const unsigned int nt, __global float* data, __global float* satval, __global unsigned int* satframe)
{
// nx is size of 1st dimension
// nx is size of 2nd dimension
// nt is size of 3rd dimension
// data is input data array (3D)
// satval is input array of saturation values (2D)
// satframe is output array of of the last pre-saturated frame
  
// i is the coordinate of first index
// j is coordinate of second index
// k is counter to run through last index
// pos3d is coordinate in 3D array
// pos2d is coordinate in 2D array
  
  unsigned int i = get_global_id(0);
  unsigned int j = get_global_id(1);
  unsigned int k = 0;
  unsigned long pos3d = 0;
  unsigned long pos2d = i*nx+j;
  
  float d = 0;
  float mx = satval[pos2d];
  unsigned int mxframe = nt;

  for (k=0; k<nt; k++) {
    pos3d = i*(nx*nt) + j*nt + k;
    d=data[pos3d];

    if (d >= mx){
      mxframe = k;
      break;
    }
  }
  satframe[pos2d] = mxframe;
}
