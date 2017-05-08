__kernel void compmeangrad(const unsigned int nx, const unsigned int nt, __global float* data, __global unsigned int* sat, __global float* outgrad)
{
  // *********************************************************************************************
  // A MAXIMUM OF 1000 FRAMES IS CURRENTLY HARDCODED. INCREASE THE GRAD VARIABLE ARRAY LENGTH IF MORE THAN 1000 FRAMES BELONG IN THE SEQUENCE
  // *********************************************************************************************

  // nx is size of 2nd dimension
  // nt is size of 3rd dimension
  // data is input data array (3D)
  // outgrad is output array (2D)
  
  // i is the coordinate of first index
  // j is coordinate of second index
  // k is counter to run through last index
  // pos3d is coordinate in 3D array
  // pos2d is coordinate in 2D array
  // satframe is the frame number of the first saturated frame
  
  unsigned int i = get_global_id(0);
  unsigned int j = get_global_id(1);
  unsigned int k = 0;
  unsigned int l = 0;
  unsigned long pos3d = 0;
  unsigned long pos2d = i*nx+j;
  unsigned int satframe = (int) sat[pos2d];
  
  // first determine gradient along time axis
  float grad [1000];
  float meangrad = 0.;
  float meangrad2 = 0.;
  float std=0;
  float std2=0;
  float tmp;
  float cnt;

  // only compute gradient for minimum of 2 frames
  if (satframe <1){
    meangrad = 0;
  } else {
    // if only 2 frames, compute gradient for both points as y[1]-y[0]
    if (satframe == 1){
      pos3d = i*(nx*nt) + j*nt;
      grad[0] = data[pos3d+1]-data[pos3d];
      grad[1] = grad[0];
    }
    //if more than 2 frames, compute gradient over 3-pixels
    if (satframe > 1){
      pos3d = i*(nx*nt) + j*nt;
      
      grad[0] = -(3.*data[pos3d] -4.*data[pos3d+1] + data[pos3d+2])/2.;
      
      for (k=1; k < satframe-1; k++) {
	pos3d = i*(nx*nt) + j*nt + k;
	
	grad[k] = (data[pos3d+1]-data[pos3d-1])/2.;
      }
      
      pos3d = i*(nx*nt) + j*nt + satframe-1;
      grad[satframe-1] = (3.*data[pos3d] - 4.*data[pos3d-1] + data[pos3d-2])/2.;
    }
    
    //get mean gradient
    for (k=0;k<satframe;k++){
      meangrad += grad[k];
    }
    meangrad /= (float) satframe;
    
    //get standard deviation
    for (k=0;k<satframe;k++){
      tmp = grad[k]-meangrad;
      std += tmp*tmp;
    }
    std = sqrt(std/satframe);
    
    //recompute mean rejecting all values > 1sigma from mean
    //repeat this 2 more times
    
    meangrad2 = meangrad;
    std2 = std;
    
    for (l=0;l<2;l++){
      
      meangrad = 0;
      std = 0;
      cnt = 0.;
      
      for (k=0;k<satframe;k++){
	if(fabs(grad[k]-meangrad2) < std2){
	  meangrad += grad[k];
	  tmp = (grad[k]-meangrad2);
	  std += tmp*tmp;
	  cnt += 1.;
	}
      }
      
      meangrad2 = meangrad/cnt;
      std2 = sqrt(std/cnt);
    }
    meangrad = meangrad2;
  }

  //return the mean gradient without outliers
  outgrad[pos2d] = meangrad;
}
      
   
    
