#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernal void distor(const unsigned int ny, const unsigned int nx,  __global float* dataSlc, __global float* distSlc, __global float* gradMap, __global float* xout, const float dSpat, __global float* out)
{
  
  //initialize some variables
  unsigned int k = get_global_id(0); // current row/spatial position of output slice
  unsigned int i = get_global_id(1); // current column/wavelength position
  unsigned int j; // iterator to go through spatial pixels of input
  unsigned int pos2d; // 2d position of input slice
  unsigned int pos2d_out; // 2d position of output slice
  
  double sum=0.; // hold total flux
  double w ; // flux weight
  unsigned int counter = 0; // count the number of overlapping pixels
  
  //iterate through all input pixels finding those that overlap with current output pixel

  for (j=0; j<ny; j++){

    //get 2d index
    pos2d = i*nx+j;
    
    if(distSlc[pos2d]-gradMap[pos2d]/2. <= xout[k]+dSpat/2. && xout[k]-dSpat/2. <= distSlc[pos2d]+gradMap[pos2d]/2.){

      //get fractional coverage of overlapping pixels
      w = (min(distSlc[pos2d]+gradMap[pos2d]/2.,xout[k]+dSpat/2.)-max(distSlc[pos2d]-gradMap[pos2d]/2.,xout[k]-dSpat/2.))/gradMap[j,i];

      if(isfinite(dataSlc[pos2d])){
	sum += w*dataSlc[pos2d];
	counter +=1;
      }
    }
  }
  if (counter>0){
    out[i*nx+k] = sum;
  }
}

      

	
	
