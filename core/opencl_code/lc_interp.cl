//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void distcor(const unsigned int ny, const unsigned int nx,  __global float* dataSlc, __global float* distSlc, __global float* gradMap, __global float* xout, const float dSpat, __global float* out)
{
  
  //initialize some variables
  unsigned int k = get_global_id(0); // current row/spatial position of output slice
  unsigned int j = get_global_id(1); // current column/wavelength position
  unsigned int i; // iterator to go through spatial pixels of input
  unsigned int pos2d; // 2d position of input slice
  unsigned int pos2d_out; // 2d position of output slice
  
  double sum=0.; // hold total flux
  double w ; // flux weight
  unsigned int counter = 0; // count the number of overlapping pixels

  //iterate through all input pixels finding those that overlap with current output pixel

  for (i=0; i<ny; i++){

    //get 2d index
    pos2d = i*nx+j;
    
    if(distSlc[pos2d]-gradMap[pos2d]/2. <= xout[k]+dSpat/2. && xout[k]-dSpat/2. <= distSlc[pos2d]+gradMap[pos2d]/2.){

      //get fractional coverage of overlapping pixels
      w = (min(distSlc[pos2d]+gradMap[pos2d]/2.,xout[k]+dSpat/2.)-max(distSlc[pos2d]-gradMap[pos2d]/2.,xout[k]-dSpat/2.))/gradMap[pos2d];
      counter++;

      if(isfinite(dataSlc[pos2d])){
	sum += w*dataSlc[pos2d];
      }
      else{
	sum = nan(0xFFE00000);
      }
    }
  }
  if (counter>0){
    out[k*nx+j] = (float) sum;
  }
  else{
     out[k*nx+j]=nan(0xFFE00000);
  }
}


__kernel void distcor_sig(const unsigned int ny, const unsigned int nx,  __global float* sigSlc, __global float* distSlc, __global float* gradMap, __global float* xout, const float dSpat, __global float* out)
{
  
  //initialize some variables
  unsigned int k = get_global_id(0); // current row/spatial position of output slice
  unsigned int j = get_global_id(1); // current column/wavelength position
  unsigned int i; // iterator to go through spatial pixels of input
  unsigned int pos2d; // 2d position of input slice
  unsigned int pos2d_out; // 2d position of output slice
  
  double sum=0.; // hold total flux
  double w ; // flux weight
  unsigned int counter = 0; // count the number of overlapping pixels

  //iterate through all input pixels finding those that overlap with current output pixel

  for (i=0; i<ny; i++){

    //get 2d index
    pos2d = i*nx+j;
    
    if(distSlc[pos2d]-gradMap[pos2d]/2. <= xout[k]+dSpat/2. && xout[k]-dSpat/2. <= distSlc[pos2d]+gradMap[pos2d]/2.){

      //get fractional coverage of overlapping pixels
      w = (min(distSlc[pos2d]+gradMap[pos2d]/2.,xout[k]+dSpat/2.)-max(distSlc[pos2d]-gradMap[pos2d]/2.,xout[k]-dSpat/2.))/gradMap[pos2d];
      counter++;

      if(isfinite(sigSlc[pos2d])){
	sum += (w*sigSlc[pos2d])*(w*sigSlc[pos2d]);
      }
      else{
	sum = nan(0xFFE00000);
      }
    }
  }
  if (counter>0){
    out[k*nx+j] = (float)sqrt(sum);
  }
  else{
     out[k*nx+j]=nan(0xFFE00000);
  }
}

__kernel void wavecor(const unsigned int nx_out, const unsigned int nx,  __global float* dataSlc, __global float* waveSlc, __global float* gradMap, __global float* xout, const float dWave, __global float* out)
{
  
  //initialize some variables
  unsigned int k = get_global_id(0); // current wavelength position of output slice
  unsigned int i = get_global_id(1); // current spatial position
  unsigned int j; // iterator to go through wavelength pixels of input
  unsigned int pos2d; // 2d position of input slice
  unsigned int pos2d_out; // 2d position of output slice
  
  double sum=0.; // hold total flux
  double w ; // flux weight
  unsigned int counter = 0; // count the number of overlapping pixels

  //iterate through all input pixels finding those that overlap with current output pixel

  for (j=0; j<nx; j++){

    //get 2d index
    pos2d = i*nx+j;

    //wavelength is decreasing as function of pixel, so reverse the case relative to distortion correction
    if(waveSlc[pos2d]+gradMap[pos2d]/2. <= xout[k]+dWave/2. && xout[k]-dWave/2. <= waveSlc[pos2d]-gradMap[pos2d]/2.){

      //get fractional coverage of overlapping pixels
      w = -((min(waveSlc[pos2d]-gradMap[pos2d]/2.,xout[k]+dWave/2.)-max(waveSlc[pos2d]+gradMap[pos2d]/2.,xout[k]-dWave/2.))/gradMap[pos2d]);

      counter++;
      if(isfinite(dataSlc[pos2d])){
	sum += w*dataSlc[pos2d];
      }
      else{
	sum = nan(0xFFE00000);
      }
    }
  }

  if (counter>0){
    out[i*nx_out+k] = (float) sum;
  }
  else{
    out[i*nx_out+k]= nan(0xFFE00000);
  }
  
}

__kernel void wavecor_sig(const unsigned int nx_out, const unsigned int nx,  __global float* sigSlc, __global float* waveSlc, __global float* gradMap, __global float* xout, const float dWave, __global float* out)
{
  
  //initialize some variables
  unsigned int k = get_global_id(0); // current wavelength position of output slice
  unsigned int i = get_global_id(1); // current spatial position
  unsigned int j; // iterator to go through wavelength pixels of input
  unsigned int pos2d; // 2d position of input slice
  unsigned int pos2d_out; // 2d position of output slice
  
  double sum=0.; // hold total flux
  double w ; // flux weight
  unsigned int counter = 0; // count the number of overlapping pixels

  //iterate through all input pixels finding those that overlap with current output pixel

  for (j=0; j<nx; j++){

    //get 2d index
    pos2d = i*nx+j;

    //wavelength is decreasing as function of pixel, so reverse the case relative to distortion correction
    if(waveSlc[pos2d]+gradMap[pos2d]/2. <= xout[k]+dWave/2. && xout[k]-dWave/2. <= waveSlc[pos2d]-gradMap[pos2d]/2.){

      //get fractional coverage of overlapping pixels
      w = -((min(waveSlc[pos2d]-gradMap[pos2d]/2.,xout[k]+dWave/2.)-max(waveSlc[pos2d]+gradMap[pos2d]/2.,xout[k]-dWave/2.))/gradMap[pos2d]);

      counter++;
      if(isfinite(sigSlc[pos2d])){
	sum += (w*sigSlc[pos2d])*(w*sigSlc[pos2d]);
      }
      else{
	sum = nan(0xFFE00000);
      }
    }
  }

  if (counter>0){
    out[i*nx_out+k] = (float) sqrt(sum);
  }
  else{
    out[i*nx_out+k]= nan(0xFFE00000);
  }
  
}
