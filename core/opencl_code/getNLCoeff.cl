#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void lsfit(const unsigned int nx, const unsigned int nt, __global float* data, __global float* a0,__global float* a1, __global float* a2, global float* a3, __global unsigned int* sat, __global float* zpnt, __global float* ramp)
{
// nx is size of 2nd dimension
// nt is size of 3rd dimension
// data is input array (3D)

// a0, a1, a2 a3 are the output values of the fitted coefficients (2D)
// sat is input array specifying the first saturated frame

// i is the coordinate of the first index
// j is the coordinate of the 2nd index
// k is the counter to run the polynomial order
// pos3d is the coordinate in 3D array
// pos2d is the coordinate in the 2D array

   unsigned int i = get_global_id(0);
   unsigned int j = get_global_id(1);
   unsigned int k = 0;

   unsigned long pos3d = 0;
   unsigned long pos2d = i*nx+j;
   unsigned int satframe = sat[pos2d];
 
// the goal is to solve matrix equation a = (X^T . X)^-1 . X^T . y

   // [a] = X^T . X
   double a11 =0;
   double a12 =0;
   double a13 =0;
   double a14 =0;
   double a21 =0;
   double a22 =0;
   double a23 =0;
   double a24 =0;
   double a31 =0;
   double a32 =0;
   double a33 =0;
   double a34 =0;
   double a41 =0;
   double a42 =0;
   double a43 =0;
   double a44 =0;

   double x = 0;
   double d = 0;
   double detA = 0;
   double invdetA = 0;
   
   // [b] is inverse of matrix a
   double b11 =0;
   double b12 =0;
   double b13 =0;
   double b14 =0;
   double b21 =0;
   double b22 =0;
   double b23 =0;
   double b24 =0;
   double b31 =0;
   double b32 =0;
   double b33 =0;
   double b34 =0;
   double b41 =0;
   double b42 =0;
   double b43 =0;
   double b44 =0;
   
// will contain the final values of different sums
   double term1 =0;
   double term2 =0;
   double term3 =0;
   double term4 =0;

// hold the polynomial fit to polynomial solution
   double pcoef0 = 0;
   double pcoef1 = 0;
   double pcoef2 = 0;
   double pcoef3 = 0;

// save temporary NL corrections
   float a0_tmp = 0;
   float a1_tmp = 0;
   float a2_tmp = 0;
   float a3_tmp = 0;

// only bother if there are more than 3 frames for fitting

   if (satframe > 3){
     
     for (k=0; k< satframe; k++){
       // first compute (X^T . X)
       
       d = (double)k;
       
       a11 = a11 + 1;
       a12 = a12 + d;
       a13 = a13 + d*d;
       a14 = a14 + d*d*d;
       
       a21 = a21 + d;
       a22 = a22 + d*d;
       a23 = a23 + d*d*d;
       a24 = a24 + d*d*d*d;
       
       a31 = a31 + d*d;
       a32 = a32 + d*d*d;
       a33 = a33 + d*d*d*d;
       a34 = a34 + d*d*d*d*d;
       
       a41 = a41 + d*d*d;
       a42 = a42 + d*d*d*d;
       a43 = a43 + d*d*d*d*d;
       a44 = a44 + d*d*d*d*d*d;
     }
     
// now compute the inverse of this matrix
// first the determinant
     
     detA =        a11*(a22*(a33*a44-a43*a34) - a23*(a32*a44-a42*a34) + a24*(a32*a43-a42*a33));
     detA = detA - a12*(a21*(a33*a44-a43*a34) - a23*(a31*a44-a41*a34) + a24*(a31*a43-a41*a33));
     detA = detA + a13*(a21*(a32*a44-a42*a34) - a22*(a31*a44-a41*a34) + a24*(a31*a42-a41*a32));
     detA = detA - a14*(a21*(a32*a43-a42*a33) - a22*(a31*a43-a41*a33) + a23*(a31*a42-a41*a32));
     
     if (detA != 0){
       invdetA = 1./detA;
       
       b11 = a22*a33*a44 + a23*a34*a42 + a24*a32*a43 - a22*a34*a43 - a23*a32*a44 - a24*a33*a42;
       b12 = a12*a34*a43 + a13*a32*a44 + a14*a33*a42 - a12*a33*a44 - a13*a34*a42 - a14*a32*a43;
       b13 = a12*a23*a44 + a13*a24*a42 + a14*a22*a43 - a12*a24*a43 - a13*a22*a44 - a14*a23*a42;
       b14 = a12*a24*a33 + a13*a22*a34 + a14*a23*a32 - a12*a23*a34 - a13*a24*a32 - a14*a22*a33;
       b21 = a21*a34*a43 + a23*a31*a44 + a24*a33*a41 - a21*a33*a44 - a23*a34*a41 - a24*a31*a43;
       b22 = a11*a33*a44 + a13*a34*a41 + a14*a31*a43 - a11*a34*a43 - a13*a31*a44 - a14*a33*a41;
       b23 = a11*a24*a43 + a13*a21*a44 + a14*a23*a41 - a11*a23*a44 - a13*a24*a41 - a14*a21*a43;
       b24 = a11*a23*a34 + a13*a24*a31 + a14*a21*a33 - a11*a24*a33 - a13*a21*a34 - a14*a23*a31;
       b31 = a21*a32*a44 + a22*a34*a41 + a24*a31*a42 - a21*a34*a42 - a22*a31*a44 - a24*a32*a41;
       b32 = a11*a34*a42 + a12*a31*a44 + a14*a32*a41 - a11*a32*a44 - a12*a34*a41 - a14*a31*a42;
       b33 = a11*a22*a44 + a12*a24*a41 + a14*a21*a42 - a11*a24*a42 - a12*a21*a44 - a14*a22*a41;
       b34 = a11*a24*a32 + a12*a21*a34 + a14*a22*a31 - a11*a22*a34 - a12*a24*a31 - a14*a21*a32;
       b41 = a21*a33*a42 + a22*a31*a43 + a23*a32*a41 - a21*a32*a43 - a22*a33*a41 - a23*a31*a42;
       b42 = a11*a32*a43 + a12*a33*a41 + a13*a31*a42 - a11*a33*a42 - a12*a31*a43 - a13*a32*a41;
       b43 = a11*a23*a42 + a12*a21*a43 + a13*a22*a41 - a11*a22*a43 - a12*a23*a41 - a13*a21*a42;
       b44 = a11*a22*a33 + a12*a23*a31 + a13*a21*a32 - a11*a23*a32 - a12*a21*a33 - a13*a22*a31;
       

       b11 = b11 *invdetA;
       b12 = b12 *invdetA;
       b13 = b13 *invdetA;
       b14 = b14 *invdetA;
       b21 = b21 *invdetA;
       b22 = b22 *invdetA;
       b23 = b23 *invdetA;
       b24 = b24 *invdetA;
       b31 = b31 *invdetA;
       b32 = b32 *invdetA;
       b33 = b33 *invdetA;
       b34 = b34 *invdetA;
       b41 = b41 *invdetA;
       b42 = b42 *invdetA;
       b43 = b43 *invdetA;
       b44 = b44 *invdetA;
     }
     
     
     // now carry out the final matrix multiplication (a . X^T . y)
     for (k=0; k < satframe; k++) {
       pos3d = i*(nx*nt) + j*nt + k;
       d = (double) data[pos3d];
       x = (double) k;
       
       term1 = term1 + d;
       term2 = term2 + d*x;
       term3 = term3 + d*x*x;
       term4 = term4 + d*x*x*x;
     }
     
     pcoef0 = pcoef0 + b11*term1 + b12*term2 + b13*term3 + b14*term4;
     pcoef1 = pcoef1 + b21*term1 + b22*term2 + b23*term3 + b24*term4;
     pcoef2 = pcoef2 + b31*term1 + b32*term2 + b33*term3 + b34*term4;
     pcoef3 = pcoef3 + b41*term1 + b42*term2 + b43*term3 + b44*term4;

     zpnt[pos2d] = pcoef0;
     ramp[pos2d] = pcoef1;

     //Up to now we have a polynomial solution to the non-linearity ramp
     //Now we want to repeat this process to determine the non-linearity correction coefficients
     //to the problem of polynomial/linear solution vs signal
     
     term1 =0;
     term2 =0;
     term3 =0;
     term4 =0;
     
     double ylin =0;
     double ypoly=0;
     double yterm=0;
     
     d =0;
     
     // [a] = X^T . X
     a11 = 0;
     a12 = 0;
     a13 = 0;
     a14 = 0;
     a21 = 0;
     a22 = 0;
     a23 = 0;
     a24 = 0;
     a31 = 0;
     a32 = 0;
     a33 = 0;
     a34 = 0;
     a41 = 0;
     a42 = 0;
     a43 = 0;
     a44 = 0;
     
     detA = 0;
     invdetA = 0;
     
     // [b] is inverse of matrix a
     b11 = 0;
     b12 = 0;
     b13 = 0;
     b14 = 0;
     b21 = 0;
     b22 = 0;
     b23 = 0;
     b24 = 0;
     b31 = 0;
     b32 = 0;
     b33 = 0;
     b34 = 0;
     b41 = 0;
     b42 = 0;
     b43 = 0;
     b44 = 0;
     
     // will contain the final values of different sums
     term1 =0;
     term2 =0;
     term3 =0;
     term4 =0;
     
     // first compute (X^T . X)
     
     for (k=0; k< satframe; k++){
       pos3d = i*(nx*nt) + j*nt + k;
       d = (double)data[pos3d];
       
       a11 = a11 + 1;
       a12 = a12 + d;
       a13 = a13 + d*d;
       a14 = a14 + d*d*d;
       
       a21 = a21 + d;
       a22 = a22 + d*d;
       a23 = a23 + d*d*d;
       a24 = a24 + d*d*d*d;
       
       a31 = a31 + d*d;
       a32 = a32 + d*d*d;
       a33 = a33 + d*d*d*d;
       a34 = a34 + d*d*d*d*d;
       
       a41 = a41 + d*d*d;
       a42 = a42 + d*d*d*d;
       a43 = a43 + d*d*d*d*d;
       a44 = a44 + d*d*d*d*d*d;
     }
     
     // now compute the inverse of this matrix
     // first the determinant
     
     detA =        a11*(a22*(a33*a44-a43*a34) - a23*(a32*a44-a42*a34) + a24*(a32*a43-a42*a33));
     detA = detA - a12*(a21*(a33*a44-a43*a34) - a23*(a31*a44-a41*a34) + a24*(a31*a43-a41*a33));
     detA = detA + a13*(a21*(a32*a44-a42*a34) - a22*(a31*a44-a41*a34) + a24*(a31*a42-a41*a32));
     detA = detA - a14*(a21*(a32*a43-a42*a33) - a22*(a31*a43-a41*a33) + a23*(a31*a42-a41*a32));
     
     if (detA != 0){
       invdetA = 1./detA;
       
       b11 = a22*a33*a44 + a23*a34*a42 + a24*a32*a43 - a22*a34*a43 - a23*a32*a44 - a24*a33*a42;
       b12 = a12*a34*a43 + a13*a32*a44 + a14*a33*a42 - a12*a33*a44 - a13*a34*a42 - a14*a32*a43;
       b13 = a12*a23*a44 + a13*a24*a42 + a14*a22*a43 - a12*a24*a43 - a13*a22*a44 - a14*a23*a42;
       b14 = a12*a24*a33 + a13*a22*a34 + a14*a23*a32 - a12*a23*a34 - a13*a24*a32 - a14*a22*a33;
       b21 = a21*a34*a43 + a23*a31*a44 + a24*a33*a41 - a21*a33*a44 - a23*a34*a41 - a24*a31*a43;
       b22 = a11*a33*a44 + a13*a34*a41 + a14*a31*a43 - a11*a34*a43 - a13*a31*a44 - a14*a33*a41;
       b23 = a11*a24*a43 + a13*a21*a44 + a14*a23*a41 - a11*a23*a44 - a13*a24*a41 - a14*a21*a43;
       b24 = a11*a23*a34 + a13*a24*a31 + a14*a21*a33 - a11*a24*a33 - a13*a21*a34 - a14*a23*a31;
       b31 = a21*a32*a44 + a22*a34*a41 + a24*a31*a42 - a21*a34*a42 - a22*a31*a44 - a24*a32*a41;
       b32 = a11*a34*a42 + a12*a31*a44 + a14*a32*a41 - a11*a32*a44 - a12*a34*a41 - a14*a31*a42;
       b33 = a11*a22*a44 + a12*a24*a41 + a14*a21*a42 - a11*a24*a42 - a12*a21*a44 - a14*a22*a41;
       b34 = a11*a24*a32 + a12*a21*a34 + a14*a22*a31 - a11*a22*a34 - a12*a24*a31 - a14*a21*a32;
       b41 = a21*a33*a42 + a22*a31*a43 + a23*a32*a41 - a21*a32*a43 - a22*a33*a41 - a23*a31*a42;
       b42 = a11*a32*a43 + a12*a33*a41 + a13*a31*a42 - a11*a33*a42 - a12*a31*a43 - a13*a32*a41;
       b43 = a11*a23*a42 + a12*a21*a43 + a13*a22*a41 - a11*a22*a43 - a12*a23*a41 - a13*a21*a42;
       b44 = a11*a22*a33 + a12*a23*a31 + a13*a21*a32 - a11*a23*a32 - a12*a21*a33 - a13*a22*a31;
       
       b11 = b11 *invdetA;
       b12 = b12 *invdetA;
       b13 = b13 *invdetA;
       b14 = b14 *invdetA;
       b21 = b21 *invdetA;
       b22 = b22 *invdetA;
       b23 = b23 *invdetA;
       b24 = b24 *invdetA;
       b31 = b31 *invdetA;
       b32 = b32 *invdetA;
       b33 = b33 *invdetA;
       b34 = b34 *invdetA;
       b41 = b41 *invdetA;
       b42 = b42 *invdetA;
       b43 = b43 *invdetA;
       b44 = b44 *invdetA;

       //now carry out the final matrix multiplication (b . X^T . y)
       for (k=0; k < satframe; k++) {
	 pos3d = i*(nx*nt) + j*nt + k;
	 d = (double)data[pos3d];
	 x = (double)k;
       
	 ylin = (double)pcoef0 + (double)pcoef1*x;
	 ypoly = ylin + (double)pcoef2*x*x + (double)pcoef3*x*x*x;
	 yterm = ylin/ypoly;
	 
	 term1 = term1 + yterm;
	 term2 = term2 + yterm*d;
	 term3 = term3 + yterm*d*d;
	 term4 = term4 + yterm*d*d*d;
       }
       
       a0_tmp = b11*term1 + b12*term2 + b13*term3 + b14*term4;
       a1_tmp = b21*term1 + b22*term2 + b23*term3 + b24*term4;
       a2_tmp = b31*term1 + b32*term2 + b33*term3 + b34*term4;
       a3_tmp = b41*term1 + b42*term2 + b43*term3 + b44*term4;
       
       a0[pos2d] = a0_tmp;
       a1[pos2d] = a1_tmp;
       a2[pos2d] = a2_tmp;
       a3[pos2d] = a3_tmp;
     }
   }
}
