// for jacobi gpu programming 

#include <stdio.h>
#include "support.h"
#include "kernel.cu"

int main(int argc, char* argv[])
{
    Timer timer;

   // Initialize host variables ----------------------------------------------

   printf("\nSetting up the problem...\n"); fflush(stdout);
   startTime(&timer);

	Matrix Told_h, Tnew_h; 
	Matrix Told_d, Tnew_d;
	unsigned imageHeight, imageWidth;
	cudaError_t cuda_ret;
	dim3 dim_grid, dim_block;
   int  iter = 0;

	/* Read image dimensions */
    if (argc == 1) {
        imageHeight = 1000;
        imageWidth  = 1000;
    } else if (argc == 2) {
        imageHeight = atoi(argv[1]);
        imageWidth = atoi(argv[1]);
    } else if (argc == 3) {
        imageHeight = atoi(argv[1]);
        imageWidth = atoi(argv[2]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./convolution          # Image is 600 x 1000"
           "\n    Usage: ./convolution <m>      # Image is m x m"
           "\n    Usage: ./convolution <m> <n>  # Image is m x n"
           "\n");
        exit(0);
    }

	// Allocate host memory 
	Told_h = allocateMatrix(imageHeight, imageWidth);
	Tnew_h = allocateMatrix(imageHeight, imageWidth);
      
	// Initialize filter and images 
	initMatrix(Told_h);
   
   
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Matrix size: %u x %u\n", imageHeight, imageWidth);

    // Allocate device variables ----------------------------------------------
    
    printf("Allocating device variables...\n"); fflush(stdout);
    startTime(&timer);

	Told_d = allocateDeviceMatrix(imageHeight, imageWidth);
	Tnew_d = allocateDeviceMatrix(imageHeight, imageWidth);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device...\n"); fflush(stdout);
    startTime(&timer);

	 //Copy image to device global memory 
	copyToDeviceMatrix(Told_d, Told_h);
	copyToDeviceMatrix(Tnew_d, Told_h);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel ----------------------------------------------------------
    printf("Launching kernel...\n"); fflush(stdout);
    startTime(&timer);

	dim_block.x = BLOCK_SIZE;
   dim_block.y = BLOCK_SIZE;
   dim_block.z = 1;

	dim_grid.x = imageWidth/BLOCK_SIZE;
	if(imageWidth%BLOCK_SIZE != 0) dim_grid.x++;
	dim_grid.y = imageHeight/BLOCK_SIZE;
	if(imageHeight%BLOCK_SIZE != 0) dim_grid.y++;
	dim_grid.z = 1;
   
   // loop to until L2 norm is small enough
   
   float L2 = 1.;
   float eta = 1.e-6;
   while(L2>eta)
   {
      jacobi<<<dim_grid, dim_block>>>(Told_d, Tnew_d);
      jacobi<<<dim_grid, dim_block>>>(Tnew_d, Told_d);
      iter += 2;
      // copy memory back to host
      copyFromDeviceMatrix(Told_h, Told_d);
      copyFromDeviceMatrix(Tnew_h, Tnew_d);
      // check for convergence with L2 norm
      L2 = 0.;
      for(int j=0;j<imageHeight;j++) {
          for(int i=0;i<imageWidth;i++) {
              L2 += (Tnew_h.elements[j*imageWidth+i] - Told_h.elements[j*imageWidth+i])*(Tnew_h.elements[j*imageWidth+i] - Told_h.elements[j*imageWidth+i]);
          }
      }
   } 

	cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host...\n"); fflush(stdout);
    startTime(&timer);

    copyFromDeviceMatrix(Told_h, Told_d);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    /* debug
    for(int j=0;j<imageHeight;j++) {
        for(int i=0;i<imageWidth;i++) {
            printf("%3f ",Told_h.elements[i+imageWidth*j]);
        }
        printf("\n");
    }
    printf("iterations: %d\n",iter);
    */ //end debug
    // Free memory ------------------------------------------------------------

	freeMatrix(Told_h);
	freeMatrix(Tnew_h);
	freeDeviceMatrix(Told_d);
	freeDeviceMatrix(Tnew_d);

	return 0;
}
