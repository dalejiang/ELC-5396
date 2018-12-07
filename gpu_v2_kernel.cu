__global__ void jacobi(Matrix T_old, Matrix Bo,float* reduc)
{
   int tx = threadIdx.x; 
   int ty = threadIdx.y;
   int row_o = blockIdx.y*TILE_SIZE + threadIdx.y;
   int col_o = blockIdx.x*TILE_SIZE + threadIdx.x;
   int row_i = row_o - FILTER_SIZE/2; 
   int col_i = col_o - FILTER_SIZE/2; 
   
   __shared__ float T1[BLOCK_SIZE][BLOCK_SIZE];
   __shared__ float T2[BLOCK_SIZE][BLOCK_SIZE];

   int nx= T_old.width;
   int ny= T_old.height;

   if ((row_i > 0) && (row_i<ny-1) && (col_i>0) && (col_i<nx-1))
      T1[ty][tx]=T_old.elements[row_i*nx + col_i];
   else
      T1[ty][tx]=0.0;
   
   __syncthreads();
  
   float output = 0.0;
   for(int j=1; j<BLOCK_SIZE-1;j++){
      for(int i=1; i<BLOCK_SIZE-1;i++){
         T2[j][i] = 0.25*(T2[j+1][i] + T2[j-1][i] + T2[j][i+1] +T2[j][i-1]);
         output += (T2[j][i] - T1[j][i])*(T2[j][i] - T1[j][i]);
      }
   }
   
   Bo[] = T2 
   Bo[] = T2 
   Bo[] = T2 
   Bo[] = T2 

   reduc[0] = output;
   
}
