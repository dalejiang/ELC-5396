__global__ void jacobi(Matrix T_old, Matrix T_new)
{
   int i = blockIdx.x * blockDim.x + threadIdx.x ;
   int j = blockIdx.y * blockDim.y + threadIdx.y ;
   int nx= T_old.width;
   int ny= T_old.height;
                               
   int C = i + j*nx;          
   int N = i + (j-1)*nx;     
   int S = i + (j+1)*nx;    
   int E = (i+1) + j*nx;   
   int W = (i-1) + j*nx;  
                         
   // only update "interior" node points
   if(i>0 && i<nx-1 && j>0 && j<ny-1) {
      T_new.elements[C] = 0.25*( T_old.elements[N] + T_old.elements[S] + T_old.elements[E] + T_old.elements[W] );
   }
   __syncthreads();    
}

