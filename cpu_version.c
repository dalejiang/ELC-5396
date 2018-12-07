#include <stdio.h>
#include <math.h>
#include <time.h>
#define nx 512
#define ny 512

clock_t start, end;
float cpu_time_used;

int main()
{
    clock_t start, end;
    float cpu_time_used;
    int   iter = 0;
    int   Max_iter = 1000;
    float T0[ny][nx] = {0.0};
    float T1[ny][nx] = {0.0};
    float eta        = 1.0e-6;
    float L2        = 1.;
    
    // right wall boundary conditions
    for(int j=0;j<ny;j++) {
        T0[j][nx-1]=12.0 * (float) j;
        T1[j][nx-1]=12.0 * (float) j;
    }
    
    // bottom wall boundary conditions
    for(int i=0;i<nx;i++) {
        T0[ny-1][i]=15.0 * (float) i;
        T1[ny-1][i]=15.0 * (float) i;
    }
    
    // start clock
    start = clock();
    while(L2 > eta)
    {
        for(int i=1; i<nx-1; i++)
        {
            for(int j=1; j<ny-1; j++)
            {
                float T_E = T0[j][i+1];
                float T_W = T0[j][i-1];
                float T_N = T0[j+1][i];
                float T_S = T0[j-1][i];
                T1[j][i] = 0.25*(T_E + T_W + T_N + T_S);
            }
        }
        
        for(int i=1; i<nx-1; i++)
        {
            for(int j=1; j<ny-1; j++)
            {
                float T_E = T1[j][i+1];
                float T_W = T1[j][i-1];
                float T_N = T1[j+1][i];
                float T_S = T1[j-1][i];
                T0[j][i] = 0.25*(T_E + T_W + T_N + T_S);
            }
        }
        iter+=2;
        // check for convergence
        L2 = 0;
        for(int j=0;j<ny;j++) {
            for(int i=0;i<nx;i++) {
                L2 += (T1[j][i] - T0[j][i])*(T1[j][i] -T0[j][i]);
            }
        }
        //L2 = sqrt(L2);
    }
    
    // end clock
    end = clock();
    
    /* debug
    for(int j=0;j<ny;j++) {
        for(int i=0;i<nx;i++) {
            printf("%3f ",T0[j][i]);
        }
        printf("\n");
    }
    // end debug */
    
    // check time
    cpu_time_used = ((float) (end - start)) / CLOCKS_PER_SEC;
    printf("cpu = %3.10f sec\n",cpu_time_used);
    printf("iterations = %3d \n",iter);
    
}

