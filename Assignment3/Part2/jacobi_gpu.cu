#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>


__global__ 
void jacobi_kernel1(
    double*** d_f,        /* 3D matrix "Cube" of function values, Second derivatives of temperature  */
    double*** d_u,        /* 3D matrix "Cube" of temperature estimates */
    double *** d_u_next,  /* 3D matrix "Cube" to hold new temperature estimates */
    int N,                /* #nr. interior grid points */
    double d_squared, 
    double inv              ){

    int i,j,k; 
    int N2 = N + 2; 

    for (i = 1; i < N2 - 1; i++) {
        for (j = 1; j < N2 - 1; j++) {
            for (k = 1; k < N2 - 1; k++) {
                
                d_u_next[i][j][k] = inv * (d_u[i-1][j][k] + d_u[i+1][j][k] + d_u[i][j-1][k] + d_u[i][j+1][k] + d_u[i][j][k-1] + d_u[i][j][k+1] + d_squared * d_f[i][j][k]);
                    
            }
        }
    }
    //printf("Leaving kernel function\n");
}

void jacobi_gpu_wrap1(  double*** d_f,        /* 3D matrix "Cube" of function values, Second derivatives of temperature  */
                double*** d_u,        /* 3D matrix "Cube" of temperature estimates */
                double *** d_u_next,  /* 3D matrix "Cube" to hold new temperature estimates */
                int N,              /* #nr. interior grid points */
                double tolerance,   /* threshold */
                int iter_max,       /* maximum nr. of iterations */
                int * mp){           /* #nr. the iteration needed to get a suciently small diference*/

    double delta= (double)(2.0/((double)(N+1))); // the grid spacing.
    double d_squared = delta*delta;
    double inv = 1.0/6.0;
    double *** temp; // to swipe between u and u_next.
    int m = 0;

    //printf("Entering while loop\n");
    while (m < iter_max) 
    {

        jacobi_kernel1<<<1,1>>>(d_f, d_u, d_u_next, N, d_squared,inv);    
        cudaDeviceSynchronize();          

        temp = d_u;
        d_u = d_u_next; 
        d_u_next = temp;
        
        m++;
    }

    //printf("End Jacobi wrapper\n");
}




/* *************
 EXERCISE 6 
************* */

__global__ 
void jacobi_kernel2(
    double*** d_f,        /* 3D matrix "Cube" of function values, Second derivatives of temperature  */
    double*** d_u,        /* 3D matrix "Cube" of temperature estimates */
    double *** d_u_next,  /* 3D matrix "Cube" to hold new temperature estimates */
    int N,                /* #nr. interior grid points */
    double d_squared, 
    double inv              ){

    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(0 < i && 0 < j && 0 < k && i < N+1 && j < N+1 && k < N+1)
    {    
        d_u_next[i][j][k] = inv * (d_u[i-1][j][k] + d_u[i+1][j][k] + d_u[i][j-1][k] + d_u[i][j+1][k] + d_u[i][j][k-1] + d_u[i][j][k+1] + d_squared * d_f[i][j][k]);
    }
}

void jacobi_gpu_wrap2(  double*** d_f,        /* 3D matrix "Cube" of function values, Second derivatives of temperature  */
                double*** d_u,        /* 3D matrix "Cube" of temperature estimates */
                double *** d_u_next,  /* 3D matrix "Cube" to hold new temperature estimates */
                int N,              /* #nr. interior grid points */
                double tolerance,   /* threshold */
                int iter_max,       /* maximum nr. of iterations */
                int * mp){           /* #nr. the iteration needed to get a suciently small diference*/

    double delta= (double)(2.0/((double)(N+1))); // the grid spacing.
    double d_squared = delta*delta;
    double inv = 1.0/6.0;
    double *** temp; // to swipe between u and u_next.
    int m = 0;
    
    int threads_blck = 8; //optmized to be fastest with 8 threads per block (each dim)

    dim3 dimBlock(threads_blck,threads_blck,threads_blck);// threads per block
    dim3 dimGrid(((N+2)/dimBlock.x)+1,((N+2)/dimBlock.y)+1,((N+2)/dimBlock.z)+1); // xx blocks in total

    //printf("Entering while loop\n");
    while (m < iter_max) //&& norm_result > tolerance 
    {
        jacobi_kernel2<<<dimGrid,dimBlock>>>(d_f, d_u, d_u_next, N, d_squared,inv);    
        cudaDeviceSynchronize();          

        temp = d_u;
        d_u = d_u_next; 
        d_u_next = temp;
        
        m++;
    }
}



/* *************
 EXERCISE 7 
************* */

__global__ 
void jacobi_kernel31(
    double*** d0_f,        /* 3D matrix "Cube" of function values, Second derivatives of temperature  */
    double*** d0_u,        /* 3D matrix "Cube" of temperature estimates */
    double*** d1_u,
    double *** d0_u_next,  /* 3D matrix "Cube" to hold new temperature estimates */
    int N,                /* #nr. interior grid points */
    double d_squared, 
    double inv              ){

    //I AM THE BOTTOM 
    //when highest z value = (N+2)/2 visit my sister device 

    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(0 < i && 0 < j && 0 < k && k < N+1 && j < N+1 && i < (N+2)/2)
    {    
        if (i == ((N+2)/2)-1) 
        {
            //Peer access
            d0_u_next[i][j][k] = inv * (d0_u[i-1][j][k] + d1_u[0][j][k] + d0_u[i][j-1][k] + d0_u[i][j+1][k] + d0_u[i][j][k-1] + d0_u[i][j][k+1] + d_squared * d0_f[i][j][k]);
        }
        else 
        {
            d0_u_next[i][j][k] = inv * (d0_u[i-1][j][k] + d0_u[i+1][j][k] + d0_u[i][j-1][k] + d0_u[i][j+1][k] + d0_u[i][j][k-1] + d0_u[i][j][k+1] + d_squared * d0_f[i][j][k]);
        }
    }
}

__global__ 
void jacobi_kernel32(
    double*** d1_f,        /* 3D matrix "Cube" of function values, Second derivatives of temperature  */
    double*** d1_u,        /* 3D matrix "Cube" of temperature estimates */
    double*** d0_u,
    double *** d1_u_next,  /* 3D matrix "Cube" to hold new temperature estimates */
    int N,                /* #nr. interior grid points */
    double d_squared, 
    double inv              ){

    //I AM THE TOP
    //when lowest z value = 0 visit my sister device

    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;


    if(0 <= i && 0 < j && 0 < k && k < N+1 && j < N+1 && i < (N+2/2)-1)  //if(0 < i && 0 < j && 0 <= k && i < N+1 && j < N+1 && k < (N+2/2)-1)
    {  
        if (i == 0) 
        {   
            //Peer access
            d1_u_next[i][j][k] = inv * (d0_u[((N+2)/2)-1][j][k] + d1_u[i+1][j][k] + d1_u[i][j-1][k] + d1_u[i][j+1][k] + d1_u[i][j][k-1] + d1_u[i][j][k+1] + d_squared * d1_f[i][j][k]);
        } 
        else 
        {
            d1_u_next[i][j][k] = inv * (d1_u[i-1][j][k] + d1_u[i+1][j][k] + d1_u[i][j-1][k] + d1_u[i][j+1][k] + d1_u[i][j][k-1] + d1_u[i][j][k+1] + d_squared * d1_f[i][j][k]);
        }
    }
}

void jacobi_gpu_wrap3(  double*** d0_f,        /* 3D matrix "Cube" of function values, Second derivatives of temperature  */
                double*** d0_u,        /* 3D matrix "Cube" of temperature estimates */
                double *** d0_u_next,  /* 3D matrix "Cube" to hold new temperature estimates */
                double*** d1_f,        /* 3D matrix "Cube" of function values, Second derivatives of temperature  */
                double*** d1_u,        /* 3D matrix "Cube" of temperature estimates */
                double *** d1_u_next,
                int N,              /* #nr. interior grid points */
                double tolerance,   /* threshold */
                int iter_max,       /* maximum nr. of iterations */
                int * mp){           /* #nr. the iteration needed to get a suciently small diference*/

    double delta= (double)(2.0/((double)(N+1))); 
    double d_squared = delta*delta;
    double inv = 1.0/6.0;
    double *** temp0; // to swipe between u and u_next.
    double *** temp1;
    int m = 0;

    int threads_blck = 8; 

    dim3 dimBlock(threads_blck,threads_blck,threads_blck);// threads per block
    dim3 dimGrid((((N+2)/2)/dimBlock.x)+1,((N+2)/dimBlock.y)+1,((N+2)/dimBlock.z)+1); 

    //printf("Entering while loop\n");
    while (m < iter_max) 
    {
        //DEVICE 0 
        cudaSetDevice(0);
        jacobi_kernel31<<<dimGrid,dimBlock>>>(d0_f, d0_u, d1_u, d0_u_next, N, d_squared,inv);     
        
        //DEVICE 1 
        cudaSetDevice(1);
        jacobi_kernel32<<<dimGrid,dimBlock>>>(d1_f, d1_u, d0_u, d1_u_next, N, d_squared,inv);    
        checkCudaErrors(cudaDeviceSynchronize());  
       
        cudaSetDevice(0); 
        checkCudaErrors(cudaDeviceSynchronize()); 
 
        temp0 = d0_u;
        d0_u = d0_u_next; 
        d0_u_next = temp0;

        temp1 = d1_u;
        d1_u = d1_u_next; 
        d1_u_next = temp1;
        
        m++;
    }
}



/* *************
 EXERCISE 8 
************* */

__inline__ __device__
double warpReduceSum(double value) 
{ 
    for (int i = 16; i > 0; i /= 2)
    {
        value += __shfl_down_sync(-1, value, i); 
    }
    return value;
}


__inline__ __device__
double blockReduceSum(double value) {
    return warpReduceSum(value);
}


//kernel
__global__ 
void jacobi_kernel4new(
    double*** d_f,        /* 3D matrix "Cube" of function values, Second derivatives of temperature  */
    double*** d_u,        /* 3D matrix "Cube" of temperature estimates */
    double *** d_u_next,  /* 3D matrix "Cube" to hold new temperature estimates */
    int N,                /* #nr. interior grid points */ 
    double inv,
    double d_squared,
    double * norm ){
    double inter_norm;

    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(0 < i && 0 < j && 0 < k && i < N+1 && j < N+1 && k < N+1)
    {    
        d_u_next[i][j][k] = inv * (d_u[i-1][j][k] + d_u[i+1][j][k] + d_u[i][j-1][k] + d_u[i][j+1][k] + d_u[i][j][k-1] + d_u[i][j][k+1] + d_squared * d_f[i][j][k]);
        inter_norm = (d_u_next[i][j][k] - d_u[i][j][k])*(d_u_next[i][j][k] - d_u[i][j][k]);
        //---- baseline
        //atomicAdd(norm,inter_norm);
        
        //---- v2        
        double value = blockReduceSum(inter_norm);
        if (threadIdx.x == 0){
            atomicAdd(norm, value);
        }

        //---- v3 notes: 
        // for blockReduceSum
        // defined warpSize = 32 
        // lin3axis = blockDim.y*blockDim.z*blockIdx.x + blockIdx.y*blockDim.z + blockIdx.z

    }
    

}

//host warp
void jacobi_gpu_wrap4new(  double*** d_f,   /* 3D matrix "Cube" of function values, Second derivatives of temperature  */
                double*** d_u,              /* 3D matrix "Cube" of temperature estimates */
                double *** d_u_next,        /* 3D matrix "Cube" to hold new temperature estimates */
                int N,                      /* #nr. interior grid points */
                double tolerance,           /* threshold */
                int iter_max,               /* maximum nr. of iterations */
                int * mp){                  /* #nr. the iteration needed to get a suciently small diference*/
    
    double delta= (double)(2.0/((double)(N+1))); // the grid spacing.
    double d_squared = delta*delta;
    double inv = 1.0/6.0; 
    int m = 0; 
    double *** temp; 
    double * d_norm; 
    double * h_norm; 

    cudaMalloc((void**)&d_norm,sizeof(double));
    cudaMallocHost((void**)&h_norm,sizeof(double)); 
                    
    int threads_blck = 8; 

    dim3 dimBlock(threads_blck,threads_blck,threads_blck);                          // threads per block
    dim3 dimGrid(((N+2)/dimBlock.x)+1,((N+2)/dimBlock.y)+1,((N+2)/dimBlock.z)+1);   // xx blocks in total

    printf("Calling kernel\n");
    
    *h_norm = 1.0 + tolerance; 

    while(m < iter_max &&  *h_norm > tolerance){
        *h_norm = 0.0;
        cudaMemcpy(d_norm, h_norm, sizeof(double),cudaMemcpyHostToDevice);
        jacobi_kernel4new<<<dimGrid,dimBlock>>>(d_f, d_u, d_u_next, N,inv,d_squared,d_norm);    
        cudaDeviceSynchronize(); 
        
        temp = d_u;
        d_u = d_u_next; 
        d_u_next = temp;

        cudaMemcpy(h_norm, d_norm, sizeof(double),cudaMemcpyDeviceToHost);
        *h_norm = sqrt(*h_norm);
        //printf("\n h_norm = %e", *h_norm);     
        m++;
    }        

    printf("\n m_break = %d", m-1);
    printf("\n End kernel exercise 8 \n");
    *mp = m-1;
}