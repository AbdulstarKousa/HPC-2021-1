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
    int edge_point_count = N + 2; 

    for (i = 1; i < edge_point_count - 1; i++) {
        for (j = 1; j < edge_point_count - 1; j++) {
            for (k = 1; k < edge_point_count - 1; k++) {
                
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

    double norm_result = tolerance + 0.1;        // to make sure that we enter the while loop below we add 0.01
    double delta= (double)(2.0/((double)(N+1))); // the grid spacing.
    double d_squared = delta*delta;
    double inv = 1.0/6.0;
    double *** temp; // to swipe between u and u_next.
    int i,j,k, m = 0;

    // alg. from the slides show "Assignment 2: The Poisson Problem" p 14. 

    printf("Entering while loop\n");
    while (m < iter_max) //&& norm_result > tolerance 
    {

        //insert function 
        jacobi_kernel1<<<1,1>>>(d_f, d_u, d_u_next, N, d_squared,inv);    
        cudaDeviceSynchronize();          

        temp = d_u;
        d_u = d_u_next; 
        d_u_next = temp;
        
        m++;
    }
    *mp = m;
    printf("End Jacobi wrapper\n");
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

    double norm_result = tolerance + 0.1;        // to make sure that we enter the while loop below we add 0.01
    double delta= (double)(2.0/((double)(N+1))); // the grid spacing.
    double d_squared = delta*delta;
    double inv = 1.0/6.0;
    double *** temp; // to swipe between u and u_next.
    int m = 0;

    // alg. from the slides show "Assignment 2: The Poisson Problem" p 14.
    
    int threads_blck = N; 

    dim3 dimBlock(threads_blck,threads_blck,threads_blck);// threads per block
    dim3 dimGrid((N/dimBlock.x)+1,(N/dimBlock.y)+1,(N/dimBlock.z)+1); // xx blocks in total

    printf("Entering while loop\n");
    while (m < iter_max) //&& norm_result > tolerance 
    {

        //insert function 
        jacobi_kernel2<<<dimGrid,dimBlock>>>(d_f, d_u, d_u_next, N, d_squared,inv);    
        cudaDeviceSynchronize();          

        temp = d_u;
        d_u = d_u_next; 
        d_u_next = temp;
        
        m++;
    }
    *mp = m;
    printf("End Jacobi wrapper\n");
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


    if (k == ((N+2)/2)) 
    {
        // Need to access memory of sister device
        // d0_u[?][?][k+1] becomes d1_u[?][?][0]
        d0_u_next[i][j][k] = (d0_u[i-1][j][k] + d0_u[i+1][j][k] + d0_u[i][j-1][k] + d0_u[i][j+1][k] + d0_u[i][j][k-1] + d1_u[i][j][0] + d_squared * d_f[i][j][k]);
    }


    // if(0 < i && 0 < j && 0 < k && i < N+1 && j < N+1 && k < N+1)
    if(0 < i && 0 < j && 0 < k && i < N+1 && j < N+1 && k < N+1)
    {    
        d_u_next[i][j][k] = inv * (d_u[i-1][j][k] + d_u[i+1][j][k] + d_u[i][j-1][k] + d_u[i][j+1][k] + d_u[i][j][k-1] + d_u[i][j][k+1] + d_squared * d_f[i][j][k]);
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

    if (k == 0) 
    {
        // Need to access memory of sister device
    }


    // if(0 < i && 0 < j && 0 < k && i < N+1 && j < N+1 && k < N+1)
    if(0 < i && 0 < j && 0 < k && i < N+1 && j < N+1 && k < (N/2))
    {    
        d_u_next[i][j][k] = inv * (d_u[i-1][j][k] + d_u[i+1][j][k] + d_u[i][j-1][k] + d_u[i][j+1][k] + d_u[i][j][k-1] + d_u[i][j][k+1] + d_squared * d_f[i][j][k]);
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

    double norm_result = tolerance + 0.1;        // to make sure that we enter the while loop below we add 0.01
    double delta= (double)(2.0/((double)(N+1))); // the grid spacing.
    double d_squared = delta*delta;
    double inv = 1.0/6.0;
    double *** temp0; // to swipe between u and u_next.
    double *** temp1;
    int m = 0;

    // alg. from the slides show "Assignment 2: The Poisson Problem" p 14.
    
    // TODO: Figure out whether we should actually have N blocks per thread
    int threads_blck = N; 

    dim3 dimBlock(threads_blck,threads_blck,threads_blck/2);// threads per block
    dim3 dimGrid((N/dimBlock.x)+1,(N/dimBlock.y)+1,((N/2)/dimBlock.z)+1); // xx blocks in total

    printf("Entering while loop\n");
    while (m < iter_max) //&& norm_result > tolerance 
    {

        //DEVICE 0 
        cudaSetDevice(0);
        cudaDeviceEnablePeerAccess(1, 0);
        jacobi_kernel31<<<dimGrid,dimBlock>>>(d0_f, d0_u, d0_u_next, N, d_squared,inv);    
        cudaDeviceSynchronize();   
        
        //DEVICE 1 
        cudaSetDevice(1);
        cudaDeviceEnablePeerAccess(0, 0);
        jacobi_kernel32<<<dimGrid,dimBlock>>>(d1_f, d1_u, d1_u_next, N, d_squared,inv);    
        cudaDeviceSynchronize();          

        temp0 = d0_u;
        d0_u = d0_u_next; 
        d0_u_next = temp0;

        temp1 = d1_u;
        d1_u = d1_u_next; 
        d1_u_next = temp1;
        
        m++;
    }
    *mp = m;
    printf("End Jacobi wrapper\n");
}



/* *************
 EXERCISE 8 
************* */

__global__ 
void jacobi_kernel4(
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
        
        //Add norm calculation 
    }
}

void jacobi_gpu_wrap4(  double*** d_f,        /* 3D matrix "Cube" of function values, Second derivatives of temperature  */
                double*** d_u,        /* 3D matrix "Cube" of temperature estimates */
                double *** d_u_next,  /* 3D matrix "Cube" to hold new temperature estimates */
                int N,              /* #nr. interior grid points */
                double tolerance,   /* threshold */
                int iter_max,       /* maximum nr. of iterations */
                int * mp){           /* #nr. the iteration needed to get a suciently small diference*/

    double norm_result = tolerance + 0.1;        // to make sure that we enter the while loop below we add 0.01
    double delta= (double)(2.0/((double)(N+1))); // the grid spacing.
    double d_squared = delta*delta;
    double inv = 1.0/6.0;
    double *** temp; // to swipe between u and u_next.
    int m = 0;

    // alg. from the slides show "Assignment 2: The Poisson Problem" p 14.
    
    int threads_blck = N; 

    dim3 dimBlock(threads_blck,threads_blck,threads_blck);// threads per block
    dim3 dimGrid((N/dimBlock.x)+1,(N/dimBlock.y)+1,(N/dimBlock.z)+1); // xx blocks in total

    printf("Entering while loop\n");
    while (m < iter_max) //&& norm_result > tolerance 
    {

        //insert function 
        jacobi_kernel4<<<dimGrid,dimBlock>>>(d_f, d_u, d_u_next, N, d_squared,inv);    
        cudaDeviceSynchronize();          

        temp = d_u;
        d_u = d_u_next; 
        d_u_next = temp;
        
        m++;
    }
    *mp = m;
    printf("End Jacobi wrapper\n");
}
