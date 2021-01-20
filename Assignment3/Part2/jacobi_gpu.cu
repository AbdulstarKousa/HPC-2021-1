#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>


__global__ 
void jacobi_kernel(
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
    printf("Leaving kernel function\n");
}

void jacobi_gpu_wrap(  double*** d_f,        /* 3D matrix "Cube" of function values, Second derivatives of temperature  */
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
        jacobi_kernel<<<1,1>>>(d_f, d_u, d_u_next, N, d_squared,inv);    
        cudaDeviceSynchronize();          

        temp = d_u;
        d_u = d_u_next; 
        d_u_next = temp;
        
        m++;
    }
    *mp = m;
    printf("End Jacobi wrapper\n");
}
