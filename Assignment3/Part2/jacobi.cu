/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>

/* 
jacobi:
Purpose:
    solves Sequentially the given discretized problem (Poisson problem in 3d) using 3D Jacobi iterations alg.
Return value:
    the Frobenius norm when the diference is suciently small. 
*/


double jacobi(  double*** f,        /* 3D matrix "Cube" of function values, Second derivatives of temperature  */
                double*** u,        /* 3D matrix "Cube" of temperature estimates */
                double *** u_next,  /* 3D matrix "Cube" to hold new temperature estimates */
                int N,              /* #nr. interior grid points */
                double tolerance,   /* threshold */
                int iter_max,       /* maximum nr. of iterations */
                int * mp){           /* #nr. the iteration needed to get a suciently small diference*/

    double norm_result = tolerance + 0.1;        // to make sure that we enter the while loop below we add 0.01
    double delta= (double)(2.0/((double)(N+1))); // the grid spacing.
    double d_squared = delta*delta;
    double inv = 1.0/6.0;
    int edge_point_count = N + 2; 
    double *** temp; // to swipe between u and u_next.
    int i,j,k, m = 0;

    while (m < iter_max && norm_result > tolerance ) 
    {
        norm_result = 0.0;
        for (i = 1; i < edge_point_count - 1; i++) {
            for (j = 1; j < edge_point_count - 1; j++) {
                for (k = 1; k < edge_point_count - 1; k++) {
                
                    u_next[i][j][k] = inv * (u[i-1][j][k] + u[i+1][j][k] + u[i][j-1][k] + u[i][j+1][k] + u[i][j][k-1] + u[i][j][k+1] + d_squared * f[i][j][k]);
                    
                    norm_result += (((u_next[i][j][k]) - (u[i][j][k]))*((u_next[i][j][k]) - (u[i][j][k])));
                    
                }
            }
        }
        
        temp = u;
        u = u_next; 
        u_next = temp;

        norm_result = sqrt(norm_result);
        m++;
    }
    *mp = m;
    return norm_result; 
}


void jacobi_no_norm(  double*** f,        /* 3D matrix "Cube" of function values, Second derivatives of temperature  */
                double*** u,        /* 3D matrix "Cube" of temperature estimates */
                double *** u_next,  /* 3D matrix "Cube" to hold new temperature estimates */
                int N,              /* #nr. interior grid points */
                double tolerance,   /* threshold */
                int iter_max,       /* maximum nr. of iterations */
                int * mp){           /* #nr. the iteration needed to get a suciently small diference*/

    double delta= (double)(2.0/((double)(N+1))); // the grid spacing.
    double d_squared = delta*delta;
    double inv = 1.0/6.0;
    int N2 = N + 2; 
    double *** temp; // to swipe between u and u_next.
    int i,j,k, m = 0;

    while (m < iter_max)
    {
        for (i = 1; i < N2 - 1; i++) {
            for (j = 1; j < N2 - 1; j++) {
                for (k = 1; k < N2 - 1; k++) {
                
                    u_next[i][j][k] = inv * (u[i-1][j][k] + u[i+1][j][k] + u[i][j-1][k] + u[i][j+1][k] + u[i][j][k-1] + u[i][j][k+1] + d_squared * f[i][j][k]);
                    
                }
            }
        }
        
        temp = u;
        u = u_next; 
        u_next = temp;

        m++;
    }
}