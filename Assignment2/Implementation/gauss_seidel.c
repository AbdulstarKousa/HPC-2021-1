/* gauss_seidel.c - Poisson problem in 3d
 *
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* 
gauss_seidel:
Purpose:
    solves Sequentially the given discretized problem (Poisson problem in 3d) using 3D Gauss-Seidel alg.
Return value:
    the Frobenius norm when the diference is suciently small. 
*/
double gauss_seidel(double*** f,        /* 3D matrix "Cube" of function values, Second derivatives of temperature  */
                    double*** u,        /* 3D matrix "Cube" of temperature estimates */
                    int N,              /* #nr. interior grid points */
                    double tolerance,   /* threshold */
                    int iter_max,       /* maximum nr. of iterations */
                    int * m){           /* #nr. the iteration needed to get a suciently small diference*/

    m = 0; 
    double norm_result = tolerance + 0.01; // to make sure that we enter the while loop below we add 0.01
    double delta= (double)(2.0/((double)(N+1))); // the grid spacing.
    double d_squared = delta*delta;
    double inv = 1.0/6.0;
    int edge_point_count = N + 2; 
    double u_old; // to store u old.
    int i,j,k;

    // alg. from the slides show "Assignment 2: The Poisson Problem" p 14. 
    while ( m < iter_max && norm_result > tolerance ) {
        norm_result = 0.0; //set back to 0.0
        for (i = 1; i < edge_point_count - 1; i++) {
            for (j = 1; j < edge_point_count - 1; j++) {
                for (k = 1; k < edge_point_count - 1; k++) {
                    u_old = u[i][j][k];
                    u[i][j][k] = inv * (u[i-1][j][k] + u[i+1][j][k] + u[i][j-1][k] + u[i][j+1][k] + u[i][j][k-1] + u[i][j][k+1] + d_squared * f[i][j][k]);
                    norm_result += (((u[i][j][k]) - (u_old))*((u[i][j][k]) - (u_old)));
                }
            }
        }

        norm_result = sqrt(norm_result);
        m++;
    }

    return norm_result; 
}