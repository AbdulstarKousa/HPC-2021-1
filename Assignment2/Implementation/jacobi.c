/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>


// Expected type signature: 
// int jacobi(double ***, double ***, double ***, int, int, double *);
// TODO:Still unclear what all of this is -> figure that out. My guess: 
// int jacobi(double*** f, double*** u, double*** u_next, int n, int ???, double* ???)

// TODO:
// Does not yet compile, as the type signature doesn't match
// Also, the calculations are not correct yet
void jacobi(double*** f, double*** u, double*** u_next, int edge_point_count, double delta) {
    // f: Cube of function values -> Second derivatives of temperature
    // u: Cube of temperature estimates of previous iteration
    // u_next: Cube to hold new temperature estimates
    // edge_point_count: Number of points along an axis
    // delta: Distance between two neighbor points along an axis

    double d_squared = delta*delta;
    double inv = 1.0/6.0;
    
    for (int i = 1; i < edge_point_count - 1; i++) {
        for (int j = 1; j < edge_point_count - 1; j++) {
            for (int k = 1; k < edge_point_count - 1; k++) {
                u_next[i][j][k] = inv * (u[i-1][j][k] + u[i+1][j][k] + u[i][j-1][k] + u[i][j+1][k] + u[i][j][k-1] + u[i][j][k+1] + d_squared * f[i][j][k]);
            }
        }
    }
}
