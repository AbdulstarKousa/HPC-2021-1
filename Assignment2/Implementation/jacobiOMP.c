/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <stdio.h>

double jacobiOMP(double*** f, double*** u, double *** u_next, int N, double tolerance, int iter_max) {

    double norm_result = tolerance + 0.1;
    int m = 0;
    double delta= (double)(2.0/((double)(N+1)));
    
    const double d_squared = delta*delta;
    const double inv = 1.0/6.0;
    const int edge_point_count = N + 2;

    double *** temp;
    double u_temp;
    int i; int j; int k; 
    
    // #pragma omp parallel for\
	// shared(norm_result, m, d_squared, inv, edge_point_count) private(i,j,k,temp)
    while ( m < iter_max && norm_result > tolerance ) {
        norm_result = 0.0;
        
        #pragma omp parallel for\
	    shared(f, u, u_next)\
        private(i,j,k,u_temp)\
        reduction(+:norm_result)
        for (i = 1; i < edge_point_count - 1; i++) {
            for (j = 1; j < edge_point_count - 1; j++) {
                for (k = 1; k < edge_point_count - 1; k++) {
                
                    u_temp = inv * (u[i-1][j][k] + u[i+1][j][k] + u[i][j-1][k] + u[i][j+1][k] + u[i][j][k-1] + u[i][j][k+1] + d_squared * f[i][j][k]);
                    
                    u_next[i][j][k] =u_temp;
                    
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
    printf("Number of iterations run: %d \n", m);
    return norm_result; 




}
