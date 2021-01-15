/* gauss_seidel.c - Poisson problem in 3d
 *
 */
#include <math.h>
#include <stdio.h>
#include <omp.h>



double gauss_seidelOMP(double*** f, double*** u, int N, double tolerance, int iter_max,int * mp){ 
    double norm_result = tolerance + 0.01 ;
    int m = 0;
    double delta= (double)(2.0/((double)(N+1)));
    double d_squared = delta*delta;
    double inv = 1.0/6.0;
    int edge_point_count = N + 2; 
    double u_old;
    int i, j, k; 

    while ( m < iter_max && norm_result > tolerance ) {
        norm_result = 0.0;
        
        #pragma omp parallel for ordered(2) private(u_old) \
        reduction(+: norm_result) schedule(static,1)
        for (i = 1; i < edge_point_count - 1; i++) {
            for (j = 1; j < edge_point_count - 1; j++) {
                #pragma omp ordered depend(sink:i-1,j) depend(sink:i,j-1)
                for (k = 1; k < edge_point_count - 1; k++) {
                    
                    u_old = u[i][j][k];
                    u[i][j][k] = inv * (u[i-1][j][k] + u[i+1][j][k] + 
                                        u[i][j-1][k] + u[i][j+1][k] + 
                                        u[i][j][k-1] + u[i][j][k+1] + 
                                        d_squared * f[i][j][k]);

                    norm_result += (((u[i][j][k]) - (u_old))*((u[i][j][k]) - (u_old)));
                }
                #pragma omp ordered depend(source)
            }
        }
        norm_result = sqrt(norm_result); 
        m++;
    }
    *mp = m;
    return norm_result;
}

