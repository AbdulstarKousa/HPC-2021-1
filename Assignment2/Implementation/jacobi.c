/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


double jacobi(double*** f, double*** u, double *** u_next, int N, double tolerance, int iter_max){

    double norm_result = tolerance + 0.1;
    int m = 0;
    double delta= (double)(2.0/((double)(N+1)));
    double d_squared = delta*delta;
    double inv = 1.0/6.0;
    int edge_point_count = N + 2; 
    double *** temp; 

    while ( m < iter_max && norm_result > tolerance ) {
        norm_result = 0.0;
        for (int i = 1; i < edge_point_count - 1; i++) {
            for (int j = 1; j < edge_point_count - 1; j++) {
                for (int k = 1; k < edge_point_count - 1; k++) {
                
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

    printf("Norm result: %e\n", norm_result);
    printf("Finished after %d iterations\n", m);

    return norm_result; 
}
