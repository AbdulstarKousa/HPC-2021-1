/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include <stdio.h>


double jacobiOMP(double*** f, double*** u, double *** u_next, int N, double tolerance, int iter_max, 
    int * mp){          

    double norm_result = tolerance + 0.1;
    int m = 0;
    double delta= (double)(2.0/((double)(N+1)));
    
    const double d_squared = delta*delta;
    const double inv = 1.0/6.0;
    const int edge_point_count = N + 2;

    double *** temp;
    double u_temp;
    int i; int j; int k; 
    double subtracted;

    #pragma omp parallel\
    shared(m, f, u, u_next)\
    private(i,j,k,u_temp,subtracted)
    while ( m < iter_max && norm_result > tolerance ) {

        #pragma omp barrier
        #pragma omp single
        {
            norm_result = 0.0;
        }
        
        
        //         #pragma omp parallel for\
        shared(f, u, u_next)\
        private(i,j,k,u_temp,subtracted)\
        reduction(+:norm_result)
        #pragma omp for reduction(+:norm_result)
        for (i = 1; i < edge_point_count - 1; i++) {
            for (j = 1; j < edge_point_count - 1; j++) {
                for (k = 1; k < edge_point_count - 1; k++) {
                
                    u_temp = inv * (u[i-1][j][k] + u[i+1][j][k] + u[i][j-1][k] + u[i][j+1][k] + u[i][j][k-1] + u[i][j][k+1] + d_squared * f[i][j][k]);
                    
                    u_next[i][j][k] =u_temp;

                    subtracted = u_temp - u[i][j][k];
                    
                    norm_result += subtracted*subtracted;
                }
            }
        }
        
        #pragma omp single
        {
        temp = u;
        u = u_next; 
        u_next = temp;
        norm_result = sqrt(norm_result);
        m++;
        }
    }

    *mp = m;
    return norm_result; 
}

// double jacobiOMP(double*** f, double*** u, double *** u_next, int N, double tolerance, int iter_max, 
//     int * mp){          

//     double norm_result = tolerance + 0.1;
//     int m = 0;
//     double delta= (double)(2.0/((double)(N+1)));
    
//     const double d_squared = delta*delta;
//     const double inv = 1.0/6.0;
//     const int edge_point_count = N + 2;

//     double *** temp;
//     double u_temp;
//     int i; int j; int k; 
//     double subtracted;

//     while ( m < iter_max && norm_result > tolerance ) {

//         norm_result = 0.0;       
        
//         #pragma omp parallel for\
//         shared(f, u, u_next)\
//         private(i,j,k,u_temp,subtracted)\
//         reduction(+:norm_result)
//         for (i = 1; i < edge_point_count - 1; i++) {
//             for (j = 1; j < edge_point_count - 1; j++) {
//                 for (k = 1; k < edge_point_count - 1; k++) {
                
//                     u_temp = inv * (u[i-1][j][k] + u[i+1][j][k] + u[i][j-1][k] + u[i][j+1][k] + u[i][j][k-1] + u[i][j][k+1] + d_squared * f[i][j][k]);
                    
//                     u_next[i][j][k] =u_temp;

//                     subtracted = u_temp - u[i][j][k];
                    
//                     norm_result += subtracted*subtracted;
//                 }
//             }
//         }
        
//         temp = u;
//         u = u_next; 
//         u_next = temp;
//         norm_result = sqrt(norm_result);
//         m++;

//     }

//     *mp = m;
//     return norm_result; 
// }