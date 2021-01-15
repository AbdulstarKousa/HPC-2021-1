/* gauss_seidel.c - Poisson problem in 3d
 *
 */
#include <math.h>
#include <stdio.h>
#include <omp.h>

/*
double gauss_seidelOMP(double*** f, double*** u, int N, double tolerance, int iter_max){

    double norm_result = tolerance + 0.01 ;
    int m = 0;
    double delta= (double)(2.0/((double)(N+1)));
    double d_squared = delta*delta;
    double inv = 1.0/6.0;
    int edge_point_count = N + 2; 
    double u_old;

    // k = m 
    // kmax = iter
    // threshold = tolerance
    // d = norm_result
    
    int i, j, k; 


    while ( m < iter_max && norm_result > tolerance ) {
        norm_result = 0.0; 

        #pragma omp parallel for default(none)\
            shared(u,f,inv,edge_point_count,d_squared) private(i,j,k,u_old) \
            reduction(+: norm_result)   
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

*/


double gauss_seidelOMP(double*** f, double*** u, int N, double tolerance, int iter_max){

    double norm_result = tolerance + 0.01 ;
    int m = 0;
    double delta= (double)(2.0/((double)(N+1)));
    double d_squared = delta*delta;
    double inv = 1.0/6.0;
    int edge_point_count = N + 2; 
    double u_old;
    double t_norm; 

    // k = m 
    // kmax = iter
    // threshold = tolerance
    // d = norm_result
    
    int i, j, k; 

    while ( m < iter_max && norm_result > tolerance ) {
        norm_result = 0.0;
        
        #pragma omp parallel for ordered(2) private(u_old) reduction(+: norm_result) schedule(static,1)
        for (i = 1; i < edge_point_count - 1; i++) {
            for (j = 1; j < edge_point_count - 1; j++) {
                #pragma omp ordered depend(sink:i-1,j) depend(sink:i,j-1)
                for (k = 1; k < edge_point_count - 1; k++) {
                    
                    u_old = u[i][j][k];
                    u[i][j][k] = inv * (u[i-1][j][k] + u[i+1][j][k] + u[i][j-1][k] + u[i][j+1][k] + u[i][j][k-1] + u[i][j][k+1] + d_squared * f[i][j][k]);

                    norm_result += (((u[i][j][k]) - (u_old))*((u[i][j][k]) - (u_old)));
                    
                }
                #pragma omp ordered depend(source)
            }
        }
        
        norm_result = sqrt(norm_result); 
        m++;
    
    }

    printf("Iterations %d \n",m);
    return norm_result;
}

/*

double gauss_seidelOMP(double*** f, double*** u, int N, double tolerance, int iter_max){

    double norm_result = tolerance + 0.01 ;
    int m = 0;
    double delta= (double)(2.0/((double)(N+1)));
    double d_squared = delta*delta;
    double inv = 1.0/6.0;
    int edge_point_count = N + 2; 
    double u_old;
    double t_norm; 

    // k = m 
    // kmax = iter
    // threshold = tolerance
    // d = norm_result
    
    int i, j, k; 

    omp_set_num_threads(8);



    while ( m < iter_max && norm_result > tolerance ) {
        norm_result = 0.0;

        #pragma omp parallel shared(u,norm_result,f,inv,d_squared,edge_point_count) private(i,j,k,u_old) \
        reduction(+: t_norm)
        
        t_norm = 0.0; 
        #pragma omp for ordered(2)
        for (i = 1; i < edge_point_count - 1; i++) {
            for (j = 1; j < edge_point_count - 1; j++) {
                #pragma omp ordered depend(sink:i-1,j) depend(sink:i,j-1)
                for (k = 1; k < edge_point_count - 1; k++) {
                    
                    u_old = u[i][j][k];
                    u[i][j][k] = inv * (u[i-1][j][k] + u[i+1][j][k] + u[i][j-1][k] + u[i][j+1][k] + u[i][j][k-1] + u[i][j][k+1] + d_squared * f[i][j][k]);

                    t_norm += (((u[i][j][k]) - (u_old))*((u[i][j][k]) - (u_old)));
                    
                }
                #pragma omp ordered depend(source)
            }
        }

        #pragma omp critical 
        norm_result = norm_result + t_norm; 
        
        norm_result = sqrt(norm_result); 
        m++;
    
    }

    return norm_result;
    *mp = m;
    return norm_result; 
}

*/


/*
double gauss_seidelOMP(double*** f, double*** u, int N, double tolerance, int iter_max){

    double norm_result = tolerance + 0.01 ;
    int m = 0;
    double delta= (double)(2.0/((double)(N+1)));
    double d_squared = delta*delta;
    double inv = 1.0/6.0;
    int edge_point_count = N + 2; 
    double u_old;
    double t_norm; 

    // k = m 
    // kmax = iter
    // threshold = tolerance
    // d = norm_result
    
    int i, j, k; 

    omp_set_num_threads(8);



    while ( m < iter_max && norm_result > tolerance ) {
        norm_result = 0.0;

        #pragma omp parallel shared(u,norm_result,f,inv,d_squared,edge_point_count) private(i,j,k,u_old) \
        reduction(+: t_norm)
        t_norm = 0.0; 
        #pragma omp for ordered(3)
        for (i = 1; i < edge_point_count - 1; i++) {
            for (j = 1; j < edge_point_count - 1; j++) {
                for (k = 1; k < edge_point_count - 1; k++) {
                    #pragma omp ordered depend(sink:i-1,j,k) depend(sink:i,j-1,k) depend(sink:i,j,k-1)
                    u_old = u[i][j][k];
                    u[i][j][k] = inv * (u[i-1][j][k] + u[i+1][j][k] + u[i][j-1][k] + u[i][j+1][k] + u[i][j][k-1] + u[i][j][k+1] + d_squared * f[i][j][k]);

                    t_norm += (((u[i][j][k]) - (u_old))*((u[i][j][k]) - (u_old)));
                    #pragma omp ordered depend(source)
                }
                
            }
        }

        #pragma omp critical 
        norm_result = norm_result + t_norm; 
        
        norm_result = sqrt(norm_result); 
        m++;
    
    }

    return norm_result;
}
*/

/*


double gauss_seidelOMP(double*** f, double*** u, int N, double tolerance, int iter_max){

    double norm_result = tolerance + 0.01 ;
    int m = 0;
    double delta= (double)(2.0/((double)(N+1)));
    double d_squared = delta*delta;
    double inv = 1.0/6.0;
    int edge_point_count = N + 2; 
    double u_old;

    // k = m 
    // kmax = iter
    // threshold = tolerance
    // d = norm_result
    
    int i, j, k; 


    while ( m < iter_max && norm_result > tolerance ) {
        norm_result = 0.0;

        #pragma omp parallel for ordered(3) private(i,j,k) \
        reduction(+: norm_result)
        for (i = 1; i < edge_point_count - 1; i++) {
            for (j = 1; j < edge_point_count - 1; j++) {
                for (k = 1; k < edge_point_count - 1; k++) {
                    #pragma omp ordered depend(sink:i-1,j,k) depend(sink:i,j-1,k) depend(sink:i,j,k-1)
                    u_old = u[i][j][k];
                    u[i][j][k] = inv * (u[i-1][j][k] + u[i+1][j][k] + u[i][j-1][k] + u[i][j+1][k] + u[i][j][k-1] + u[i][j][k+1] + d_squared * f[i][j][k]);

                    norm_result += (((u[i][j][k]) - (u_old))*((u[i][j][k]) - (u_old)));
                    #pragma omp ordered depend(source)
                }
                
            }
        }

        norm_result = sqrt(norm_result); 
        m++;
    
    }

    return norm_result;
}
*/
