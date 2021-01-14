#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "jacobi.h"
#include "alloc3d.h"



void collector(double *** f, double *** u, double *** u_next, int N, int iter_max, double tolerance, double start_T){

    //Start_T, initial guess for grid points 
    //tolerance, threshold for Frobinius norm 
    //iter_max, max iterations for jocobi loop 

    double grid_space = (double)(2.0/((double)(N+1)));


    //JACOBIAN VERSION 

    //#ifndef _JACOBI

    //START LOOP 
    int kk = 0; 
    //set norm = inf 
    double norm_check = 1000000000.0;  
    //while norm > threshold && k < iter_max 
    printf("Entering jocobi while loop \n");
    double start = omp_get_wtime();
    while (norm_check > tolerance && kk < iter_max){

        //call Jacobi 
        norm_check = jacobi(f, u, u_next, N, grid_space);
        //printf("Norm result from collector: %e\n",norm_check);

        //call Fro Norm function
        //norm_check = wrapper_norm(u, u_next, N); 

        //reset matrices 
        double *** temp = u; 
        u = u_next; 
        u_next = temp; 
        //increment 
        kk += 1; 
    }

    double end = omp_get_wtime();

    printf("Wall time %f \n", (end-start) );
    printf("Norm result from collector: %e\n",norm_check);
    printf("\n");
    printf("grid space: %.3f\n",grid_space);


    //#endif 


    //GAUSS_SEIDEL VERSION 
    #ifndef _GAUSS_SEIDEL

    //copy above, insert gauss_seidel function 

    #endif  

    
    

}