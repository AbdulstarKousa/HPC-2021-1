#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "jacobi.h"
#include "alloc3d.h"
#include "Norm_Fro.h"
#include "init.h"


void collector(double *** f, double *** u, double *** u_next, int N, int iter_max, double tolerance, double start_T){

    //Start_T, initial guess for grid points 
    //tolerance, threshold for Frobinius norm 
    //iter_max, max iterations for jocobi loop 

    double grid_space = 2.0/((double)(N-1));


    //JACOBIAN VERSION 

    #ifndef _JACOBI

    //START LOOP 
    int k = 0; 
    //set norm = inf 
    double norm_check = 1000000000000000000;  
    //while norm > threshold && k < iter_max 
    double start = omp_get_wtime();
    while (norm_check > tolerance && k < iter_max){

        //call Jacobi 
        jacobi(f, u, u_next, N, grid_space);

        //reset matrices 
        double *** temp = u; 
        u = u_next; 
        u_next = temp; 
    
        //call Fro Norm function
        norm_check = wrapper_norm(u, u_next, j); 

        //increment 
        k += 1; 
    }
    double end = omp_get_wtime();

    printf("Wall time %f \n", (end-start) );

    #endif 


    //GAUSS_SEIDEL VERSION 
    #ifndef _GAUSS_SEIDEL

    //copy above, insert gauss_seidel function 

    #endif  

    
    

}