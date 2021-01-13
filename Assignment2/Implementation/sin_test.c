#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "jacobi.h"
#include "alloc3d.h"


void sin_test(){

    int j = 6; //size of the cube 

    int k = j; 
    int m = j; 
    int n = j; 

    double *** f = d_malloc_3d(m, n, k); 
    double *** u = d_malloc_3d(m, n, k); 
    double *** u_next = d_malloc_3d(m, n, k);   
    double *** correct = d_malloc_3d(m, n, k);   

    printf("Initializing matrices\n");

    for (int x = 0; x < j; x++)
    {
        for (int y = 0; y < j; y++)
        {
            for (int z = 0; z < j; z++)
            {
                f[x][y][z] = 3*(M_PI*M_PI)*(sin(M_PI*x)*sin(M_PI*y)*sin(M_PI*z));
                u[x][y][z] = 0.0; 
                u_next[x][y][z] = 0.0; 
                correct[x][y][z] = (sin(M_PI*x)*sin(M_PI*y)*sin(M_PI*z));
            }
            
        }
        
    }

    printf("Initialization finished\n");

    int iter = 20; 
    double grid_s = (double)(2.0/j);

    printf("Entering Jacobi loop\n");
    for (int i = 0; i < iter; i++)
    {

        jacobi(f, u, u_next, j, grid_s);

        double *** temp = u; 
        u = u_next; 
        u_next = temp; 
        
    }

    printf("Done with Jacobi loop\n");


    for (int x = 0; x < j; x++)
    {
        for (int y = 0; y < j; y++)
        {
            for (int z = 0; z < j; z++)
            {
                printf("coorect vs jacobi: %.2f %.2f \n", correct[x][y][z], u[x][y][z]);
            }
            
        }
        
    }

    printf("Done with printing results\n");
    
    
    free(f); 
    free(u);
    free(u_next);
    free(correct);   

    

}