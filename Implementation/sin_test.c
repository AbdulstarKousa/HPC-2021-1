#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "alloc3d.h"


void sin_init(double*** f, double*** u, double*** u_next, int N) {
    // f: Cube where function values will be stored
    // u: Cube where current u-estimates will be stored
    // u_next: Cube where next u-estimates will be stored
    // N: Interior point count

    double edge_width = 2.0;
    double grid_space = edge_width/((double)(N + 1));

    double xx = -1.0; 
    double yy = -1.0; 
    double zz = -1.0; 

    double array[N];
    array[0] = -1.0; 
    array[N-1] = 1.0; 
    for(int i = 1; i < N; i++){
        array[i] = array[i-1] + grid_space; 
    }


    for (int x = 0; x < N+2; x++)
    {
        xx = array[x];
        for (int y = 0; y < N+2; y++)
        {
            yy = array[y];
            for (int z = 0; z < N+2; z++)
            {
                zz = array[z];

                f[x][y][z] = 3.0*(M_PI*M_PI)*(sin(M_PI*xx)*sin(M_PI*yy)*sin(M_PI*zz));
                u[x][y][z] = 0.0; 
                u_next[x][y][z] = 0.0; 
            }
        }
    }
}

double sin_check(double*** u, int N) {
    // u: Cube of estimated values
    // N: Interior point count

    // Allocate cube storing expected values
    double*** correct = d_malloc_3d(N+2, N+2, N+2);

    // Setup for filling in expected values
    double xx = -1.0; 
    double yy = -1.0; 
    double zz = -1.0; 

    double grid_space = 2.0/((double)(N+1)); 
    double array[N];

    array[0] = -1.0; 
    array[N-1] = 1.0; 
    for(int i = 1; i < N; i++){
        array[i] = array[i-1] + grid_space; 
    }

    // Insert expected values
    for (int x = 0; x < N+2; x++)
    {
        xx = array[x];
        for (int y = 0; y < N+2; y++)
        {
            yy = array[y];
            for (int z = 0; z < N+2; z++)
            {
                zz = array[z];

                correct[x][y][z] = (sin(M_PI*xx)*sin(M_PI*yy)*sin(M_PI*zz));
            }
        }
    }

    // Compare values in u-matrix with expected values
    double norm = 0.0;

    for (int x = 0; x < N+2; x++)
    {
        for (int y = 0; y < N+2; y++)
        {
            for (int z = 0; z < N+2; z++)
            {
                norm += (u[x][y][z] - correct[x][y][z]) * (u[x][y][z] - correct[x][y][z]);
            }
        }
    }

    free(correct);

    return sqrt(norm);
}




// #ifndef M_PI
//     #define M_PI 3.14159265358979323846
// #endif

// #include <math.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <omp.h>

// #include "jacobi.h"
// #include "alloc3d.h"

// void sin_test(){
    
//     int N = 50; //size of the cube 

//     int k = N + 2; 
//     int m = N + 2; 
//     int n = N + 2; 

//     double *** f = d_malloc_3d(m, n, k); 
//     double *** u = d_malloc_3d(m, n, k); 
//     double *** u_next = d_malloc_3d(m, n, k);   
//     double *** correct = d_malloc_3d(m, n, k);   

//     // printf("Initializing matrices\n");

//     double xx = -1.0; 
//     double yy = -1.0; 
//     double zz = -1.0; 

//     double grid_space = 2.0/((double)(N+1)); 
//     double array[N];

//     array[0] = -1.0; 
//     array[N-1] = 1.0; 
//     for(int i = 1; i < N; i++){
//         array[i] = array[i-1] + grid_space; 
//     }


//     for (int x = 0; x < N+2; x++)
//     {
//         xx = array[x];
//         for (int y = 0; y < N+2; y++)
//         {
//             yy = array[y];
//             for (int z = 0; z < N+2; z++)
//             {
//                 zz = array[z];

//                 f[x][y][z] = 3.0*(M_PI*M_PI)*(sin(M_PI*xx)*sin(M_PI*yy)*sin(M_PI*zz));
//                 //printf("f val: %.2f \n", f[x][y][z]);
//                 u[x][y][z] = 0.0; 
//                 u_next[x][y][z] = 0.0; 
//                 correct[x][y][z] = (sin(M_PI*xx)*sin(M_PI*yy)*sin(M_PI*zz));

                
//             }
//         }
//     }

//     // printf("Initialization finished\n");

//     int iter_max = 1000; 
//     double tolerance = 1.0e-3; 
//     double grid_s = (double)(2.0/((double)(N+1)));
//     double norm_result = 0.0; 

//     // printf("Entering Jacobi loop\n");
//     double start = omp_get_wtime();

//     norm_result = jacobi(f, u, u_next, N, tolerance, iter_max);

//     double end = omp_get_wtime();

//     //subtract matrices

//     // printf("Done with Jacobi loop\n");


//     // for (int x = 20; x < 40; x++)
//     // {
//     //     for (int y = 20; y < 40; y++)
//     //     {
//     //         for (int z = 20; z < 40; z++)
//     //         {
//     //             printf("correct vs jacobi vs f: [%d %d %d] %.4f %.4f %.1f \n", x,y,z,correct[x][y][z], u[x][y][z], f[x][y][z]);
//     //         }
            
//     //     }
        
//     // }

//     // printf("Done with printing results\n");

//     printf("Wall time %f \n", (end-start) );

//     printf("Done with printing results: %lf \n", norm_result);
    
    
//     free(f); 
//     free(u);
//     free(u_next);
//     free(correct);   

    

// }