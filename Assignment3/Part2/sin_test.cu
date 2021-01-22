#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "alloc3d.h"
#include <cuda_runtime_api.h>
#include <helper_cuda.h>


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

