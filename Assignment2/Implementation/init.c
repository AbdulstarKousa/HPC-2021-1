#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Function for initializing f, u, and u_next according to given initial conditions
void init(double*** f, double*** u, double*** u_next, int N, double start_T) {
    // f: Cube of function values -> Second derivatives of temperature
    // u: Cube of temperature estimates of previous iteration
    // u_next: Cube to hold new temperature estimates
    // edge_point_count: Number of points along an axis -> N 

    double edge_width = 2.0;  // edge_width: Physical width of an edge (-1..1 = 2)
    int min_point = 0;
    int edge_point_count = N + 2; 
    int max_point = edge_point_count - 1;
    

    // Initialize u to 0 degrees everywhere except for the walls
    for (int x = 0; x < edge_point_count; x++) {
        for (int y = 0; y < edge_point_count; y++) {
            for (int z = 0; z < edge_point_count; z++) {
                // TODO: Not too sure about these boundary conditions
                // Initializing the 5 walls to 20 degrees. Every other point becomes 0 degrees
                if( y == max_point || x == max_point || x == min_point || z == min_point || z == max_point ) 
                {
                    u[x][y][z] = 20.0;
                }
                else if (y == min_point )
                {
                    u[x][y][z] = 0.0;
                }
                 else {
                    u[x][y][z] = start_T;
                }
            }
        }
    }

    // Initialize u_next to 0.0
    for (int x = 0; x < edge_point_count; x++) {
        for (int y = 0; y < edge_point_count; y++) {
            for (int z = 0; z < edge_point_count; z++) {
                u_next[x][y][z] = start_T;
            }
        }
    }

    int x_min = 0;
    int x_max = (int)(floor(((double)(edge_point_count/2.0))*(5.0/8.0)));
    int y_min = 0;
    int y_max = (int)(floor(((double)(edge_point_count/2.0))*(1.0/2.0)));
    int z_min = (int)(ceil(((double)(edge_point_count/2.0))*(1.0/3.0)));
    int z_max = (int)floor((edge_point_count/2.0));


    //printf("printing: %d \n", (int)(floor(((double)(edge_point_count/2.0))*(1.0/3.0))));
    //printf("printing: [%d %d] [%d %d] [%d %d] \n", x_min,x_max,y_min,y_max,z_min,z_max);

    for (int x = 0; x < edge_point_count; x++) {
        for (int y = 0; y < edge_point_count; y++) {
            for (int z = 0; z < edge_point_count; z++) {
                f[x][y][z] = 0.0;
                if (x_min <= x && x <= x_max) {
                    if (y_min <= y && y <= y_max) {
                        if (z_min <= z && z <= z_max) {
                            f[x][y][z] = 200.0;
                        }
                    }
                }
            }
        }
    }

    /*
    for (int x = 0; x < edge_point_count; x++) {
        for (int y = 0; y < edge_point_count; y++) {
            for (int z = 0; z < edge_point_count; z++) {
                printf("printing f: [%d %d %d] %.3f \n", x,y,z,f[x][y][z]);
            }
        }
    }
    */
    



}