#include <math.h>

// Function for initializing f, u, and u_next according to given initial conditions
void init(double*** f, double*** u, double*** u_next, int N, double start_T) {
    // f: Cube of function values -> Second derivatives of temperature
    // u: Cube of temperature estimates of previous iteration
    // u_next: Cube to hold new temperature estimates
    // edge_point_count: Number of points along an axis -> N 

    double edge_width = 2.0;  // edge_width: Physical width of an edge (-1..1 = 2)
    int min_point = 0;
    int edge_point_count = N; 
    int max_point = edge_point_count - 1;
    

    // Initialize u to 0 degrees everywhere except for the walls
    for (int x = 0; x < edge_point_count; x++) {
        for (int y = 0; y < edge_point_count; y++) {
            for (int z = 0; z < edge_point_count; z++) {
                // TODO: Not too sure about these boundary conditions
                // Initializing the 5 walls to 20 degrees. Every other point becomes 0 degrees
                if(
                   y == max_point || x == max_point || x == min_point || z == min_point || z == max_point 
                ) {
                    u[x][y][z] = 20.0;
                } else {
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

    // Initialize f (based on radiator)
    int x_min = 0;
    int x_max = (int)floor((-3.0/8.0 + 1.0)/edge_width * ((double)max_point)); // -1..1 = 0..2 => -1..-3/8 = 0..-3/8 + 8/8 = 0..5/8
    int y_min = 0;
    int y_max = (int)floor((-1.0/2.0 + 1.0)/edge_width * ((double)max_point));
    int z_min = (int)ceil((-2.0/3.0 + 1.0)/edge_width * ((double)max_point));
    int z_max = max_point;

    for (int x = 0; x < edge_point_count; x++) {
        for (int y = 0; y < edge_point_count; y++) {
            for (int z = 0; z < edge_point_count; z++) {
                f[x][y][z] = start_T;
                if (x_min <= x && x <= x_max) {
                    if (y_min <= y && y <= y_max) {
                        if (z <= z_min && z <= z_max) {
                            f[x][y][z] = 200.0;
                        }
                    }
                }
            }
        }
    }

}