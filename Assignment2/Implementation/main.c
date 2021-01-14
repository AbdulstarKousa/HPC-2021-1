/* main.c - Poisson problem in 3D
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "alloc3d.h"
#include "print.h"
#include "sin_test.h"
#include "init.h"
#include "collector.h"

#include "gauss_seidel.h"

#ifdef _JACOBI
#include "jacobi.h"
#endif

#ifdef _GAUSS_SEIDEL
#include "gauss_seidel.h"
#endif

#define N_DEFAULT 100

int
main(int argc, char *argv[]) {

    // printf("Entering main\n");

    int 	N = N_DEFAULT;
    int 	iter_max = 1000;
    double	tolerance;
    double	start_T;
    int		output_type = 0;
    char	*output_prefix = "poisson_res";
    char        *output_ext    = "";
    char	output_filename[FILENAME_MAX];
    double 	***u = NULL;
    double 	***f = NULL;
    double 	***u_next = NULL;


    /* get the paramters from the command line */
    N         = atoi(argv[1]);	// grid size
    iter_max  = atoi(argv[2]);  // max. no. of iterations
    tolerance = atof(argv[3]);  // tolerance
    start_T   = atof(argv[4]);  // start T for all inner grid points
    if (argc == 6) {
	output_type = atoi(argv[5]);  // ouput type
    }

    // printf("Allocating memory for u\n");

    int N2 = N + 2; 

    // allocate memory
    if ( (u = d_malloc_3d(N2, N2, N2)) == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }
    if ( (u_next = d_malloc_3d(N2, N2, N2)) == NULL ) {
        perror("array u_next: allocation failed");
        exit(-1);
    }
    if ( (f = d_malloc_3d(N2, N2, N2)) == NULL ) {
        perror("array f: allocation failed");
        exit(-1);
    }

    printf("Done allocating memory for u\n");


    // TEST
    //printf("Starting sin test\n");
    //sin_test();
    //printf("Done with sin test\n");

    printf("Running Gauss \n");
    init(f, u, u_next, N, start_T); 
    gauss_seidel(f, u, N, tolerance, iter_max); 
   
    printf("Running Jacobi \n");
    init(f, u, u_next, N, start_T); 
    jacobi(f, u, u_next, N, tolerance, iter_max); 



    // dump  results if wanted 
    switch(output_type) {
	case 0:
	    // no output at all
	    break;
	case 3:
	    output_ext = ".bin";
	    sprintf(output_filename, "%s_%d%s", output_prefix, N+2, output_ext);
	    fprintf(stderr, "Write binary dump to %s: ", output_filename);
	    print_binary(output_filename, N+2, u);
	    break;
	case 4:
	    output_ext = ".vtk";
	    sprintf(output_filename, "%s_%d%s", output_prefix, N+2, output_ext);
	    fprintf(stderr, "Write VTK file to %s: ", output_filename);
	    print_vtk(output_filename, N+2, u);
	    break;
	default:
	    fprintf(stderr, "Non-supported output type!\n");
	    break;
    }

    // de-allocate memory
    free(u);
    free(u_next);
    free(f);

    return(0);
}
