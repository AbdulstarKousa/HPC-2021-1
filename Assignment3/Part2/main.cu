/* main.c - Poisson problem in 3D
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime_api.h>
//#include <helper_cuda.h>
#include "alloc3d.h"
#include "alloc3d_gpu.h"
#include "transfer3d_gpu.h"
#include "print.h"
#include "sin_test.h"
#include "init.h"


#ifdef _JACOBI
#include "jacobi.h"
#endif

////#ifdef _JACOBI_OMP
///#include "jacobi_omp.h"
//#endif

//#ifdef _GAUSS_SEIDEL
//#include "gauss_seidel.h"
//#endif

//#ifdef _GAUSS_SEIDEL_OMP
//#include "gauss_seidel_omp.h"
//#endif

#define N_DEFAULT 100

int
main(int argc, char *argv[]) {

    int 	N = N_DEFAULT;
    int 	iter_max = 1000;
    double	tolerance;
    double	start_T;
    int		output_type = 0;
    char	*output_prefix = "poisson_res";
    char    *output_ext    = "";
    char	output_filename[FILENAME_MAX];
    double 	***h_u = NULL;
    double 	***h_f = NULL;
    double 	***h_u_next = NULL;
    double 	***d_u = NULL;
    double 	***d_f = NULL;
    double 	***d_u_next = NULL;
    int m;

    /* get the paramters from the command line */
    N         = atoi(argv[1]);	// grid size
    iter_max  = atoi(argv[2]);  // max. no. of iterations
    tolerance = atof(argv[3]);  // tolerance
    start_T   = atof(argv[4]);  // start T for all inner grid points
    if (argc == 6) {
	output_type = atoi(argv[5]);  // ouput type
    }

    //Allocate memory on HOST
    int N2 = N + 2; 
    printf("Allocating mem_space on CPU\n");

    if ( (h_u = d_malloc_3d(N2, N2, N2)) == NULL ) {
        perror("array h_u: allocation failed");
        exit(-1);
    }
    if ( (h_u_next = d_malloc_3d(N2, N2, N2)) == NULL ) {
        perror("array h_u_next: allocation failed");
        exit(-1);
    }
    if ( (h_f = d_malloc_3d(N2, N2, N2)) == NULL ) {
        perror("array h_f: allocation failed");
        exit(-1);
    }

    //Allocate memory on DEVICE
    printf("Allocating mem_space on GPU\n");

    if ( (d_u = d_malloc_3d_gpu(N2, N2, N2)) == NULL ) {
        perror("array d_u: allocation failed");
        exit(-1);
    }
    if ( (d_u_next = d_malloc_3d_gpu(N2, N2, N2)) == NULL ) {
        perror("array d_u_next: allocation failed");
        exit(-1);
    }
    if ( (d_f = d_malloc_3d_gpu(N2, N2, N2)) == NULL ) {
        perror("array d_f: allocation failed");
        exit(-1);
    }


    //Iniliazie matrices on HOST  
    printf("Iniliazie matrices on HOST\n");
    init(h_f, h_u, h_u_next, N, start_T);  

    //Transfer data to DEVICE 
    printf("Transfer data to DEVICE \n");
    transfer_3d(d_u, h_u, N2, N2, N2, cudaMemcpyHostToDevice); 
    transfer_3d(d_u_next, h_u_next, N2, N2, N2, cudaMemcpyHostToDevice); 
    transfer_3d(d_f, h_f, N2, N2, N2, cudaMemcpyHostToDevice); 


/*
    //Iniliazie matrices 
    #ifdef _SIN_TEST
    printf("Running sin_test \n");
    sin_init(f, u, u_next, N);
    #else
    init(f, u, u_next, N, start_T);
    #endif
    


    #ifdef _JACOBI
    printf("Running jacobi sequential\n");
    double start = omp_get_wtime();
    double norm_check = jacobi(f, u, u_next, N, tolerance, iter_max, &m); 
    double end = omp_get_wtime();
    printf("Wall time %f \n", (end-start) );
    //printf("Number of iterations run: %d \n", p);
    printf("Norm result from norm %e\n",norm_check);
    printf("#Nr. iterations= %d\n",m);
    #endif
*/




/*
    #ifdef _JACOBI_OMP
    printf("Running Jacobi OMP\n");
    double start = omp_get_wtime();
    double norm_check = jacobiOMP(f, u, u_next, N, tolerance, iter_max, &m); 
    double end = omp_get_wtime();
    printf("Wall time %f \n", (end-start) );
    //printf("Number of iterations run: %d \n", p);
    printf("Norm result from norm %e\n",norm_check);
    printf("#Nr. iterations= %d\n",m);
    #endif

    #ifdef _GAUSS_SEIDEL
    printf("Running Gauss sequential\n");
    double start = omp_get_wtime();
    double norm_check = gauss_seidel(f, u, N, tolerance, iter_max, &m); 
    double end = omp_get_wtime();
    printf("Wall time %f \n", (end-start) );
    //printf("Number of iterations run: %d \n", p);
    printf("Norm result from norm %e\n",norm_check);
    printf("#Nr. iterations= %d\n",m);
    #endif

    #ifdef _GAUSS_SEIDEL_OMP
    printf("Running Gauss OMP \n");
    double start = omp_get_wtime();
    double norm_check = gauss_seidelOMP(f, u, N, tolerance, iter_max, &m); 
    double end = omp_get_wtime();
    printf("Wall time %f \n", (end-start) );
    //printf("Number of iterations run: %d \n", p);
    printf("Norm result from norm %e\n",norm_check);
    printf("#Nr. iterations= %d\n",m);
    #endif

    #ifdef _SIN_TEST
    printf("Running sin_check \n");
    printf("Results of sin_check = %lf\n",sin_check(u, N));
    #endif  
*/

    // dump  results if wanted 
    switch(output_type) {
	case 0:
	    // no output at all
	    break;
	case 3:
	    output_ext = ".bin";
	    sprintf(output_filename, "%s_%d%s", output_prefix, N+2, output_ext);
	    fprintf(stderr, "Write binary dump to %s: ", output_filename);
	    print_binary(output_filename, N+2, h_u);
	    break;
	case 4:
	    output_ext = ".vtk";
	    sprintf(output_filename, "%s_%d%s", output_prefix, N+2, output_ext);
	    fprintf(stderr, "Write VTK file to %s: ", output_filename);
	    print_vtk(output_filename, N+2, h_u);
	    break;
	default:
	    fprintf(stderr, "Non-supported output type!\n");
	    break;
    }

    // de-allocate memory
    free(h_u);
    free(h_u_next);
    free(h_f);
    free_gpu(d_u);
    free_gpu(d_u_next);
    free_gpu(d_f);

    return(0);
}