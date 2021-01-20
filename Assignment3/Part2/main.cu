/* main.c - Poisson problem in 3D
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime_api.h>
#include "alloc3d.h"
#include "alloc3d_gpu.h"
#include "transfer3d_gpu.h"
#include "print.h"
#include "sin_test.h"
#include "init.h"
#include "jacobi_gpu.h"

#ifdef _JACOBI
#include "jacobi.h"
#endif

#define N_DEFAULT 100

void warmUp()
{
    const int device = 0;
        
    // Wake up GPU from power save state.
    printf("Warming up device %i ... ", device); fflush(stdout);
    double time = omp_get_wtime();
    cudaSetDevice(device);           // Set the device to 0 or 1.
    double *dummy_d;
    cudaMalloc((void**)&dummy_d, 0); // We force the creation of context on the
                                         // device by allocating a dummy variable.
    printf("time = %lf seconds\n", omp_get_wtime() - time);
}

int
main(int argc, char *argv[]) {

    int 	N = N_DEFAULT;
    int 	iter_max = 1000;
    double	tolerance;
    double	start_T;
    int		output_type = 0;
    int     jacobi_type = 0; 
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
    jacobi_type = atoi(argv[5]);  // exercise number 
    }
    if (argc == 7) {
    output_type = atoi(argv[6]);  // ouput type
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


    // Call different exercises 
    switch(jacobi_type) {
        case 11:
            {
            //warm up GPU
            warmUp(); 

            //Iniliazie matrices on HOST  
            printf("Iniliazie matrices on HOST\n");
            init(h_f, h_u, h_u_next, N, start_T);  

            double time_t = omp_get_wtime();
            
            //Transfer data to DEVICE 
            printf("Transfer data to DEVICE \n");
            transfer_3d(d_u, h_u, N2, N2, N2, cudaMemcpyHostToDevice); 
            transfer_3d(d_u_next, h_u_next, N2, N2, N2, cudaMemcpyHostToDevice); 
            transfer_3d(d_f, h_f, N2, N2, N2, cudaMemcpyHostToDevice); 


            jacobi_gpu_wrap(d_f,d_u,d_u_next,N,tolerance,iter_max,&m);
            printf("Out of Jabobi\n");
            
            printf("Transfer data back to HOST \n");
            transfer_3d(h_u,d_u, N2, N2, N2, cudaMemcpyDeviceToHost); 
            //transfer_3d( h_u_next,d_u_next, N2, N2, N2, cudaMemcpyDeviceToHost); 
            //transfer_3d( h_f, d_f,N2, N2, N2, cudaMemcpyDeviceToHost); 
            
            printf("total time = %lf seconds, with N=%d and %d iterations \n", (omp_get_wtime() - time_t),N,iter_max);
            
            break;
            }
        
        case 12: 
            {
            //Initialize matrices
            init(h_f, h_u, h_u_next, N, start_T);  

            //Call reference jacobi 
            printf("Calling reference jacobi\n");
            double time_t1 = omp_get_wtime();

            jacobi_no_norm(h_f,h_u,h_u_next,N,tolerance,iter_max,&m);

            printf("total time = %lf seconds, with N=%d and %d iterations \n", (omp_get_wtime() - time_t1),N,iter_max);
            printf("Out of reference jacobi\n");

            break;
            } 
        default:
            fprintf(stderr, "Non-supported output type!\n");
            break;
        }








    /*
    int i,j,k; 
    for (i = 0; i < N+2; i++) {
        for (j = 0; j < N+2; j++) {
            for (k = 0; k < N+2; k++) { 
                printf("%lf ",h_u[i][j][k]);
            }
        }
        printf("\n");
    }
    */
    

/*
    //Iniliazie matrices 
    #ifdef _SIN_TEST
    printf("Running sin_test \n");
    sin_init(f, u, u_next, N);
    #else
    init(f, u, u_next, N, start_T);
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