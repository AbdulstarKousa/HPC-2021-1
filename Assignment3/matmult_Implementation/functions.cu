extern "C" {

    #include <stdio.h>     // for in-out purposes
    #include <stdlib.h>    // for memory purposes
    #include "cblas.h"     // for cblas_dgemm Prototype

    /* matmult_lib calls cblas_dgemm from cblas library */
    void matmult_lib(int m,int n,int k,double *A,double *B,double *C) {
        double alpha, beta;
        alpha = 1.0; beta = 0.0;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k, B, n, beta, C, n);
        
        }


    void matmult_gpulib(int m,int n,int k,double *A,double *B,double *C){
        
    }
    
    
} // end extern "C"    