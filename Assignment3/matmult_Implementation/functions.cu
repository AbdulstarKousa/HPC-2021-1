extern "C" {

<<<<<<< HEAD
#include "cublas_v2.h"

void matmult_gpulib(int m,int n,int k,double *A,double *B,double *C){
    
    double *d_A;
    double *d_B;
    double *d_C;

    double alpha = 1.0;
    double beta = 0.0;

    int dimA = k * m*sizeof(double);
    int dimB = k * n*sizeof(double);
    int dimC = n * m*sizeof(double);

    cudaMalloc((void **)&d_A, dimA);
    cudaMalloc((void **)&d_B, dimB);
    cudaMalloc((void **)&d_C, dimC);

    cublasStatus_t status;

    cublasHandle_t handle;

    status = cublasCreate(&handle);
  
    cudaMemcpy(d_A, A, dimA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, dimB, cudaMemcpyHostToDevice);

    /* Performs operation using cublas */
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B,
                         k, d_A, m, &beta, d_C, m);

    /* Read the result back */
    cudaMemcpy(C, d_C, dimC, cudaMemcpyDeviceToHost);

    cudaFree(d_A); 
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
  }
=======
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
    
>>>>>>> 70abe9cc23653bb37dd4facdbb2f5b923f3677ab
    
} // end extern "C"    