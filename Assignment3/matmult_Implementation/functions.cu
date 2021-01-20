#include "cublas_v2.h" // for cublas_dgemm
#include <stdio.h>     // for in-out purposes
#include <stdlib.h>    // for memory purposes
extern "C" {

    #include "cblas.h"     // for cblas_dgemm Prototype
    #include <omp.h>
    
    __global__ void matmult_gpu2_kernel(int m,int n,int k,double *A,double *B,double *C){

        int j = blockIdx.x * blockDim.x + threadIdx.x;
        int i = blockIdx.y * blockDim.y + threadIdx.y;

        double sum =0.0;

        if (i < m && j < n){
            for (int l = 0; l < k; l++) {
                sum += A[i * k +l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }   
    }    

    void matmult_gpu2(int m,int n,int k,double *A,double *B,double *C){
        
        double *d_A;
        double *d_B;
        double *d_C;

        int dimA = m * k * sizeof(double);
        int dimB = k * n * sizeof(double);
        int dimC = m * n * sizeof(double);

        cudaMalloc((void **)&d_A, dimA);
        cudaMalloc((void **)&d_B, dimB);
        cudaMalloc((void **)&d_C, dimC);

        cudaMemcpy(d_A, A, dimA, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, dimB, cudaMemcpyHostToDevice);

        int numThreads = 32;

        dim3 threadsPerBlock = dim3(numThreads,numThreads);
        dim3 blocks = dim3(n/numThreads+1, m/numThreads+1);

        double GPUstart = omp_get_wtime();

        matmult_gpu2_kernel<<<blocks,threadsPerBlock>>>(m, n, k, d_A, d_B, d_C);

        cudaDeviceSynchronize();

        cudaMemcpy(C, d_C, dimC, cudaMemcpyDeviceToHost);

        cudaFree(d_A); 
        cudaFree(d_B);
        cudaFree(d_C);
    }

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
                            n, d_A, k, &beta, d_C, n);

        /* Read the result back */
        cudaMemcpy(C, d_C, dimC, cudaMemcpyDeviceToHost);

        cudaFree(d_A); 
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);

    }


    /* matmult_lib calls cblas_dgemm from cblas library */
    void matmult_lib(int m,int n,int k,double *A,double *B,double *C) {

        double alpha, beta;
        alpha = 1.0; beta = 0.0;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k, B, n, beta, C, n);
        
    }
   
    
} // end extern "C"    