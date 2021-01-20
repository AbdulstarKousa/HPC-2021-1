#include <helper_cuda.h>    // checkCudaErrors
#include "cublas_v2.h"      // cublas_dgemm
#include <omp.h>            // parallel, timing , ..etc. 
#include <stdio.h>          // in-out purposes
#include <stdlib.h>         // memory purposes
extern "C" {                // c++ syntax purposes "in matmult_f.nvcc"
#include "cblas.h"          // cblas_dgemm Prototype


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


/*  matmult_lib:
        calls cblas_dgemm from cblas library, the provided driver(matmult_f.nvcc) will link it to a multithreaded version of CBLAS.
    */
void matmult_lib(int m,int n,int k,double *A,double *B,double *C) {

    double alpha, beta;
    alpha = 1.0; beta = 0.0;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k, B, n, beta, C, n);
    
    }



//  Two functions for the first sequential implementation of matrix multiplication on the GPU, useing only a single thread:
//         - matmult_gpu1_kernel:  see the comment attached to the function below.
//         - matmult_gpu1:         see the comment attached to the function below. 
    
/*  matmult_gpu1_kernel:  
        helper function, that takes care of the calculations, for the sequential single threaded matmult_gpu1 function.
*/
__global__ void matmult_gpu1_kernel(int m,int n,int k,double *A,double *B,double *C){
    // Matrices Dimensions: 
        // A: m x k 
        // B: k x n
        // C: m x n
    // Here you don't need to think about the columns major as it's one threaded.
    double sum = 0.0;
    for (int i = 0; i != m; i++) {
        for (int j = 0; j != n; j++) {
            sum = 0.0;
            for (int l = 0; l != k; l++) {
                sum += A[i * k +l] * B[l * n  + j];
                // A[i][l] = A[i * k +l] = A[the_current_row(i) * the_number_of_A_columns(k) + the_current_column(l)] 
            }
            C[i * n + j] = sum;
        }
    }
}

/* matmult_gpu1: 
        sequential single threaded function to solve matrxi-matrix multiplication C=AB on the GPU.
*/
void matmult_gpu1(int m,int n,int k,double *A,double *B,double *C){

    // Allocate host memory (here we don't need to allocate host memory as it's already givin as arguments)
    // But as an example bolw is how to Allocate host memory: 
    // double *h_A, *h_B, *h_C;
    // cudaMallocHost((void**)&h_A, m*k*sizeof(double));
    // cudaMallocHost((void**)&h_B, k*n*sizeof(double));
    // cudaMallocHost((void**)&h_C, m*n*sizeof(double));
    
    // Allocate device memory
    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m*k*sizeof(double));
    cudaMalloc((void**)&d_B, k*n*sizeof(double));
    cudaMalloc((void**)&d_C, m*n*sizeof(double));        

    // Transfer data from host to device memory
    cudaMemcpy(d_A, A, m*k*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k*n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, m*n*sizeof(double), cudaMemcpyHostToDevice);

    // Executing kernel 
    matmult_gpu1_kernel<<<1,1>>>(m,n,k,d_A,d_B,d_C); //single threaded (1 block, 1 thread per block)
    checkCudaErrors(cudaDeviceSynchronize());

    // Transfer data back to host memory
    cudaMemcpy(C, d_C, m*n*sizeof(double), cudaMemcpyDeviceToHost);

    // Deallocate device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Deallocate host memory (here we don't need to Deallocate host memory as it was givin as arguments)
    // but as an example bolw is how to Deallocate host memory:
    // cudaFreeHost(h_A);
    // cudaFreeHost(h_B);
    // cudaFreeHost(h_C);        
}



} // end extern "C"    