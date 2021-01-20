#ifndef _JACOBI_GPU_H
#define _JACOBI_GPU_H

// includes CUDA Runtime
#include <cuda_runtime.h>

__global__ 
void jacobi_kernel(double*** d_f,double*** d_u,double *** d_u_next,int N,double d_squared,double inv);

void jacobi_gpu_wrap(double*** d_f,double*** d_u,double *** d_u_next,int N,double tolerance,int iter_max,int * mp); 

#endif