#ifndef _JACOBI_GPU_H
#define _JACOBI_GPU_H

// includes CUDA Runtime
#include <cuda_runtime.h>

__global__ 
void jacobi_kernel1(double*** d_f,double*** d_u,double *** d_u_next,int N,double d_squared,double inv);
void jacobi_gpu_wrap1(double*** d_f,double*** d_u,double *** d_u_next,int N,double tolerance,int iter_max,int * mp); 


__global__ 
void jacobi_kernel2(double*** d_f,double*** d_u,double *** d_u_next,int N,double d_squared,double inv);
void jacobi_gpu_wrap2(double*** d_f,double*** d_u,double *** d_u_next,int N,double tolerance,int iter_max,int * mp); 

__global__ 
void jacobi_kernel3(double*** d_f,double*** d_u,double *** d_u_next,int N,double d_squared,double inv);
void jacobi_gpu_wrap3(double*** d_f,double*** d_u,double *** d_u_next,int N,double tolerance,int iter_max,int * mp); 


__global__ 
void jacobi_kernel4(double*** d_f,double*** d_u,double *** d_u_next,int N,double d_squared,double inv);
void jacobi_gpu_wrap4(double*** d_f,double*** d_u,double *** d_u_next,int N,double tolerance,int iter_max,int * mp); 


#endif