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
void jacobi_kernel31(double*** d0_f,double*** d0_u,double*** d1_u, double *** d0_u_next,int N,double d_squared,double inv, int Nh);
void jacobi_kernel32(double*** d1_f,double*** d1_u,double*** d0_u,double *** d1_u_next,int N,double d_squared,double inv, int Nh);
void jacobi_gpu_wrap3(double*** d0_f,double*** d0_u,double *** d0_u_next,double*** d1_f,double*** d1_u,double *** d1_u_next,int N,double tolerance,int iter_max,int * mp); 

__global__ 
void jacobi_kernel4new(double*** d_f,double*** d_u,double *** d_u_next,int N,double inv, double d_squares,double * norm);
void jacobi_gpu_wrap4new(double*** d_f,double*** d_u,double *** d_u_next,int N,double tolerance,int iter_max,int * mp); 


#endif