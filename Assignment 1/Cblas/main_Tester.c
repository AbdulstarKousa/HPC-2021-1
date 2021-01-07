#include <stdio.h>     // for in-out purposes  
#include <stdlib.h>    // for memory purposes 
#include "cblas.h"     // for cblas_dgemm Prototype 

#define min(x,y) (((x) < (y)) ? (x) : (y))

// dgemm prototype 
void cblas_dgemm(
  const enum CBLAS_ORDER __Order,      //reserved enum (CblasRowMajor)
  const enum CBLAS_TRANSPOSE __TransA, //reserved enum (CblasNoTrans)
  const enum CBLAS_TRANSPOSE __TransB, //reserved enum (CblasNoTrans)
  const int __M,        // C #Row. 
  const int __N,        // C #Col.
  const int __K,        // A #Col. || B #Row.  
  const double __alpha, // C=alpha*A*B+beta*C => alpha = 1 for C=A*B
  const double *__A,    // A[0]
  const int __lda,      // A #Col.
  const double *__B,    // B[0]
  const int __ldb,      // B #Col.
  const double __beta,  // C=alpha*A*B+beta*C => beta = 0 for C=A*B
  double *__C,          // C[0]
  const int __ldc);     // C #Col.


int main() {
  int m, n, k;
  double alpha, beta;
  m = 20;k = 20; n = 20;
  alpha = 1.0; beta = 0.0;

  // A (m,k)
  double **A = malloc(m * sizeof(double *));
  if (A == NULL) {return -1;}
  A[0] = malloc(m*k*sizeof(double));
  if (A[0] == NULL) {free(A); return -1;}
  for (size_t i = 1; i < m; i++)
  A[i] = A[0] + i * k;
  
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
          A[i][j] = (double)rand();
    }
  }




  // B (k,n)
  double **B = malloc(k * sizeof(double *));
  if (B == NULL) {return -1;}
  B[0] = malloc(k*n*sizeof(double));
  if (B[0] == NULL) {free(B); return -1;}
  for (size_t i = 1; i < k; i++)
  B[i] = B[0] + i * n;

  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n; j++) {
          B[i][j] = (double)rand();
    }
  }


  // C (m,n)  
  double **C = malloc(m * sizeof(double *));
  if (C == NULL) {return -1;}
  C[0] = malloc(m*n*sizeof(double));
  if (C[0] == NULL) {free(C); return -1;}
  for (size_t i = 1; i < m; i++)
  C[i] = C[0] + i * n;

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
          C[i][j] = 0.0;
    }
  }


  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A[0], k, B[0], n, beta, C[0], n);  

  printf ("\n Computations completed.\n\n");
  
  printf ("\n Top left corner of matrix C: \n");
  for (int i=0; i<min(m,6); i++) {
    for (int j=0; j<min(n,6); j++) {
      printf ("%12.5G", C[i][j]);
    }
    printf ("\n");
  }


  // Deallocating memory
  free(A);
  free(B);
  free(C);  
  return 0;
}
