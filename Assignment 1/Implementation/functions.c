#include <stdio.h>     // for in-out purposes  
#include <stdlib.h>    // for memory purposes 
#include "cblas.h"     // for cblas_dgemm Prototype 


void matmult_nat(int m, int n, int k, double **A, double **B, double **C) {
    double sum = 0.0;

    for (int i_m = 0; i_m != m; i_m++) {
        for (int i_n = 0; i_n != n; i_n++) {
            sum = 0.0;
            for (int i_k = 0; i_k != k; i_k++) {
                sum += A[i_m][i_k] * B[i_k][i_n];
            }
            C[i_m][i_n] = sum;
        }
    }
}


// void matmult_mnk_fast(int m, int n, int k, double **A, double **B, double **C) {
//     double sum = 0;

//     for (int i_m = 0; i_m != m; i_m++) {
//         for (int i_n = 0; i_n != n; i_n++) {
//             sum = 0;
//             for (int i_k = 0; i_k != k; i_k++) {
//                 // C[i_m][i_n] += A[i_m][i_k] * B[i_k][i_n];
//                 sum += A[i_m][i_k] * B[i_k][i_n];
//             }
//             C[i_m][i_n] = sum;
//         }
//     }
// }

void matmult_mnk(int m, int n, int k, double **A, double **B, double **C) {
    // Reset C Matrix before storing result
    for (int i_m = 0; i_m < m; i_m++) {
        for (int i_n = 0; i_n < n; i_n++) {
            C[i_m][i_n] = 0.0;
        }
    }

    for (int i_m = 0; i_m != m; i_m++) {
        for (int i_n = 0; i_n != n; i_n++) {
            // sum = 0;
            for (int i_k = 0; i_k != k; i_k++) {
                // C[i_m][i_n] += A[i_m][i_k] * B[i_k][i_n];
                // sum += A[i_m][i_k] * B[i_k][i_n];
                C[i_m][i_n] += A[i_m][i_k] * B[i_k][i_n];
            }
            // C[i_m][i_n] = sum;
        }
    }
}


void matmult_mkn(int m, int n, int k, double **A, double **B, double **C) {
    // printf("\n\nC-MATRIX VALUE(1,1): %e\n\n", C[1][1]);

    for (int i_m = 0; i_m < m; i_m++) {
        for (int i_n = 0; i_n < n; i_n++) {
            C[i_m][i_n] = 0.0;
        }
    }

    // printf("\n\nC-MATRIX VALUE(0,0) top: %e\n\n", C[0][0]);

    for (int i_m = 0; i_m != m; i_m++) {
        for (int i_k = 0; i_k != k; i_k++) {
            for (int i_n = 0; i_n != n; i_n++) {
                C[i_m][i_n] += A[i_m][i_k] * B[i_k][i_n];
            }
        }
    }

    // printf("\n\nC-MATRIX VALUE(0,0) bottom: %e\n\n", C[0][0]);
}


// void matmult_nmk_fast(int m, int n, int k, double **A, double **B, double **C) {
//     double sum = 0;
//     for (int i_n = 0; i_n != n; i_n++) {
//         for (int i_m = 0; i_m != m; i_m++) {
//             sum = 0;
//             for (int i_k = 0; i_k != k; i_k++) {
//                 sum += A[i_m][i_k] * B[i_k][i_n];
//             }
//             C[i_m][i_n] = sum;
//         }
//     }
// }

void matmult_nmk(int m, int n, int k, double **A, double **B, double **C) {
    for (int i_m = 0; i_m < m; i_m++) {
        for (int i_n = 0; i_n < n; i_n++) {
            C[i_m][i_n] = 0.0;
        }
    }


    for (int i_n = 0; i_n != n; i_n++) {
        for (int i_m = 0; i_m != m; i_m++) {
            // sum = 0;
            for (int i_k = 0; i_k != k; i_k++) {
                // sum += A[i_m][i_k] * B[i_k][i_n];
                C[i_m][i_n] += A[i_m][i_k] * B[i_k][i_n];
            }
            // C[i_m][i_n] = sum;
        }
    }
}


void matmult_nkm(int m, int n, int k, double **A, double **B, double **C) {
    for (int i_m = 0; i_m < m; i_m++) {
        for (int i_n = 0; i_n < n; i_n++) {
            C[i_m][i_n] = 0.0;
        }
    }

    for (int i_n = 0; i_n != n; i_n++) {
        for (int i_k = 0; i_k != k; i_k++) {
            for (int i_m = 0; i_m != m; i_m++) {
                C[i_m][i_n] += A[i_m][i_k] * B[i_k][i_n];
            }
        }
    }
}


void matmult_kmn(int m, int n, int k, double **A, double **B, double **C) {
    for (int i_m = 0; i_m < m; i_m++) {
        for (int i_n = 0; i_n < n; i_n++) {
            C[i_m][i_n] = 0.0;
        }
    }

    for (int i_k = 0; i_k != k; i_k++) {
        for (int i_m = 0; i_m != m; i_m++) {
            for (int i_n = 0; i_n != n; i_n++) {
                C[i_m][i_n] += A[i_m][i_k] * B[i_k][i_n];
            }
        }
    }
}


void matmult_knm(int m, int n, int k, double **A, double **B, double **C) {
    for (int i_m = 0; i_m < m; i_m++) {
        for (int i_n = 0; i_n < n; i_n++) {
            C[i_m][i_n] = 0.0;
        }
    }

    for (int i_k = 0; i_k != k; i_k++) {
        for (int i_n = 0; i_n != n; i_n++) {
            for (int i_m = 0; i_m != m; i_m++) {
                C[i_m][i_n] += A[i_m][i_k] * B[i_k][i_n];
            }
        }
    }
}



void matmult_blk(int m,int n,int k,double **A,double **B,double **C, int bs) {
    for (int i_m = 0; i_m < m; i_m++) {
        for (int i_n = 0; i_n < n; i_n++) {
            C[i_m][i_n] = 0.0;
        }
    }

    // OBS, the block size needs to be a multiple of m, n and k in order for this to work!
    // ie. m, n and k also need to be the same 
    int block = bs * (n / bs); //Defining the block size 

    for (int i_kk = 0; i_kk < block; i_kk += block){
        for (int i_nn = 0; i_nn < block; i_nn += block){
            for (int i_m = 0; i_m != m; i_m++) {
                for (int i_n = i_nn; i_n < i_nn + block ; i_n++) {
                    for (int i_k = i_kk; i_n < i_kk + block ; i_k++) {
                        C[i_m][i_n] += A[i_m][i_k] * B[i_k][i_n];
                    }
                }
            }
        }
    }
}

// #include <stdlib.h>

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

/* Scale the diagonal elements of a square two-dimensional row-major array */
void matmult_lib(int m,int n,int k,double **A,double **B,double **C) {
  double alpha, beta;
  alpha = 1.0; beta = 0.0;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A[0], k, B[0], n, beta, C[0], n);  

}