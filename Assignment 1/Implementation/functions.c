#include <stdlib.h>
#include <stdio.h>


void matmult_nat(int m, int n, int k, double **A, double **B, double **C) {
    double sum = 0;

    for (int i_m = 0; i_m != m; i_m++) {
        for (int i_n = 0; i_n != n; i_n++) {
            sum = 0;
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

}

// #include <stdlib.h>

/* DGEMM (double generel matrix matrix multiplication)                           */
void dgemm_(
    const char * transa,    /* a transposed?        */
    const char * transb,    /* b transposed?        */
    const int * m,          /* rows of a,c          */
    const int * n,          /* columns of b,c       */
    const int * k,          /* columns a, rows c    */
    const double * alpha,   /* scalar alpha         */

    double * a,             /* array a              */
    const int * inca,       /* array a, stride      */

    double * b,             /* array b              */
    const int * incb,       /* array b, stride      */

    double * c,             /* array c              */
    const int * incc        /* array c, stride      */
);

/* Scale the diagonal elements of a square two-dimensional row-major array */
void matmult_lib(int m,int n,int k,double **A,double **B,double **C) {

    // if (k ==0) return -1;
    // if (m ==0) return -1;
    // if (n ==0) return -1;
    // if (A == 0) return -1;
    // if (B == 0) return -1;
    // if (C == 0) return -1;

    // "Unpack" A
    double *dataA;
    dataA = malloc(k*m*sizeof(double));
    if(dataA == NULL){
      free(dataA);
    }
    int N=0;
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < m; j++) {
            dataA[N++] = A[j][i];
        }
    }

    // "Unpack" B
    double *dataB;
    dataB = malloc(n*k*sizeof(double));
    if(dataB == NULL){
      free(dataB);
    }
    N=0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            dataB[N++] = B[j][i];
        }
    }

    // "Unpack" C
    double *dataC;
    dataC = malloc(m*n*sizeof(double));
    if(dataC == NULL){
      free(dataC);
    }
    N=0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            dataC[N++] = C[j][i];
        }
    }

    char trans = 'N';
    double alpha = 1.0;

    dgemm_(&trans, &trans, &m, &n, &k, &alpha, *A, &k, *B, &n, *C, &m);

    free(dataA);
    free(dataB);
    free(dataC);

    // return 0;
}