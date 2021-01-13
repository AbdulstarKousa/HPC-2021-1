#include "NORM_FRO.h"

/* norm_fro
Purpose:
  Computes the Frobenius norm of a matrix.

Arguments:
  pA         a pointer to ** double
  m          cube dim. int   

Return value:
  a double with one of the following values:
   - ILLEGAL_INPUT if the input matrix is NULL
   - ILLEGAL_DIMENSION if the matrix dim. is illegal 
   - Frobenius norm if no error occurred
*/


double norm_fro(double *** pA,int dim) {
    if ( pA == NULL) {
        INPUT_ERR;
        return ILLEGAL_INPUT;
    }
    
    if ( dim  < 1) {
        DIMENSION_ERR;
        return ILLEGAL_DIMENSION;
    }


    size_t i,j,k;
    double nrm=0.0;
    
    for (i=0;i<dim; i++) {
        for (j=0;j<dim; j++) {
            for (k=0;k<dim; k++) {
                nrm += (pA[i][j][k])*(pA[i][j][k]);
            }
            
        }
    }
    
    nrm = sqrt(nrm);
    return nrm;
}