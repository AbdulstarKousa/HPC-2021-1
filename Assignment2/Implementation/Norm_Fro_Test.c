#include "NORM_FRO.h"
#include "alloc3d.h"
#include <assert.h>

/* test norm_fro(double *** pA, int dim);*/ 
int main(void) {
    
    /*allcolate a 3x3x3 cube "All elements are set to one except one element is set to 2"*/
    int dim=3;  
    double *** B = d_malloc_3d( dim,  dim,  dim);    
    
    size_t i,j,k;
    for (i=0;i<dim; i++) {
        for (j=0;j<dim; j++) {
            for (k=0;k<dim; k++) {
                B[i][j][k] = 1.0;
            }
        }
    }    
    B[0][0][0]=2;


    
    
    assert(norm_fro(NULL,dim) == ILLEGAL_INPUT);
    assert(norm_fro(B,0) == ILLEGAL_DIMENSION);
    assert(round(norm_fro(B,dim) * 100.0)/100.0 == 5.48);
    printf("norm_fro = %lf \n\n",norm_fro(B,dim));



    /* Deallocates a matrix */
	free(B);       
    return 0;
}