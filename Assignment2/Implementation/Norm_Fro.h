#ifndef NORM_FRO
#define NORM_FRO

/* Libraries */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "alloc3d.h"

/* Macros */
#define ILLEGAL_INPUT -1.0 
#define ILLEGAL_DIMENSION -2.0
#define INPUT_ERR fprintf(stderr,"%s: received NULL pointer as input\n",__func__)
#define DIMENSION_ERR fprintf(stderr, "%s: dimension mismatch error\n", __func__)

/* Prototype */
double norm_fro(double *** pA, int dim);
double wrapper_norm(double *** m1, double *** m2, int dim);

#endif
