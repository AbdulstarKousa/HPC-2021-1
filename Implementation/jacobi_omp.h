/* jacobi.h - Poisson problem 
 *
 * $Id: jacobi.h,v 1.1 2006/09/28 10:12:58 bd Exp bd $
 */

#ifndef _JACOBI_H
#define _JACOBI_H

// int jacobi(double ***, double ***, double ***, int, int, double *);
double jacobiOMP(double*** f, double*** u, double *** u_next, int N, double tolerance, int iter_max, int * m);

#endif
