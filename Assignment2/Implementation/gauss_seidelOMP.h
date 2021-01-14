/* gauss_seidel.h - Poisson problem
 *
 */
#ifndef _GAUSS_SEIDEL_H
#define _GAUSS_SEIDEL_H

// define your function prototype here
double gauss_seidelOMP(double*** f, double*** u, int N, double tolerance, int iter_max);

#endif
