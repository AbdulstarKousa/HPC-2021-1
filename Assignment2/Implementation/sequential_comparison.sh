#!/bin/bash
# Script running experiments and collecting data to compare sequential implementations of Jacobi and Gauss-Seidel

#BSUB -J sequential_comparison
#BSUB -o sequential_comparison_%J.out
#BSUB -q hpcintro
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2048]"
#BSUB -W 59

JACOBI=poisson_j
GAUSSSEIDEL=poisson_gs

LOG_JACOBI=../Results/sequential_comparison_JAC.dat
LOG_GAUSSSEIDEL=../Results/sequential_comparison_GS.dat

PROBLEM_SIZES="10 30 50 70 100 130 150 170 200"
ITER="100000"
TOLE="0.001"
START_T="0"
IMG="0"  #image disabled -> 0 

for SIZE in $PROBLEM_SIZES
do
    { ./$JACOBI $SIZE $ITER $TOLE $START_T $IMG; } |& grep -v CPU >>$LOG_JACOBI
    { ./$GAUSSSEIDEL $SIZE $ITER $TOLE $START_T $IMG; } |& grep -v CPU >>$LOG_GAUSSSEIDEL
done

echo sizes $PROBLEM_SIZES iterations $ITER tolerance $TOLE initial guess $START_T  |  grep -v CPU >>$LOG_JACOBI

echo size $PROBLEM_SIZES iterations $ITER tolerance $TOLE initial guess $START_T  |  grep -v CPU >>$LOG_GAUSSSEIDEL