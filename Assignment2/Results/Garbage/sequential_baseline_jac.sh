#!/bin/bash
# Script running experiments and collecting data to compare sequential implementations of Jacobi and Gauss-Seidel

#BSUB -J sequential_baseline_gcc9_jac
#BSUB -o sequential_baseline_gcc9_jac_%J.out
#BSUB -q hpcintro
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2048]"
#BSUB -W 59

JACOBI=poisson_j

LOG_JACOBI=../Results/sequential_baseline_gcc9_JAC.dat

PROBLEM_SIZES="150"
ITER="1000"
TOLE="0.001"
START_T="0"
IMG="0"  #image disabled -> 0 

for SIZE in $PROBLEM_SIZES
do
    { ./$JACOBI $SIZE $ITER $TOLE $START_T $IMG; } |& grep -v CPU >>$LOG_JACOBI
done

echo sizes $PROBLEM_SIZES iterations $ITER tolerance $TOLE initial guess $START_T  |  grep -v CPU >>$LOG_JACOBI