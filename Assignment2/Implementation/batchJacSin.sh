#!/bin/bash
# 02614 - High-Performance Computing, January 2018
#
# batch script to run matmult on a decidated server in the hpcintro
# queue
#
# Author: Bernd Dammann <bd@cc.dtu.dk>
#
#BSUB -J opmJ_batch
#BSUB -o opmJ_batch_%J.out
#BSUB -q hpcintro
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2048]"
#BSUB -W 15

EXECUTABLE=poisson_j

THREADS="24 16 12 8 4 2 1"

LOGEXT=../Results/datjacobOPMsinTest.dat

for T in $THREADS
do
	{ OMP_NUM_THREADS=$T ./$EXECUTABLE 5 5 5 5; } |& grep -v CPU >>$LOGEXT
	echo $T |  grep -v CPU >>$LOGEXT
done

