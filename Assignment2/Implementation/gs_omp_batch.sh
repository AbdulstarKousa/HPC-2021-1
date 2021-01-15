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
#BSUB -n 12
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2048]"
#BSUB -W 15

EXECUTABLE=poisson_gs_omp

THREADS="12 8 4 2 1"

# SCHEDULE="static static,5 static,10 dynamic dynamic,5 dynamic,25 guided guided,5"
LOGEXT=../Results/datgsOMP_1.dat

SIZE_N="100"
ITER="2000"
TOLE="0.001"
START_T="0.0"
IMG="0"  #image disabled -> 0 

for T in $THREADS
do
		{ OMP_NUM_THREADS=$T ./$EXECUTABLE $SIZE_N $ITER $TOLE $START_T $IMG; } |& grep -v CPU >>$LOGEXT  
		echo $T |  grep -v CPU >>$LOGEXT
		echo $S |  grep -v CPU >>$LOGEXT
done

echo size $SIZE_N iterations $ITER tolerance $TOLE initial guess $START_T  |  grep -v CPU >>$LOGEXT