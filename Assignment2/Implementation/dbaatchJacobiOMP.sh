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
#BSUB -n 24
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2048]"
#BSUB -W 15

EXECUTABLE=poisson_j_omp

THREADS="24 16 12 8 4 2 1"
# THREADS="12"

# SCHEDULE="static static,5 static,10  static,25 dynamic dynamic,5 dynamic,25 guided guided,5"
# SCHEDULE="static static,4 static,8 static,10"
SCHEDULE="static"



SIZE_N="500"
ITER="100"
TOLE="0.001"
START_T="0"
IMG="0"  #image disabled -> 0 



export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_WAIT_POLICY=active

for T in $THREADS
do
	for S in $SIZE_N
	do
		LOGEXT=../Results/Jac_Parallel_region_${S}_${ITER}.dat
		{ OMP_NUM_THREADS=${T} ./$EXECUTABLE $S $ITER $TOLE $START_T $IMG; } |& grep -v CPU >>$LOGEXT
		echo threads: $T |  grep -v CPU >>$LOGEXT
		echo size $S iterations $ITER tolerance $TOLE initial guess $START_T  |  grep -v CPU >>$LOGEXT
	done
done




