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
#BSUB -q hpcintrogpu
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2048]"
#BSUB -W 59
#BSUB -gpu "num=1:mode=exclusive_process"

module load cuda/11.1
module load gcc/9.2.0
numactl --physcpubind=1

EXECUTABLE=poisson_j_omp

THREADS="16"


SIZE_N="30 40 50 60 100 150 200 250 300 400 500"
ITER="100"
TOLE="0.001"
START_T="0.0"
IMG="0"  #image disabled -> 0 



export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_WAIT_POLICY=active

LOGEXT1=Results_ex6/Exercise6_CPU_${ITER}.dat

for T in $THREADS
do
	for S in $SIZE_N
	do
		{ OMP_NUM_THREADS=${T} ./$EXECUTABLE $S $ITER $TOLE $START_T $IMG ; } |& grep -v CPU >>$LOGEXT1
		echo size $S iterations $ITER |  grep -v CPU >>$LOGEXT1
	done
done




