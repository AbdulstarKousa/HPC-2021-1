#!/bin/bash
# 02614 - High-Performance Computing, January 2018
# 
# batch script to run matmult on a decidated server in the hpcintro
# queue
#
# Author: Bernd Dammann <bd@cc.dtu.dk>
#
#BSUB -J mm_batch
#BSUB -o ../jobfiles/mm_batch_%J.out
#BSUB -q hpcintrogpu
#BSUB -n 16
#BSUB -R "rusage[mem=2048]"
#BSUB -R "span[hosts=1]"
#BSUB -W 20
#BSUB -gpu "num=1:mode=exclusive_process"

module load cuda/11.1
module load gcc/9.2.0
numactl --cpunodebind=1

EXECUTABLE=matmult_f.nvcc
# SIZES="2000"
# SIZES="16 32 64 128 256 512 1024"
# SIZES="2048 4096 8192"
SIZES="16 32 64 128 256 512 1024 2048 4096 8192"

PERMUTATIONS="gpu2"
# PERMUTATIONS="lib gpu1 gpu2 gpu3 gpu4 gpu5 gpulib"

for P in $PERMUTATIONS
do
	for S in $SIZES
	do
		LOGEXT=../matmult_Results/datmatmult_time_${P}.dat
		MFLOPS_MIN_T=3 MFLOPS_MAX_IT=3 ./$EXECUTABLE $P $S $S $S |& grep -v CPU >> $LOGEXT
		echo permutation: $P size $S |  grep -v CPU >>$LOGEXT
	done
done
# MFLOPS_MIN_T=3 MFLOPS_MAX_IT=3 .