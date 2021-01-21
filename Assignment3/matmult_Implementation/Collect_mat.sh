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

# LOGEXT=../matmult_Results/GPUandCPUInfowe.dat

EXECUTABLE=matmult_f.nvcc
# SIZES="2000"
S="10 20 50 100 200 500 1000 2000 5000 10000"
P="gpu5"
# PERMUTATIONS="lib gpu1 gpu2 gpu3 gpu4 gpulib"

LOGEXT=../matmult_Results/datmatmult_${P}.dat
./$EXECUTABLE $P $S $S $S |& grep -v CPU >> $LOGEXT
echo permutation: $P size $S |  grep -v CPU >>$LOGEXT
