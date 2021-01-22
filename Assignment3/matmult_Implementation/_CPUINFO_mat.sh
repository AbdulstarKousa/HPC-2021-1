#!/bin/bash
# 02614 - High-Performance Computing, January 2018
# 
# batch script to run matmult on a decidated server in the hpcintro
# queue
#
# Author: Bernd Dammann <bd@cc.dtu.dk>
#
#BSUB -J mm_batch13
#BSUB -o mm_batch_%J.out
#BSUB -q hpcintrogpu
#BSUB -n 12
#BSUB -R "rusage[mem=2048]"
#BSUB -R "span[hosts=1]"
#BSUB -W 20
#BSUB -gpu "num=1:mode=exclusive_process"

LOGEXT=../matmult_Results/GPUandCPUInfo.dat

numactl --cpunodebind=0
lscpu |& grep -v BLA >> $LOGEXT
/appl/cuda/11.1/samples/bin/x86_64/linux/release/deviceQuery |& grep -v BLA >> $LOGEXT