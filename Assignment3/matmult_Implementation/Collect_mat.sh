#!/bin/bash

#BSUB -J proftest
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
export TMPDIR=$__LSF_JOB_TMPDIR__

export MFLOPS_MAX_IT=1 
# LOGEXT=../matmult_Results/GPUandCPUInfowe.dat

EXECUTABLE=matmult_f.nvcc
S="2048"
P="gpulib"

nv-nsight-cu-cli -o profile_${P}_$LSB_JOBID \
    --section MemoryWorkloadAnalysis \
    --section MemoryWorkloadAnalysis_Chart \
    --section ComputeWorkloadAnalysis \
    --section SchedulerStats \
    --section LaunchStats \
    --section SOL \
    ./matmult_f.nvcc $P $S $S $S