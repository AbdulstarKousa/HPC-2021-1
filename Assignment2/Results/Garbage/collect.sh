#!/bin/bash
# 02614 - High-Performance Computing, January 2018
#
# batch script to run collect on a decidated server in the hpcintro
# queue
#
# Author: Bernd Dammann <bd@cc.dtu.dk>
#BSUB -J opmJ_batch
#BSUB -o opmJ_batch_%J.out
#BSUB -q hpcintro
#BSUB -n 24
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2048]"
#BSUB -W 15

module load studio

EXECUTABLE=poisson_gs_omp

# THREADS="12 8 4 2 1"
THREADS="4"

# SCHEDULE="static static,5 static,10  static,25 dynamic dynamic,5 dynamic,25 guided guided,5"
# SCHEDULE="static static,4 static,8 static,10"
#SCHEDULE="static"

SIZE_N="150"
ITER="1000"
TOLE="0.001"
START_T="0.0"
IMG="0"  #image disabled -> 0
# define the max no. of iterations the driver should use - adjust to
# get a reasonable run time.  You can get an estimate by trying this
# on the command line, i.e. "MFLOPS_MAX_IT=10 ./matmult_...." for the
# problem size you want to analyze.
#
export MFLOPS_MAX_IT=1000
export MATMULT_COMPARE=0
export OMP_NUM_THREADS=${THREADS}
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# experiment name
#
JID=${LSB_JOBID}
EXPOUT="../Results/$LSB_JOBNAME.${JID}_GS_Reduc.er"

# uncomment the HWCOUNT line, if you want to use hardware counters
# define an option string for the harwdware counters (see output of
# 'collect -h' for valid values.  The format is:
# -h cnt1,on,cnt2,on,...  (up to four counters at a time)
#
# the example below is for L1 hits, L1 misses, L2 hits, L2 misses
#
HWCOUNT="-h dch,on,dcm,on,l2h,on,l2m,on"

# start the collect command with the above settings
collect -o $EXPOUT $HWCOUNT ./$EXECUTABLE $SIZE_N $ITER $TOLE $START_T $IMG;
