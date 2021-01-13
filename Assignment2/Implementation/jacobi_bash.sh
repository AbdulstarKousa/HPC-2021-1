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
#BSUB -q hpcintro
#BSUB -n 1
#BSUB -R "rusage[mem=2048]"
#BSUB -W 20

# define the driver name to use
# valid values: matmult_c.studio, matmult_f.studio, matmult_c.gcc or
# matmult_f.gcc
#
EXECUTABLE=poisson_j
LOGEXT=test_jacobi.dat

# define the mkn values in the MKN variable
#
# SIZES="100 200 500 1000 2000"
##SIZES="10 20 50 100 200 500 1000 2000"

# define the permutation type in PERM
#
#PERMUTATIONS="nat mnk mkn nmk nkm kmn knm lib"
# PERMUTATIONS="lib"

# OPTIMIZATIONLEVELS="-Ofast -O2 -O3"
# Additional flag: mfma

# uncomment and set a reasonable BLKSIZE for the blk version
#
# BLKSIZE=1

# enable(1)/disable(0) result checking
#export MATMULT_COMPARE=1

# start the collect command with the above settings

# LOGEXT=""
# MAKEOPTIONS=""
# for OPT in $OPTIMIZATIONLEVELS
# do
    # No further optimization
    # LOGEXT="matmult_"
    # LOGEXT+="${OPT}"
    # LOGEXT+="_.dat"
    # LOGEXT="$OPT_matmult_.dat"
/bin/rm -f $LOGEXT

    # make clean
    # MAKEOPTIONS="OPT=-g ${OPT} -std=c11"
    # make $MAKEOPTIONS
    # make OPT=-g $OPT -std=c11

./$EXECUTABLE 10 5 5 5 | grep -v CPU >> $LOGEXT
