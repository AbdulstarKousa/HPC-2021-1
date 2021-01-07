#!/bin/bash
# 02614 - High-Performance Computing, January 2018
# 
# batch script to run matmult on a decidated server in the hpcintro
# queue
#
# Author: Bernd Dammann <bd@cc.dtu.dk>
#
#BSUB -J mm_batch
#BSUB -o mm_batch_Ofast_funroll_loops_flto_%J.out
#BSUB -q hpcintro
#BSUB -n 1
#BSUB -R "rusage[mem=2048]"
#BSUB -W 59

# define the driver name to use
# valid values: matmult_c.studio, matmult_f.studio, matmult_c.gcc or
# matmult_f.gcc
#
EXECUTABLE=matmult_c.gcc
LOGEXT=matmult_Ofast_funroll_loops_flto.dat

# define the mkn values in the MKN variable
#
SIZES="100 200 500 1000 2000"

# define the permutation type in PERM
#
PERMUTATIONS="nat mnk mkn nmk nkm kmn knm"
# PERMUTATIONS="lib"

# uncomment and set a reasonable BLKSIZE for the blk version
#
# BLKSIZE=1

# enable(1)/disable(0) result checking
export MATMULT_COMPARE=0

# start the collect command with the above settings
/bin/rm -f $LOGEXT
for PERM in $PERMUTATIONS
do
    for S in $SIZES
    do
        ./$EXECUTABLE $PERM $S $S $S $BLKSIZE | grep -v CPU >> $LOGEXT
    done
done


# ./aos.${CC} $LOOPS $particles | grep -v CPU >> aos.$LOGEXT