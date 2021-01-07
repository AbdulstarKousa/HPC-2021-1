#!/bin/bash
# 02614 - High-Performance Computing, January 2018
# 
# batch script to run matmult on a decidated server in the hpcintro
# queue
#
# Author: Bernd Dammann <bd@cc.dtu.dk>
#
#BSUB -J mm_batch
#BSUB -o mm_batch_%J.out
#BSUB -q hpcintro
#BSUB -n 1
#BSUB -R "rusage[mem=2048]"
#BSUB -W 59

# define the driver name to use
# valid values: matmult_c.studio, matmult_f.studio, matmult_c.gcc or
# matmult_f.gcc
#
EXECUTABLE=matmult_c.gcc
# LOGEXT=matmult_Ofast_funroll_loops_flto.dat

# define the mkn values in the MKN variable
#
# SIZES="100 200 500 1000 2000"
SIZES="10 20 50 100"

# define the permutation type in PERM
#
PERMUTATIONS="nat mnk mkn nmk nkm kmn knm lib"
# PERMUTATIONS="lib"

OPTIMIZATIONLEVELS="-Ofast -O2 -O3"

# uncomment and set a reasonable BLKSIZE for the blk version
#
# BLKSIZE=1

# enable(1)/disable(0) result checking
export MATMULT_COMPARE=0

# start the collect command with the above settings

LOGEXT=""

for OPT in $OPTIMIZATIONLEVELS
do

    # No further optimization
    LOGEXT="matmult_"
    LOGEXT+="${OPT}"
    LOGEXT+="_.dat"
    # LOGEXT="$OPT_matmult_.dat"
    /bin/rm -f $LOGEXT

    make clean
    make OPT=-g $OPT -std=c11

    for PERM in $PERMUTATIONS
    do
        for S in $SIZES
        do
            ./$EXECUTABLE $PERM $S $S $S $BLKSIZE | grep -v CPU >> $LOGEXT
            echo %$SIZES >> $LOGEXT
        done
    done

    # funroll-loops
    # LOGEXT="${OPT}_funroll_loops_matmult_.dat"
    LOGEXT="matmult_"
    LOGEXT+="${OPT}"
    LOGEXT+="_funroll_loops_.dat"
    /bin/rm -f $LOGEXT

    make clean
    make OPT=-g -funroll_loops $OPT -std=c11

    for PERM in $PERMUTATIONS
    do
        for S in $SIZES
        do
            ./$EXECUTABLE $PERM $S $S $S $BLKSIZE | grep -v CPU >> $LOGEXT
            echo %$SIZES >> $LOGEXT
        done
    done

    # flto
    # LOGEXT="${OPT}_flto_matmult_.dat"
    LOGEXT="matmult_"
    LOGEXT+="${OPT}"
    LOGEXT+="_flto_.dat"
    /bin/rm -f $LOGEXT

    make clean
    make OPT=-g -flto $OPT -std=c11

    for PERM in $PERMUTATIONS
    do
        for S in $SIZES
        do
            ./$EXECUTABLE $PERM $S $S $S $BLKSIZE | grep -v CPU >> $LOGEXT
            echo %$SIZES >> $LOGEXT
        done
    done

    # funroll-loops and flto
    # LOGEXT="${OPT}_funroll_loops_flto_matmult_.dat"
    LOGEXT="matmult_"
    LOGEXT+="${OPT}"
    LOGEXT+="_funroll_loops_flto_.dat"
    /bin/rm -f $LOGEXT

    make clean
    make OPT=-g -funroll_loops -flto $OPT -std=c11

    for PERM in $PERMUTATIONS
    do
        for S in $SIZES
        do
            ./$EXECUTABLE $PERM $S $S $S $BLKSIZE | grep -v CPU >> $LOGEXT
            echo %$SIZES >> $LOGEXT
        done
    done


    # /bin/rm -f $LOGEXT

    # for PERM in $PERMUTATIONS
    # do
    #     for S in $SIZES
    #     do
    #         ./$EXECUTABLE $PERM $S $S $S $BLKSIZE | grep -v CPU >> $LOGEXT
    #     done
    # done
done


# ./aos.${CC} $LOOPS $particles | grep -v CPU >> aos.$LOGEXT