#!/bin/bash
# Script running experiments and collecting data to compare sequential implementations of Jacobi and Gauss-Seidel

#BSUB -J sequential_comparison
#BSUB -o sequential_comparison_%J.out
#BSUB -q hpcintro
#BSUB -n 1
#BSUB -R"span[hosts=1]"
#BSUB -R "rusage[mem=2048]"
#BSUB -W 59

JACOBI=poisson_j
GAUSSSEIDEL=poisson_gs

LOG_JACOBI=../Results/sequential_comparison_JAC.dat
LOG_GAUSSSEIDEL=../Results/sequential_comparison_GS.dat

PROBLEM_SIZES="10 30 50 70 100 130 150 170 200"
ITER="100000"
TOLE="0.001"
START_T="0"
IMG="0"  #image disabled -> 0 

for SIZE in $PROBLEM_SIZES
do
    { ./$JACOBI $SIZE $ITER $TOLE $START_T $IMG; } |& grep -v CPU >>$LOG_JACOBI
    { ./$GAUSSSEIDEL $SIZE $ITER $TOLE $START_T $IMG; } |& grep -v CPU >>$LOG_GAUSSSEIDEL
done

# for T in $THREADS
# do
# 	for S in $SCHEDULE
# 	do
# 		{ OMP_SCHEDULE=$S OMP_NUM_THREADS=$T ./$EXECUTABLE $SIZE_N $ITER $TOLE $START_T $IMG; } |& grep -v CPU >>$LOGEXT
# 		echo $T |  grep -v CPU >>$LOGEXT
# 		echo $S |  grep -v CPU >>$LOGEXT
		
# 	done
# done

echo size $SIZE_N iterations $ITER tolerance $TOLE initial guess $START_T  |  grep -v CPU >>$LOG_JACOBI

echo size $SIZE_N iterations $ITER tolerance $TOLE initial guess $START_T  |  grep -v CPU >>$LOG_GAUSSSEIDEL









# #!/bin/bash
# # 02614 - High-Performance Computing, January 2018
# #
# # batch script to run matmult on a decidated server in the hpcintro
# # queue
# #
# # Author: Bernd Dammann <bd@cc.dtu.dk>
# #
# #BSUB -J opmJ_batch
# #BSUB -o opmJ_batch_%J.out
# #BSUB -q hpcintro
# #BSUB -n 12
# #BSUB -R "span[hosts=1]"
# #BSUB -R "rusage[mem=2048]"
# #BSUB -W 15

# EXECUTABLE=poisson_j

# THREADS="12 8 4 2 1"

# #SCHEDULE="static static,5 static,10 dynamic dynamic,5 dynamic,25 guided guided,5"
# SCHEDULE="static,10 dynamic,10"

# LOGEXT=../Results/datjacobOPMsinTest3.dat

# SIZE_N="100"
# ITER="10000"
# TOLE="0.001"
# START_T="0"
# IMG="4"  #image disabled -> 0 

# for T in $THREADS
# do
# 	for S in $SCHEDULE
# 	do
# 		{ OMP_SCHEDULE=$S OMP_NUM_THREADS=$T ./$EXECUTABLE $SIZE_N $ITER $TOLE $START_T $IMG; } |& grep -v CPU >>$LOGEXT
# 		echo $T |  grep -v CPU >>$LOGEXT
# 		echo $S |  grep -v CPU >>$LOGEXT
		
# 	done
# done

# echo size $SIZE_N iterations $ITER tolerance $TOLE initial guess $START_T  |  grep -v CPU >>$LOGEXT


