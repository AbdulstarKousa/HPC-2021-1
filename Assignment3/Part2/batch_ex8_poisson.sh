
#BSUB -J mm_batch
#BSUB -o ../jobfiles/mm_batch_%J.out
#BSUB -q hpcintrogpu
#BSUB -n 16
#BSUB -R "rusage[mem=2048]"
#BSUB -R "span[hosts=1]"
#BSUB -W 59
#BSUB -gpu "num=1:mode=exclusive_process"

module load cuda/11.1
module load gcc/9.2.0
numactl --cpunodebind=1

EXECUTABLE=poisson_j

SIZE_N="200 230 250"
ITER="1000000"
TOLE="0.001"
START_T="0"
IMG="0"  #image disabled -> 0 
EXERCISE11="41"

LOGEXT0=Results_ex8/Exercise8_GPU_${ITER}.dat
for S in $SIZE_N
do
		{ ./$EXECUTABLE $S $ITER $TOLE $START_T $EXERCISE11; } |& grep -v CPU >>$LOGEXT0
		echo size $S iterations $ITER |  grep -v CPU >>$LOGEXT0
done
