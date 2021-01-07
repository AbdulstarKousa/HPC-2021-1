#!/bin/bash
#BSUB -J CPU_info
#BSUB -o CPU_info.txt
#BSUB -q hpcintro
#BSUB -W 1
#BSUB -R "rusage[mem=512MB]"

lscpu