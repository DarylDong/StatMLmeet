#!/bin/sh
#SBATCH -p cp100

echo [$SECONDS] setting up environment

conda activate tf_gpu_14

srun -p cp100 python zeldoG_quick.py

echo [$SECONDS] End job 
