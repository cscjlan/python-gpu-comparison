#!/bin/bash
#SBATCH --account=project_2013477
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:1

ml pytorch/2.7
ml gcc/11.2.0
ml cuda/11.5.0

srun python3 02_Numba_indexing.py 4000 4000
srun python3 numba_lbm.py 4000 4000
