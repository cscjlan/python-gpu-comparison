#!/bin/bash
#SBATCH --account=project_462000915
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --time=1:00:00
#SBATCH --gpus-per-node=1

ml PrgEnv-cray/8.5.0
ml craype-accel-amd-gfx90a
ml rocm/6.0.3
srun python3 python/hop_lbm.py 1000 1000 data/input.json hop_lumi

ml use /appl/local/csc/modulefiles/
ml pytorch/2.7
srun python3 python/torch_lbm.py 1000 1000 data/input.json torch_lumi

ml LUMI
ml partition/G
ml gnuplot

srun gnuplot scripts/plot_profile.gnuplot
