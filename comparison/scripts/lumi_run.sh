#!/bin/bash
#SBATCH --account=project_462000915
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --time=1:00:00
#SBATCH --gpus-per-node=1

ml purge

ml PrgEnv-cray/8.5.0
ml craype-accel-amd-gfx90a
ml rocm/6.0.3
srun python3 python/hop_lbm.py 1000 1000 data/input.json hop_lumi

ml LUMI
ml partition/container
ml PyTorch/2.7.0-rocm-6.2.4-python-3.12-singularity-20250527

srun python3 python/torch_lbm.py 1000 1000 data/input.json torch_lumi

ml partition/G
ml gnuplot

srun gnuplot \
    -e "filename1='data/profiles_hop_lumi_float32.csv'" \
    -e "filename2='data/profiles_torch_lumi_float32.csv'" \
    -e "outfile='data/profiles_lumi.png'" \
    scripts/plot_profile.gnuplot
