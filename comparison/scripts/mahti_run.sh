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

srun python3 python/hop_lbm.py 1000 1000 data/input.json hop_mahti
srun python3 python/torch_lbm.py 1000 1000 data/input.json torch_mahti
srun python3 python/numba_lbm.py 1000 1000 data/input.json numba_mahti
#srun python3 python/02_Numba_indexing.py 1000 1000 data/input.json numba_original_mahti
#srun python3 python/numpy_lbm.py 1000 1000 data/input.json numpy_mahti

ml gnuplot

srun gnuplot \
    -e "filename1='data/profiles_hop_mahti_float32.csv'" \
    -e "filename2='data/profiles_torch_mahti_float32.csv'" \
    -e "outfile='data/profiles_mahti.png'" \
    scripts/plot_profile.gnuplot
