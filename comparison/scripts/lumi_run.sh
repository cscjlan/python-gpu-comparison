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

# Numba not available on LUMI
#srun python3 python/numba_lbm.py 1000 1000 data/input.json numba_lumi
#srun python3 python/02_Numba_indexing.py 1000 1000 data/input.json numba_original_lumi
#srun python3 python/numpy_lbm.py 1000 1000 data/input.json numpy_lumi
