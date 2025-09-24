#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --partition=standard-g
#SBATCH --mem=480G
#SBATCH --time=00:10:00
#SBATCH --output="output_%x_%j.txt"
#SBATCH --account=project_462000915

ml purge

ml PrgEnv-cray/8.5.0
ml craype-accel-amd-gfx90a
ml rocm/6.0.3
ml LUMI
ml partition/container
ml PyTorch/2.7.0-rocm-6.2.4-python-3.12-singularity-20250527

# Optional: Inject the environment variables for NCCL debugging into the container.   
# This will produce a lot of debug output!     
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL

CPU_BIND="mask_cpu:fe000000000000,fe00000000000000"
CPU_BIND="${CPU_BIND},fe0000,fe000000"
CPU_BIND="${CPU_BIND},fe,fe00"
CPU_BIND="${CPU_BIND},fe00000000,fe0000000000"

# Note that `conda-python-distributed` automatically sets
# ROCR_VISIBLE_DEVICES=$SLURM_LOCALID, so only one GPU is visible
# per rank
#srun \
srun --cpu-bind=${CPU_BIND} \
  singularity exec $SIFPYTORCH \
    conda-python-distributed -u main.py

