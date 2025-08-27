## Mahti

### Load modules

```bash
python-data/3.10-24.04
gcc/11.2.0
cuda/11.5.0
```

### Run with
`srun -A project_2013477 -p gpusmall -t 00:10:00 --gres=gpu:a100:1 -N 1 -n 1 python 02_Numba_indexing.py`

## LUMI

```bash
module use /appl/local/csc/modulefiles/
module load pytorch
```
