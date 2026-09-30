#!/bin/bash
# One job per seed; the published run used seeds 17-21, one NVIDIA A100 each.
# Submit from the repository root after filling in the <...> fields:  sbatch slurm/run_ood.sbatch
#SBATCH --account=<account>
#SBATCH --partition=<gpu-partition>
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=ood_seed_%A_%a.log
set -e
source <environment>/bin/activate          # Python environment with requirements.txt installed
export GEOCROP_DATA=<input directory>      # layout in README.md
export GEOCROP_SEED=$((17 + SLURM_ARRAY_TASK_ID))
python -u run_ood.py
