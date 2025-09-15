#!/bin/bash
#SBATCH --job-name=vae-oasis
#SBATCH --partition=comp3710
#SBATCH --qos=comp3710            # if this errors, replace with:  #SBATCH --account=comp3710
#SBATCH --gres=gpu:a100:1         # match comp3710's GRES exactly
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Activate your env
source ~/.bashrc
conda activate vae

# (Usually no module load needed if your PyTorch wheel includes CUDA)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p logs out_oasis

python -u train_vae.py \
  --data_root /home/groups/comp3710/OASIS_preprocessed \
  --epochs 30 --batch 64 --zdim 16 --lr 1e-3 --out out_oasis
