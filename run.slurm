#!/bin/bash
#SBATCH --job-name=vae-oasis
#SBATCH --partition=comp3710
#SBATCH --qos=comp3710             
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
source ~/.bashrc
conda activate vae
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -u train_vae.py \
  --data_root /home/groups/comp3710/OASIS_preprocessed \
  --epochs 30 --batch 64 --zdim 16 --lr 1e-3 --out out_oasis