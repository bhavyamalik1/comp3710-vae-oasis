#!/bin/bash
#SBATCH -J vae-oasis
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 06:00:00
#SBATCH --mem=24G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

source ~/.bashrc
conda activate vae
module load cuda/12.2

# if PyTorch isn't installed in your env on the node, uncomment next line once:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

python -u train_vae.py \
  --data_root /home/groups/comp3710/OASIS_preprocessed \
  --epochs 30 --batch 64 --zdim 16 --lr 1e-3 --out out_oasis
