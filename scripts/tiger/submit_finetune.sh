#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --output=logs/%A_finetune.out
#SBATCH --error=logs/%A_finetune.err
#SBATCH --time=48:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --partition=all

set -euo pipefail
mkdir -p logs

export JAX_PLATFORMS=cuda

cd /n/fs/robot-data/openpi-cot

uv run --group cuda scripts/train.py pi05_tiger_finetune_local_low_mem \
  --exp-name finetune_low_mem_64_add_norm \
  --fsdp-devices=4 \
  --batch-size=64
