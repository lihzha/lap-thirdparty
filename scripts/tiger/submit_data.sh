#!/bin/bash
#SBATCH --job-name=prep_tiger
#SBATCH --output=logs/%A_prep.out
#SBATCH --error=logs/%A_prep.err
#SBATCH --time=06:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=all

set -euo pipefail
mkdir -p logs

# Force CPU to avoid grabbing GPUs accidentally
export JAX_PLATFORMS=cpu

uv run scripts/tiger/convert_tiger_data_to_lerobot.py --data_dir all_data/data_tiger/demos
uv run scripts/compute_norm_stats.py --config-name pi05_tiger_finetune_local
