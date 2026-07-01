#!/bin/bash
#SBATCH --job-name=expand_fs
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/expand_fs_%j.out
#SBATCH --error=logs/expand_fs_%j.out

source ~/probe_env/bin/activate
cd ~/Impossible_projects/"Mechanistic Interpretability"/"Information Flow Analysis"

echo "=== Gutenberg ==="
python3 expand_fullshuffle_ifa.py --n 600 --corpus gutenberg

echo ""
echo "=== BNC ==="
python3 expand_fullshuffle_ifa.py --n 600 --corpus bnc
