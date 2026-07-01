#!/bin/bash
#SBATCH --job-name=patch_fullshuffle
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/patch_fullshuffle_%j.out
#SBATCH --error=logs/patch_fullshuffle_%j.out

source ~/probe_env/bin/activate
cd ~/Impossible_projects/"Mechanistic Interpretability"/"Activation Patching"

echo "=== fullshuffle / GUTENBERG ==="
python3 run_patching.py fullshuffle --corpus gutenberg

echo ""
echo "=== fullshuffle / BNC ==="
python3 run_patching.py fullshuffle --corpus bnc

echo "=== DONE: fullshuffle ==="
