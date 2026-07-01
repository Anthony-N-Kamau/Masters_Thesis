#!/bin/bash
#SBATCH --job-name=patch_reverse
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/patch_reverse_%j.out
#SBATCH --error=logs/patch_reverse_%j.out

source ~/probe_env/bin/activate
cd ~/Impossible_projects/"Mechanistic Interpretability"/"Activation Patching"

echo "=== reverse / GUTENBERG ==="
python3 run_patching.py reverse --corpus gutenberg

echo ""
echo "=== reverse / BNC ==="
python3 run_patching.py reverse --corpus bnc

echo "=== DONE: reverse ==="
