#!/bin/bash
#SBATCH --job-name=patch_hop
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/patch_hop_%j.out
#SBATCH --error=logs/patch_hop_%j.out

source ~/probe_env/bin/activate
cd ~/Impossible_projects/"Mechanistic Interpretability"/"Activation Patching"

echo "=== hop / GUTENBERG ==="
python3 run_patching.py hop --corpus gutenberg

echo ""
echo "=== hop / BNC ==="
python3 run_patching.py hop --corpus bnc

echo "=== DONE: hop ==="
