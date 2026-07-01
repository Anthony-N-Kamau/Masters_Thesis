#!/bin/bash
#SBATCH --job-name=patch_localw5
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/patch_localw5_%j.out
#SBATCH --error=logs/patch_localw5_%j.out

source ~/probe_env/bin/activate
cd ~/Impossible_projects/"Mechanistic Interpretability"/"Activation Patching"

echo "=== localw5 / GUTENBERG ==="
python3 run_patching.py localw5 --corpus gutenberg

echo ""
echo "=== localw5 / BNC ==="
python3 run_patching.py localw5 --corpus bnc

echo "=== DONE: localw5 ==="
