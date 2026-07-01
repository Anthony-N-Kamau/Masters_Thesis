#!/bin/bash
#SBATCH --job-name=patch_localw3
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/patch_localw3_%j.out
#SBATCH --error=logs/patch_localw3_%j.out

source ~/probe_env/bin/activate
cd ~/Impossible_projects/"Mechanistic Interpretability"/"Activation Patching"

echo "=== localw3 / GUTENBERG ==="
python3 run_patching.py localw3 --corpus gutenberg

echo ""
echo "=== localw3 / BNC ==="
python3 run_patching.py localw3 --corpus bnc

echo "=== DONE: localw3 ==="
