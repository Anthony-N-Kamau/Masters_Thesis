#!/bin/bash
#SBATCH --job-name=prep_prompts
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/prep_prompts_%j.out
#SBATCH --error=logs/prep_prompts_%j.out

source ~/probe_env/bin/activate
cd ~/Impossible_projects/"Mechanistic Interpretability"/"Activation Patching"

for pert in hop reverse localw3 localw5 fullshuffle; do
    echo "=== $pert ==="
    python3 prepare_prompts.py --pert $pert --corpus gutenberg
    python3 prepare_prompts.py --pert $pert --corpus bnc
    echo ""
done

echo "=== ALL PROMPTS PREPARED ==="
