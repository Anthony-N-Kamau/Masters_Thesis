#!/bin/bash
#SBATCH --job-name=blimp_full_all
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/akamau/Impossible_projects/logs/blimp_full_all_%j.out
#SBATCH --error=/home/akamau/Impossible_projects/logs/blimp_full_all_%j.err

source ~/.bashrc
source ~/probe_env/bin/activate
cd ~/Impossible_projects/BLiMP_full

for PERT in reverse localw3 localw5 fullshuffle; do
    echo "============================================================"
    echo "Starting perturbation: $PERT"
    echo "============================================================"
    python3 full_blimp_eval.py $PERT 1000
done

echo "All perturbations complete."
