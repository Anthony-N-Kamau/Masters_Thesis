#!/bin/bash
#SBATCH --job-name=blimp_full
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/akamau/Impossible_projects/logs/blimp_full_%j.out
#SBATCH --error=/home/akamau/Impossible_projects/logs/blimp_full_%j.err

source ~/.bashrc
source ~/probe_env/bin/activate
cd ~/Impossible_projects/BLiMP_full

python3 full_blimp_eval.py $1 $2
