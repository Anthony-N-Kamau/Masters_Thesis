#!/bin/bash
#SBATCH --job-name=ifa_f_fullshuffle_bnc
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/ifa_full_fullshuffle_bnc_%j.out
#SBATCH --error=logs/ifa_full_fullshuffle_bnc_%j.out

source ~/probe_env/bin/activate
cd ~/Impossible_projects/"Mechanistic Interpretability"/"Information Flow Analysis"
python3 ifa_phase1_textprep.py --pert fullshuffle --corpus bnc --n 150
