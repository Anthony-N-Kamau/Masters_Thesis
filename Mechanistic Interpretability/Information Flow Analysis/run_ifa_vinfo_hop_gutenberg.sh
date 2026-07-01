#!/bin/bash
#SBATCH --job-name=ifa_v_hop_gutenberg
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/ifa_vinfo_hop_gutenberg_%j.out
#SBATCH --error=logs/ifa_vinfo_hop_gutenberg_%j.out

source ~/probe_env/bin/activate
cd ~/Impossible_projects/"Mechanistic Interpretability"/"Information Flow Analysis"
python3 ifa_phase2_vinfo.py --pert hop --corpus gutenberg
