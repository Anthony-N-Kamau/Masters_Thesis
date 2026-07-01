#!/bin/bash
#SBATCH --job-name=head_probe_full
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/akamau/Impossible_projects/logs/head_probe_full_%j.out
#SBATCH --error=/home/akamau/Impossible_projects/logs/head_probe_full_%j.err

source ~/.bashrc
source ~/probe_env/bin/activate

cd ~/Impossible_projects
python3 linear_probing_2/unified_head_probe_full.py
