#!/bin/bash
#SBATCH --job-name=ifa_p1_fs600
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/ifa_p1_fs600_%j.out
#SBATCH --error=logs/ifa_p1_fs600_%j.out

source ~/probe_env/bin/activate
cd ~/Impossible_projects/"Mechanistic Interpretability"/"Information Flow Analysis"

echo "=== GUTENBERG ==="
python3 ifa_phase1_textprep.py --pert fullshuffle --corpus gutenberg --n 600 \
  --pairs-override "$(pwd)/fullshuffle/pairs_fullshuffle_gutenberg_ifa600.json"

echo ""
echo "=== BNC ==="
python3 ifa_phase1_textprep.py --pert fullshuffle --corpus bnc --n 600 \
  --pairs-override "$(pwd)/fullshuffle/pairs_fullshuffle_bnc_ifa600.json"
