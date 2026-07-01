#!/bin/bash
#SBATCH --job-name=ifa_p2_fs600
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --output=logs/ifa_p2_fs600_%j.out
#SBATCH --error=logs/ifa_p2_fs600_%j.out

source ~/probe_env/bin/activate
cd ~/Impossible_projects/"Mechanistic Interpretability"/"Information Flow Analysis"

echo "=== GUTENBERG ==="
python3 ifa_phase2_vinfo.py --pert fullshuffle --corpus gutenberg \
  --textprep-override "$(pwd)/fullshuffle/ifa_textprep_fullshuffle_gutenberg_n600.json"

echo ""
echo "=== BNC ==="
python3 ifa_phase2_vinfo.py --pert fullshuffle --corpus bnc \
  --textprep-override "$(pwd)/fullshuffle/ifa_textprep_fullshuffle_bnc_n600.json"
