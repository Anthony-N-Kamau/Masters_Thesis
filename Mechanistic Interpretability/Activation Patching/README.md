# Activation Patching

Head-level restoration patching across Base GPT-2, Kallini, and Recovery models. Two steps: build prompt pairs, then patch.

## Setup

See the repo root README for environment setup. Once `probe_env` is active:

```bash
cd "Mechanistic Interpretability/Activation Patching"
```

## Step 1: Generate prompt pairs

```bash
sbatch run_prepare_prompts.sh
```

Calls `generate_pairs.py` (all perturbations) and `generate_hop_filtered.py` (hop only, filtered). Output: `<perturbation>/pairs_<perturbation>_<corpus>.json`.

## Step 2: Run patching

```bash
sbatch run_full_hop.sh
sbatch run_full_reverse.sh
sbatch run_full_localw3.sh
sbatch run_full_localw5.sh
sbatch run_full_fullshuffle.sh
```

Each wraps `run_patching.py`.

## Output

`<perturbation>/patching_<perturbation>_<corpus>_<model>.json`: 12x12 layer-head restoration score matrix, plus best (layer, head, score).

## Generate plots

```bash
python plot_patching.py
```

Produces layer-head heatmaps, top-k tracking, and violin plots in `plots/`.

Check `logs/` if a job fails.
