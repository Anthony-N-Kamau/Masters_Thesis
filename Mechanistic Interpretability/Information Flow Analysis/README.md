# Information Flow Analysis

Token-to-token V-information flow analysis across Base GPT-2, Kallini, and Recovery models. Two phases per perturbation and corpus.

## Setup

See the repo root README for environment setup. Once `probe_env` is active:

```bash
cd "Mechanistic Interpretability/Information Flow Analysis"
```

## Phase 1: Text preparation

```bash
sbatch run_ifa_full_<perturbation>_<corpus>.sh
```

Calls `ifa_phase1_textprep.py`. Output: `<perturbation>/ifa_textprep_<perturbation>_<corpus>.json`.

## Phase 2: V-information computation

```bash
sbatch run_ifa_vinfo_<perturbation>_<corpus>.sh
```

Calls `ifa_phase2_vinfo.py`. Output: `<perturbation>/ifa_vinfo_<perturbation>_<corpus>.json`.

For fullshuffle with the expanded 600-sentence sample, run `run_expand_fullshuffle.sh` first, then `run_ifa_phase1_fs_expanded.sh` and `run_ifa_phase2_fs_expanded.sh`.

## Generate plots

```bash
python plot_ifa.py
```

Produces combined curves, per-model curves, and model-comparison plots in `plots/`.

Check `logs/` if a job fails.
