# BLiMP_full

Evaluates Base GPT-2, Kallini, and Recovery models on BLiMP minimal pairs, for each perturbation type.

## Setup

See the repo root README for environment setup. Once `probe_env` is active:

```bash
cd BLiMP_full
```

## Run

All perturbations and models:

```bash
sbatch run_full_blimp_all.sh
```

Single perturbation:

```bash
sbatch run_full_blimp.sh <perturbation>
```

Both call `full_blimp_eval.py`.

## Output

`<perturbation>/blimp_full_<Model>.json` per model, plus heatmaps and accuracy plots in `<perturbation>/plots/`.

## Regenerate plots only

```bash
python replot_blimp.py
```

Check `logs/` if a job fails.
