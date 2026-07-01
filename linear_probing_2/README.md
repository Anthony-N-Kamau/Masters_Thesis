# linear_probing_2

Head-level linear probing (selectivity, Hewitt & Liang 2019) across Base GPT-2, Kallini, and Recovery models, for four syntactic features: POS, dependency relation, arc direction, phrase role.

## Setup

See the repo root README for environment setup. Once `probe_env` is active:

```bash
cd linear_probing_2
```

## Run

```bash
sbatch run_head_probe_full.sh
```

Calls `unified_head_probe_full.py`. Covers all perturbations, models, and features.

## Output

`<perturbation>/head_probe_<perturbation>_<corpus>_<Model>_<feature>.json`: 12x12 layer-head selectivity matrix (`real_acc`, `ctrl_acc`, `selectivity`), per feature.

## Analysis and plots

Find best-selectivity layer/head per model and feature:

```bash
python analyse_best_layers.py
```

Regenerate BNC vs Gutenberg split plots:

```bash
python replot_bnc_split.py
```

Plots are saved per perturbation in `<perturbation>/plots_<perturbation>/` (heatmaps: `B_sel_heatmap_<feature>.pdf/png`).

Check `logs/` if a job fails.
