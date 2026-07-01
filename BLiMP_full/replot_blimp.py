import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PERT = sys.argv[1] if len(sys.argv) > 1 else 'hop'
BASE = os.path.expanduser('~/Impossible_projects')
OUT_DIR = os.path.join(BASE, 'BLiMP_full', PERT)
plots_dir = os.path.join(OUT_DIR, 'plots')

MODELS_PER_PERT = {
    'hop':         ['Base GPT-2', 'Kallini', 'Recovery (BNC)', 'Recovery (Gutenberg)'],
    'reverse':     ['Base GPT-2', 'Kallini', 'Recovery (BNC)', 'Recovery (Gutenberg)'],
    'localw3':     ['Base GPT-2', 'Kallini', 'Recovery (BNC)', 'Recovery (Gutenberg)'],
    'localw5':     ['Base GPT-2', 'Kallini', 'Recovery (Gutenberg)'],
    'fullshuffle': ['Base GPT-2', 'Kallini', 'Recovery (Gutenberg)'],
}
MODELS = MODELS_PER_PERT[PERT]

MODEL_COLORS = {
    'Base GPT-2':           '#2166ac',
    'Kallini':               '#4dac26',
    'Recovery (BNC)':        '#d01c8b',
    'Recovery (Gutenberg)':  '#f1a340',
}

matplotlib.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['DejaVu Serif'],
    'font.size':         9,
    'axes.titlesize':    10,
    'axes.labelsize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
})

def pert_display_name(pert):
    return pert.replace('localw', 'Local-w').replace('fullshuffle', 'Full Shuffle').title()

def save_fig(fig, path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f'  Saved: {path}')

# Load existing JSON results 
all_results = {}
for model_name in MODELS:
    safe_name = model_name.replace(" ", "").replace("(", "").replace(")", "")
    json_path = os.path.join(OUT_DIR, f'blimp_full_{safe_name}.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            all_results[model_name] = json.load(f)
        print(f'Loaded: {json_path}')
    else:
        print(f'MISSING: {json_path}')

if not all_results:
    print('No results found — nothing to replot.')
    sys.exit(1)

# Overall bar chart 
def plot_overall_bar():
    models = [m for m in MODELS if m in all_results]
    means  = [all_results[m]['overall_mean'] for m in models]
    colors = [MODEL_COLORS[m] for m in models]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(len(models)), means, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    ax.axhline(0.5, color='gray', linewidth=0.7, linestyle='--', alpha=0.6,
               label='Chance (0.5)')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylabel('Overall BLiMP accuracy', fontsize=9)
    ax.set_ylim(0, 1.0)
    n_para = all_results[models[0]]['n_paradigms']
    ax.set_title(f'BLiMP Suite ({n_para} paradigms) | {pert_display_name(PERT)}',
                fontsize=10, fontweight='bold')
    ax.legend(fontsize=7)
    save_fig(fig, os.path.join(plots_dir, f'overall_blimp_{PERT}.pdf'))

# Field heatmap 
def plot_field_heatmap():
    models = [m for m in MODELS if m in all_results]
    fields = sorted(set(f for m in models for f in all_results[m]['field_means'].keys()))
    mat = np.full((len(models), len(fields)), np.nan)
    for mi, m in enumerate(models):
        for fi, f in enumerate(fields):
            if f in all_results[m]['field_means']:
                mat[mi, fi] = all_results[m]['field_means'][f]

    fig, ax = plt.subplots(figsize=(1.5*len(fields)+2, 1.2*len(models)+1))
    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(len(fields)))
    ax.set_xticklabels(fields, rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    for mi in range(len(models)):
        for fi in range(len(fields)):
            val = mat[mi, fi]
            if not np.isnan(val):
                ax.text(fi, mi, f'{val:.2f}', ha='center', va='center', fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label='Accuracy')
    ax.set_title(f'BLiMP Accuracy by Field | {pert_display_name(PERT)}', fontsize=10, fontweight='bold')
    save_fig(fig, os.path.join(plots_dir, f'field_heatmap_{PERT}.pdf'))

# Phenomenon heatmap 
def plot_phenomenon_heatmap():
    models = [m for m in MODELS if m in all_results]
    phenomena = sorted(set(
        p for m in models for p in all_results[m]['phenomenon_means'].keys()
    ))
    mat = np.full((len(models), len(phenomena)), np.nan)
    for mi, m in enumerate(models):
        for pi, p in enumerate(phenomena):
            if p in all_results[m]['phenomenon_means']:
                mat[mi, pi] = all_results[m]['phenomenon_means'][p]

    fig, ax = plt.subplots(figsize=(1.3*len(phenomena)+2, 1.2*len(models)+1.5))
    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(len(phenomena)))
    ax.set_xticklabels(phenomena, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    for mi in range(len(models)):
        for pi in range(len(phenomena)):
            val = mat[mi, pi]
            if not np.isnan(val):
                ax.text(pi, mi, f'{val:.2f}', ha='center', va='center', fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='Accuracy')
    n_para = next(iter(all_results.values()))['n_paradigms']
    ax.set_title(f'BLiMP Accuracy by Phenomenon ({n_para} paradigms) | {pert_display_name(PERT)}',
                fontsize=10, fontweight='bold')
    save_fig(fig, os.path.join(plots_dir, f'phenomenon_heatmap_{PERT}.pdf'))

# Run 
plot_overall_bar()
plot_field_heatmap()
plot_phenomenon_heatmap()
print('\nDone — all plots regenerated with corrected titles.')
