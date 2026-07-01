import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE     = os.path.expanduser('~/Impossible_projects')
PROBE_DIR = os.path.join(BASE, 'linear_probing_2')
BNC_DIR   = os.path.join(BASE, 'BNC_plots')   # new output root

N_LAYERS = 12
N_HEADS  = 12

FEATURES = ['pos', 'dep_relation', 'arc_direction', 'phrase_role']
FEATURE_LABELS = {
    'pos':           'POS (noun/verb)',
    'dep_relation':  'Dep Rel (nsubj/obj)',
    'arc_direction': 'Arc Dir (L/R)',
    'phrase_role':   'Phrase Role (subj/obj NP)',
}

# Perturbation -> (subfolder under linear_probing_2, corpus tag used in filenames)
PERTURBATIONS = {
    'hop':         {'subfolder': 'hop',     'corpus_guten': 'gutenberg', 'corpus_bnc': 'bnc'},
    'reverse':     {'subfolder': 'reverse', 'corpus_guten': 'gutenberg', 'corpus_bnc': 'bnc'},
    'localw3':     {'subfolder': 'shuffle', 'corpus_guten': 'gutenberg', 'corpus_bnc': 'bnc'},
    'localw5':     {'subfolder': 'shuffle', 'corpus_guten': 'gutenberg', 'corpus_bnc': None},
    'fullshuffle': {'subfolder': 'shuffle', 'corpus_guten': 'gutenberg', 'corpus_bnc': None},
}

GUTEN_MODEL_LABELS = ['Base GPT-2', 'Kallini', 'Recovery (Guten)']
BNC_MODEL_LABEL    = 'Recovery (BNC)'


FILE_SAFE_NAME = {
    'Base GPT-2':       'Base_GPT-2',
    'Kallini':          'Kallini',
    'Recovery (BNC)':   'Translator_BNC',
    'Recovery (Guten)': 'Translator_Guten',
}

def safe_name(model_name):
    return FILE_SAFE_NAME.get(
        model_name,
        model_name.replace(' ', '_').replace('(', '').replace(')', '').strip('_')
    )

def load_result(pert, subfolder, corpus, model_name, feature):
    fname = f'head_probe_{pert}_{corpus}_{safe_name(model_name)}_{feature}.json'
    fpath = os.path.join(PROBE_DIR, subfolder, fname)
    if not os.path.exists(fpath):
        return None
    with open(fpath, 'r') as f:
        d = json.load(f)
    return {
        'real_acc':    np.array(d['real_acc']),
        'ctrl_acc':    np.array(d['ctrl_acc']),
        'selectivity': np.array(d['selectivity']),
    }

def pert_label_str(pert):
    return pert.replace('_', ' ').title()

def save_fig(fig, path_no_ext):
    plt.tight_layout()
    fig.savefig(path_no_ext + '.pdf', bbox_inches='tight')
    fig.savefig(path_no_ext + '.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'  Saved: {path_no_ext}.pdf / .png')

# Gutenberg-trio selectivity heatmap (per feature, per perturbation) 
def plot_gutenberg_selectivity(pert, cfg):
    subfolder = cfg['subfolder']
    corpus    = cfg['corpus_guten']
    plots_dir = os.path.join(PROBE_DIR, subfolder, f'plots_{pert}')
    os.makedirs(plots_dir, exist_ok=True)

    for feature in FEATURES:
        results = {}
        for model_name in GUTEN_MODEL_LABELS:
            r = load_result(pert, subfolder, corpus, model_name, feature)
            if r is not None:
                results[model_name] = r
        if not results:
            print(f'  [skip] {pert} / {feature}: no Gutenberg-trio results found')
            continue

        all_sel = [results[m]['selectivity'] for m in results]
        vmax = max(0.1, float(np.nanmax(np.abs(np.concatenate([s.flatten() for s in all_sel])))))

        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4.0), sharey=False)
        if n_models == 1:
            axes = [axes]

        for ax, (model_name, r) in zip(axes, results.items()):
            im = ax.imshow(r['selectivity'], aspect='auto', cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax, origin='upper')
            cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
            cbar.ax.tick_params(labelsize=7)
            ax.set_xticks(range(N_HEADS))
            ax.set_xticklabels([h+1 for h in range(N_HEADS)], fontsize=7)
            ax.set_yticks(range(N_LAYERS))
            ax.set_ylabel('Layer', fontsize=9, labelpad=5)
            ax.set_yticklabels([l+1 for l in range(N_LAYERS)], fontsize=7)
            ax.set_title(model_name, fontsize=10, fontweight='bold', pad=8)

        fig.text(0.5, -0.02, 'Attention head', ha='center', fontsize=9)
        fig.suptitle(f'Selectivity | {FEATURE_LABELS[feature]} | {pert_label_str(pert)}',
                    fontsize=11, fontweight='bold', y=1.02, x=0.45)

        save_fig(fig, os.path.join(plots_dir, f'B_sel_heatmap_{feature}'))

# BNC_heatmap: 2x2 grid of all 4 features, BNC only 
def plot_bnc_2x2(pert, cfg, out_dir):
    if cfg['corpus_bnc'] is None:
        print(f'  [skip] {pert}: no BNC model available')
        return
    subfolder = cfg['subfolder']
    corpus    = cfg['corpus_bnc']

    results = {}
    for feature in FEATURES:
        r = load_result(pert, subfolder, corpus, BNC_MODEL_LABEL, feature)
        if r is not None:
            results[feature] = r

    if not results:
        print(f'  [skip] {pert}: no BNC results found for 2x2 grid')
        return

    all_sel = [results[f]['selectivity'] for f in results]
    vmax = max(0.1, float(np.nanmax(np.abs(np.concatenate([s.flatten() for s in all_sel])))))

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.0), sharey=False, sharex=False)
    axes_flat = axes.flatten()

    for ax, feature in zip(axes_flat, FEATURES):
        if feature not in results:
            ax.axis('off')
            continue
        r = results[feature]
        im = ax.imshow(r['selectivity'], aspect='auto', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, origin='upper')
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
        ax.set_xticks(range(N_HEADS))
        ax.set_xticklabels([h+1 for h in range(N_HEADS)], fontsize=7)
        ax.set_yticks(range(N_LAYERS))
        ax.set_ylabel('Layer', fontsize=9, labelpad=5)
        ax.set_yticklabels([l+1 for l in range(N_LAYERS)], fontsize=7)
        ax.set_xlabel('Attention head', fontsize=9)
        ax.set_title(FEATURE_LABELS[feature], fontsize=10, fontweight='bold', pad=8)

    fig.suptitle(f'Recovery (BNC) Selectivity | All Features | {pert_label_str(pert)}',
                fontsize=12, fontweight='bold', y=1.02)

    save_fig(fig, os.path.join(out_dir, 'BNC_heatmap'))

# ── 3. BNC_pos_arc: POS and Arc Direction grouped, BNC only ─────────────────
def plot_bnc_pos_arc(pert, cfg, out_dir):
    if cfg['corpus_bnc'] is None:
        print(f'  [skip] {pert}: no BNC model available')
        return
    subfolder = cfg['subfolder']
    corpus    = cfg['corpus_bnc']

    results = {}
    for feature in ['pos', 'arc_direction']:
        r = load_result(pert, subfolder, corpus, BNC_MODEL_LABEL, feature)
        if r is not None:
            results[feature] = r

    if not results:
        print(f'  [skip] {pert}: no BNC POS/Arc results found')
        return

    all_sel = [results[f]['selectivity'] for f in results]
    vmax = max(0.1, float(np.nanmax(np.abs(np.concatenate([s.flatten() for s in all_sel])))))

    n_feat = len(results)
    fig, axes = plt.subplots(1, n_feat, figsize=(5.5 * n_feat, 4.0), sharey=False)
    if n_feat == 1:
        axes = [axes]

    for ax, feature in zip(axes, ['pos', 'arc_direction']):
        if feature not in results:
            continue
        r = results[feature]
        im = ax.imshow(r['selectivity'], aspect='auto', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, origin='upper')
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
        ax.set_xticks(range(N_HEADS))
        ax.set_xticklabels([h+1 for h in range(N_HEADS)], fontsize=7)
        ax.set_yticks(range(N_LAYERS))
        ax.set_ylabel('Layer', fontsize=9, labelpad=5)
        ax.set_yticklabels([l+1 for l in range(N_LAYERS)], fontsize=7)
        ax.set_title(FEATURE_LABELS[feature], fontsize=10, fontweight='bold', pad=8)

    fig.text(0.5, -0.02, 'Attention head', ha='center', fontsize=9)
    fig.suptitle(f'Recovery (BNC) Selectivity | POS \\& Arc Direction | {pert_label_str(pert)}',
                fontsize=11, fontweight='bold', y=1.02, x=0.45)

    save_fig(fig, os.path.join(out_dir, 'BNC_pos_arc'))

# Main 
if __name__ == '__main__':
    os.makedirs(BNC_DIR, exist_ok=True)

    for pert, cfg in PERTURBATIONS.items():
        print(f'\n{"="*60}')
        print(f'Perturbation: {pert.upper()}')
        print(f'{"="*60}')

        # Gutenberg-trio plots (overwrite existing B_sel_heatmap_*.pdf/.png)
        plot_gutenberg_selectivity(pert, cfg)

        # BNC plots, grouped by perturbation in BNC_plots/{pert}/
        pert_bnc_dir = os.path.join(BNC_DIR, pert)
        os.makedirs(pert_bnc_dir, exist_ok=True)

        plot_bnc_2x2(pert, cfg, pert_bnc_dir)
        plot_bnc_pos_arc(pert, cfg, pert_bnc_dir)

    print('\nDone.')
