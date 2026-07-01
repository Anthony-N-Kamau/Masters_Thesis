import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.titlesize': 8.5,
    'axes.labelsize': 7.5,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'legend.fontsize': 6.5,
    'figure.titlesize': 10,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.6,
    'axes.edgecolor': '#444444',
})

BASE = os.path.expanduser('~/Impossible_projects')
IFA_BASE = os.path.join(BASE, 'Mechanistic Interpretability', 'Information Flow Analysis')
PLOT_DIR = os.path.join(IFA_BASE, 'plots')

PERTS = ['hop', 'reverse', 'localw3', 'localw5', 'fullshuffle']
PERT_DISPLAY = {
    'hop': 'Hop', 'reverse': 'Reverse', 'localw3': 'Local w3',
    'localw5': 'Local w5', 'fullshuffle': 'Full Shuffle',
}
CORPORA = ['gutenberg', 'bnc']
CORPUS_DISPLAY = {'gutenberg': 'Gutenberg', 'bnc': 'BNC Spoken'}

VALID_COMBOS = [(p, c) for p in PERTS for c in CORPORA
                if not (p == 'localw5' and c == 'bnc')]

MODELS = ['Base_GPT-2', 'Kallini', 'Recovery']
MODEL_DISPLAY = {'Base_GPT-2': 'Base GPT-2', 'Kallini': 'Kallini', 'Recovery': 'Recovery'}
MODEL_COLORS = {'Base_GPT-2': '#3b6e8f', 'Kallini': '#7a9b57', 'Recovery': '#c98a3e'}

CONDITIONS = ['original', 'impossible', 'recovered']
CONDITION_DISPLAY = {'original': 'Original', 'impossible': 'Impossible', 'recovered': 'Recovered'}
CONDITION_COLORS = {'original': '#2c5f6f', 'impossible': '#b33951', 'recovered': '#5a8a3c'}
CONDITION_STYLES = {'original': '-', 'impossible': '--', 'recovered': ':'}

FEATURES = ['pos', 'dep_rel', 'arc_dir', 'phrase_role']
FEATURE_DISPLAY = {
    'pos': 'POS', 'dep_rel': 'Dep. Relation',
    'arc_dir': 'Arc Direction', 'phrase_role': 'Phrase Role',
}

N_LAYERS_TOTAL = 13


def load_vinfo(pert, corpus):
    # Load the V-information results for one perturbation-corpus combo.
    
    expanded_path = os.path.join(IFA_BASE, pert, f'ifa_vinfo_{pert}_{corpus}_expanded.json')
    standard_path = os.path.join(IFA_BASE, pert, f'ifa_vinfo_{pert}_{corpus}.json')

    if os.path.exists(expanded_path):
        path = expanded_path
        print(f'  [{pert}/{corpus}] using EXPANDED data: {os.path.basename(path)}')
    elif os.path.exists(standard_path):
        path = standard_path
    else:
        return None

    with open(path) as f:
        return json.load(f)


def get_curve(d, model, condition, feature):
    """Returns (layers, vinfo) lists, or ([], []) if missing/empty."""
    try:
        c = d['curves'][model][condition][feature]
        return c['layers'], c['v_information']
    except (KeyError, TypeError):
        return [], []


def clean_axes(ax):
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)



# per (pert, corpus, model) feature grid

def plot_curves_single_model(pert, corpus, model, d):
    fig, axes = plt.subplots(2, 2, figsize=(5.2, 4.2))
    any_data = False

    for fi, feature in enumerate(FEATURES):
        ax = axes[fi // 2, fi % 2]
        for condition in CONDITIONS:
            layers, vinfo = get_curve(d, model, condition, feature)
            if not layers:
                continue
            any_data = True
            ax.plot(layers, vinfo, CONDITION_STYLES[condition],
                    color=CONDITION_COLORS[condition],
                    label=CONDITION_DISPLAY[condition],
                    marker='o', markersize=2.2, linewidth=1.1)
        ax.axhline(0, color='#bbbbbb', linewidth=0.5, zorder=0)
        ax.set_xlim(-0.5, N_LAYERS_TOTAL - 0.5)
        ax.set_xticks(range(N_LAYERS_TOTAL))
        ax.tick_params(axis="x", labelsize=5.5)
        ax.set_title(FEATURE_DISPLAY[feature], fontsize=8, fontweight='bold', pad=4)
        clean_axes(ax)
        if fi // 2 == 1:
            ax.set_xlabel('Layer', fontsize=7)
        if fi % 2 == 0:
            ax.set_ylabel('V-information', fontsize=7)

    if not any_data:
        plt.close(fig)
        return

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False,
                  bbox_to_anchor=(0.5, 1.06), fontsize=7)

    fig.suptitle(f'{MODEL_DISPLAY[model]} \u2014 {PERT_DISPLAY[pert]} ({CORPUS_DISPLAY[corpus]})',
                 fontsize=9.5, fontweight='bold', y=1.13)

    out_dir = os.path.join(PLOT_DIR, 'per_model')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'curves_{pert}_{corpus}_{model}.pdf')
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'  Saved: {out}')



# per (pert, corpus) combined grid, models (cols) times features (rows)

def plot_curves_combined(pert, corpus, d):
    fig, axes = plt.subplots(len(FEATURES), len(MODELS),
                              figsize=(2.1 * len(MODELS), 1.5 * len(FEATURES)),
                              squeeze=False)
    any_data = False

    for fi, feature in enumerate(FEATURES):
        for mi, model in enumerate(MODELS):
            ax = axes[fi, mi]
            for condition in CONDITIONS:
                layers, vinfo = get_curve(d, model, condition, feature)
                if not layers:
                    continue
                any_data = True
                ax.plot(layers, vinfo, CONDITION_STYLES[condition],
                        color=CONDITION_COLORS[condition],
                        label=CONDITION_DISPLAY[condition],
                        marker='o', markersize=1.8, linewidth=0.9)
            ax.axhline(0, color='#cccccc', linewidth=0.4, zorder=0)
            ax.set_xlim(-0.5, N_LAYERS_TOTAL - 0.5)
            ax.set_xticks(range(N_LAYERS_TOTAL))
            ax.tick_params(axis="x", labelsize=4.5)
            clean_axes(ax)
            if fi == 0:
                ax.set_title(MODEL_DISPLAY[model], fontsize=8, fontweight='bold',
                             color=MODEL_COLORS[model], pad=4)
            if mi == 0:
                ax.set_ylabel(FEATURE_DISPLAY[feature], fontsize=7, fontweight='bold')
            if fi == len(FEATURES) - 1:
                ax.set_xlabel('Layer', fontsize=7)

    if not any_data:
        plt.close(fig)
        return

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False,
                  bbox_to_anchor=(0.5, 1.02), fontsize=7)

    fig.suptitle(f'V-information curves \u2014 {PERT_DISPLAY[pert]} ({CORPUS_DISPLAY[corpus]})',
                 fontsize=9.5, fontweight='bold', y=1.06)
    fig.subplots_adjust(top=0.88, hspace=0.5, wspace=0.35)

    out = os.path.join(PLOT_DIR, f'curves_combined_{pert}_{corpus}.pdf')
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)

    # layer-by-layer V-information curves 
    plots2_dir = os.path.join(IFA_BASE, 'plots_2')
    os.makedirs(plots2_dir, exist_ok=True)
    out2 = os.path.join(plots2_dir, f'layer_curves_{pert}_{corpus}.pdf')
    fig.savefig(out2, bbox_inches='tight')
    fig.savefig(out2.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)

    plt.close(fig)
    print(f'  Saved: {out}')
    print(f'  Saved: {out2}')



# model-comparison

def plot_model_comparison_single(pert, corpus, condition, d):
    fig, axes = plt.subplots(2, 2, figsize=(5.2, 4.2))
    any_data = False

    for fi, feature in enumerate(FEATURES):
        ax = axes[fi // 2, fi % 2]
        for model in MODELS:
            layers, vinfo = get_curve(d, model, condition, feature)
            if not layers:
                continue
            any_data = True
            ax.plot(layers, vinfo, '-', color=MODEL_COLORS[model],
                    label=MODEL_DISPLAY[model], marker='o', markersize=2.2, linewidth=1.1)
        ax.axhline(0, color='#bbbbbb', linewidth=0.5, zorder=0)
        ax.set_xlim(-0.5, N_LAYERS_TOTAL - 0.5)
        ax.set_xticks(range(N_LAYERS_TOTAL))
        ax.tick_params(axis="x", labelsize=5.5)
        ax.set_title(FEATURE_DISPLAY[feature], fontsize=8, fontweight='bold', pad=4)
        clean_axes(ax)
        if fi // 2 == 1:
            ax.set_xlabel('Layer', fontsize=7)
        if fi % 2 == 0:
            ax.set_ylabel('V-information', fontsize=7)

    if not any_data:
        plt.close(fig)
        return

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False,
                  bbox_to_anchor=(0.5, 1.06), fontsize=7)

    fig.suptitle(f'Model comparison ({CONDITION_DISPLAY[condition]}) \u2014 '
                 f'{PERT_DISPLAY[pert]} ({CORPUS_DISPLAY[corpus]})',
                 fontsize=9.5, fontweight='bold', y=1.13)

    out_dir = os.path.join(PLOT_DIR, 'model_comparison')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'modelcmp_{pert}_{corpus}_{condition}.pdf')
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'  Saved: {out}')


def plot_model_comparison_grid(condition):
    """All (pert, corpus) combos x all 4 features, fixed condition, 3 model lines."""
    n_rows = len(VALID_COMBOS)
    n_cols = len(FEATURES)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(1.7 * n_cols, 1.25 * n_rows), squeeze=False)

    for ri, (pert, corpus) in enumerate(VALID_COMBOS):
        d = load_vinfo(pert, corpus)
        for fi, feature in enumerate(FEATURES):
            ax = axes[ri, fi]
            if d is None:
                ax.text(0.5, 0.5, '\u2013', ha='center', va='center',
                        transform=ax.transAxes, color='#999999')
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                continue
            for model in MODELS:
                layers, vinfo = get_curve(d, model, condition, feature)
                if not layers:
                    continue
                ax.plot(layers, vinfo, '-', color=MODEL_COLORS[model],
                        label=MODEL_DISPLAY[model], linewidth=0.8)
            ax.axhline(0, color='#dddddd', linewidth=0.4, zorder=0)
            ax.set_xlim(-0.5, N_LAYERS_TOTAL - 0.5)
            ax.set_xticks(range(N_LAYERS_TOTAL))
            ax.tick_params(labelsize=3.5, length=1, axis='x')
            ax.tick_params(labelsize=5.5, length=1.5, axis='y')
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)

            if ri == 0:
                ax.set_title(FEATURE_DISPLAY[feature], fontsize=7.5, fontweight='bold', pad=3)
            if fi == 0:
                ax.set_ylabel(f'{PERT_DISPLAY[pert]}\n{CORPUS_DISPLAY[corpus][:4]}',
                             fontsize=6, fontweight='bold')

    handles = [plt.Line2D([0], [0], color=MODEL_COLORS[m], label=MODEL_DISPLAY[m])
               for m in MODELS]
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False,
              bbox_to_anchor=(0.5, 1.0), fontsize=7)
    fig.suptitle(f'Model comparison across all conditions ({CONDITION_DISPLAY[condition]})',
                 fontsize=9.5, fontweight='bold', y=1.04)
    fig.subplots_adjust(top=0.90, hspace=0.55, wspace=0.3)

    out = os.path.join(PLOT_DIR, f'modelcmp_grid_{condition}.pdf')
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'  Saved: {out}')


# peak-layer summary heatmap

def plot_peak_summary(condition):
    rows = VALID_COMBOS
    cols = [(m, f) for m in MODELS for f in FEATURES]

    peak_layer = np.full((len(rows), len(cols)), np.nan)
    peak_val = np.full((len(rows), len(cols)), np.nan)

    for ri, (pert, corpus) in enumerate(rows):
        d = load_vinfo(pert, corpus)
        if d is None:
            continue
        for ci, (model, feature) in enumerate(cols):
            layers, vinfo = get_curve(d, model, condition, feature)
            if not layers:
                continue
            best_i = int(np.argmax(vinfo))
            peak_layer[ri, ci] = layers[best_i]
            peak_val[ri, ci] = vinfo[best_i]

    fig, ax = plt.subplots(figsize=(0.55 * len(cols) + 1.5, 0.4 * len(rows) + 1.0))
    cmap = LinearSegmentedColormap.from_list('layers', ['#f7f5f0', '#1f4e63'], N=13)
    im = ax.imshow(peak_layer, aspect='auto', cmap=cmap, vmin=0, vmax=12)

    for ri in range(len(rows)):
        for ci in range(len(cols)):
            if not np.isnan(peak_layer[ri, ci]):
                val = peak_val[ri, ci]
                lum = peak_layer[ri, ci] / 12
                color = 'white' if lum > 0.55 else 'black'
                ax.text(ci, ri, f'{int(peak_layer[ri, ci])}\n({val:.2f})',
                        ha='center', va='center', fontsize=4.5, color=color)
            else:
                ax.text(ci, ri, '\u2013', ha='center', va='center', fontsize=6, color='#999999')

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f'{MODEL_DISPLAY[m]}\n{FEATURE_DISPLAY[f]}' for m, f in cols],
                       rotation=60, ha='right', fontsize=5.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f'{PERT_DISPLAY[p]} ({CORPUS_DISPLAY[c][:4]})' for p, c in rows],
                       fontsize=6.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label('Peak layer', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax.set_title(f'Peak V-information layer ({CONDITION_DISPLAY[condition]})',
                 fontsize=9, fontweight='bold', pad=8)

    out = os.path.join(PLOT_DIR, f'peak_summary_{condition}.pdf')
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'  Saved: {out}')



# MAIN

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pert', default=None)
    ap.add_argument('--corpus', default=None)
    ap.add_argument('--only', choices=['curves_single', 'curves_combined',
                                        'modelcmp_single', 'modelcmp_grid', 'peak_summary'],
                    default=None)
    args = ap.parse_args()

    os.makedirs(PLOT_DIR, exist_ok=True)

    combos = VALID_COMBOS
    if args.pert and args.corpus:
        combos = [(args.pert, args.corpus)]

    for pert, corpus in combos:
        d = load_vinfo(pert, corpus)
        if d is None:
            print(f'{pert}/{corpus}: NO DATA, skipping')
            continue
        print(f'\n=== {pert}/{corpus} ===')

        if args.only in (None, 'curves_single'):
            for model in MODELS:
                plot_curves_single_model(pert, corpus, model, d)

        if args.only in (None, 'curves_combined'):
            plot_curves_combined(pert, corpus, d)

        if args.only in (None, 'modelcmp_single'):
            for condition in CONDITIONS:
                plot_model_comparison_single(pert, corpus, condition, d)

    if args.only in (None, 'modelcmp_grid'):
        print('\n=== Model comparison grids (all combos) ===')
        for condition in CONDITIONS:
            plot_model_comparison_grid(condition)

    if args.only in (None, 'peak_summary'):
        print('\n=== Peak-layer summaries (all combos) ===')
        for condition in CONDITIONS:
            plot_peak_summary(condition)

    print('\nAll plots done.')


if __name__ == '__main__':
    main()
