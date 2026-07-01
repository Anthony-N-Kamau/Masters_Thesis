import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ACL-style typography 
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
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'text.color': '#222222',
    'axes.labelcolor': '#222222',
})


_CMAP_COLORS = ['#f7f5f0', '#cfe0e8', '#8fb6c9', '#4a7f9e', '#1f4e63', '#0c2733']
RESTRAINED_CMAP = LinearSegmentedColormap.from_list('acl_seq', _CMAP_COLORS, N=256)

# Paths and constants 
BASE = os.path.expanduser('~/Impossible_projects')
OUT_BASE = os.path.join(BASE, 'Mechanistic Interpretability', 'Activation Patching')
PLOT_DIR = os.path.join(OUT_BASE, 'plots')

PERTS = ['hop', 'reverse', 'localw3', 'localw5', 'fullshuffle']
PERT_DISPLAY = {
    'hop': 'Hop', 'reverse': 'Reverse', 'localw3': 'Local w3',
    'localw5': 'Local w5', 'fullshuffle': 'Full Shuffle',
}

MODEL_DISPLAY = {
    'BaseGPT-2': 'Base GPT-2', 'Kallini': 'Kallini',
    'RecoveryGuten': 'Recovery (Gutenberg)',
    'RecoveryBNC': 'Recovery (BNC)',
}
# Muted, print-safe accent colors 
MODEL_COLORS = {
    'BaseGPT-2': '#3b6e8f', 'Kallini': '#7a9b57',
    'RecoveryGuten': '#c98a3e', 'RecoveryBNC': '#a85d7a',
}

MODELS_BY_CORPUS = {
    'gutenberg': ['BaseGPT-2', 'Kallini', 'RecoveryGuten'],
    'bnc':       ['BaseGPT-2', 'Kallini', 'RecoveryBNC'],
}

CORPORA = ['gutenberg', 'bnc']
CORPUS_DISPLAY = {'gutenberg': 'Gutenberg', 'bnc': 'BNC Spoken'}

N_LAYERS = 12
N_HEADS = 12
TOP_K = 5


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
def load_matrix(pert, corpus, model):
    path = os.path.join(OUT_BASE, pert, f'patching_{pert}_{corpus}_{model}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    if 'restoration' not in d:
        return None
    return np.array(d['restoration'])


def load_all(corpus):
    models = MODELS_BY_CORPUS.get(corpus, [])
    data = {}
    for pert in PERTS:
        data[pert] = {}
        for model in models:
            mat = load_matrix(pert, corpus, model)
            if mat is not None:
                data[pert][model] = mat
    return data


def clean_axes(ax):
    """Strip default matplotlib chrome to something quieter for print."""
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION 1: Faceted 12×12 Heatmap Matrix
# ═══════════════════════════════════════════════════════════════════════════
def plot_heatmap_grid(data, corpus):
    """5 rows (perturbations) x N columns (models). No per-cell numbers --
    at this many panels, numeric overlays are unreadable and just add noise.
    A shared colorbar carries the scale instead."""
    models = MODELS_BY_CORPUS.get(corpus, [])
    n_rows = len(PERTS)
    n_cols = len(models)

    # Use the TRUE global max across every panel in this corpus so the
    # shared colorbar never visually clips a real value (e.g. hop's 0.396
    # would be capped at 0.30 under a fixed ceiling).
    all_vals = []
    for pert in PERTS:
        for model in models:
            mat = data.get(pert, {}).get(model)
            if mat is not None:
                all_vals.append(np.nanmax(mat))
    vmax = max(0.05, max(all_vals)) if all_vals else 0.3

    # ACL double-column width (~7in) scaled to row count
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(2.0 * n_cols + 0.6, 1.55 * n_rows + 0.3),
                              squeeze=False)

    im = None
    for ri, pert in enumerate(PERTS):
        for ci, model in enumerate(models):
            ax = axes[ri, ci]
            mat = data.get(pert, {}).get(model)

            if mat is None:
                ax.text(0.5, 0.5, '\u2013', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='#999999')
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                continue

            im = ax.imshow(mat, aspect='auto', cmap=RESTRAINED_CMAP,
                           vmin=0.0, vmax=vmax, origin='upper')

            # All 12 ticks shown (not sparse) per the requirement that every
            # layer/head number be visible, even in the small multiples.
            # Smaller font needed since 12 labels must fit where 3 did before.
            ax.set_xticks(range(N_HEADS))
            ax.set_xticklabels(range(1, N_HEADS + 1), fontsize=4.8)
            ax.set_yticks(range(N_LAYERS))
            ax.set_yticklabels(range(1, N_LAYERS + 1), fontsize=4.8)
            ax.tick_params(length=1.5)

            if ri == 0:
                ax.set_title(MODEL_DISPLAY[model], fontsize=8, fontweight='bold',
                             color=MODEL_COLORS[model], pad=4)
            if ci == 0:
                ax.set_ylabel(PERT_DISPLAY[pert], fontsize=7.5, fontweight='bold')
            if ri == n_rows - 1:
                ax.set_xlabel('Head', fontsize=7)
            if ci == 0 and ri == n_rows - 1:
                ax.set_xlabel('Head', fontsize=7)

    # shared y-label for "Layer", aligned to the row of plots (not floating)
    fig.text(0.005, 0.50, 'Layer', va='center', rotation='vertical', fontsize=7.5)

    fig.suptitle(f'Head-level activation patching ({CORPUS_DISPLAY[corpus]})',
                 fontsize=9.5, fontweight='bold', y=0.995)
    fig.subplots_adjust(top=0.93, bottom=0.10, hspace=0.45, wspace=0.25)

    if im is not None:
        # Place the colorbar in its own reserved strip below the grid,
        # rather than stealing space from the subplot axes.
        cbar_ax = fig.add_axes([0.32, 0.025, 0.38, 0.018])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Restoration score', fontsize=7.5, labelpad=2)
        cbar.ax.tick_params(labelsize=6, length=2)
        cbar.outline.set_linewidth(0.5)

    out = os.path.join(PLOT_DIR, f'heatmap_grid_{corpus}.pdf')
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    plt.close()
    print(f'  Saved: {out}')


def plot_perturbation_heatmap(data, corpus, pert):
    """One perturbation, one row of panels (Base / Kallini / Recovery), each
    panel fully annotated with its restoration scores. Meant to stand alone
    as a single figure -- e.g. for main-text inclusion of one locality tier,
    or for a dedicated appendix figure per perturbation."""
    models = MODELS_BY_CORPUS.get(corpus, [])
    n_cols = len(models)

    pert_data = data.get(pert, {})
    all_vals = [np.nanmax(m) for m in pert_data.values()]
    vmax = max(0.05, max(all_vals)) if all_vals else 0.3

    # Larger panels than before -- real data has far more non-trivial cells
    # than the synthetic test data did, so each cell needs more room.
    fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.2), squeeze=False)
    axes = axes[0]

    im = None
    for ci, model in enumerate(models):
        ax = axes[ci]
        mat = pert_data.get(model)

        if mat is None:
            ax.text(0.5, 0.5, 'No model for this\nperturbation/corpus',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=7, color='#999999')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            continue

        im = ax.imshow(mat, aspect='auto', cmap=RESTRAINED_CMAP,
                       vmin=0.0, vmax=vmax, origin='upper')

        # Only annotate cells that are a meaningful fraction of THIS panel's
        # own max -- real matrices are noisy (many small nonzero cells), and
        # labeling all 144 makes the figure illegible. A per-panel relative
        # threshold keeps each panel's own standout cells legible without
        # drowning them in near-zero noise values.
        panel_max = np.nanmax(mat)
        label_threshold = max(0.01, panel_max * 0.35)
        for li in range(N_LAYERS):
            for hi in range(N_HEADS):
                val = mat[li, hi]
                if np.isnan(val) or val < label_threshold:
                    continue
                color = '#f5f5f5' if val > vmax * 0.55 else '#1a1a1a'
                ax.text(hi, li, f'{val:.2f}', ha='center', va='center',
                        fontsize=5.2, color=color)

        ax.set_xticks(range(N_HEADS))
        ax.set_xticklabels(range(1, N_HEADS + 1), fontsize=6)
        ax.set_yticks(range(N_LAYERS))
        ax.set_yticklabels(range(1, N_LAYERS + 1), fontsize=6)
        ax.set_title(MODEL_DISPLAY[model], fontsize=9, fontweight='bold',
                     color=MODEL_COLORS[model], pad=6)
        ax.set_xlabel('Attention head', fontsize=7.5)
        if ci == 0:
            ax.set_ylabel('Layer', fontsize=7.5)
        for s in ('top', 'right', 'left', 'bottom'):
            ax.spines[s].set_visible(True)
            ax.spines[s].set_linewidth(0.5)
        ax.tick_params(length=2)

    if im is not None:
        cbar = fig.colorbar(im, ax=list(axes), fraction=0.035, pad=0.025)
        cbar.set_label('Restoration score', fontsize=7.5)
        cbar.ax.tick_params(labelsize=6)
        cbar.outline.set_linewidth(0.5)

    fig.suptitle(f'{PERT_DISPLAY[pert]} \u2014 head-level activation patching '
                 f'({CORPUS_DISPLAY[corpus]})',
                 fontsize=9, fontweight='bold', y=1.06)

    out = os.path.join(PLOT_DIR, f'heatmap_{pert}_{corpus}.pdf')
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    plt.close()
    print(f'  Saved: {out}')


def plot_all_perturbation_heatmaps(data, corpus):
    """Generate the standalone per-perturbation figure for every perturbation
    that has at least one model's data in this corpus."""
    for pert in PERTS:
        if data.get(pert):
            plot_perturbation_heatmap(data, corpus, pert)


def plot_single_heatmap(pert, corpus, model):
    """A standalone, fully-annotated heatmap for one model/perturbation --
    use this when a value needs to be read precisely off the figure,
    e.g. as a supplementary or single-result figure."""
    mat = load_matrix(pert, corpus, model)
    if mat is None:
        print(f'  No data for {pert}/{corpus}/{model}')
        return

    fig, ax = plt.subplots(figsize=(3.3, 2.9))  # ACL single-column width
    vmax = max(0.3, np.nanmax(mat))
    im = ax.imshow(mat, aspect='auto', cmap=RESTRAINED_CMAP, vmin=0.0, vmax=vmax,
                   origin='upper')

    for li in range(N_LAYERS):
        for hi in range(N_HEADS):
            val = mat[li, hi]
            if np.isnan(val):
                continue
            color = '#f5f5f5' if val > vmax * 0.55 else '#1a1a1a'
            ax.text(hi, li, f'{val:.2f}', ha='center', va='center',
                    fontsize=4.3, color=color)

    ax.set_xticks(range(N_HEADS))
    ax.set_xticklabels(range(1, N_HEADS + 1))
    ax.set_yticks(range(N_LAYERS))
    ax.set_yticklabels(range(1, N_LAYERS + 1))
    ax.set_xlabel('Attention head')
    ax.set_ylabel('Layer')
    clean_axes(ax)
    for s in ('top', 'right', 'left', 'bottom'):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(0.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Restoration score', fontsize=7.5)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_linewidth(0.5)

    ax.set_title(f'{MODEL_DISPLAY[model]} \u2014 {PERT_DISPLAY[pert]} ({CORPUS_DISPLAY[corpus]})',
                 fontsize=8.5, fontweight='bold', pad=6)

    out = os.path.join(PLOT_DIR, f'heatmap_single_{pert}_{corpus}_{model}.pdf')
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    plt.close()
    print(f'  Saved: {out}')


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION 2: Top-K Causal Head Tracking Plots
# ═══════════════════════════════════════════════════════════════════════════
def plot_topk_tracking(data, corpus):
    models = MODELS_BY_CORPUS.get(corpus, [])
    if 'BaseGPT-2' not in models:
        return
    x_labels = [MODEL_DISPLAY[m] for m in models]

    # A restrained, distinguishable line palette (not matplotlib's tab10 defaults)
    line_colors = ['#1f4e63', '#c98a3e', '#7a9b57', '#a85d7a', '#6b6b6b']

    for pert in PERTS:
        base_mat = data.get(pert, {}).get('BaseGPT-2')
        if base_mat is None:
            continue

        flat = base_mat.flatten()
        top_indices = np.argsort(flat)[::-1][:TOP_K]
        top_heads = [np.unravel_index(idx, (N_LAYERS, N_HEADS)) for idx in top_indices]

        fig, ax = plt.subplots(figsize=(3.3, 2.6))  # single-column width

        all_scores = []
        for rank, (layer, head) in enumerate(top_heads):
            scores = []
            for model in models:
                mat = data.get(pert, {}).get(model)
                scores.append(mat[layer, head] if mat is not None else np.nan)

            label = f'L{layer + 1}H{head + 1}'
            ax.plot(range(len(models)), scores, 'o-',
                    color=line_colors[rank % len(line_colors)],
                    label=label, markersize=4, linewidth=1.1,
                    markeredgewidth=0)
            all_scores.extend([s for s in scores if not np.isnan(s)])

        y_max = max(0.15, max(all_scores) * 1.15) if all_scores else 0.5
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(x_labels, fontsize=6.5)
        ax.set_ylabel('Restoration score')
        ax.set_ylim(-0.02, y_max)
        ax.axhline(0, color='#bbbbbb', linewidth=0.6, zorder=0)
        clean_axes(ax)
        ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.0, 1.0),
                  fontsize=6, handlelength=1.4, labelspacing=0.4)
        ax.set_title(f'{PERT_DISPLAY[pert]} ({CORPUS_DISPLAY[corpus]})',
                     fontsize=8.5, fontweight='bold', pad=5)

        out = os.path.join(PLOT_DIR, f'topk_tracking_{pert}_{corpus}.pdf')
        fig.savefig(out, bbox_inches='tight')
        fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
        plt.close()
        print(f'  Saved: {out}')


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION 3: Layer-Wise Causal Density Violins
# ═══════════════════════════════════════════════════════════════════════════
def plot_layer_violins(data, corpus):
    models = MODELS_BY_CORPUS.get(corpus, [])
    n_rows = len(PERTS)
    n_cols = len(models)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(2.0 * n_cols + 0.5, 1.35 * n_rows + 0.3),
                              squeeze=False)

    for ri, pert in enumerate(PERTS):
        for ci, model in enumerate(models):
            ax = axes[ri, ci]
            mat = data.get(pert, {}).get(model)

            if mat is None:
                ax.text(0.5, 0.5, '\u2013', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='#999999')
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                continue

            layer_data = [mat[layer, :] for layer in range(N_LAYERS)]
            layer_data_clean, positions = [], []
            for li, ld in enumerate(layer_data):
                vals = ld[~np.isnan(ld)]
                if len(vals) > 1:
                    layer_data_clean.append(vals)
                    positions.append(li + 1)

            if layer_data_clean:
                vp = ax.violinplot(layer_data_clean, positions=positions,
                                   showmeans=False, showmedians=False,
                                   showextrema=False, widths=0.75)
                for body in vp['bodies']:
                    body.set_facecolor(MODEL_COLORS[model])
                    body.set_alpha(0.55)
                    body.set_edgecolor(MODEL_COLORS[model])
                    body.set_linewidth(0.4)

                means = [np.mean(v) for v in layer_data_clean]
                ax.plot(positions, means, '-', color='#222222', linewidth=0.7,
                        marker='o', markersize=1.6, zorder=3)

            ax.set_xlim(0.5, N_LAYERS + 0.5)
            # All 12 layer numbers shown (not sparse); smaller font to fit.
            ax.set_xticks(range(1, N_LAYERS + 1))
            ax.set_xticklabels(range(1, N_LAYERS + 1), fontsize=5.0)
            ax.axhline(0, color='#cccccc', linewidth=0.5, zorder=0)
            clean_axes(ax)

            if ri == 0:
                ax.set_title(MODEL_DISPLAY[model], fontsize=8, fontweight='bold',
                             color=MODEL_COLORS[model], pad=4)
            if ci == 0:
                ax.set_ylabel(PERT_DISPLAY[pert], fontsize=7.5, fontweight='bold')
            if ri == n_rows - 1:
                ax.set_xlabel('Layer', fontsize=7)

    fig.suptitle(f'Layer-wise restoration score distribution ({CORPUS_DISPLAY[corpus]})',
                 fontsize=9.5, fontweight='bold', y=0.995)
    fig.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)

    out = os.path.join(PLOT_DIR, f'violin_grid_{corpus}.pdf')
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    plt.close()
    print(f'  Saved: {out}')


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════
def print_summary_table(data, corpus):
    models = MODELS_BY_CORPUS.get(corpus, [])
    print(f'\n  Best-head summary ({CORPUS_DISPLAY[corpus]}, 1-indexed):')
    print(f'  {"Perturbation":<14} {"Model":<22} {"Best Head":<10} {"RS":<8} {"Mean RS"}')
    print('  ' + '-' * 66)
    for pert in PERTS:
        for model in models:
            mat = data.get(pert, {}).get(model)
            if mat is None:
                continue
            li, hi = np.unravel_index(np.nanargmax(mat), mat.shape)
            print(f'  {PERT_DISPLAY[pert]:<14} {MODEL_DISPLAY[model]:<22} '
                  f'L{li+1}H{hi+1:<8} {mat[li,hi]:<8.4f} {np.nanmean(mat):.4f}')


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', choices=['gutenberg', 'bnc', 'both'], default='both')
    ap.add_argument('--only', choices=['heatmap', 'perpert', 'topk', 'violin'],
                    default=None,
                    help="'heatmap'=faceted grid, 'perpert'=one figure per "
                         "perturbation, 'topk'=head tracking, 'violin'=layer violins")
    ap.add_argument('--single', nargs=3, metavar=('PERT', 'CORPUS', 'MODEL'),
                    help='produce one fully-annotated standalone heatmap for one model')
    args = ap.parse_args()

    os.makedirs(PLOT_DIR, exist_ok=True)

    if args.single:
        pert, corpus, model = args.single
        plot_single_heatmap(pert, corpus, model)
        return

    corpora = CORPORA if args.corpus == 'both' else [args.corpus]
    for corpus in corpora:
        print(f'\n=== {CORPUS_DISPLAY[corpus]} ===')
        data = load_all(corpus)
        for pert in PERTS:
            found = list(data.get(pert, {}).keys())
            print(f'  {pert}: {", ".join(MODEL_DISPLAY[m] for m in found) if found else "NO DATA"}')
        print_summary_table(data, corpus)

        if args.only in (None, 'heatmap'):
            print('\n  Generating heatmap grid (all perturbations, faceted)...')
            plot_heatmap_grid(data, corpus)
        if args.only in (None, 'perpert'):
            print('\n  Generating per-perturbation standalone heatmaps...')
            plot_all_perturbation_heatmaps(data, corpus)
        if args.only in (None, 'topk'):
            print('\n  Generating top-K tracking plots...')
            plot_topk_tracking(data, corpus)
        if args.only in (None, 'violin'):
            print('\n  Generating violin plots...')
            plot_layer_violins(data, corpus)

    print('\nAll plots done.')


if __name__ == '__main__':
    main()
