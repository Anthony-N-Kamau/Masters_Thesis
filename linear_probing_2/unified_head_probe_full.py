import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import spacy
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformer_lens import HookedTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

# Config 
SMOKE_TEST    = False
MAX_SENTENCES = 50 if SMOKE_TEST else 2000
TOP_K_LAYERS  = 2
N_LAYERS      = 12
N_HEADS       = 12
HEAD_DIM      = 64
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE     = os.path.expanduser('~/Impossible_projects')
DATA_DIR = os.path.join(BASE, 'datasets')
OUT_DIR  = os.path.join(BASE, 'linear_probing_2')

CORPUS_PATHS = {
    'bnc':       os.path.join(DATA_DIR, '2k_bnc_spoken.test'),
    'gutenberg': os.path.join(DATA_DIR, '2k_gutenberg.test'),
}

PERTURBATIONS = {
    'hop': {
        'kallini':          'mission-impossible-lms/word-hop-gpt2',
        'bnc_translator':   'amirhoseinMhmD/bnc_spoken-wordHop',
        'guten_translator': 'amirhoseinMhmD/gutenberg-wordHop',
        'corpora':          ['bnc', 'gutenberg'],
        'subfolder':        'hop',
    },
    'reverse': {
        'kallini':          'mission-impossible-lms/partial-reverse-gpt2',
        'bnc_translator':   'amirhoseinMhmD/bnc_spoken-partialReverse',
        'guten_translator': 'amirhoseinMhmD/gutenberg-partialReverse',
        'corpora':          ['bnc', 'gutenberg'],
        'subfolder':        'reverse',
    },
    'localw3': {
        'kallini':          'mission-impossible-lms/local-shuffle-w3-gpt2',
        'bnc_translator':   'amirhoseinMhmD/bnc_spoken-localShuffle-w3',
        'guten_translator': 'amirhoseinMhmD/gutenberg-localShuffle-w3',
        'corpora':          ['bnc', 'gutenberg'],
        'subfolder':        'shuffle',
    },
    'localw5': {
        'kallini':          'mission-impossible-lms/local-shuffle-w5-gpt2',
        'bnc_translator':   None,
        'guten_translator': 'amirhoseinMhmD/gutenberg-localShuffle-w5',
        'corpora':          ['gutenberg'],
        'subfolder':        'shuffle',
    },
    'fullshuffle': {
        'kallini':          'mission-impossible-lms/deterministic-shuffle-s57-gpt2',
        'bnc_translator':   None,
        'guten_translator': 'amirhoseinMhmD/gutenberg-fullshuffle-s57',
        'corpora':          ['gutenberg'],
        'subfolder':        'shuffle',
    },
}

FEATURES = ['pos', 'dep_relation', 'arc_direction', 'phrase_role']

FEATURE_LABELS = {
    'pos':           'POS (noun/verb)',
    'dep_relation':  'Dep Rel (nsubj/obj)',
    'arc_direction': 'Arc Dir (L/R)',
    'phrase_role':   'Phrase Role (subj/obj NP)',
}

MODEL_COLORS = {
    'Base GPT-2':         '#2166ac',
    'Kallini':            '#4dac26',
    'Recovery (BNC)':   '#d01c8b',
    'Recovery (Guten)': '#f1a340',
}

matplotlib.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['DejaVu Serif'],
    'font.size':         9,
    'axes.titlesize':    10,
    'axes.labelsize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linewidth':    0.5,
})

nlp       = spacy.load('en_core_web_sm')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# ── Data ─────────────────────────────────────────────────────────────────────
def load_sentences(path):
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)
            if len(sentences) >= MAX_SENTENCES:
                break
    return sentences

# Labels 
def get_word_labels(sentence, feature):
    doc = nlp(sentence)
    labels = []
    for token in doc:
        if feature == 'pos':
            l = 0 if token.pos_ == 'NOUN' else (1 if token.pos_ == 'VERB' else -1)
        elif feature == 'dep_relation':
            # Fix 2: keep -1 as third class instead of discarding
            l = 0 if token.dep_ == 'nsubj' else (1 if token.dep_ == 'obj' else 2)
        elif feature == 'arc_direction':
            l = 0 if token.i < token.head.i else (1 if token.i > token.head.i else -1)
        elif feature == 'phrase_role':
            # Fix 2: keep third class instead of discarding
            subtree_deps = {a.dep_ for a in token.subtree}
            l = 0 if 'nsubj' in subtree_deps else (1 if 'obj' in subtree_deps else 2)
        labels.append(l)
    return labels

def align_to_tokens(sentence, word_labels):
    enc      = tokenizer(sentence, return_tensors='pt')
    word_ids = enc.word_ids()
    tok_labels = []
    for wid in word_ids:
        if wid is None:
            tok_labels.append(-100)
        else:
            tok_labels.append(word_labels[wid] if wid < len(word_labels) else -100)
    return tok_labels

def build_label_lists(sentences, feature):
    return [align_to_tokens(s, get_word_labels(s, feature)) for s in sentences]

# Model 
def load_model(model_id):
    print(f'  Loading {model_id} ...')
    hf_model = GPT2LMHeadModel.from_pretrained(model_id)
    model    = HookedTransformer.from_pretrained('gpt2', hf_model=hf_model)
    model.eval()
    model.to(DEVICE)
    return model

# Activations 
def extract_hook_z(model, sentences):
    all_acts = []
    for sent in sentences:
        inputs     = tokenizer(sent, return_tensors='pt').to(DEVICE)
        layer_acts = {}
        def make_hook(layer):
            def hook_fn(value, hook):
                layer_acts[layer] = value[0].detach().cpu()
            return hook_fn
        hooks = [(f'blocks.{l}.attn.hook_z', make_hook(l)) for l in range(N_LAYERS)]
        with torch.no_grad():
            model.run_with_hooks(inputs['input_ids'], fwd_hooks=hooks)
        stacked = torch.stack([layer_acts[l] for l in range(N_LAYERS)], dim=1)
        all_acts.append(stacked)
    return all_acts

# Probing 
def balance_classes(X, y):
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    X_bal, y_bal = [], []
    for cls in classes:
        idx         = np.where(y == cls)[0]
        idx_sampled = resample(idx, n_samples=min_count, random_state=42, replace=False)
        X_bal.append(X[idx_sampled])
        y_bal.append(y[idx_sampled])
    return np.vstack(X_bal), np.concatenate(y_bal)

def train_probe(X, y):
    X, y  = balance_classes(X, y)
    clf   = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
    split = max(1, int(0.8 * len(X)))
    clf.fit(X[:split], y[:split])
    return accuracy_score(y[split:], clf.predict(X[split:]))

def run_head_probing(activations, label_lists):
    real_acc = np.full((N_LAYERS, N_HEADS), np.nan)
    ctrl_acc = np.full((N_LAYERS, N_HEADS), np.nan)
    for layer in range(N_LAYERS):
        for head in range(N_HEADS):
            X, y = [], []
            for acts, labels in zip(activations, label_lists):
                for t in range(acts.shape[0]):
                    lbl = labels[t] if t < len(labels) else -100
                    if lbl == -100 or lbl == -1:
                        continue
                    X.append(acts[t, layer, head, :].numpy())
                    y.append(lbl)
            # Fix 1: lower threshold from 20 to 5
            if len(set(y)) < 2 or len(X) < 5:
                continue
            X      = np.array(X)
            y      = np.array(y)
            y_ctrl = np.random.permutation(y)
            real_acc[layer, head] = train_probe(X, y)
            ctrl_acc[layer, head] = train_probe(X, y_ctrl)
    selectivity = real_acc - ctrl_acc
    return real_acc, ctrl_acc, selectivity

def get_important_layers(selectivity, top_k=TOP_K_LAYERS):
    layer_avg  = np.nanmean(selectivity, axis=1)
    top_layers = np.argsort(layer_avg)[::-1][:top_k]
    return sorted(top_layers.tolist()), layer_avg

# Plots 
def plot_accuracy_heatmap(results_by_model, important_layers_by_model, title, save_path):
    n_models = len(results_by_model)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4.0))
    if n_models == 1:
        axes = [axes]
    for ax, (model_name, (real_acc, _, _)) in zip(axes, results_by_model.items()):
        im = ax.imshow(real_acc, aspect='auto', cmap='YlOrRd',
                       vmin=0.4, vmax=1.0, origin='upper')
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
        ax.set_xlabel('Attention head', fontsize=9, labelpad=5)
        ax.set_ylabel('Layer', fontsize=9, labelpad=5)
        ax.set_xticks(range(N_HEADS))
        ax.set_xticklabels([h+1 for h in range(N_HEADS)], fontsize=7)
        ax.set_yticks(range(N_LAYERS))
        ax.set_yticklabels([l+1 for l in range(N_LAYERS)], fontsize=7)
        ax.set_title(model_name, fontsize=10, fontweight='bold', pad=8)
    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.02, x=0.45)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f'  Saved: {save_path}')


def plot_selectivity_heatmap(results_by_model, important_layers_by_model, title, save_path):
    n_models = len(results_by_model)
    all_sel  = [results_by_model[m][2] for m in results_by_model]
    vmax     = max(0.1, float(np.nanmax(np.abs(np.concatenate([s.flatten() for s in all_sel])))))
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4.0))
    if n_models == 1:
        axes = [axes]
    for ax, (model_name, (_, _, selectivity)) in zip(axes, results_by_model.items()):
        im = ax.imshow(selectivity, aspect='auto', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, origin='upper')
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
        ax.set_xlabel('Attention head', fontsize=9, labelpad=5)
        ax.set_ylabel('Layer', fontsize=9, labelpad=5)
        ax.set_xticks(range(N_HEADS))
        ax.set_xticklabels([h+1 for h in range(N_HEADS)], fontsize=7)
        ax.set_yticks(range(N_LAYERS))
        ax.set_yticklabels([l+1 for l in range(N_LAYERS)], fontsize=7)
        ax.set_title(model_name, fontsize=10, fontweight='bold', pad=8)
    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.02, x=0.45)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f'  Saved: {save_path}')


def plot_accuracy_curves(results_by_feature, imp_layers_by_feature, title, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.0), sharey=False, sharex=False)
    axes   = axes.flatten()
    layers = list(range(N_LAYERS))
    for i, feature in enumerate(FEATURES):
        ax               = axes[i]
        results_by_model = results_by_feature[feature]
        for model_name, (real_acc, _, _) in results_by_model.items():
            layer_avg = np.nanmean(real_acc, axis=1)
            ax.plot(layers, layer_avg, marker='o', markersize=3.5,
                    color=MODEL_COLORS[model_name], linewidth=1.4,
                    markeredgewidth=0.4, markeredgecolor='white')
        ax.set_title(FEATURE_LABELS[feature], fontsize=10, fontweight='bold', pad=6)
        ax.set_xticks(layers)
        ax.set_xticklabels([l+1 for l in layers], fontsize=7)
        ax.tick_params(labelsize=7)
        if i in (0, 2):
            ax.set_ylabel('Mean probe accuracy', fontsize=9, labelpad=5)
        ax.set_xlabel('Layer', fontsize=9, labelpad=5)
    handles = [
        plt.Line2D([0],[0], color=MODEL_COLORS[m], marker='o', markersize=4,
                   linewidth=1.4, markeredgewidth=0.4, markeredgecolor='white', label=m)
        for m in MODEL_COLORS
    ]
    fig.legend(handles=handles, fontsize=7, loc='center right',
               bbox_to_anchor=(1.13, 0.5), frameon=True,
               framealpha=0.9, edgecolor='#cccccc')
    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f'  Saved: {save_path}')


def plot_selectivity_curves(results_by_feature, imp_layers_by_feature, title, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.0), sharey=False, sharex=False)
    axes   = axes.flatten()
    layers = list(range(N_LAYERS))
    for i, feature in enumerate(FEATURES):
        ax               = axes[i]
        results_by_model = results_by_feature[feature]
        for model_name, (_, _, selectivity) in results_by_model.items():
            layer_avg = np.nanmean(selectivity, axis=1)
            ax.plot(layers, layer_avg, marker='s', markersize=3.5,
                    color=MODEL_COLORS[model_name], linewidth=1.4,
                    markeredgewidth=0.4, markeredgecolor='white')
        ax.axhline(0, color='#555555', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.set_title(FEATURE_LABELS[feature], fontsize=10, fontweight='bold', pad=6)
        ax.set_xticks(layers)
        ax.set_xticklabels([l+1 for l in layers], fontsize=7)
        ax.tick_params(labelsize=7)
        if i in (0, 2):
            ax.set_ylabel('Mean selectivity', fontsize=9, labelpad=5)
        ax.set_xlabel('Layer', fontsize=9, labelpad=5)
    handles = [
        plt.Line2D([0],[0], color=MODEL_COLORS[m], marker='s', markersize=4,
                   linewidth=1.4, markeredgewidth=0.4, markeredgecolor='white', label=m)
        for m in MODEL_COLORS
    ]
    fig.legend(handles=handles, fontsize=7, loc='center right',
               bbox_to_anchor=(1.13, 0.5), frameon=True,
               framealpha=0.9, edgecolor='#cccccc')
    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f'  Saved: {save_path}')


# Summary 
def print_summary(all_results, pert_label):
    print(f'\n{"="*70}')
    print(f'SUMMARY: Peak Selectivity | {pert_label}')
    print(f'{"="*70}')
    print(f'{"Model":<25} {"Feature":<25} {"Top Layers":<15} {"Peak Sel."}')
    print('-'*70)
    for model_name, features in all_results.items():
        for feature, (_, _, sel) in features.items():
            imp, lavg = get_important_layers(sel)
            peak      = round(float(np.nanmax(lavg)), 4) if not np.all(np.isnan(lavg)) else float('nan')
            print(f'{model_name:<25} {FEATURE_LABELS[feature]:<25} {str(imp):<15} {peak}')
    print('='*70)


# Main 
def run_perturbation(pert, cfg):
    print(f'\n{"="*60}')
    print(f'Perturbation: {pert}')
    print(f'{"="*60}')

    subfolder = cfg['subfolder']
    out_dir   = os.path.join(OUT_DIR, subfolder)
    plots_dir = os.path.join(out_dir, f'plots_{pert}')
    os.makedirs(plots_dir, exist_ok=True)

    MODELS = {'Base GPT-2': 'gpt2', 'Kallini': cfg['kallini']}
    for corpus in cfg['corpora']:
        key   = 'bnc_translator' if corpus == 'bnc' else 'guten_translator'
        label = 'Recovery (BNC)' if corpus == 'bnc' else 'Recovery (Guten)'
        if cfg.get(key):
            MODELS[label] = cfg[key]

    corpus_sentences = {c: load_sentences(CORPUS_PATHS[c]) for c in cfg['corpora']}

    model_corpus = {
        'Base GPT-2':         'gutenberg' if 'gutenberg' in cfg['corpora'] else cfg['corpora'][0],
        'Kallini':            'gutenberg' if 'gutenberg' in cfg['corpora'] else cfg['corpora'][0],
        'Recovery (BNC)':   'bnc',
        'Recovery (Guten)': 'gutenberg',
    }

    all_results = {}

    for model_name, model_id in MODELS.items():
        print(f'\n  Model: {model_name}')
        corpus    = model_corpus.get(model_name, cfg['corpora'][0])
        if corpus not in corpus_sentences:
            corpus = cfg['corpora'][0]
        sentences = corpus_sentences[corpus]
        model     = load_model(model_id)

        print(f'  Extracting hook_z ({len(sentences)} sentences, corpus={corpus})...')
        activations = extract_hook_z(model, sentences)

        all_results[model_name] = {}
        for feature in FEATURES:
            print(f'    Probing: {feature}')
            label_lists             = build_label_lists(sentences, feature)
            real_acc, ctrl_acc, sel = run_head_probing(activations, label_lists)
            all_results[model_name][feature] = (real_acc, ctrl_acc, sel)

            safe      = model_name.replace(' ','_').replace('(','').replace(')','').strip('_')
            json_path = os.path.join(out_dir, f'head_probe_{pert}_{corpus}_{safe}_{feature}.json')
            with open(json_path, 'w') as f:
                json.dump({
                    'real_acc':    real_acc.tolist(),
                    'ctrl_acc':    ctrl_acc.tolist(),
                    'selectivity': sel.tolist(),
                }, f)

        del model
        torch.cuda.empty_cache()

    pert_label            = pert.replace('_', ' ').title()
    results_by_feature    = {}
    imp_layers_by_feature = {}

    for feature in FEATURES:
        feat_label          = FEATURE_LABELS[feature]
        imp_layers_by_model = {}
        results_by_model    = {}

        for model_name in MODELS:
            if model_name not in all_results:
                continue
            _, _, sel                       = all_results[model_name][feature]
            imp, _                          = get_important_layers(sel)
            imp_layers_by_model[model_name] = imp
            results_by_model[model_name]    = all_results[model_name][feature]

        results_by_feature[feature]    = results_by_model
        imp_layers_by_feature[feature] = imp_layers_by_model

        plot_accuracy_heatmap(
            results_by_model, imp_layers_by_model,
            f'Probe Accuracy | {feat_label} | {pert_label}',
            os.path.join(plots_dir, f'A_acc_heatmap_{feature}.pdf'))

        plot_selectivity_heatmap(
            results_by_model, imp_layers_by_model,
            f'Selectivity | {feat_label} | {pert_label}',
            os.path.join(plots_dir, f'B_sel_heatmap_{feature}.pdf'))

    plot_accuracy_curves(
        results_by_feature, imp_layers_by_feature,
        f'Layer-Averaged Accuracy | {pert_label}',
        os.path.join(plots_dir, 'C_acc_curves_all_features.pdf'))

    plot_selectivity_curves(
        results_by_feature, imp_layers_by_feature,
        f'Layer-Averaged Selectivity | {pert_label}',
        os.path.join(plots_dir, 'D_sel_curves_all_features.pdf'))

    print_summary(all_results, pert_label)


if __name__ == '__main__':
    for pert, cfg in PERTURBATIONS.items():
        run_perturbation(pert, cfg)
    print('\nAll perturbations complete.')
