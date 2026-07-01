import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import warnings
warnings.filterwarnings('ignore')

gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Config 
PERT = sys.argv[1] if len(sys.argv) > 1 else 'hop'
BASE    = os.path.expanduser('~/Impossible_projects')
OUT_DIR = os.path.join(BASE, 'BLiMP_full', PERT)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'plots'), exist_ok=True)

N_SAMPLES_PER_PARADIGM = 1000   # full paradigm size; override via 2nd CLI arg for budget control
if len(sys.argv) > 2:
    N_SAMPLES_PER_PARADIGM = int(sys.argv[2])

MODELS_PER_PERT = {
    'hop': {
        'Base GPT-2':           'gpt2',
        'Kallini':               'mission-impossible-lms/word-hop-gpt2',
        'Recovery (BNC)':        'amirhoseinMhmD/bnc_spoken-wordHop',
        'Recovery (Gutenberg)':  'amirhoseinMhmD/gutenberg-wordHop',
    },
    'reverse': {
        'Base GPT-2':           'gpt2',
        'Kallini':               'mission-impossible-lms/partial-reverse-gpt2',
        'Recovery (BNC)':        'amirhoseinMhmD/bnc_spoken-partialReverse',
        'Recovery (Gutenberg)':  'amirhoseinMhmD/gutenberg-partialReverse',
    },
    'localw3': {
        'Base GPT-2':           'gpt2',
        'Kallini':               'mission-impossible-lms/local-shuffle-w3-gpt2',
        'Recovery (BNC)':        'amirhoseinMhmD/bnc_spoken-localShuffle-w3',
        'Recovery (Gutenberg)':  'amirhoseinMhmD/gutenberg-localShuffle-w3',
    },
    'localw5': {
        'Base GPT-2':           'gpt2',
        'Kallini':               'mission-impossible-lms/local-shuffle-w5-gpt2',
        'Recovery (Gutenberg)':  'amirhoseinMhmD/gutenberg-localShuffle-w5',
    },
    'fullshuffle': {
        'Base GPT-2':           'gpt2',
        'Kallini':               'mission-impossible-lms/deterministic-shuffle-s57-gpt2',
        'Recovery (Gutenberg)':  'amirhoseinMhmD/gutenberg-fullshuffle-s57',
    },
}
MODELS = MODELS_PER_PERT[PERT]

MODEL_COLORS = {
    'Base GPT-2':           '#2166ac',
    'Kallini':               '#4dac26',
    'Recovery (BNC)':        '#d01c8b',
    'Recovery (Gutenberg)':  '#f1a340',
}

# Grouped here by the paper's 12 PHENOMENA categories

BLIMP_PHENOMENA = {
    'anaphor_agreement': [
        'anaphor_number_agreement',
    ],
    'subject_verb_agreement': [
        'regular_plural_subject_verb_agreement_1',
    ],
    'determiner_noun_agreement': [
        'determiner_noun_agreement_1',
    ],
    'argument_structure': [
        'transitive',
    ],
    'binding': [
        'principle_A_domain_1',
    ],
    'filler_gap': [
        'wh_questions_subject_gap',
    ],
    'island_effects': [
        'adjunct_island',
    ],
}

# Flatten to single list (67 total) — this is what actually gets loaded
ALL_PARADIGM_UIDS = [uid for group in BLIMP_PHENOMENA.values() for uid in group]
assert len(ALL_PARADIGM_UIDS) == 7, f'Expected 7 paradigms, got {len(ALL_PARADIGM_UIDS)}'

# phenomenon, used for phenomenon-level reporting
UID_TO_PHENOMENON = {
    uid: phenom for phenom, uids in BLIMP_PHENOMENA.items() for uid in uids
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

# Load full BLiMP suite 
def load_full_blimp():
    print(f'Loading all {len(ALL_PARADIGM_UIDS)} BLiMP paradigms...')
    blimp_data = {}
    failed = []
    for uid in ALL_PARADIGM_UIDS:
        try:
            ds = load_dataset('nyu-mll/blimp', uid, split='train',
                              trust_remote_code=True)
            blimp_data[uid] = ds
            print(f'  Loaded {uid}: {len(ds)} pairs '
                  f'(field={ds[0]["field"]}, phenomenon={UID_TO_PHENOMENON[uid]})')
        except Exception as e:
            print(f'  FAILED {uid}: {e}')
            failed.append(uid)
    if failed:
        print(f'\n  WARNING: {len(failed)}/{len(ALL_PARADIGM_UIDS)} paradigms failed to load: {failed}')
    print(f'\n  Successfully loaded {len(blimp_data)}/{len(ALL_PARADIGM_UIDS)} paradigms.')
    return blimp_data

# Scoring
def score_sentence(model, sentence):
    inputs = gpt2_tokenizer(sentence, return_tensors='pt')
    if inputs['input_ids'].shape[1] == 0:
        return None
    with torch.no_grad():
        outputs  = model(**inputs, labels=inputs['input_ids'])
        log_prob = -outputs.loss.item()
    return log_prob

def evaluate_paradigm(model, dataset, n_samples=N_SAMPLES_PER_PARADIGM):
    correct = 0
    total   = 0
    n       = min(n_samples, len(dataset))
    for i in range(n):
        good_sent = dataset[i]['sentence_good']
        bad_sent  = dataset[i]['sentence_bad']
        try:
            good_score = score_sentence(model, good_sent)
            bad_score  = score_sentence(model, bad_sent)
            if good_score is None or bad_score is None:
                continue
            if good_score > bad_score:
                correct += 1
            total += 1
        except Exception:
            continue
    return correct / total if total > 0 else float('nan')

def evaluate_model_full(model_id, model_name, blimp_data):
    print(f'\n  Loading {model_name} ({model_id})...')
    try:
        hf_model = GPT2LMHeadModel.from_pretrained(model_id)
        hf_model.eval()
    except Exception as e:
        print(f'  Failed to load: {e}')
        return {}

    per_paradigm  = {}
    per_field     = {}
    per_phenomenon = {}

    for uid, ds in blimp_data.items():
        field      = ds[0]['field']
        phenomenon = UID_TO_PHENOMENON[uid]
        print(f'    {uid} (field={field}, phenomenon={phenomenon})...')
        acc = evaluate_paradigm(hf_model, ds)
        per_paradigm[uid] = {
            'field': field,
            'phenomenon': phenomenon,
            'accuracy': round(acc, 4),
        }
        per_field.setdefault(field, []).append(acc)
        per_phenomenon.setdefault(phenomenon, []).append(acc)
        print(f'      Accuracy: {acc:.4f}')

    field_means = {f: round(float(np.nanmean(a)), 4) for f, a in per_field.items()}
    phenomenon_means = {p: round(float(np.nanmean(a)), 4) for p, a in per_phenomenon.items()}
    overall_mean = round(float(np.nanmean(
        [v['accuracy'] for v in per_paradigm.values()])), 4)

    del hf_model
    return {
        'per_paradigm':       per_paradigm,
        'field_means':        field_means,
        'phenomenon_means':   phenomenon_means,
        'overall_mean':       overall_mean,
        'n_paradigms':        len(per_paradigm),
        'n_samples_per_paradigm': N_SAMPLES_PER_PARADIGM,
    }

# Plots 
def save_fig(fig, path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f'  Saved: {path}')

def pert_display_name(pert):
    return pert.replace('localw', 'Local-w').replace('fullshuffle', 'Full Shuffle').title()

def plot_overall_bar(all_results, pert, plots_dir):
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
    n_para = next(iter(all_results.values()))['n_paradigms']
    ax.set_title(f'BLiMP Suite ({n_para} paradigms) | {pert_display_name(pert)}',
                fontsize=10, fontweight='bold')
    ax.legend(fontsize=7)
    save_fig(fig, os.path.join(plots_dir, f'overall_blimp_{pert}.pdf'))

def plot_field_heatmap(all_results, pert, plots_dir):
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
    ax.set_title(f'BLiMP Accuracy by Field | {pert_display_name(pert)}', fontsize=10, fontweight='bold')
    save_fig(fig, os.path.join(plots_dir, f'field_heatmap_{pert}.pdf'))

def plot_phenomenon_heatmap(all_results, pert, plots_dir):
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
    ax.set_title(f'BLiMP Accuracy by Phenomenon (12 categories) | {pert_display_name(pert)}',
                fontsize=10, fontweight='bold')
    save_fig(fig, os.path.join(plots_dir, f'phenomenon_heatmap_{pert}.pdf'))

# Main 
if __name__ == '__main__':
    print(f'\n{"="*70}')
    print(f'FULL BLiMP EVALUATION (67 paradigms) | Perturbation: {PERT.upper()} '
          f'| N={N_SAMPLES_PER_PARADIGM}/paradigm')
    print(f'{"="*70}')

    blimp_data = load_full_blimp()

    all_results = {}
    for model_name, model_id in MODELS.items():
        results = evaluate_model_full(model_id, model_name, blimp_data)
        if results:
            all_results[model_name] = results
            safe_name = model_name.replace(" ", "").replace("(", "").replace(")", "")
            out_path = os.path.join(OUT_DIR, f'blimp_full_{safe_name}.json')
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f'  Saved: {out_path}')

    plots_dir = os.path.join(OUT_DIR, 'plots')
    if all_results:
        plot_overall_bar(all_results, PERT, plots_dir)
        plot_field_heatmap(all_results, PERT, plots_dir)
        plot_phenomenon_heatmap(all_results, PERT, plots_dir)

    print(f'\n{"="*70}')
    print(f'SUMMARY | {PERT.upper()}')
    print(f'{"="*70}')
    for model_name in MODELS:
        if model_name not in all_results:
            continue
        r = all_results[model_name]
        print(f'\n{model_name}: overall = {r["overall_mean"]:.4f} '
              f'({r["n_paradigms"]} paradigms)')
        for field, val in sorted(r['field_means'].items()):
            print(f'    {field:<20} {val:.4f}')
    print('\n' + '='*70)
    print('Done.')
