import json
import numpy as np
import os

# Config 
BASE    = os.path.expanduser('~/Impossible_projects/linear_probing_2')
TOP_K   = 2

PERTURBATIONS = {
    'hop':         'hop',
    'reverse':     'reverse',
    'localw3':     'shuffle',
    'localw5':     'shuffle',
    'fullshuffle': 'shuffle',
}

FEATURES = ['pos', 'dep_relation', 'arc_direction', 'phrase_role']

FEATURE_LABELS = {
    'pos':           'POS (noun/verb)',
    'dep_relation':  'Dep Rel (nsubj/obj)',
    'arc_direction': 'Arc Dir (L/R)',
    'phrase_role':   'Phrase Role (subj/obj NP)',
}

MODELS = [
    'Base_GPT-2',
    'Kallini',
    'Translator_BNC',
    'Translator_Guten',
]

MODEL_LABELS = {
    'Base_GPT-2':       'Base GPT-2',
    'Kallini':          'Kallini',
    'Translator_BNC':   'Translator (BNC)',
    'Translator_Guten': 'Translator (Guten)',
}

CORPUS_MAP = {
    'Base_GPT-2':       'gutenberg',
    'Kallini':          'gutenberg',
    'Translator_BNC':   'bnc',
    'Translator_Guten': 'gutenberg',
}

# Analysis 
def get_top_layers(selectivity, top_k=TOP_K):
    layer_avg  = np.nanmean(selectivity, axis=1)
    valid      = ~np.isnan(layer_avg)
    if not valid.any():
        return [], float('nan'), float('nan')
    top_layers = np.argsort(layer_avg)[::-1][:top_k]
    peak       = float(np.nanmax(layer_avg))
    mean       = float(np.nanmean(layer_avg[valid]))
    return [l+1 for l in sorted(top_layers.tolist())], round(peak, 4), round(mean, 4)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

# Main 
print(f'\n{"="*90}')
print(f'BEST LAYERS ANALYSIS — Previous Run (linear_probing_2)')
print(f'{"="*90}')
print(f'{"Perturbation":<14} {"Model":<22} {"Feature":<25} {"Top Layers":<15} {"Peak Sel.":<12} {"Mean Sel."}')
print('-'*90)

summary = {}  # for cross-perturbation comparison

for pert, subfolder in PERTURBATIONS.items():
    pert_dir = os.path.join(BASE, subfolder)
    summary[pert] = {}

    for model_key in MODELS:
        corpus     = CORPUS_MAP[model_key]
        model_label = MODEL_LABELS[model_key]
        summary[pert][model_label] = {}

        for feature in FEATURES:
            fname = f'head_probe_{pert}_{corpus}_{model_key}_{feature}.json'
            fpath = os.path.join(pert_dir, fname)

            if not os.path.exists(fpath):
                continue

            data        = load_json(fpath)
            selectivity = np.array(data['selectivity'])
            top_layers, peak, mean = get_top_layers(selectivity)

            summary[pert][model_label][feature] = {
                'top_layers': top_layers,
                'peak':       peak,
                'mean':       mean,
            }

            print(f'{pert:<14} {model_label:<22} {FEATURE_LABELS[feature]:<25} '
                  f'{str(top_layers):<15} {peak:<12} {mean}')

print('='*90)

# Cross-perturbation summary: best layers per feature
print(f'\n{"="*90}')
print(f'CROSS-PERTURBATION: Most Consistent Top Layers per Feature')
print(f'{"="*90}')

for feature in FEATURES:
    print(f'\n  Feature: {FEATURE_LABELS[feature]}')
    print(f'  {"Model":<22} {"hop":<12} {"reverse":<12} {"localw3":<12} {"localw5":<12} {"fullshuffle"}')
    print(f'  {"-"*78}')
    for model_label in MODEL_LABELS.values():
        row = f'  {model_label:<22}'
        for pert in PERTURBATIONS:
            if (pert in summary and
                model_label in summary[pert] and
                feature in summary[pert][model_label]):
                layers = summary[pert][model_label][feature]['top_layers']
                row += f'{str(layers):<12}'
            else:
                row += f'{"N/A":<12}'
        print(row)

# Best single head per feature per model 
print(f'\n{"="*90}')
print(f'BEST SINGLE HEAD (highest selectivity) per Feature per Model — hop perturbation')
print(f'{"="*90}')
print(f'{"Model":<22} {"Feature":<25} {"Layer":<8} {"Head":<8} {"Selectivity"}')
print('-'*70)

pert     = 'hop'
subfolder = PERTURBATIONS[pert]
pert_dir  = os.path.join(BASE, subfolder)

for model_key in MODELS:
    corpus      = CORPUS_MAP[model_key]
    model_label = MODEL_LABELS[model_key]
    for feature in FEATURES:
        fname = f'head_probe_{pert}_{corpus}_{model_key}_{feature}.json'
        fpath = os.path.join(pert_dir, fname)
        if not os.path.exists(fpath):
            continue
        data        = load_json(fpath)
        selectivity = np.array(data['selectivity'])
        flat_idx    = np.nanargmax(selectivity)
        layer, head = np.unravel_index(flat_idx, selectivity.shape)
        peak_val    = round(float(selectivity[layer, head]), 4)
        print(f'{model_label:<22} {FEATURE_LABELS[feature]:<25} {layer+1:<8} {head+1:<8} {peak_val}')

print('='*70)
print('\nAnalysis complete.')
