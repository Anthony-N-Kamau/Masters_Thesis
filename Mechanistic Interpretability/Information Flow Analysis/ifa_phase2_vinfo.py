import os
import json
import argparse
import random

import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AddedToken
from transformer_lens import HookedTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss

BASE = os.path.expanduser('~/Impossible_projects')
IFA_BASE = os.path.join(BASE, 'Mechanistic Interpretability', 'Information Flow Analysis')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_LAYERS_TOTAL = 13   # 0 = embeddings, 1..12 = block outputs
MIN_TEST_TOKENS = 20
TRAIN_FRACTION = 0.8
SPLIT_SEED = 57         # fixed seed -> reproducible global split

FEATURES = ['pos', 'dep_rel', 'arc_dir', 'phrase_role']
CONDITIONS = ['original', 'impossible', 'recovered']

MARK_SG, MARK_PL, MARK_REV = '\U0001F142', '\U0001F13F', '\U0001F141'
PERT_MARKERS = {
    'hop': [MARK_SG, MARK_PL], 'reverse': [MARK_REV],
    'localw3': [], 'localw5': [], 'fullshuffle': [],
}

MODEL_IDS = {
    'Base_GPT-2': {'hop': 'gpt2', 'reverse': 'gpt2', 'localw3': 'gpt2',
                   'localw5': 'gpt2', 'fullshuffle': 'gpt2'},
    'Kallini':    {'hop': 'mission-impossible-lms/word-hop-gpt2',
                   'reverse': 'mission-impossible-lms/partial-reverse-gpt2',
                   'localw3': 'mission-impossible-lms/local-shuffle-w3-gpt2',
                   'localw5': 'mission-impossible-lms/local-shuffle-w5-gpt2',
                   'fullshuffle': 'mission-impossible-lms/deterministic-shuffle-s57-gpt2'},
}
RECOVERY_MODEL_IDS = {
    'gutenberg': {
        'hop': 'amirhoseinMhmD/gutenberg-wordHop',
        'reverse': 'amirhoseinMhmD/gutenberg-partialReverse',
        'localw3': 'amirhoseinMhmD/gutenberg-localShuffle-w3',
        'localw5': 'amirhoseinMhmD/gutenberg-localShuffle-w5',
        'fullshuffle': 'amirhoseinMhmD/gutenberg-fullshuffle-s57',
    },
    'bnc': {
        'hop': 'amirhoseinMhmD/bnc_spoken-wordHop',
        'reverse': 'amirhoseinMhmD/bnc_spoken-partialReverse',
        'localw3': 'amirhoseinMhmD/bnc_spoken-localShuffle-w3',
        'fullshuffle': 'amirhoseinMhmD/bnc_spoken-fullShuffle-s57',
    },
}
MODEL_LABELS = {'Base_GPT-2': 'Base GPT-2', 'Kallini': 'Kallini', 'Recovery': 'Recovery'}


def make_tokenizer(markers):
    tok = GPT2Tokenizer.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    if markers:
        tok.add_tokens([AddedToken(m, lstrip=True, rstrip=False) for m in markers])
    return tok


def load_model(model_id, tokenizer):
    hf = GPT2LMHeadModel.from_pretrained(model_id)
    hf.resize_token_embeddings(len(tokenizer))
    m = HookedTransformer.from_pretrained('gpt2', hf_model=hf)
    m.eval()
    m.to(DEVICE)
    return m


def extract_layer_activations(model, tokenizer, prompt_text):
    """One forward pass. Returns a dict layer_idx -> (seq_len, d_model) numpy array,
    for layer_idx in 0..12 (0 = embeddings, i = output of block i-1, i.e.
    resid_post of block (i-1))."""
    ids = tokenizer(prompt_text, return_tensors='pt')['input_ids'].to(DEVICE)
    with torch.no_grad():
        _, cache = model.run_with_cache(ids)
    layers = {}
    # Layer 0: embeddings (token + positional), before any block runs.
    embed = cache['hook_embed'] + cache['hook_pos_embed']
    layers[0] = embed[0].cpu().numpy()
    # Layers 1..12: resid_post of blocks 0..11
    n_blocks = model.cfg.n_layers
    for b in range(n_blocks):
        layers[b + 1] = cache[f'blocks.{b}.hook_resid_post'][0].cpu().numpy()
    return layers, ids.shape[1]


def get_label_value(word_record, feature):
    return word_record.get(feature)  # None if not applicable to this word


def compute_baseline_loss(feature, all_labels_for_feature):
    """Train a probe on an empty/zero-vector dataset (dimension 1, all zeros)
    -- it can only ever learn the marginal label distribution. Trained ONCE
    globally per feature, since it never depends on layer/condition/model."""
    le = LabelEncoder()
    y = le.fit_transform(all_labels_for_feature)
    if len(set(y)) < 2:
        return None  # degenerate: feature has only one class in this dataset
    X_zero = np.zeros((len(y), 1))
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_zero, y)
    probs = clf.predict_proba(X_zero)
    return log_loss(y, probs, labels=list(range(len(le.classes_))))


def train_layer_probe(X_train, y_train, X_test, y_test, n_classes):
    """Simple linear classifier (logistic regression), with standardization
    and L2 regularization. Returns test cross-entropy.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=2000, C=0.1)
    clf.fit(X_train_s, y_train)
    probs = clf.predict_proba(X_test_s)
    # predict_proba may omit classes unseen in training; align columns
    if probs.shape[1] != n_classes:
        full_probs = np.full((probs.shape[0], n_classes), 1e-9)
        for i, c in enumerate(clf.classes_):
            full_probs[:, c] = probs[:, i]
        full_probs /= full_probs.sum(axis=1, keepdims=True)
        probs = full_probs
    return log_loss(y_test, probs, labels=list(range(n_classes)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pert', required=True, choices=list(PERT_MARKERS.keys()))
    ap.add_argument('--corpus', choices=['gutenberg', 'bnc'], default='gutenberg')
    ap.add_argument('--validate', action='store_true',
                    help='verbose single-combination validation mode')
    ap.add_argument('--textprep-override', default=None,
                    help='full path to an alternate ifa_textprep JSON file '
                         '(e.g. the expanded fullshuffle/Gutenberg run). '
                         'Output filename gets a matching suffix so it does '
                         'not overwrite the default result.')
    args = ap.parse_args()

    if args.pert not in RECOVERY_MODEL_IDS[args.corpus]:
        print(f'No Recovery model for {args.pert} on {args.corpus}. Aborting.')
        return

    markers = PERT_MARKERS[args.pert]
    tokenizer = make_tokenizer(markers)

    if args.textprep_override:
        textprep_path = args.textprep_override
        print(f'Using OVERRIDE textprep file: {textprep_path}')
    else:
        textprep_path = os.path.join(IFA_BASE, args.pert,
                                      f'ifa_textprep_{args.pert}_{args.corpus}.json')
    textprep = json.load(open(textprep_path, encoding='utf-8'))
    print(f'Loaded textprep: {textprep["n_sentences"]} sentences total')

    active_sentences = [s for s in textprep['sentences'] if s['classification'] == 'active']
    print(f'Active sentences (used for main curves): {len(active_sentences)}')
    if len(active_sentences) < 10:
        print('WARNING: very few active sentences -- curves may be unreliable.')

    # --- ONE global 80/20 sentence-level split, reused everywhere ---
    rng = random.Random(SPLIT_SEED)
    indices = list(range(len(active_sentences)))
    rng.shuffle(indices)
    n_train = int(len(indices) * TRAIN_FRACTION)
    train_idx = set(indices[:n_train])
    test_idx = set(indices[n_train:])
    print(f'Global split: {len(train_idx)} train sentences, {len(test_idx)} test sentences')

    # model IDs for this perturbation/corpus 
    model_ids = {
        'Base_GPT-2': MODEL_IDS['Base_GPT-2'][args.pert],
        'Kallini': MODEL_IDS['Kallini'][args.pert],
        'Recovery': RECOVERY_MODEL_IDS[args.corpus][args.pert],
    }

    # baseline loss per feature, computed ONCE globally from all active
    #     sentences' word_records (label distribution only, no activations)
    all_labels_by_feature = {f: [] for f in FEATURES}
    for s in active_sentences:
        for wr in s['word_records']:
            for f in FEATURES:
                val = get_label_value(wr, f)
                if val is not None:
                    all_labels_by_feature[f].append(val)

    baseline_loss = {}
    for f in FEATURES:
        bl = compute_baseline_loss(f, all_labels_by_feature[f])
        baseline_loss[f] = bl
        print(f'  baseline_loss[{f}] = {bl}  (n={len(all_labels_by_feature[f])})')

    curves = {}

    for model_key, model_id in model_ids.items():
        print(f'\n=== Model: {MODEL_LABELS[model_key]} ({model_id}) ===')
        model = load_model(model_id, tokenizer)
        curves[model_key] = {}

        for condition in CONDITIONS:
            print(f'  Condition: {condition}')
            # --- collect, per layer, the (activation, label) pairs for each
            #     feature, tagged train/test by sentence index ---
            # layer -> feature -> {'train': [(vec, label)], 'test': [(vec, label)]}
            layer_feat_data = {l: {f: {'train': [], 'test': []} for f in FEATURES}
                               for l in range(N_LAYERS_TOTAL)}

            for si, s in enumerate(active_sentences):
                prompt_key = f'prompt_{condition}'
                colon_key = f'colon_idx_{condition}'
                prompt_text = s[prompt_key]
                colon_idx = s[colon_key]
                if colon_idx is None:
                    continue

                layers, seq_len = extract_layer_activations(model, tokenizer, prompt_text)

                split = 'train' if si in train_idx else 'test'

                for wr in s['word_records']:
                    offset_key = ('offset_from_colon' if condition != 'recovered'
                                  else 'recovered_offset_from_colon')
                    offset = wr.get(offset_key)
                    if offset is None:
                        continue
                    pos_in_seq = colon_idx + offset
                    if pos_in_seq >= seq_len or pos_in_seq < 0:
                        continue  # out of bounds (e.g. recovered text shorter than offset)

                    for f in FEATURES:
                        val = get_label_value(wr, f)
                        if val is None:
                            continue
                        for layer_idx in range(N_LAYERS_TOTAL):
                            vec = layers[layer_idx][pos_in_seq]
                            layer_feat_data[layer_idx][f][split].append((vec, val))

            # (model is reused across all 3 conditions, only deleted after the loop)

            # train probes per layer per feature, apply min-token threshold 
            curves[model_key][condition] = {}
            for f in FEATURES:
                if baseline_loss[f] is None:
                    continue
                layer_list, vinfo_list, ntest_list = [], [], []
                for layer_idx in range(N_LAYERS_TOTAL):
                    train_data = layer_feat_data[layer_idx][f]['train']
                    test_data = layer_feat_data[layer_idx][f]['test']
                    if len(test_data) < MIN_TEST_TOKENS or len(train_data) < MIN_TEST_TOKENS:
                        continue  # DROP this point entirely (per spec)

                    le = LabelEncoder()
                    all_vals = [v for _, v in train_data] + [v for _, v in test_data]
                    le.fit(all_vals)
                    if len(le.classes_) < 2:
                        continue

                    X_train = np.array([v for v, _ in train_data])
                    y_train = le.transform([v for _, v in train_data])
                    X_test = np.array([v for v, _ in test_data])
                    y_test = le.transform([v for _, v in test_data])

                    try:
                        l_layer = train_layer_probe(X_train, y_train, X_test, y_test,
                                                     len(le.classes_))
                    except Exception as e:
                        print(f'    [skip] layer={layer_idx} feature={f}: {e}')
                        continue

                    v_info = baseline_loss[f] - l_layer
                    layer_list.append(layer_idx)
                    vinfo_list.append(float(v_info))
                    ntest_list.append(len(test_data))

                curves[model_key][condition][f] = {
                    'layers': layer_list, 'v_information': vinfo_list,
                    'n_test_tokens': ntest_list,
                }
                if layer_list:
                    print(f'    {f}: {len(layer_list)}/{N_LAYERS_TOTAL} layers kept, '
                          f'peak V-info={max(vinfo_list):.4f} at layer {layer_list[int(np.argmax(vinfo_list))]}')
                else:
                    print(f'    {f}: NO layers passed the {MIN_TEST_TOKENS}-token threshold')

        del model
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    out_dir = os.path.join(IFA_BASE, args.pert)
    suffix = '_expanded' if args.textprep_override else ''
    out_path = os.path.join(out_dir, f'ifa_vinfo_{args.pert}_{args.corpus}{suffix}.json')
    payload = {
        'perturbation': args.pert, 'corpus': args.corpus,
        'n_active_sentences': len(active_sentences),
        'n_train_sentences': len(train_idx), 'n_test_sentences': len(test_idx),
        'min_test_tokens': MIN_TEST_TOKENS,
        'baseline_loss': baseline_loss,
        'curves': curves,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
