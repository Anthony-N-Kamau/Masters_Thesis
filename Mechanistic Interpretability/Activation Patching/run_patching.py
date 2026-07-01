import os
import sys
import json
import argparse
import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AddedToken
from transformer_lens import HookedTransformer

N_LAYERS = 12
N_HEADS  = 12
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
CLIP_LO, CLIP_HI = -1.0, 2.0

BASE     = os.path.expanduser('~/Impossible_projects')
OUT_BASE = os.path.join(BASE, 'Mechanistic Interpretability', 'Activation Patching')

# markers per perturbation family (from utils.py)
MARK_SG, MARK_PL, MARK_REV = '🅂', '🄿', '🅁'
PERT_MARKERS = {
    'hop':         [MARK_SG, MARK_PL],
    'reverse':     [MARK_REV],
    'localw3':     [],
    'localw5':     [],
    'fullshuffle': [],
}

MODEL_IDS = {
    'Base_GPT-2':    {'hop': 'gpt2', 'reverse': 'gpt2', 'localw3': 'gpt2',
                      'localw5': 'gpt2', 'fullshuffle': 'gpt2'},
    'Kallini':       {'hop': 'mission-impossible-lms/word-hop-gpt2',
                      'reverse': 'mission-impossible-lms/partial-reverse-gpt2',
                      'localw3': 'mission-impossible-lms/local-shuffle-w3-gpt2',
                      'localw5': 'mission-impossible-lms/local-shuffle-w5-gpt2',
                      'fullshuffle': 'mission-impossible-lms/deterministic-shuffle-s57-gpt2'},
    'Recovery_Guten':{'hop': 'amirhoseinMhmD/gutenberg-wordHop',
                      'reverse': 'amirhoseinMhmD/gutenberg-partialReverse',
                      'localw3': 'amirhoseinMhmD/gutenberg-localShuffle-w3',
                      'localw5': 'amirhoseinMhmD/gutenberg-localShuffle-w5',
                      'fullshuffle': 'amirhoseinMhmD/gutenberg-fullshuffle-s57'},
    'Recovery_BNC':  {'hop': 'amirhoseinMhmD/bnc_spoken-wordHop',
                      'reverse': 'amirhoseinMhmD/bnc_spoken-partialReverse',
                      'localw3': 'amirhoseinMhmD/bnc_spoken-localShuffle-w3',
                      'fullshuffle': 'amirhoseinMhmD/bnc_spoken-fullShuffle-s57'},
}
MODEL_LABELS = {'Base_GPT-2': 'Base GPT-2', 'Kallini': 'Kallini',
                'Recovery_Guten': 'Recovery (Gutenberg)',
                'Recovery_BNC': 'Recovery (BNC)'}

# Each corpus is evaluated ONLY with models trained on that corpus
# (Base GPT-2 and Kallini are general; Recovery models are corpus-specific).
# No BNC localw5 Recovery model exists (BNC sentences too short for w5 windows).
MODELS_BY_CORPUS = {
    'gutenberg': ['Base_GPT-2', 'Kallini', 'Recovery_Guten'],
    'bnc':       ['Base_GPT-2', 'Kallini', 'Recovery_BNC'],
}

def make_tokenizer(markers):
    tok = GPT2Tokenizer.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    if markers:
        tok.add_tokens([AddedToken(m, lstrip=True, rstrip=False) for m in markers])
    return tok

def load_model(model_id, tokenizer):
    hf = GPT2LMHeadModel.from_pretrained(model_id)
    # resize embeddings to match the marker-augmented vocab
    hf.resize_token_embeddings(len(tokenizer))
    m = HookedTransformer.from_pretrained('gpt2', hf_model=hf)
    m.eval(); m.to(DEVICE)
    return m

def mean_lp(logits, ids):
    lp = torch.log_softmax(logits[0], dim=-1)
    seq = ids[0]
    return lp[:-1].gather(1, seq[1:, None]).squeeze(1).mean().item()

def patch_sentence(model, tok, clean_sent, corrupt_sent, verbose=False):
    clean_ids   = tok(clean_sent,   return_tensors='pt')['input_ids'].to(DEVICE)
    corrupt_ids = tok(corrupt_sent, return_tensors='pt')['input_ids'].to(DEVICE)
    if clean_ids.shape[1] < 2 or corrupt_ids.shape[1] < 2:
        return None

    _, cache = model.run_with_cache(clean_ids)
    LL_clean   = mean_lp(model(clean_ids), clean_ids)
    LL_corrupt = mean_lp(model(corrupt_ids), corrupt_ids)
    denom = LL_clean - LL_corrupt
    L = min(clean_ids.shape[1], corrupt_ids.shape[1])
    if verbose:
        print(f'      clean_tok={clean_ids.shape[1]} corrupt_tok={corrupt_ids.shape[1]} '
              f'overlap={L}  LL_clean={LL_clean:.4f} LL_corrupt={LL_corrupt:.4f} denom={denom:.4f}')
    if abs(denom) < 0.05:
        if verbose: print(f'      denom={denom:.4f} < 0.05, skipping (perturbation not degrading this model)')
        return None

    res = np.full((N_LAYERS, N_HEADS), np.nan)
    for layer in range(N_LAYERS):
        clean_z = cache[f'blocks.{layer}.attn.hook_z']
        for head in range(N_HEADS):
            def hook(value, hook, _h=head):
                value[:, :L, _h, :] = clean_z[:, :L, _h, :]
                return value
            pl = model.run_with_hooks(corrupt_ids,
                    fwd_hooks=[(f'blocks.{layer}.attn.hook_z', hook)])
            LL_p = mean_lp(pl, corrupt_ids)
            res[layer, head] = float(np.clip((LL_p - LL_corrupt) / denom, CLIP_LO, CLIP_HI))
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pert', choices=list(PERT_MARKERS.keys()))
    ap.add_argument('--corpus', choices=['gutenberg', 'bnc'], default='gutenberg')
    ap.add_argument('--smoke', action='store_true', help='1 sentence, verbose')
    ap.add_argument('--n', type=int, default=150)
    args = ap.parse_args()

    n = 1 if args.smoke else args.n
    markers = PERT_MARKERS[args.pert]
    tok = make_tokenizer(markers)

    pair_file = os.path.join(OUT_BASE, args.pert, f'pairs_{args.pert}_{args.corpus}.json')
    pairs = json.load(open(pair_file, encoding='utf-8'))[:n]
    print(f'=== PATCHING {args.pert} / {args.corpus} | smoke={args.smoke} | n={len(pairs)} ===')
    print(f'    markers: {markers if markers else "none"}')

    for mk in MODELS_BY_CORPUS[args.corpus]:
        if args.pert not in MODEL_IDS[mk]:
            print(f'\n  {MODEL_LABELS[mk]}: no model for {args.pert} on this corpus, skipping')
            continue
        print(f'\n  Model: {MODEL_LABELS[mk]}  ({MODEL_IDS[mk][args.pert]})')
        model = load_model(MODEL_IDS[mk][args.pert], tok)
        acc, cnt = np.zeros((N_LAYERS, N_HEADS)), 0
        for i, p in enumerate(pairs):
            r = patch_sentence(model, tok, p['actual'], p['input'], verbose=args.smoke)
            if r is None:
                continue
            acc += np.nan_to_num(r); cnt += 1
            if args.smoke:
                bl, bh = np.unravel_index(np.nanargmax(r), r.shape)
                print(f'      [sent {i}] best L{bl}H{bh} RS={r[bl,bh]:.4f} '
                      f'meanRS={np.nanmean(r):.4f} minRS={np.nanmin(r):.4f}')
        if cnt == 0:
            print('    no usable sentences'); del model; continue
        mat = acc / cnt
        bl, bh = np.unravel_index(np.nanargmax(mat), mat.shape)
        out = {'model': MODEL_LABELS[mk], 'perturbation': args.pert, 'corpus': args.corpus,
               'n_layers': N_LAYERS, 'n_heads': N_HEADS, 'n_sentences': cnt,
               'restoration': mat.tolist(), 'best_layer': int(bl), 'best_head': int(bh),
               'best_score': float(mat[bl, bh]), 'mean_restoration': float(np.nanmean(mat))}
        tag = 'SMOKE_' if args.smoke else ''
        outdir = os.path.join(OUT_BASE, args.pert)
        path = os.path.join(outdir, f'{tag}patching_{args.pert}_{args.corpus}_{mk.replace("_","")}.json')
        json.dump(out, open(path, 'w'), indent=2)
        print(f'    best L{bl}H{bh} score={mat[bl,bh]:.4f} mean={np.nanmean(mat):.4f}  -> {path}')
        del model
        if DEVICE == 'cuda': torch.cuda.empty_cache()
    print('\nDone.')

if __name__ == '__main__':
    main()
