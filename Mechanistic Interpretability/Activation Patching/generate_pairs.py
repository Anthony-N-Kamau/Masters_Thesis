# -*- coding: utf-8 -*-
import os
import json
import argparse
from pathlib import Path

from shuffle import full_shuffle, local_shuffle
from reverse import partial_reverse

BASE = os.path.expanduser('~/Impossible_projects')
DATA_DIR = os.path.join(BASE, 'datasets')
OUT_BASE = os.path.join(BASE, 'Mechanistic Interpretability', 'Activation Patching')

CORPUS_PATHS = {
    'bnc':       os.path.join(DATA_DIR, '1k_bnc_spoken.test'),
    'gutenberg': os.path.join(DATA_DIR, '1k_gutenberg.test'),
}

def load_sentences(path):
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    return [line.strip() for line in lines if line.strip()]

def perturb_one(s, pert):
    if pert == 'reverse':
        return partial_reverse(s)
    elif pert == 'fullshuffle':
        return full_shuffle(s, seed=57)
    elif pert == 'localw3':
        return local_shuffle(s, window_size=3)
    elif pert == 'localw5':
        return local_shuffle(s, window_size=5)
    elif pert == 'hop':
        return wordhop(s)
    else:
        raise ValueError(pert)

def apply_and_save(sentences, pert, out_path):
    pairs, skipped = [], 0
    for s in sentences:
        try:
            perturbed = perturb_one(s, pert)
            if perturbed is not None:
                pairs.append({'input': perturbed, 'actual': s})
            else:
                skipped += 1
        except Exception:
            skipped += 1
    Path(out_path).write_text(
        json.dumps(pairs, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(f'    {len(pairs)} pairs saved to {out_path}  ({skipped} skipped)')
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=150)
    ap.add_argument('--corpus', choices=['gutenberg', 'bnc'], default='gutenberg')
    ap.add_argument('--skip-hop', action='store_true')
    args = ap.parse_args()

    perts = ['fullshuffle', 'localw3', 'localw5', 'reverse']
    if not args.skip_hop:
        global wordhop
        from hop import wordhop
        perts.append('hop')

    sentences = load_sentences(CORPUS_PATHS[args.corpus])[:args.n]
    print(f'Loaded {len(sentences)} {args.corpus} sentences\n')

    for pert in perts:
        print(f'=== {pert} ===')
        out_dir = os.path.join(OUT_BASE, pert)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'pairs_{pert}_{args.corpus}.json')
        pairs = apply_and_save(sentences, pert, out_path)
        if pairs:
            print(f'    spot-check actual: {pairs[0]["actual"][:90]}')
            print(f'    spot-check input : {pairs[0]["input"][:90]}\n')

    print('Done.')

if __name__ == '__main__':
    main()
