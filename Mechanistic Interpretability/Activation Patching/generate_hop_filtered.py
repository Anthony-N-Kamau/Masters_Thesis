# -*- coding: utf-8 -*-
import os
import json
import argparse
from pathlib import Path

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=int, default=150)
    ap.add_argument('--corpus', choices=['gutenberg', 'bnc'], default='gutenberg')
    ap.add_argument('--pool', type=int, default=500)
    args = ap.parse_args()

    from hop import wordhop

    sentences = load_sentences(CORPUS_PATHS[args.corpus])[:args.pool]
    print(f'Drawing from {len(sentences)} {args.corpus} sentences, targeting {args.target} CHANGED hop pairs')

    pairs, skipped_none, unchanged, errored = [], 0, 0, 0
    for s in sentences:
        if len(pairs) >= args.target:
            break
        try:
            h = wordhop(s)
        except Exception:
            errored += 1
            continue
        if h is None:
            skipped_none += 1
            continue
        if h.strip() == s.strip():
            unchanged += 1
            continue
        pairs.append({'input': h, 'actual': s})

    out_dir = os.path.join(OUT_BASE, 'hop')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'pairs_hop_{args.corpus}.json')
    Path(out_path).write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'  kept {len(pairs)} CHANGED pairs (dropped: {unchanged} unchanged, {skipped_none} none, {errored} errored)')
    print(f'  saved {out_path}')
    if len(pairs) < args.target:
        print(f'  WARNING: only {len(pairs)} found in pool of {len(sentences)}; increase --pool')
    if pairs:
        print(f'  spot-check actual: {pairs[0]["actual"][:90]}')
        print(f'  spot-check input : {pairs[0]["input"][:90]}')

if __name__ == '__main__':
    main()
