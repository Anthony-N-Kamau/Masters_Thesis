import os
import json
import argparse
from pathlib import Path

import sys
AP_FOLDER = os.path.expanduser(
    '~/Impossible_projects/Mechanistic Interpretability/Activation Patching')
sys.path.insert(0, AP_FOLDER)
from shuffle import full_shuffle

BASE = os.path.expanduser('~/Impossible_projects')
DATA_DIR = os.path.join(BASE, 'datasets')
IFA_BASE = os.path.join(BASE, 'Mechanistic Interpretability', 'Information Flow Analysis')

CORPUS_PATHS = {
    'gutenberg': os.path.join(DATA_DIR, '1k_gutenberg.test'),
    'bnc': os.path.join(DATA_DIR, '1k_bnc_spoken.test'),
}


def load_sentences(path, n):
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    out = [line.strip() for line in lines if line.strip()]
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=600,
                    help='number of sentences to draw from the 1k corpus file')
    ap.add_argument('--corpus', choices=['gutenberg', 'bnc'], default='gutenberg')
    args = ap.parse_args()

    corpus_path = CORPUS_PATHS[args.corpus]
    sentences = load_sentences(corpus_path, args.n)
    print(f'Loaded {len(sentences)} sentences from {corpus_path} '
          f'(requested {args.n})')

    if len(sentences) < args.n:
        print(f'WARNING: corpus only has {len(sentences)} usable lines, '
              f'fewer than requested {args.n}.')

    pairs, skipped = [], 0
    for s in sentences:
        try:
            perturbed = full_shuffle(s, seed=57)
            if perturbed is not None:
                pairs.append({'input': perturbed, 'actual': s})
            else:
                skipped += 1
        except Exception:
            skipped += 1

    out_dir = os.path.join(IFA_BASE, 'fullshuffle')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'pairs_fullshuffle_{args.corpus}_ifa{args.n}.json')
    Path(out_path).write_text(
        json.dumps(pairs, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'{len(pairs)} pairs saved to {out_path}  ({skipped} skipped)')
    if pairs:
        print(f'  spot-check actual: {pairs[0]["actual"][:90]}')
        print(f'  spot-check input : {pairs[0]["input"][:90]}')


if __name__ == '__main__':
    main()
