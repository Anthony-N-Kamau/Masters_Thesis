import os
import json
import argparse

import torch
import spacy
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AddedToken

if spacy.prefer_gpu():
    print('spaCy: using GPU')
else:
    print('spaCy: using CPU')

nlp = spacy.load('en_core_web_trf')

BASE = os.path.expanduser('~/Impossible_projects')
AP_BASE = os.path.join(BASE, 'Mechanistic Interpretability', 'Activation Patching')
IFA_BASE = os.path.join(BASE, 'Mechanistic Interpretability', 'Information Flow Analysis')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MARK_SG, MARK_PL, MARK_REV = '\U0001F142', '\U0001F13F', '\U0001F141'
PERT_MARKERS = {
    'hop': [MARK_SG, MARK_PL], 'reverse': [MARK_REV],
    'localw3': [], 'localw5': [], 'fullshuffle': [],
}
ALL_MARKERS = {MARK_SG, MARK_PL, MARK_REV}


def is_marker_token(text):
    stripped = text.strip()
    return len(stripped) > 0 and all(ch in ALL_MARKERS for ch in stripped)


def strip_markers(text):
    return ''.join(ch for ch in text if ch not in ALL_MARKERS)


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

BRITTLE_DROP_THRESHOLD = 0.30

PROMPT_PREFIX = "Fix this text: "
PROMPT_MID = "\nCorrected: "


# --- syntactic feature extraction (unchanged definitions) ---
def get_arc_direction(tok):
    if tok.dep_ == 'ROOT':
        return None
    return 'left' if tok.head.i < tok.i else 'right'


def is_in_subj_or_obj_np(tok):
    current = tok
    visited = set()
    while current.dep_ not in ('nsubj', 'nsubjpass', 'dobj', 'ROOT'):
        if current.i in visited or current.head == current:
            break
        visited.add(current.i)
        current = current.head
    if current.dep_ in ('nsubj', 'nsubjpass'):
        return 'subj_np'
    if current.dep_ == 'dobj':
        return 'obj_np'
    return None


def extract_word_labels(doc):
    records = []
    for i, tok in enumerate(doc):
        if tok.is_punct or tok.is_space:
            continue
        pos = tok.pos_ if tok.pos_ in ('NOUN', 'VERB') else None
        dep_rel = tok.dep_ if tok.dep_ in ('nsubj', 'dobj') else None
        arc_dir = get_arc_direction(tok)
        phrase_role = is_in_subj_or_obj_np(tok)
        records.append({
            'orig_idx': i, 'word': tok.text, 'lemma': tok.lemma_.lower(),
            'pos': pos, 'dep_rel': dep_rel, 'arc_dir': arc_dir,
            'phrase_role': phrase_role,
        })
    return records


# marker-agnostic matching for the Recovered side only

def find_recovered_index(lemma, recovered_doc, used_indices):
    ll = lemma.lower()
    for i, tok in enumerate(recovered_doc):
        if i in used_indices or is_marker_token(tok.text):
            continue
        if tok.lemma_.lower() == ll:
            return i
    if len(ll) >= 2:
        for i, tok in enumerate(recovered_doc):
            if i in used_indices or is_marker_token(tok.text):
                continue
            t = strip_markers(tok.lemma_).lower()
            if len(t) > len(ll) and ll in t:
                return i
    for i in range(len(recovered_doc) - 1):
        if i in used_indices or (i + 1) in used_indices:
            continue
        if is_marker_token(recovered_doc[i].text) or is_marker_token(recovered_doc[i + 1].text):
            continue
        joined = (strip_markers(recovered_doc[i].text) +
                  strip_markers(recovered_doc[i + 1].text)).lower()
        if joined == ll:
            return i
    return None


def find_colon_token_index(tokenizer, prompt_text):
    """Index of the FINAL colon token in the tokenized prompt (the anchor
    right after '\\nCorrected:'). Tokenizes incrementally to find the exact
    token position regardless of subword boundaries."""
    ids = tokenizer.encode(prompt_text)
    # Find the colon that follows "Corrected" 
    # "Fix this text:" also contains a colon earlier in the prompt.
    colon_variants = {tokenizer.encode(':')[0]}
    # GPT-2 BPE often encodes ":" attached to the previous token (e.g. "Corrected:")
    # so also check for tokens whose decoded string ends with ':'.
    for i in range(len(ids) - 1, -1, -1):
        decoded = tokenizer.decode([ids[i]])
        if decoded.strip().endswith(':'):
            return i
    return None


# generation 
def make_tokenizer(markers):
    tok = GPT2Tokenizer.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    if markers:
        tok.add_tokens([AddedToken(m, lstrip=True, rstrip=False) for m in markers])
    return tok


def load_recovery_model(model_id, tokenizer):
    hf = GPT2LMHeadModel.from_pretrained(model_id)
    hf.resize_token_embeddings(len(tokenizer))
    hf.eval()
    hf.to(DEVICE)
    return hf


GPT2_MAX_CONTEXT = 1024  # hard positional-embedding limit; exceeding this on
                          # generate() causes a CUDA gather-kernel index error


def generate_recovered_text(model, tokenizer, impossible_sent, orig_token_len, verbose=False):
    prompt = PROMPT_PREFIX + impossible_sent + PROMPT_MID.rstrip()
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
    prompt_len = input_ids.shape[1]

    # Cap max_new_tokens so prompt_len + max_new never exceeds GPT-2's hard
    
    desired = max(8, 2 * orig_token_len)
    headroom = GPT2_MAX_CONTEXT - prompt_len - 1  
    max_new = max(1, min(desired, headroom))

    if max_new < desired and verbose:
        print(f'      [context-cap] prompt_len={prompt_len}, desired_new={desired}, '
              f'capped to {max_new} to stay under {GPT2_MAX_CONTEXT} total tokens')

    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=max_new, do_sample=False, num_beams=1,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0][input_ids.shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    if verbose:
        print(f'      generated="{gen_text[:80]}"')
    return gen_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pert', required=True, choices=list(PERT_MARKERS.keys()))
    ap.add_argument('--corpus', choices=['gutenberg', 'bnc'], default='gutenberg')
    ap.add_argument('--n', type=int, default=150)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--pairs-override', default=None,
                    help='full path to an alternate pairs JSON file to use '
                         'instead of the default Activation Patching pairs '
                         '(e.g. a larger IFA-only sentence set). Output is '
                         'still saved under the normal ifa_textprep_<pert>_'
                         '<corpus>.json name, OVERWRITING the existing file '
                         '-- back it up first if you want to keep both.')
    args = ap.parse_args()

    n = 1 if args.smoke else args.n
    markers = PERT_MARKERS[args.pert]
    tokenizer = make_tokenizer(markers)

    if args.pert not in RECOVERY_MODEL_IDS[args.corpus]:
        print(f'No Recovery model for {args.pert} on {args.corpus}. Aborting.')
        return

    model_id = RECOVERY_MODEL_IDS[args.corpus][args.pert]
    print(f'Loading Recovery model: {model_id}')
    model = load_recovery_model(model_id, tokenizer)

    if args.pairs_override:
        pair_file = args.pairs_override
        print(f'Using OVERRIDE pairs file: {pair_file}')
    else:
        pair_file = os.path.join(AP_BASE, args.pert, f'pairs_{args.pert}_{args.corpus}.json')
    pairs = json.load(open(pair_file, encoding='utf-8'))[:n]
    print(f'Loaded {len(pairs)} pairs from {pair_file}')

    results = []
    total_words, total_dropped = 0, 0
    total_markers, total_markers_survived = 0, 0
    n_skipped_errors = 0

    for si, p in enumerate(pairs):
        original = p['actual']
        impossible = p['input']

        try:
            orig_doc = nlp(original)
            orig_token_len = len(tokenizer.encode(original))

            recovered = generate_recovered_text(model, tokenizer, impossible,
                                                 orig_token_len, verbose=args.smoke)
            recovered_doc = nlp(recovered) if recovered else nlp("")

            # build the three prompts (confirmed structures)
            prompt_original = f"{PROMPT_PREFIX}{original}{PROMPT_MID}{original}"
            prompt_impossible = f"{PROMPT_PREFIX}{impossible}{PROMPT_MID}{original}"
            prompt_recovered = f"{PROMPT_PREFIX}{impossible}{PROMPT_MID}{recovered}"

            colon_idx_original = find_colon_token_index(
                tokenizer, f"{PROMPT_PREFIX}{original}{PROMPT_MID.rstrip()}")
            colon_idx_impossible = find_colon_token_index(
                tokenizer, f"{PROMPT_PREFIX}{impossible}{PROMPT_MID.rstrip()}")
            colon_idx_recovered = colon_idx_impossible

            # word labels from Original, offsets relative to colon 
            word_labels = extract_word_labels(orig_doc)

            used_rec = set()
            word_records = []
            for wl in word_labels:
                rec_idx = find_recovered_index(wl['lemma'], recovered_doc, used_rec)
                if rec_idx is not None:
                    used_rec.add(rec_idx)

                total_words += 1
                if rec_idx is None:
                    total_dropped += 1
                    continue

                word_records.append({
                    **wl,
                    'offset_from_colon': wl['orig_idx'] + 1,
                    'recovered_offset_from_colon': rec_idx + 1,
                })

            # marker survival tracking (separate analysis stream)
            marker_records = []
            if markers:
                impossible_doc = nlp(impossible)
                for i, tok in enumerate(impossible_doc):
                    if is_marker_token(tok.text):
                        survived = any(is_marker_token(rt.text) and rt.text == tok.text
                                        for rt in recovered_doc)
                        marker_records.append({
                            'marker': tok.text, 'position_in_impossible': i,
                            'survived_in_recovered': survived,
                        })
                        total_markers += 1
                        if survived:
                            total_markers_survived += 1

            n_labeled = len(word_labels)
            n_dropped_this_sentence = n_labeled - len(word_records)
            sentence_drop_rate = (n_dropped_this_sentence / n_labeled) if n_labeled else 0.0
            classification = 'brittle' if sentence_drop_rate >= BRITTLE_DROP_THRESHOLD else 'active'

            results.append({
                'original': original, 'impossible': impossible, 'recovered': recovered,
                'prompt_original': prompt_original,
                'prompt_impossible': prompt_impossible,
                'prompt_recovered': prompt_recovered,
                'colon_idx_original': colon_idx_original,
                'colon_idx_impossible': colon_idx_impossible,
                'colon_idx_recovered': colon_idx_recovered,
                'word_records': word_records,
                'marker_records': marker_records,
                'sentence_drop_rate': sentence_drop_rate,
                'classification': classification,
            })

            if args.smoke:
                print(f'  [{si}] original : {original[:80]}')
                print(f'       impossible: {impossible[:80]}')
                print(f'       recovered : {recovered[:80]}')
                print(f'       colon_idx (orig/imp): {colon_idx_original} / {colon_idx_impossible}')
                ids_o = tokenizer.encode(f"{PROMPT_PREFIX}{original}{PROMPT_MID.rstrip()}")
                if colon_idx_original is not None:
                    ctx_before = [tokenizer.decode([t]) for t in ids_o[max(0, colon_idx_original-3):colon_idx_original]]
                    ctx_at = tokenizer.decode([ids_o[colon_idx_original]])
                    print(f'       COLON CHECK (original prompt): ...{ctx_before}[{ctx_at!r}]<-colon_idx  '
                          f'(total tokens={len(ids_o)}, colon should be LAST token)')
                ids_i = tokenizer.encode(f"{PROMPT_PREFIX}{impossible}{PROMPT_MID.rstrip()}")
                if colon_idx_impossible is not None:
                    ctx_before = [tokenizer.decode([t]) for t in ids_i[max(0, colon_idx_impossible-3):colon_idx_impossible]]
                    ctx_at = tokenizer.decode([ids_i[colon_idx_impossible]])
                    print(f'       COLON CHECK (impossible prompt): ...{ctx_before}[{ctx_at!r}]<-colon_idx  '
                          f'(total tokens={len(ids_i)}, colon should be LAST token)')
                print(f'       words: {n_labeled} total, {n_dropped_this_sentence} dropped, '
                      f'{len(word_records)} kept, drop_rate={sentence_drop_rate:.3f}, '
                      f'classification={classification}')
                if marker_records:
                    print(f'       markers: {marker_records}')

        except Exception as e:
            n_skipped_errors += 1
            print(f'  [{si}] SKIPPED due to error: {type(e).__name__}: {e}')
            print(f'        original  : {original[:120]}')
            print(f'        impossible: {impossible[:120]}')
            # If CUDA raised a device-side assert
            if 'CUDA' in str(e) or 'cuda' in type(e).__name__.lower():
                print('  CUDA error detected -- GPU context likely corrupted. '
                      'Stopping this run; rerun to pick up remaining sentences '
                      '(or exclude the triggering sentence and rerun).')
                break
            continue

    match_rate = 1 - (total_dropped / total_words) if total_words else 0.0
    n_active = sum(1 for r in results if r['classification'] == 'active')
    n_brittle = sum(1 for r in results if r['classification'] == 'brittle')
    marker_survival_rate = (total_markers_survived / total_markers) if total_markers else None

    print(f'\nTotal words: {total_words}  Dropped: {total_dropped}  Match rate: {match_rate:.4f}')
    print(f'Sentences: {n_active} active, {n_brittle} brittle (threshold={BRITTLE_DROP_THRESHOLD})')
    print(f'Skipped due to errors: {n_skipped_errors} / {len(pairs)} attempted')
    if total_markers:
        print(f'Markers: {total_markers} total, {total_markers_survived} survived, '
              f'survival_rate={marker_survival_rate:.4f}')

    out_dir = os.path.join(IFA_BASE, args.pert)
    os.makedirs(out_dir, exist_ok=True)
    tag = 'SMOKE_' if args.smoke else ''
    suffix = f'_n{args.n}' if args.pairs_override else ''
    out_path = os.path.join(out_dir, f'{tag}ifa_textprep_{args.pert}_{args.corpus}{suffix}.json')
    payload = {
        'perturbation': args.pert, 'corpus': args.corpus,
        'n_sentences': len(results),
        'n_attempted': len(pairs), 'n_skipped_errors': n_skipped_errors,
        'total_words': total_words, 'total_dropped': total_dropped,
        'match_rate': match_rate,
        'n_active': n_active, 'n_brittle': n_brittle,
        'brittle_threshold': BRITTLE_DROP_THRESHOLD,
        'total_markers': total_markers, 'total_markers_survived': total_markers_survived,
        'marker_survival_rate': marker_survival_rate,
        'sentences': results,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
