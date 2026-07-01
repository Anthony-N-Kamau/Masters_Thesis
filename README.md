# Impossible Language Models: Lightweight Explainability vs Mechanistic Interpretability

Code and results for "Lightweight Explainability vs Mechanistic Interpretability: Probing and Patching Syntactic Encoding in Impossible Language Models".

Investigates whether GPT-2 models trained on impossible languages (Kallini et al., 2024) and their recovery-fine-tuned counterparts (Mohammadi et al., 2026) encode syntactic structure, and whether encoding it causally translates into using it.

## Repo structure

```
Impossible_projects/
├── datasets/                        # BNC Spoken and Gutenberg corpora, perturbed variants
├── utils/                           # shared helper functions used across pipelines
├── BLiMP_full/                      # behavioural evaluation on BLiMP minimal pairs
├── linear_probing_2/                # head-level linear probing (selectivity)
└── Mechanistic Interpretability/
    ├── Activation Patching/         # head-level restoration patching
    └── Information Flow Analysis/   # token-to-token V-information flow
```

Each of `BLiMP_full`, `linear_probing_2`, `Activation Patching`, and `Information Flow Analysis` has its own README with setup and run instructions specific to that pipeline.

## Setup

Clone the repo, naming the folder `Impossible_projects` (required, since all SLURM scripts assume this exact path):

```bash
git clone https://github.com/Anthony-N-Kamau/Masters_Thesis.git Impossible_projects
cd Impossible_projects
```

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv probe_env
source probe_env/bin/activate
pip install -r requirements.txt
```

spaCy's English model is required separately:

```bash
python -m spacy download en_core_web_md
```

Runs on Snellius (SURF), `gpu_a100` partition, NVIDIA A100 GPUs. Models are loaded via TransformerLens and Hugging Face.

## Snellius requirements

- **Clone location matters.** All SLURM scripts use `~/Impossible_projects` as a fixed path. Clone the repo into your home directory under exactly this name, not elsewhere or under a different folder name, or every `sbatch` job will fail on `cd`.
- **Environment location matters too.** Scripts run `source ~/probe_env/bin/activate`, so the virtual environment from the Setup section must be created at `~/probe_env`, not inside the repo or elsewhere.
- **`gpu_a100` partition access.** SLURM scripts do not set `#SBATCH --account=`, relying on your default Snellius account. Confirm your account has access to the `gpu_a100` partition before submitting jobs; if not, add `#SBATCH --account=<your_account>` to the relevant script.

Three model families are compared throughout: Base GPT-2, Kallini (impossible-language-trained), and Recovery (fine-tuned to translate impossible language back to natural language), on the BNC Spoken and Gutenberg corpora, under five perturbation types (hop, localw3, localw5, reverse, fullshuffle).

## Run order

1. **BLiMP_full** — behavioural sanity check, independent of the other two, can run anytime. See `BLiMP_full/README.md`.
2. **linear_probing_2** — head-level selectivity per feature (POS, dependency relation, arc direction, phrase role). See `linear_probing_2/README.md`.
3. **Mechanistic Interpretability/Activation Patching** — head-level restoration scores. Uses the same models and perturbations as probing but is otherwise independent. See its README.
4. **Mechanistic Interpretability/Information Flow Analysis** — token-to-token information flow, complements probing and patching. See its README.

Probing and patching results are combined post hoc (not part of either pipeline) to identify heads that are both representationally informative and causally sufficient.

## Notes

- Recovery (BNC) only exists for hop, reverse, and localw3. LocalW5 and fullshuffle Recovery models exist only for Gutenberg.
- All patching results in this thesis use restoration (denoising) patching, testing causal sufficiency, not necessity.
- Check each folder's `logs/` directory for SLURM output if a job fails.
