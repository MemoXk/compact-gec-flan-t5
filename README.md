# Compact GEC with FLAN-T5-small: controlled baselines, full-budget ablations and two-stage transfer

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23326147.svg)](https://doi.org/10.5281/zenodo.23326147)

Archived on Zenodo: https://doi.org/10.5281/zenodo.23326147 (all versions; v1.0.0 is https://doi.org/10.5281/zenodo.23326148).

Code, configurations, data splits, system outputs and analysis results for

> M. A. ElTaweel and N. Negied, *Rethinking FLAN-T5 for Grammatical Error Correction: Controlled Baselines,
> Full-Budget Ablations and Two-Stage Transfer with Little Human Data* (manuscript under review, 2026).

The paper studies what a 77M-parameter FLAN-T5-small model achieves for grammatical error correction with synthetic
data (C4-200M) and a small human-annotated stage (W&I+LOCNESS), and which commonly added design choices actually help.
Every number in the paper can be traced to a file in this repository.

## Repository map

| Path | Contents |
|---|---|
| `notebooks/` | The Kaggle notebooks that produced every result (see *Reproducing*). `notebooks/earlier/` keeps the earlier-configuration notebooks for provenance. |
| `generators/` | Python scripts that generate the round-2 notebooks (`nb_common.py` holds the shared cells). |
| `configs/` | Complete training, decoding and baseline configurations, split definitions and hashes. |
| `splits/` | C4-200M split indices (shard, line, split) for the cleaned and uncleaned samples, the 5,000-pair in-domain test subset, and the W&I+LOCNESS stage-2 holdout indices. |
| `outputs/<benchmark>/<system>.txt` | System outputs exactly as scored (one line per benchmark sentence, benchmark order, after the benchmark retokeniser). `outputs/manifest.json` lists them. |
| `results/` | Notebook result files, analysis JSONs with all paired bootstraps, trainer logs, official CodaBench scores for the blind BEA-2019 test, and the harm audit. |
| `analysis/verify.py` | Re-scores every output with the official scorers and recomputes the paired bootstraps of Tables 4 and 6. |
| `analysis/provenance/` | The exact analysis scripts used for the paper (they contain local paths; `verify.py` is the portable equivalent). |
| `third_party/m2scorer_py3/` | Python-3 port of the official MaxMatch (M2) scorer (GPL-2.0, see its LICENSE). |

Systems in `outputs/`: `final` (the paper's final system), its decode variants (`final_no_threshold`,
`final_no_refinement`, `final_tau*`), `final_stage1*`, `nogate_stage1` and `nogate_two_stage*` (no-edit ablation),
`noisy_s42`, `noisy_s1337`, `clean_s1337` (cleaning ablation), `earlier_config/*` (systems trained in the earlier
configuration: copy-gate arms, length-penalty arms, earlier final system), `superseded/*` (see *Corrections*) and
`baselines/*` (Qwen2.5-1.5B-Instruct 0/4-shot, gec-t5-small/base-clang8).

## Verifying the paper's numbers

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python analysis/get_data.py          # CoNLL-2014 test, W&I+LOCNESS v2.1, JFLEG (public downloads)
python analysis/verify.py --check    # re-scores all outputs, recomputes the bootstraps, compares with results/verified/
```

`verify.py` asserts that the per-sentence counts used for the bootstrap reproduce the official M2 totals. ERRANT scoring
of all systems takes a few hours on one CPU core (`--only <system> ...` restricts the run; several runs can share the
cache in `.cache/`). JFLEG is scored only if `data/jfleg/gleu.py` is present.

The blind BEA-2019 test set cannot be scored locally. `results/bea19_test_codabench/` holds the official CodaBench
results of the three trained versions of the final system, each submitted once to CodaBench competition 10960:
`earlier_config_final` (0.591, submission 975012), `superseded_final_dupstage2` (0.601, submission 978739) and `final`
(0.596, submission 981124; the system reported in the paper, see *Corrections*).

## Reproducing the experiments

All training ran on the free Kaggle tier (2 x NVIDIA T4). Run the notebooks with GPU T4 x2 and Internet on:

| Notebook | What | Inputs | Time |
|---|---|---|---|
| `R2_A_cleaning_full_budget.ipynb` | cleaning ablation at full budget (noisy/clean x seeds 42, 1337) | Kaggle dataset `dariocioni/c4200m` | 2 sessions x ~7 h |
| `R2_C_retrain_core.ipynb` | stage-1 models (`final`, `nogate`; `gate` needs > 12 h, see paper) | `dariocioni/c4200m` | ~4-6 h per job |
| `R2_D_stage2_dedup.ipynb` | stage 2 on W&I+LOCNESS, threshold tuning, full evaluation, blind BEA-2019 test file | outputs of `R2_C` | ~4 h |
| `R2_B_final_system_eval.ipynb` | baselines (Qwen2.5, gec-t5), latency, decode factors of the earlier configuration | earlier checkpoints | one session (< 12 h) |

Each notebook downloads the official evaluation data and scorers itself and writes a single results archive. Every
training run includes an automatic check that stops training if the training/validation loss ratio indicates that the
loss is not normalised by gradient accumulation.

## Checkpoints

The final system (stage 1 and stage 2) and the no-edit ablation checkpoints are archived on Zenodo:
https://doi.org/10.5281/zenodo.23328229 (four archives: `final_stage2`, `final_stage1`, `nogate_stage2`, `nogate_stage1`; CC BY-NC 4.0, because stage 2 was trained on W&I+LOCNESS).

## Corrections relative to earlier versions of this work

* **Loss normalisation.** The Kaggle image used for the earlier runs did not divide the training loss by the number of
  gradient-accumulation steps, so gradient clipping was active at almost every step. All models of the final pipeline
  were retrained with an automatic check (observed loss ratios 1.04-1.17). Earlier-configuration systems are kept
  under `outputs/earlier_config/` for the comparisons that use them.
* **Duplicated stage-2 data.** The earlier notebooks read the W&I+LOCNESS files `A`, `B`, `C` *and* the combined
  `ABC` file, so every training sentence appeared twice and 94% of the 5% holdout also occurred in training. Stage 2
  was retrained on A+B+C with a holdout disjoint by source sentence (`R2_D_stage2_dedup.ipynb`). The outputs of the
  affected models are kept under `outputs/superseded/`.

## Data and licences

* Code and documentation written for this project: MIT (see `LICENSE`). It does not cover
  `third_party/m2scorer_py3/` (GPL-2.0, its own `LICENSE`) or the data described in `LICENSE-DATA.md`.
* Benchmark data are **not** redistributed; `analysis/get_data.py` downloads them from their official sources.
  System outputs on W&I+LOCNESS (BEA-2019), CoNLL-2014 and JFLEG are derived from those datasets and are provided for
  non-commercial research only, under the terms of the respective datasets (see `LICENSE-DATA.md`).
* `splits/c4_test_subset_5k.tsv` redistributes 5,000 sentence pairs of C4-200M with attribution.

## Citation

See `CITATION.cff`. Please also cite C4-200M (Stahlberg and Kumar, 2021), W&I+LOCNESS (Bryant et al., 2019;
Yannakoudakis et al., 2018), CoNLL-2014 (Ng et al., 2014), JFLEG (Napoles et al., 2017) and ERRANT (Bryant et al., 2017).
