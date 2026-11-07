# What matters for a compact grammatical error corrector? FLAN-T5-small with little human data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23326147.svg)](https://doi.org/10.5281/zenodo.23326147)

Archived on Zenodo: https://doi.org/10.5281/zenodo.23326147 (all versions). Trained checkpoints:
https://doi.org/10.5281/zenodo.23328228 (all versions; version 2 holds the checkpoints of the current paper,
one archive per run of Table 2, listed in `results/run_manifest.json`).

Code, configurations, data splits, system outputs, per-sentence edit counts and analysis results for

> M. A. ElTaweel and N. Negied, *What Matters for a Compact Grammatical Error Corrector? Seed-Replicated Ablations of
> FLAN-T5-small with Little Human Data* (manuscript under review, 2026).

The paper asks which commonly added design choices change the accuracy of a 77M-parameter FLAN-T5-small corrector
trained on synthetic data (C4-200M) and 32,314 human-annotated learner pairs (W&I+LOCNESS). Every number in the paper
can be traced to a file in this repository through `results/run_manifest.json`.

## Repository map

| Path | Contents |
|---|---|
| `notebooks/` | The Kaggle notebooks that produced the results. `R2_E_round3.ipynb` trains and evaluates every stage-2 run of the paper (three sessions), the extra stage-1 seed, the data-budget curve, CoEdIT-large and the latency measurements. `notebooks/earlier/` keeps earlier notebooks for provenance only. |
| `generators/` | Scripts that generate the notebooks (`nb_common.py` holds shared cells). |
| `configs/` | Training, decoding and baseline configurations and split definitions. |
| `splits/` | C4-200M split indices for both samples, the 5,000-pair in-domain test subset and its 626 uncontaminated rows, the essay of every W&I+LOCNESS training sentence and the essay-level holdout. |
| `outputs/<benchmark>/<system>.txt` | System outputs exactly as scored (one line per benchmark sentence, after the benchmark retokeniser). `outputs/manifest.json` lists them. |
| `results/run_manifest.json` | For every table of the paper: the systems behind each row, the training runs behind each system (notebook, data, identity pairs, seeds, selected checkpoint, loss ratio, median gradient norm, selected threshold), the decoder, the output files and the per-sentence counts, and the scorer versions. |
| `results/verified/verified_scores.json` | Output of `analysis/verify.py`: every score and every paired comparison of the paper. |
| `results/per_sentence_counts/` | Per-sentence TP/FP/FN behind every edit-level score (written by `verify.py`). |
| `results/` (other) | Notebook result files, trainer logs, official CodaBench results of the blind BEA-2019 test, overlap audits, human-evaluation key and judgments. |
| `analysis/verify.py` | Re-scores every output with the official scorers and recomputes all comparisons in `analysis/table_spec.json`. |
| `analysis/provenance/` | The exact analysis and paper-building scripts (local paths masked). |
| `third_party/m2scorer_py3/` | Python-3 port of the official MaxMatch (M2) scorer (GPL-2.0, see its LICENSE). |

### Systems in `outputs/`

* `E/final_e{42,1337,2024}`: the **primary system** (filtered C4-200M sample with 15% identity pairs, stage 1; stage 2
  on the essay-level W&I+LOCNESS split with three seeds; beam 5, length penalty 0.7, refinement, threshold tau = 0),
  with decoding variants `_no_threshold`, `_no_refinement`, `E/final_e42_tau*` and `E/final_e42_lp1.0`.
* `E/nogate_e*`: the same without identity pairs; `E/noedit_s1337_*` and `E/clean_s1337_e1337`: the second stage-1 seed
  pair; `E/final_curve{25,50}`: the learner-data budget curve.
* `final_stage1*`, `nogate_stage1`, `clean_s1337`, `noisy_s42`, `noisy_s1337`: stage-1 models (identity-pair and
  filtering contrasts).
* `baselines/*`: CoEdIT-large, gec-t5-small/base-clang8, Qwen2.5-1.5B-Instruct (0- and 4-shot).
* `round2/*`, `superseded/*`, `earlier_config/*`: systems of earlier versions of this work (see *Corrections*); they
  are not used in the current paper.

## Verifying the paper's numbers

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python analysis/get_data.py          # CoNLL-2014 test, W&I+LOCNESS v2.1, JFLEG (public downloads)
python analysis/verify.py --check    # re-scores all outputs, recomputes the comparisons, compares with results/verified/
```

`verify.py` asserts that the per-sentence counts reproduce the official M2 totals. Comparisons are paired bootstraps
over sentences (2,000 resamples, generator seeded by comparison name); a comparison with several seed pairs reports the
mean paired difference, with 95% and 90% intervals, Holm-adjusted p within each table and the TOST p for a +/-0.01 F0.5
margin. ERRANT scoring of all systems takes a few hours on one CPU core (`--only <system> ...` restricts the run;
parallel runs share the cache in `.cache/`).

The blind BEA-2019 test set cannot be scored locally. `results/bea19_test_codabench/E_primary/` holds the official
CodaBench results of the three primary runs (submissions 1035236, 1035237, 1035238 to competition 10960; F0.5 59.25,
59.76 and 59.53).

## Reproducing the experiments

All training ran on the free Kaggle tier (2 x NVIDIA T4), GPU T4 x2 and Internet on.

| Notebook | What | Inputs | Time |
|---|---|---|---|
| `R2_A_cleaning_full_budget.ipynb` | stage-1 runs of the filtering contrast (filtered/unfiltered, seeds 42 and 1337) | Kaggle dataset `dariocioni/c4200m` | 2 sessions x ~7 h |
| `R2_C_retrain_core.ipynb` | stage-1 models with (`final`) and without (`nogate`) identity pairs, seed 42 | `dariocioni/c4200m` | ~4 h per job |
| `R2_E_round3.ipynb` | session 1: primary stage 2 x 3 seeds, CoEdIT-large, latency; session 2: no-identity stage 2 x 3 seeds, budget curve; session 3: second stage-1 seed pair | outputs of `R2_C` (and `R2_A` for session 3) | 3 sessions, < 12 h each |
| `R2_B_final_system_eval.ipynb` | Qwen2.5 and gec-t5 baselines (and decode factors of the earlier configuration) | earlier checkpoints | one session |

Every training run in the paper uses the corrected loss normalisation; the ratio of training to validation loss is
recorded for every run in `results/run_manifest.json` (1.04-1.17).

## Corrections relative to earlier versions of this work

* **Loss normalisation.** The Kaggle image used for the earliest runs did not divide the training loss by the number
  of gradient-accumulation steps, so gradient clipping was active at almost every step. All runs in the paper were
  retrained with corrected normalisation; systems from the earlier configuration are kept under `outputs/earlier_config/`.
* **Duplicated stage-2 data.** Earlier notebooks read the W&I+LOCNESS files `A`, `B`, `C` and the combined `ABC` file,
  so every stage-2 pair appeared twice (`outputs/superseded/`). Fixed: A, B and C only (34,308 pairs).
* **Sentence-level holdout.** The round-2 stage-2 holdout was drawn by source sentence, and 1,704 of its 1,708 pairs
  came from essays that also supplied training pairs (`results/analysis/D_holdout_essay_overlap.json`;
  `outputs/round2/`). Fixed: essay-level holdout (`splits/wi_holdout_essay_index.txt`).
* **In-domain contamination of the unfiltered sample.** Its split was drawn independently, so 4,374 of the 5,000
  in-domain test pairs are in its training split. The filtering contrast is scored in domain on the 626 rows of
  `splits/c4_test_subset_uncontaminated.txt` (`results/analysis/overlap_audit.json`).

## Licences

Code: MIT (`LICENSE`). Third-party M2 scorer: GPL-2.0. Data-derived files: see `LICENSE-DATA.md`. Benchmark sources
and references are not redistributed; `analysis/get_data.py` downloads them from their official locations.
