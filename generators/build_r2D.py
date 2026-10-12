"""Build R2_D_stage2_dedup.ipynb -- retrain STAGE 2 on a correctly reconstructed W&I+LOCNESS training set.

Defect fixed: the earlier notebooks globbed '*.train.gold.bea19.m2', which matches A, B, C *and* the combined ABC file,
so every W&I training sentence appeared twice (68,616 = 2 x 34,308) and 94% of the 5% 'holdout' also occurred in the
training part. Here: A, B and C only (34,308 pairs); the holdout is 5% of the UNIQUE source sentences, so no source
sentence occurs in both parts. Stage 1 is NOT retrained: the stage-1 checkpoints of Notebook C are reused.

Jobs: final  = stage-1 model WITH no-edit pairs (Notebook C 'final', t5v3_model/)   -> the paper's final system
      nogate = stage-1 model WITHOUT no-edit pairs (Notebook C 'nogate', model_nogate/) -> no-edit ablation, end to end
"""
import ast, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); P = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import build_r2C as BC          # rebuilds R2_C identically and embeds the ported m2scorer into nb_common.DATA
C = BC.C

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}

CFG = r'''# ============================== CONFIG ==============================
import os; os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
JOBS = ["final", "nogate"]       # a job runs only if its stage-1 model is found among the inputs
QUICK_TEST = False               # True: ~25-min smoke test on small subsets
SESSION_BUDGET_H = 11.6
RESULTS_NAME, EXPORT_NAME = "r2D_results.json", "r2D_results.zip"
SEED = 42
TEST_N, C4_EVAL_N, SUBSET_SEED = 25_000, (5_000 if not QUICK_TEST else 300), 2026   # same C4 subset as notebooks A-C
PER_DEVICE_BATCH, GRAD_ACCUM, EPOCHS = 16, 4, 1
LEARNING_RATE, WARMUP_FRAC, WEIGHT_DECAY = 3e-4, 0.06, 0.01
LABEL_SMOOTHING, SRC_DROPOUT_P = 0.1, 0.05
EVAL_SAVE_STEPS = 2500
STAGE2_EPOCHS, STAGE2_LR, STAGE2_GA = 3, 1e-4, 2         # stage-2 global batch 64 (unchanged recipe)
HOLDOUT_FRAC = 0.05                                      # of the unique W&I source sentences
GATE_INIT_BIAS = 2.0
TAUS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
MAX_TRAIN_EVAL_RATIO = 2.0
RUN2_CLEAN_TEST_SHA256 = "%s"
''' % BC.CLEAN_TEST_SHA

LOCATE = r'''# ---------------------------------------------------------------- inputs: stage-1 models + the clean C4 test split
import hashlib
def find_dirs(name):
    hits = []
    for root, dirs, files in os.walk("/kaggle/input", followlinks=True):
        if Path(root).name == name and "config.json" in files and any(f.endswith((".safetensors", ".bin")) for f in files):
            hits.append(Path(root))
    return hits
S1 = {"final": find_dirs("t5v3_model"), "nogate": find_dirs("model_nogate")}
for job, hits in S1.items(): print(f"stage-1 model for '{job}':", [str(h) for h in hits] or "NOT FOUND")
S1 = {job: hits[0] for job, hits in S1.items() if hits}
JOBS = [j for j in JOBS if j in S1]
assert JOBS, ("No stage-1 model found. Add Input -> Your Work -> R2_C_retrain_core: the version that ran JOBS=['final'] "
              "(contains t5v3_model/) and/or the version that ran JOBS=['nogate'] (contains model_nogate/).")
te_path = None
for root, dirs, files in os.walk("/kaggle/input", followlinks=True):
    if "clean_test.parquet" in files: te_path = Path(root) / "clean_test.parquet"; break
assert te_path is not None, "clean_test.parquet not found in the attached Notebook C output"
te = pq.read_table(te_path).to_pandas()
sha = hashlib.sha256("\n".join(f"{s}\t{t}" for s, t in zip(te.src, te.tgt)).encode("utf-8")).hexdigest()
assert sha == RUN2_CLEAN_TEST_SHA256, "the attached clean test split differs from the paper's"
SUB_IDX = np.sort(np.random.default_rng(SUBSET_SEED).permutation(TEST_N)[:C4_EVAL_N])
C4_SRC = [te.src[i] for i in SUB_IDX]; C4_TGT = [te.tgt[i] for i in SUB_IDX]; del te
print("jobs to run:", JOBS, "| clean C4 test split verified (sha256) | C4 subset", len(C4_SRC))

# ---------------------------------------------------------------- W&I+LOCNESS stage-2 data, reconstructed CORRECTLY
wi_files = [p for p in sorted((EV / "wi+locness" / "m2").glob("*.train.gold.bea19.m2")) if not p.name.startswith("ABC.")]
assert [p.name for p in wi_files] == ["A.train.gold.bea19.m2", "B.train.gold.bea19.m2", "C.train.gold.bea19.m2"], wi_files
pairs = [(s, apply_m2_edits(s, ed.get(0, []))) for p_ in wi_files for s, ed in read_m2(p_)]
assert len(pairs) == 34_308, len(pairs)
srcs_u = sorted(set(s for s, _ in pairs))
perm = np.random.default_rng(SEED).permutation(len(srcs_u)); n_hold = int(round(len(srcs_u) * HOLDOUT_FRAC))
hold_src = set(srcs_u[i] for i in perm[:n_hold])
tr_idx = [i for i, (s, _) in enumerate(pairs) if s not in hold_src]
ho_idx = [i for i, (s, _) in enumerate(pairs) if s in hold_src]
assert not (set(pairs[i][0] for i in tr_idx) & hold_src) and len(tr_idx) + len(ho_idx) == len(pairs)
if QUICK_TEST: tr_idx, ho_idx = tr_idx[:3000], ho_idx[:200]
pq.write_table(pa.table({"src": [pairs[i][0] for i in tr_idx], "tgt": [pairs[i][1] for i in tr_idx]}), OUT / "wi_train_dedup.parquet")
pq.write_table(pa.table({"src": [pairs[i][0] for i in ho_idx], "tgt": [pairs[i][1] for i in ho_idx]}), OUT / "wi_holdout_dedup.parquet")
HOLD_SRC, HOLD_TGT = [pairs[i][0] for i in ho_idx], [pairs[i][1] for i in ho_idx]
(OUT / "wi_holdout_index.txt").write_text("\n".join(map(str, ho_idx)) + "\n")      # indices into A+B+C file order
RESULTS["wi_split"] = {"files": [p.name for p in wi_files], "pairs": len(pairs), "unique_sources": len(srcs_u),
                       "unique_pairs": len(set(pairs)), "holdout_sources": n_hold, "train_pairs": len(tr_idx),
                       "holdout_pairs": len(ho_idx), "holdout_rule": "5% of unique source sentences, seed 42"}
save_results(); print("W&I+LOCNESS:", json.dumps(RESULTS["wi_split"]))
'''

JOBS = r'''# ---------------------------------------------------------------- run the jobs
NEED_H = {"final": 2.8, "nogate": 2.0} if not QUICK_TEST else {"final": 0.4, "nogate": 0.3}
for job in JOBS:
    tag = f"{job}_s2dedup"
    if RESULTS.get(f"done|{job}"): print("done already:", job); continue
    if not budget_ok(NEED_H[job], job): continue
    save = OUT / f"{job}_stage2_dedup"
    train(tag, OUT / "wi_train_dedup.parquet", OUT / "wi_holdout_dedup.parquet", S1[job], save, stage=2)
    tok, m = load_model(save)
    tau, curve = tune_tau(tok, m, HOLD_SRC, HOLD_TGT); RESULTS[f"tau|{tag}"] = {"tau": tau, "curve": curve}; save_results()
    if job == "final":
        score_all(tag, tok, m, taus=tuple(dict.fromkeys([tau, None, 0.05, 0.1, 0.2])), refines=(False, True),
                  with_test=True, test_tau=tau)
    else:
        score_all(tag, tok, m, taus=tuple(dict.fromkeys([tau, None])), refines=(True,))
    del m; gc.collect(); torch.cuda.empty_cache()
    RESULTS[f"done|{job}"] = True; save_results()
'''

EVAL = BC.EVAL.replace("bea19_test_submission_retrained.zip", "bea19_test_submission_dedup.zip")
assert EVAL.count("bea19_test_submission_dedup.zip") == 2

INTRO = md("""# Round 2 - Notebook D: retrain stage 2 on a correctly reconstructed W&I+LOCNESS set

The earlier notebooks read **A, B, C and the combined ABC** W&I training files, so every sentence appeared twice and 94% of
the 5% holdout also occurred in training. This notebook re-runs **only stage 2** (about 15 min of training per model) on
A+B+C (34,308 pairs) with a holdout of 5% of the unique source sentences, reusing the stage-1 models of Notebook C,
then evaluates exactly as before and writes a new blind BEA-2019 test file.

**Setup (Kaggle):** GPU **T4 x2** - Internet **on** - **Add Input -> Your Work -> `R2_C_retrain_core`**:
* the version that ran `JOBS = ["final"]` (its output has `t5v3_model/` and `clean_test.parquet`), and
* the version that ran `JOBS = ["nogate"]` (its output has `model_nogate/`).

If Kaggle lets you attach only one version at a time, run this notebook twice (once with each); the notebook runs
whichever job's model it finds. No C4 dataset is needed. ~2.5 h for `final`, ~1.5 h for `nogate`.

Send back **`r2D_results.zip`**, and upload **`bea19_test_submission_dedup.zip`** to CodaBench as before.""")

cells = [INTRO, code(CFG), code(C.INSTALL), code(C.IMPORTS), code(C.DATA), code(C.SCORE), code(C.T5DECODE),
         code(LOCATE), code(BC.GATE), code(BC.TRAIN), code(EVAL), code(JOBS), code(C.EXPORT)]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = f"{P}/R2_D_stage2_dedup.ipynb"
json.dump(nb, open(out, "w", encoding="utf-8"), indent=1)
for c in cells:
    if c["cell_type"] == "code": ast.parse("".join(c["source"]))
print("wrote", out, "-", len(cells), "cells | syntax OK")
