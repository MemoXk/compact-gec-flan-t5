"""Assemble the two round-2 Kaggle notebooks.

  R2_A_cleaning_full_budget.ipynb  -- reference-side cleaning at the FINAL training budget, 2 seeds
  R2_B_final_system_eval.ipynb     -- blind BEA test, decode-only factors at full budget, no-edit factor,
                                      GEC-specific baselines, fair Qwen re-run, official JFLEG GLEU,
                                      latency/memory on one T4, reference-free harm audit
"""
import base64, gzip, hashlib, csv, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nb_common as C

P = r"<PROJECT>"
V3 = ["".join(c.get("source", "")) for c in json.load(open(f"{P}/t5v3_final_system.ipynb", encoding="utf-8"))["cells"]]


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
def write(name, cells):
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(f"{P}/{name}", "w", encoding="utf-8"), indent=1)
    print("wrote", name, "-", len(cells), "cells")
def embed(obj): return base64.b64encode(gzip.compress(json.dumps(obj).encode())).decode()


# ---------------------------------------------------------------- facts computed locally
run2 = list(csv.DictReader(open(f"{P}/run2_results/t5max_examples.csv", encoding="utf-8", errors="replace")))
v3c4 = list(csv.DictReader(open(f"{P}/v3_results/t5v3_examples.csv", encoding="utf-8", errors="replace")))
assert [(r["src"], r["tgt"]) for r in run2] == [(r["src"], r["tgt"]) for r in v3c4], "run2 and v3 test sets differ"
CLEAN_TEST_SHA = hashlib.sha256("\n".join(f"{r['src']}\t{r['tgt']}" for r in run2).encode("utf-8")).hexdigest()
print("clean test set sha256 (run2 == v3):", CLEAN_TEST_SHA[:16], "...")

# the final system's noise filters, VERBATIM (cell 4 up to its demo block)
FILTERS = V3[4].split("# Sanity check on real noise examples")[0]
assert "def pair_noise_reason" in FILTERS and "_RE_PUNCT_RUN" in FILTERS


import io as _io, tarfile as _tf
_buf = _io.BytesIO()
with _tf.open(fileobj=_buf, mode="w") as _t:
    _t.add(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval", "m2scorer"), arcname="m2scorer")
M2_B64 = base64.b64encode(gzip.compress(_buf.getvalue())).decode()
C.DATA = C.DATA.replace("__M2_PORTED_B64__", M2_B64)
print("embedded ported m2scorer:", round(len(M2_B64) / 1e3), "KB")
# ================================================================= NOTEBOOK A
A_CFG = r'''# ============================== CONFIG ==============================
# Reference-side cleaning at the FINAL training budget (reviewer comments R1.4 / R2.3).
# Each run = one epoch over 1,150,000 pairs with the exact recipe of the paper's "no gate" run (run 2),
# changing ONE factor: whether the final system's noise filters are applied when sampling C4-200M.
#   clean, seed 42  = run 2 itself (its predictions are already in the paper's archive -> not re-run)
# One run ~= 2.8 h training + ~0.8 h evaluation on 2x T4.  Two runs fit one Kaggle session.
RUNS = [("noisy", 42), ("noisy", 1337)]      # SESSION 1.   SESSION 2: RUNS = [("clean", 1337)]
QUICK_TEST = False                            # True = ~25-min smoke test on tiny data (do this first!)
SESSION_BUDGET_H = 11.6
RESULTS_NAME, EXPORT_NAME = "r2A_results.json", "r2A_results.zip"

MODEL_NAME = "google/flan-t5-small"
PER_SHARD = 120_000 if not QUICK_TEST else 3_000        # 10 shards -> 1.2M pairs
VAL_N = TEST_N = 25_000 if not QUICK_TEST else 1_000
C4_EVAL_N = 5_000 if not QUICK_TEST else 300             # fixed subset of the 25k clean test
SPLIT_SEED, SUBSET_SEED = 42, 2026
# run-2 recipe, verified against t5max_final_experiment.ipynb and its trainer log
PER_DEVICE_BATCH, GRAD_ACCUM, EPOCHS = 16, 4, 1           # global batch 128 on 2 GPUs
LEARNING_RATE, WARMUP_FRAC, WEIGHT_DECAY = 3e-4, 0.06, 0.01
LABEL_SMOOTHING, SRC_DROPOUT_P = 0.1, 0.05
EVAL_SAVE_STEPS = 2500 if not QUICK_TEST else 50          # best checkpoint by validation loss (as run 2)
MIN_W, MAX_W, MAX_LEN_DIFF, MIN_BAG_OVERLAP, MAX_NONASCII_RATIO = 4, 50, 10, 0.5, 0.25
RUN2_CLEAN_TEST_SHA256 = "%s"
''' % CLEAN_TEST_SHA

A_SAMPLE = r'''# ---------------------------------------------------------------- C4-200M sampling, clean vs noisy
# Identical to the final system's sampler (t5v3_final_system.ipynb, cell 5) except that `apply_filters`
# can switch the noise filters OFF.  Every kept pair also records (shard, line) for the public index.
import hashlib
def require_c4_shards():
    for root in (Path("/kaggle/input/datasets/dariocioni/c4200m"), Path("/kaggle/input")):
        f = sorted(root.rglob("C4_200M.tsv-*")) if root.exists() else []
        if f: return f
    raise FileNotFoundError("Attach the C4-200M dataset (Add Input -> dariocioni/c4200m)")

def build_corpus(tag, apply_filters):
    if (OUT / f"{tag}_test.parquet").exists(): print(tag, "splits exist"); return
    files = require_c4_shards(); stats = Counter(); seen = set(); parts = []
    for i, fpath in enumerate(files):
        kept = 0; S, T, L = [], [], []
        with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
            for ln_no, line in enumerate(fh):
                stats["scanned"] += 1
                cols = line.rstrip("\r\n").split("\t")
                if len(cols) != 2: stats["drop_format"] += 1; continue
                src, tgt = normalize_ws(cols[0]), normalize_ws(cols[1])
                if not src or not tgt or src == tgt: stats["drop_identical"] += 1; continue
                ls, lt = len(src.split()), len(tgt.split())
                if not (MIN_W <= ls <= MAX_W and MIN_W <= lt <= MAX_W): stats["drop_length"] += 1; continue
                h = hashlib.md5(src.encode("utf-8")).digest()
                if h in seen: stats["drop_duplicate"] += 1; continue
                if apply_filters:
                    reason = pair_noise_reason(src, tgt)
                    if reason is not None: stats["drop_" + reason] += 1; continue
                seen.add(h); S.append(src); T.append(tgt); L.append(ln_no); kept += 1; stats["kept"] += 1
                if kept >= PER_SHARD: break
        parts.append(pa.table({"src": pa.array(S, type=pa.large_string()), "tgt": pa.array(T, type=pa.large_string()),
                               "shard": pa.array([i] * len(S), type=pa.int8()), "line": pa.array(L, type=pa.int64())}))
        print(f"  [{tag}] shard {i}: kept {kept:,} | total {stats['kept']:,}", flush=True)
    full = pa.concat_tables(parts); n = full.num_rows
    perm = np.random.default_rng(SPLIT_SEED).permutation(n)          # same split rule as the paper
    idx = {"test": perm[:TEST_N], "val": perm[TEST_N:TEST_N + VAL_N], "train": perm[TEST_N + VAL_N:]}
    rows = []
    for name, ix in idx.items():
        tb = full.take(pa.array(ix)); pq.write_table(tb.select(["src", "tgt"]), OUT / f"{tag}_{name}.parquet")
        rows += [(int(s), int(l), name) for s, l in zip(tb.column("shard").to_pylist(), tb.column("line").to_pylist())]
    pd.DataFrame(rows, columns=["shard", "line", "split"]).to_csv(OUT / f"split_index_{tag}.tsv.gz", sep="\t", index=False)
    json.dump({"tag": tag, "apply_filters": apply_filters, "filter_stats": dict(stats)},
              open(OUT / f"c4_meta_{tag}.json", "w"), indent=1)
    print(f"[{tag}] {dict(stats)}")

need_noisy = any(a == "noisy" for a, _ in RUNS)
build_corpus("clean", True)
if need_noisy: build_corpus("noisy", False)
te = pq.read_table(OUT / "clean_test.parquet").to_pandas()
sha = hashlib.sha256("\n".join(f"{s}\t{t}" for s, t in zip(te.src, te.tgt)).encode("utf-8")).hexdigest()
RESULTS["clean_test_reproduces_run2"] = (sha == RUN2_CLEAN_TEST_SHA256); save_results()
print("clean test set byte-identical to the paper's run 2 / final system:", sha == RUN2_CLEAN_TEST_SHA256)
SUB_IDX = np.sort(np.random.default_rng(SUBSET_SEED).permutation(TEST_N)[:C4_EVAL_N])
C4_SRC = [te.src[i] for i in SUB_IDX]; C4_TGT = [te.tgt[i] for i in SUB_IDX]
pd.DataFrame({"test_index": SUB_IDX}).to_csv(OUT / "c4_eval_subset_index.csv", index=False)
del te; gc.collect()
'''

A_TRAIN = r'''# ---------------------------------------------------------------- training (run-2 recipe, one factor changed)
def _drop(text, p, rng):
    w = str(text).split()
    if p <= 0 or len(w) < 4: return text
    k = [x for x in w if rng.random() > p]
    return " ".join(k) if k else text
class GECDataset(Dataset):
    def __init__(self, path, tok, dp=0.0, seed=42):
        t = pq.read_table(path); self.s, self.t, self.tok, self.dp, self.seed = t.column("src"), t.column("tgt"), tok, dp, seed
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        s, t = str(self.s[i].as_py()), str(self.t[i].as_py())
        if self.dp > 0 and s != t: s = _drop(s, self.dp, random.Random(self.seed * 1_000_003 + i))
        e = self.tok(PREFIX + s, max_length=MAX_INPUT_LEN, truncation=True)
        e["labels"] = self.tok(text_target=t, max_length=MAX_TARGET_LEN, truncation=True)["input_ids"]
        return e

def train_arm(arm, seed):
    tag = f"{arm}_s{seed}"; save_dir = OUT / f"model_{tag}"
    if (save_dir / "config.json").exists(): return tag, save_dir
    import inspect
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME); m = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    tr = GECDataset(OUT / f"{arm}_train.parquet", tok, SRC_DROPOUT_P, seed)
    va = GECDataset(OUT / f"{arm}_val.parquet", tok, 0.0, seed + 1)
    if QUICK_TEST: va = torch.utils.data.Subset(va, range(200))
    n_gpu = max(1, torch.cuda.device_count())
    total = math.ceil(len(tr) / (PER_DEVICE_BATCH * GRAD_ACCUM * n_gpu)) * EPOCHS
    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8   # T4 -> fp16, as run 2
    a = dict(output_dir=str(OUT / f"ck_{tag}"), learning_rate=LEARNING_RATE, lr_scheduler_type="cosine",
             per_device_train_batch_size=PER_DEVICE_BATCH, per_device_eval_batch_size=64, gradient_accumulation_steps=GRAD_ACCUM,
             num_train_epochs=EPOCHS, warmup_steps=max(1, int(total * WARMUP_FRAC)), weight_decay=WEIGHT_DECAY,
             label_smoothing_factor=LABEL_SMOOTHING, logging_steps=500 if not QUICK_TEST else 10,
             save_steps=EVAL_SAVE_STEPS, eval_steps=EVAL_SAVE_STEPS, save_total_limit=2, predict_with_generate=False,
             fp16=torch.cuda.is_available() and not bf16, bf16=bf16, dataloader_num_workers=2, report_to=[], seed=seed,
             load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False, save_strategy="steps")
    a["eval_strategy" if "eval_strategy" in inspect.signature(Seq2SeqTrainingArguments.__init__).parameters else "evaluation_strategy"] = "steps"
    t = Seq2SeqTrainer(model=m, args=Seq2SeqTrainingArguments(**a), train_dataset=tr, eval_dataset=va,
                       data_collator=DataCollatorForSeq2Seq(tok, model=m, label_pad_token_id=-100))
    t0 = time.time(); o = t.train(); dt = time.time() - t0
    assert o.training_loss > 1e-6, "training loss is zero -- labels are not reaching the model"
    t.save_model(str(save_dir)); tok.save_pretrained(str(save_dir))
    RESULTS[f"train|{tag}"] = {"train_rows": len(tr), "steps": int(o.global_step), "planned_steps": total,
        "warmup_steps": max(1, int(total * WARMUP_FRAC)), "train_seconds": round(dt, 1),
        "examples_per_s": round(len(tr) * EPOCHS / dt, 2), "train_loss": float(o.training_loss),
        "best_checkpoint": t.state.best_model_checkpoint, "best_eval_loss": t.state.best_metric, "seed": seed}
    json.dump(t.state.log_history, open(OUT / f"trainer_log_{tag}.json", "w"))
    save_results(); shutil.rmtree(OUT / f"ck_{tag}", ignore_errors=True)
    del t, m; gc.collect(); torch.cuda.empty_cache()
    return tag, save_dir
'''

A_EVAL = r'''# ---------------------------------------------------------------- evaluation with run 2's exact decoding
# run 2 decoding = beam 5, length penalty 0.7, no-repeat 3, guarded refinement, NO confidence threshold.
def evaluate(tag, mdir):
    if f"eval|{tag}" in RESULTS: print("already evaluated:", tag); return
    tok, m = load_t5(mdir); r = {}
    def run(srcs, name):
        p = refine(tok, m, srcs, t5_gen(tok, m, srcs, desc=f"{tag}/{name}"))
        return [norm_ws(x) for x in p]
    p = run(C4_SRC, "c4"); m1, _ = single_ref_errant(C4_SRC, C4_TGT, p)
    r["c4_clean_subset"] = {**m1, **outcome_table(C4_SRC, C4_TGT, p)}
    pd.DataFrame({"test_index": SUB_IDX, "src": C4_SRC, "tgt": C4_TGT, "pred": p}).to_csv(OUT / f"pred_c4_{tag}.csv", index=False)
    bs = BEA_SRC[:BEA_N]; p = run(bs, "bea_dev"); pr = [RETOK["bea"](x) for x in p]
    r["bea_dev_errant"] = errant_official(bs, pr, BEA_DEV_M2, f"bea_{tag}")
    r["bea_dev_outcome"] = outcome_table(bs, BEA_TGT[:BEA_N], pr)
    pd.DataFrame({"src": bs, "pred": p, "pred_retok": pr}).to_csv(OUT / f"pred_bea_dev_{tag}.csv", index=False)
    p = run(CONLL_SRC, "conll"); pr = [RETOK["conll"](x) for x in p]
    r["conll_m2"] = official_m2(pr, f"conll_{tag}")
    pd.DataFrame({"src": CONLL_SRC, "pred": p, "pred_retok": pr}).to_csv(OUT / f"pred_conll_{tag}.csv", index=False)
    js = JF_SRC[:JF_N]; p = run(js, "jfleg"); pr = [RETOK["jfleg"](x) for x in p]
    r["jfleg_gleu_official"] = jfleg_gleu(pr, JF_N)
    pd.DataFrame({"src": js, "pred": p, "pred_retok": pr}).to_csv(OUT / f"pred_jfleg_{tag}.csv", index=False)
    RESULTS[f"eval|{tag}"] = r; save_results()
    print(tag, json.dumps({k: (v.get("F05") or v.get("GLEU")) for k, v in r.items() if isinstance(v, dict)}, default=str))
    del m; gc.collect(); torch.cuda.empty_cache()

if "jfleg_do_nothing" not in RESULTS:
    RESULTS["jfleg_do_nothing"] = jfleg_gleu([RETOK["jfleg"](x) for x in JF_SRC[:JF_N]], JF_N); save_results()
print("JFLEG do-nothing (official GLEU):", RESULTS["jfleg_do_nothing"])
for arm, seed in RUNS:
    need = 3.9 if not QUICK_TEST else 0.3
    if f"eval|{arm}_s{seed}" in RESULTS: continue
    if not budget_ok(need, f"{arm} seed {seed}"): break
    tag, mdir = train_arm(arm, seed)
    evaluate(tag, mdir)
'''

A_INTRO = md("""# Round 2 - Notebook A: reference-side cleaning at the final training budget

Answers reviewer comments **R1.4** and **R2.3**: the paper's cleaning ablation ran at a reduced 150k-pair budget (F0.5 ~ 0.13)
and with an earlier variant of the filters. Here each run trains for **one epoch over 1,150,000 pairs** with the exact recipe of the
paper's run 2, changing only whether the final system's noise filters are applied. The clean seed-42 arm *is* run 2, so it is not retrained.

**Setup (Kaggle):** Accelerator **GPU T4 x2** - Internet **on** - Add Input -> dataset **`dariocioni/c4200m`**.

**Order of runs**
1. First set `QUICK_TEST = True`, *Save & Run All* (~25 min). Check that the last cell prints results without errors.
2. Set `QUICK_TEST = False`, keep `RUNS = [("noisy", 42), ("noisy", 1337)]`, *Save & Run All* (~7.5 h).
3. New version: `RUNS = [("clean", 1337)]`, *Save & Run All* (~4 h). The two sessions can run at the same time.
4. Download **`r2A_results.zip`** from each version's Output tab and send both back.

Model weights stay in the version output (needed later for the public reproducibility package).""")

nbA = [A_INTRO, code(A_CFG), code(C.INSTALL), code(C.IMPORTS), code(C.DATA), code(C.SCORE), code(C.T5DECODE),
       code("# ---------------------------------------------------------------- the final system's noise filters (verbatim)\n" + FILTERS),
       code(A_SAMPLE), code(A_TRAIN), code(A_EVAL), code(C.EXPORT)]
write("R2_A_cleaning_full_budget.ipynb", nbA)

# ================================================================= NOTEBOOK B
ours_bea = list(csv.DictReader(open(f"{P}/v3_results/t5v3_bea19_examples.csv", encoding="utf-8", errors="replace")))
ours_conll = list(csv.DictReader(open(f"{P}/v3_results/t5v3_conll14_examples.csv", encoding="utf-8", errors="replace")))
STORED = embed({"bea_dev_pred": [r["pred"] for r in ours_bea], "bea_dev_src": [r["src"] for r in ours_bea],
                "conll_pred": [r["pred"] for r in ours_conll]})
print("embedded stored predictions:", round(len(STORED) / 1e3), "KB")

B_CFG = r'''# ============================== CONFIG ==============================
QUICK_TEST = False
SESSION_BUDGET_H = 11.6
RESULTS_NAME, EXPORT_NAME = "r2B_results.json", "r2B_results.zip"
RUN = dict(bea_test=True, decode_factors=True, noedit_factor=True, gec_baselines=True,
           qwen=True, jfleg=True, latency=True, judge=True)
C4_EVAL_N, SUBSET_SEED = (5_000 if not QUICK_TEST else 300), 2026        # same subset as notebook A
LPS_C4, LPS_BENCH = (0.7, 1.0, 1.2), (0.7, 1.0)
TAUS = (None, 0.0, 0.05, 0.1, 0.2)                                       # None = no threshold; paper uses 0.0
GEC_BASELINES = ["gotutiyan/gec-t5-small-clang8", "gotutiyan/gec-t5-base-clang8", "gotutiyan/gector-roberta-base-5k"]
LLM_NAME, LLM_MAX_NEW, LLM_BATCH = "Qwen/Qwen2.5-1.5B-Instruct", 96, 32
JUDGE_NAME, JUDGE_FALLBACK = "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-3B-Instruct"
JUDGE_N = {"harm": 150, "lateral": 150, "improved": 100} if not QUICK_TEST else {"harm": 8, "lateral": 8, "improved": 4}
LAT_N, LAT_WARM = (300, 10) if not QUICK_TEST else (30, 3)
'''

B_STORED = 'STORED = json.loads(__import__("gzip").decompress(__import__("base64").b64decode("' + STORED + '")))\n' + \
           'print({k: len(v) for k, v in STORED.items()})\n'

B_LOCATE = r'''# ---------------------------------------------------------------- the paper's final checkpoints
# Add Input -> "Your Work" -> the t5v3_final_system notebook output. Every model folder found is listed; if the final
# folders t5v3_model / t5v3_model_stage2 are absent, the trainer's BEST checkpoint (lowest validation loss) is used.
MODELS = []
for root, dirs, files in os.walk("/kaggle/input", followlinks=True):
    if "config.json" in files and any(f in files for f in ("model.safetensors", "pytorch_model.bin")):
        MODELS.append(Path(root))
print("model folders under /kaggle/input:")
for mdl in MODELS: print("   ", mdl)
for top in sorted(Path("/kaggle/input").glob("*/*/*")) if Path("/kaggle/input").exists() else []:
    if top.is_dir(): print("  input dir:", top, "->", sorted(p.name for p in top.iterdir())[:40])
def pick(final_name, ckpt_parent):
    for mdl in MODELS:
        if mdl.name == final_name: return mdl, "final folder"
    cks = [mdl for mdl in MODELS if mdl.parent.name == ckpt_parent and mdl.name.startswith("checkpoint-")]
    if not cks: return None, "missing"
    newest = max(cks, key=lambda p: int(p.name.split("-")[-1]))
    st = newest / "trainer_state.json"
    best = json.loads(st.read_text()).get("best_model_checkpoint") if st.exists() else None
    for mdl in cks:
        if best and mdl.name == Path(best).name: return mdl, "best checkpoint (trainer_state)"
    return newest, "newest checkpoint (no trainer_state)"
(S1_DIR, S1_HOW), (S2_DIR, S2_HOW) = pick("t5v3_model", "t5v3_ckpts"), pick("t5v3_model_stage2", "t5v3_ckpts_stage2")
if S1_DIR is not None and (S1_DIR.parent / "test.parquet").exists():
    TEST_PQ = S1_DIR.parent / "test.parquet"
else:
    TEST_PQ = next(iter(Path("/kaggle/input").rglob("test.parquet")), None) if Path("/kaggle/input").exists() else None
HAVE_CKPT = S1_DIR is not None and S2_DIR is not None
RESULTS["checkpoints"] = {"stage1": str(S1_DIR), "stage1_how": S1_HOW, "stage2": str(S2_DIR), "stage2_how": S2_HOW,
                          "all_model_folders": [str(x) for x in MODELS]}; save_results()
print("stage-1:", S1_DIR, f"[{S1_HOW}]", "| stage-2:", S2_DIR, f"[{S2_HOW}]", "| test.parquet:", TEST_PQ)
if not HAVE_CKPT:
    print("!! Final checkpoints NOT found -- see the folder list above and send it to Claude. Baselines, Qwen,")
    print("!! JFLEG do-nothing and the judge (on the paper's stored predictions) still run.")
C4_SRC = C4_TGT = SUB_IDX = None
if TEST_PQ is not None:
    te = pq.read_table(TEST_PQ).to_pandas()
    SUB_IDX = np.sort(np.random.default_rng(SUBSET_SEED).permutation(len(te))[:C4_EVAL_N])
    C4_SRC = [te.src[i] for i in SUB_IDX]; C4_TGT = [te.tgt[i] for i in SUB_IDX]; del te
'''

B_P2 = r'''# ---------------------------------------------------------------- P2: blind BEA-2019 test + reproduction check
# Final configuration of the paper: stage-2 model, beam 5, lp 0.7, refinement, confidence threshold tau = 0.0
if RUN["bea_test"] and HAVE_CKPT and "bea_test" not in RESULTS and budget_ok(0.7, "BEA test"):
    tok, m = load_t5(S2_DIR)
    def final(srcs, name): return decode_variants(tok, m, srcs, (0.7,), (True,), (0.0,), desc=name)[(0.7, True, 0.0)]
    bs = BEA_SRC[:BEA_N]; p = final(bs, "bea-dev"); pr = [RETOK["bea"](x) for x in p]
    RESULTS["bea_dev_reproduction"] = {"errant": errant_official(bs, pr, BEA_DEV_M2, "bea_repro"),
        "paper_reported_F05": 0.4288,
        "identical_to_stored_outputs_pct": round(100 * np.mean([a == b for a, b in zip(p, STORED["bea_dev_pred"][:len(p)])]), 2)}
    pd.DataFrame({"src": bs, "pred": p, "pred_retok": pr}).to_csv(OUT / "pred_bea_dev_final.csv", index=False)
    tsrc = BEA_TEST_SRC if not QUICK_TEST else BEA_TEST_SRC[:50]
    p = final(tsrc, "bea-test"); pr = [RETOK["bea"](x) for x in p]
    (OUT / "bea19_test_hyp.txt").write_text("\n".join(pr) + "\n", encoding="utf-8")
    with zipfile.ZipFile(OUT / "bea19_test_submission.zip", "w") as z: z.write(OUT / "bea19_test_hyp.txt", "bea19_test_hyp.txt")
    RESULTS["bea_test"] = {"n": len(pr), "file": "bea19_test_submission.zip", "edited_pct": outcome_table(tsrc, tsrc, pr)["edited_pct"]}
    save_results(); print(RESULTS["bea_dev_reproduction"]); del m; gc.collect(); torch.cuda.empty_cache()
'''

B_P3 = r'''# ---------------------------------------------------------------- P3: decode-only factors at FULL budget
# length penalty x refinement x confidence threshold, on the paper's final checkpoints (R1.4 / R2.3)
if RUN["decode_factors"] and HAVE_CKPT and "factors_c4" not in RESULTS and C4_SRC is not None and budget_ok(1.3, "decode factors"):
    tok, m = load_t5(S1_DIR)
    V = decode_variants(tok, m, C4_SRC, LPS_C4, (False, True), TAUS, desc="stage1-c4")
    rows, res = {}, {}
    for (lp, rf, tau), p in V.items():
        k = vkey(lp, rf, tau); mm, _ = single_ref_errant(C4_SRC, C4_TGT, p)
        res[k] = {**mm, **outcome_table(C4_SRC, C4_TGT, p)}; rows[k] = p
    pd.DataFrame({"test_index": SUB_IDX, "src": C4_SRC, "tgt": C4_TGT, **rows}).to_csv(OUT / "pred_c4_stage1_variants.csv", index=False)
    RESULTS["factors_c4"] = res; save_results(); del m; gc.collect(); torch.cuda.empty_cache()
if RUN["decode_factors"] and HAVE_CKPT and "factors_bench" not in RESULTS and budget_ok(1.0, "benchmark factors"):
    tok, m = load_t5(S2_DIR); res = {}
    bs = BEA_SRC[:BEA_N]
    for name, srcs in (("bea_dev", bs), ("conll", CONLL_SRC)):
        V = decode_variants(tok, m, srcs, LPS_BENCH, (False, True), (None, 0.0, 0.1), desc=f"stage2-{name}")
        rows = {}
        for (lp, rf, tau), p in V.items():
            k = vkey(lp, rf, tau); pr = [RETOK["bea" if name == "bea_dev" else "conll"](x) for x in p]; rows[k] = pr
            res[f"{name}|{k}"] = (errant_official(bs, pr, BEA_DEV_M2, f"f_{name}_{k}") if name == "bea_dev"
                                  else official_m2(pr, f"f_{name}_{k}"))
        pd.DataFrame({"src": srcs, **rows}).to_csv(OUT / f"pred_{name}_stage2_variants.csv", index=False)
    RESULTS["factors_bench"] = res; save_results(); del m; gc.collect(); torch.cuda.empty_cache()
'''

B_P4 = r'''# ---------------------------------------------------------------- P4: no-edit pairs at FULL budget
# stage-1 (trained WITH no-edit pairs) decoded exactly like run 2 (lp 0.7 + refinement, no threshold);
# compared offline, sentence by sentence, with run 2 (same data, same recipe, WITHOUT no-edit pairs).
if RUN["noedit_factor"] and HAVE_CKPT and "noedit_bench" not in RESULTS and budget_ok(0.6, "no-edit factor"):
    tok, m = load_t5(S1_DIR); res = {}
    for name, srcs in (("bea_dev", BEA_SRC[:BEA_N]), ("conll", CONLL_SRC)):
        p = [norm_ws(x) for x in refine(tok, m, srcs, t5_gen(tok, m, srcs, desc=f"stage1-{name}"))]
        pr = [RETOK["bea" if name == "bea_dev" else "conll"](x) for x in p]
        res[name] = (errant_official(srcs, pr, BEA_DEV_M2, f"ne_{name}") if name == "bea_dev" else official_m2(pr, f"ne_{name}"))
        pd.DataFrame({"src": srcs, "pred": p, "pred_retok": pr}).to_csv(OUT / f"pred_{name}_stage1_run2decode.csv", index=False)
    RESULTS["noedit_bench"] = res; save_results(); del m; gc.collect(); torch.cuda.empty_cache()
'''

B_P5 = r'''# ---------------------------------------------------------------- P5: GEC-specific public baselines, same splits + scorers
pip("--no-deps", "git+https://github.com/gotutiyan/gec-t5", "git+https://github.com/gotutiyan/gector")
VERB = fetch("https://github.com/grammarly/gector/raw/master/data/verb-form-vocab.txt", EV / "verb-form-vocab.txt")
def load_gector(mid, device=DEV):
    """GECToR.__init__ calls AutoModel.from_pretrained itself, which recent transformers forbids inside
    GECToR.from_pretrained (meta-device init). Build from the config instead, then load the released weights."""
    from gector import GECToR, GECToRConfig
    from huggingface_hub import hf_hub_download
    m = GECToR(GECToRConfig.from_pretrained(mid))
    try:
        from safetensors.torch import load_file
        sd = load_file(hf_hub_download(mid, "model.safetensors"))
    except Exception:
        sd = torch.load(hf_hub_download(mid, "pytorch_model.bin"), map_location="cpu")
    missing, unexpected = m.load_state_dict(sd, strict=False)
    bad = [k for k in missing if "position_ids" not in k]
    assert not bad, f"GECToR weights missing: {bad[:5]}"
    print(f"  {mid}: loaded {len(sd)} tensors (missing {len(missing)}, unexpected {len(unexpected)})")
    return m.to(device).eval()
def run_baseline(mid, srcs):
    if "gector" in mid:
        from gector import predict, load_verb_dict
        tok = AutoTokenizer.from_pretrained(mid); m = load_gector(mid)
        enc, dec = load_verb_dict(str(VERB))
        out = predict(m, tok, srcs, enc, dec, keep_confidence=0.3, min_error_prob=0.6, n_iteration=5, batch_size=64)
        how = "gector.predict(keep_confidence=0.3, min_error_prob=0.6, n_iteration=5)"
    else:
        tok = AutoTokenizer.from_pretrained(mid); m = AutoModelForSeq2SeqLM.from_pretrained(mid).to(DEV).eval()
        try:
            from gec_t5 import generate as g5
            out = g5(model=m, tokenizer=tok, sources=srcs, batch_size=32); how = "gec_t5.generate (authors' defaults)"
        except Exception as e:
            out = []; how = f"model.generate(num_beams=5, max_length=128) [gec_t5 import failed: {e!r}]"
            with torch.no_grad():
                for i in range(0, len(srcs), 32):
                    x = tok(srcs[i:i + 32], return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEV)
                    out += tok.batch_decode(m.generate(**x, num_beams=5, max_length=128), skip_special_tokens=True)
    del m; gc.collect(); torch.cuda.empty_cache()
    return [norm_ws(x) for x in out], how
if RUN["gec_baselines"]:
    for mid in GEC_BASELINES:
        key = f"baseline|{mid}"
        if key in RESULTS or not budget_ok(0.5, mid): continue
        r = {}; short = mid.split("/")[-1]
        try:
            for name, srcs in (("conll", CONLL_SRC), ("bea_dev", BEA_SRC[:BEA_N]), ("jfleg", JF_SRC[:JF_N])):
                p, how = run_baseline(mid, srcs); pr = [RETOK["bea" if name == "bea_dev" else name](x) for x in p]
                r[name] = (official_m2(pr, f"b_{short}") if name == "conll" else
                           errant_official(srcs, pr, BEA_DEV_M2, f"b_{short}") if name == "bea_dev" else jfleg_gleu(pr, JF_N))
                r[name] = r[name] or {"error": "scorer returned nothing"}
                r[name]["edited_pct"] = outcome_table(srcs, srcs, pr)["edited_pct"]; r["inference"] = how
                pd.DataFrame({"src": srcs, "pred": p, "pred_retok": pr}).to_csv(OUT / f"pred_{name}_{short}.csv", index=False)
        except Exception as e:
            r["error"] = repr(e)
        RESULTS[key] = r; save_results(); print(mid, json.dumps(r, default=str)[:600])
'''

B_P6 = r'''# ---------------------------------------------------------------- P6: Qwen2.5-1.5B-Instruct, fair re-run (R1.6)
# Fixes from round 1: (i) exemplars drawn by a stated rule from W&I+LOCNESS TRAIN (round 1 used two BEA-dev sentences),
# (ii) model revision pinned and recorded, (iii) outputs retokenized like every other system.
from huggingface_hub import model_info
SYSTEM = ("You are a grammatical error correction system. Return ONLY the corrected sentence. "
          "Make the minimum number of edits. If the sentence is already correct, repeat it unchanged.")
def pick_exemplars():
    pool = []
    for p in sorted((EV / "wi+locness" / "m2").glob("*.train.gold.bea19.m2")):
        for s, ed in read_m2(p):
            e0 = ed.get(0, []); t = apply_m2_edits(s, e0); n = len(s.split())
            if 8 <= n <= 20 and 1 <= len(e0) <= 3 and s != t and s.isascii(): pool.append((s, t))
    rng = random.Random(2024); ex = rng.sample(pool, 4)
    banned = {re.sub(r"\W+", " ", x.lower()).strip() for x in BEA_SRC + CONLL_SRC}
    assert not any(re.sub(r"\W+", " ", s.lower()).strip() in banned for s, _ in ex), "exemplar overlaps an evaluation set"
    return ex, len(pool)
def prompts(tok, srcs, shots):
    out = []
    for s in srcs:
        msgs = [{"role": "system", "content": SYSTEM}]
        for a, b in shots: msgs += [{"role": "user", "content": a}, {"role": "assistant", "content": b}]
        msgs.append({"role": "user", "content": s})
        out.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    return out
@torch.no_grad()
def llm_gen(m, tok, srcs, shots):
    P = prompts(tok, srcs, shots); out = []
    for i in range(0, len(P), LLM_BATCH):
        enc = tok(P[i:i + LLM_BATCH], return_tensors="pt", padding=True).to(m.device)
        g = m.generate(**enc, max_new_tokens=LLM_MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
        out += [norm_ws(tok.decode(x[enc.input_ids.shape[1]:], skip_special_tokens=True).split("\n")[0]) for x in g]
    return out
if RUN["qwen"] and "qwen" not in RESULTS and budget_ok(1.0, "Qwen"):
    EX, pool_n = pick_exemplars(); rev = model_info(LLM_NAME).sha
    tok = AutoTokenizer.from_pretrained(LLM_NAME, revision=rev); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(LLM_NAME, revision=rev, torch_dtype=torch.float16, device_map={"": 0}).eval()
    r = {"model": LLM_NAME, "revision": rev, "dtype": "float16 (no quantization)", "decoding": "greedy, max_new_tokens=96",
         "postprocessing": "first output line, whitespace-normalized, then benchmark retokenizer", "system_prompt": SYSTEM,
         "exemplar_rule": "W&I+LOCNESS train (first annotator), 8-20 tokens, 1-3 edits, ASCII; random.Random(2024).sample(pool, 4)",
         "exemplar_pool_size": pool_n, "exemplars": EX}
    for shots_name, shots in (("zeroshot", []), ("fourshot", EX)):
        for name, srcs in (("conll", CONLL_SRC), ("bea_dev", BEA_SRC[:BEA_N])):
            p = llm_gen(m, tok, srcs, shots); pr = [RETOK["bea" if name == "bea_dev" else "conll"](x) for x in p]
            s = (official_m2(pr, f"q_{shots_name}") if name == "conll" else errant_official(srcs, pr, BEA_DEV_M2, f"q_{shots_name}"))
            s = s or {"error": "scorer returned nothing"}; s.update(outcome_table(srcs, srcs, pr)); s["raw_glued_punct_pct"] = round(100 * np.mean([bool(re.search(r"\w[.,!?;:](\s|$)", x)) for x in p]), 1)
            r[f"{name}|{shots_name}"] = s
            pd.DataFrame({"src": srcs, "pred": p, "pred_retok": pr}).to_csv(OUT / f"pred_{name}_qwen_{shots_name}.csv", index=False)
            print(name, shots_name, s)
    RESULTS["qwen"] = r; save_results(); del m; gc.collect(); torch.cuda.empty_cache()
'''

B_P7 = r'''# ---------------------------------------------------------------- P7: JFLEG with the OFFICIAL GLEU (the paper used NLTK's MT GLEU)
if RUN["jfleg"] and "jfleg" not in RESULTS and budget_ok(0.4, "JFLEG"):
    r = {"do_nothing": jfleg_gleu([RETOK["jfleg"](x) for x in JF_SRC[:JF_N]], JF_N)}
    if HAVE_CKPT:
        tok, m = load_t5(S2_DIR); js = JF_SRC[:JF_N]
        p = decode_variants(tok, m, js, (0.7,), (True,), (0.0,), desc="jfleg")[(0.7, True, 0.0)]
        pr = [RETOK["jfleg"](x) for x in p]; r["ours_final"] = jfleg_gleu(pr, JF_N)
        pd.DataFrame({"src": js, "pred": p, "pred_retok": pr}).to_csv(OUT / "pred_jfleg_final.csv", index=False)
        del m; gc.collect(); torch.cuda.empty_cache()
    RESULTS["jfleg"] = r; save_results(); print(r)
'''

B_P8 = r'''# ---------------------------------------------------------------- P8: latency + memory, same GPU for every system (R2.5)
# One T4 (cuda:0). Batch-1 latency (median / p90 ms per sentence) and batch-32 throughput on the same CoNLL sentences.
def measure(name, fn_batch, params, n=LAT_N, warm=LAT_WARM):
    srcs = CONLL_SRC[:n]; torch.cuda.synchronize(); fn_batch(srcs[:warm])
    torch.cuda.reset_peak_memory_stats(0); lat = []
    for s in srcs:
        torch.cuda.synchronize(); t = time.perf_counter(); fn_batch([s]); torch.cuda.synchronize(); lat.append(1000 * (time.perf_counter() - t))
    torch.cuda.synchronize(); t = time.perf_counter()
    for i in range(0, n, 32): fn_batch(srcs[i:i + 32])
    torch.cuda.synchronize(); thr = n / (time.perf_counter() - t)
    return {"system": name, "params_M": round(params / 1e6, 1), "lat_median_ms": round(float(np.median(lat)), 1),
            "lat_p90_ms": round(float(np.percentile(lat, 90)), 1), "throughput_sent_s_b32": round(thr, 2),
            "peak_mem_MB": round(torch.cuda.max_memory_allocated(0) / 2**20, 0)}
def nparams(m): return sum(p.numel() for p in m.parameters())
if RUN["latency"] and "latency" not in RESULTS and torch.cuda.is_available() and budget_ok(0.8, "latency"):
    rows = []
    if HAVE_CKPT:
        tok, m = load_t5(S2_DIR)
        rows.append(measure("ours, final pipeline (beam 5 + refinement + threshold)",
                            lambda b: decode_variants(tok, m, b, (0.7,), (True,), (0.0,)), nparams(m)))
        rows.append(measure("ours, greedy single pass", lambda b: t5_gen(tok, m, b, {"num_beams": 1, "length_penalty": 1.0}), nparams(m)))
        del m; gc.collect(); torch.cuda.empty_cache()
    for mid in GEC_BASELINES:
        m = None
        try:
            if "gector" in mid:
                from gector import predict, load_verb_dict
                tok = AutoTokenizer.from_pretrained(mid); m = load_gector(mid, "cuda:0"); e, d = load_verb_dict(str(VERB))
                f = lambda b: predict(m, tok, b, e, d, keep_confidence=0.3, min_error_prob=0.6, n_iteration=5, batch_size=32)
            else:
                tok = AutoTokenizer.from_pretrained(mid); m = AutoModelForSeq2SeqLM.from_pretrained(mid).to("cuda:0").eval()
                def f(b):
                    with torch.no_grad():
                        x = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda:0")
                        return m.generate(**x, num_beams=5, max_length=128)
            rows.append(measure(mid, f, nparams(m)))
        except Exception as ex: rows.append({"system": mid, "error": repr(ex)})
        m = None; gc.collect(); torch.cuda.empty_cache()
    try:
        tok = AutoTokenizer.from_pretrained(LLM_NAME); tok.padding_side = "left"
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        m = AutoModelForCausalLM.from_pretrained(LLM_NAME, torch_dtype=torch.float16, device_map={"": 0}).eval()
        shots = RESULTS.get("qwen", {}).get("exemplars", [])
        rows.append(measure(f"{LLM_NAME}, 4-shot greedy", lambda b: llm_gen(m, tok, b, shots), nparams(m)))
        del m; gc.collect(); torch.cuda.empty_cache()
    except Exception as ex: rows.append({"system": LLM_NAME, "error": repr(ex)})
    RESULTS["latency"] = {"gpu": torch.cuda.get_device_name(0), "rows": rows}; save_results()
    print(pd.DataFrame(rows).to_string())
'''

B_P9 = r'''# ---------------------------------------------------------------- P9: is reference-distance "harm" genuinely harmful? (R2.6)
# Blind pairwise judgement by an instruction model that never sees the reference: source vs our output, both orders.
# Sample = the paper's stored BEA-dev outputs of the final system, stratified by the five-way class.
pip("bitsandbytes")
from transformers import BitsAndBytesConfig
J_SYS = "You are an expert English editor. You compare two versions of the same learner sentence."
def j_better(a, b): return (f"Version 1: {a}\nVersion 2: {b}\n\nWhich version is better English (more grammatical and "
                            "natural) while keeping the same meaning? Reply with exactly one word: 1, 2, or same.")
def j_meaning(a, b): return (f"Original: {a}\nEdited: {b}\n\nDoes the edited sentence keep the meaning of the original? "
                             "Reply with exactly one word: yes or no.")
@torch.no_grad()
def judge(m, tok, users):
    out = []
    for i in range(0, len(users), 16):
        P = [tok.apply_chat_template([{"role": "system", "content": J_SYS}, {"role": "user", "content": u}],
                                     tokenize=False, add_generation_prompt=True) for u in users[i:i + 16]]
        enc = tok(P, return_tensors="pt", padding=True).to(m.device)
        g = m.generate(**enc, max_new_tokens=3, do_sample=False, pad_token_id=tok.pad_token_id)
        out += [tok.decode(x[enc.input_ids.shape[1]:], skip_special_tokens=True).strip().lower() for x in g]
    return out
if RUN["judge"] and "judge" not in RESULTS and budget_ok(0.8, "judge"):
    src, pred = STORED["bea_dev_src"], STORED["bea_dev_pred"]; tgt = BEA_TGT[:len(src)]
    assert src == BEA_SRC[:len(src)], "stored predictions are not in BEA-dev gold order"
    cls = []
    for s, t, p in zip(src, tgt, pred):
        if p.strip() == t.strip(): cls.append("perfect")
        elif p.strip() == s.strip(): cls.append("unchanged")
        else:
            ds, dp = lev(s, t), lev(p, t); cls.append("improved" if dp < ds else ("lateral" if dp == ds else "harm"))
    rng = random.Random(7); items = []
    for c, n in JUDGE_N.items():
        pool = [i for i, x in enumerate(cls) if x == c]; items += [(i, c) for i in rng.sample(pool, min(n, len(pool)))]
    try:
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        tok = AutoTokenizer.from_pretrained(JUDGE_NAME); m = AutoModelForCausalLM.from_pretrained(JUDGE_NAME, quantization_config=q, device_map={"": 0}); used = JUDGE_NAME + " (4-bit nf4)"
    except Exception as ex:
        print("7B judge unavailable:", ex); tok = AutoTokenizer.from_pretrained(JUDGE_FALLBACK)
        m = AutoModelForCausalLM.from_pretrained(JUDGE_FALLBACK, torch_dtype=torch.float16, device_map={"": 0}); used = JUDGE_FALLBACK + " (fp16)"
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    a1 = judge(m, tok, [j_better(src[i], pred[i]) for i, _ in items])     # output shown as version 2
    a2 = judge(m, tok, [j_better(pred[i], src[i]) for i, _ in items])     # output shown as version 1
    mean = judge(m, tok, [j_meaning(src[i], pred[i]) for i, _ in items])
    def norm(x): return "1" if x.startswith("1") else "2" if x.startswith("2") else "same" if x.startswith("same") else "?"
    rows = []
    for (i, c), x, y, z in zip(items, a1, a2, mean):
        x, y = norm(x), norm(y)
        v = ("better" if (x == "2" and y == "1") else "worse" if (x == "1" and y == "2") else
             "equal" if (x == "same" and y == "same") else "inconsistent")
        rows.append({"bea_dev_index": i, "class": c, "src": src[i], "pred": pred[i], "ref": tgt[i],
                     "order_pred_second": x, "order_pred_first": y, "verdict_for_output": v,
                     "meaning_kept": z.startswith("yes")})
    df = pd.DataFrame(rows); df.to_csv(OUT / "judge_items.csv", index=False)
    summ = {c: {**df[df["class"] == c]["verdict_for_output"].value_counts(normalize=True).round(3).to_dict(),
                "meaning_kept": round(float(df[df["class"] == c]["meaning_kept"].mean()), 3), "n": int((df["class"] == c).sum())}
            for c in JUDGE_N}
    RESULTS["judge"] = {"judge_model": used, "summary": summ, "class_counts_all_dev": dict(Counter(cls))}
    save_results(); print(json.dumps(RESULTS["judge"], indent=1))
    # blinded sheet for an OPTIONAL human check by the authors (key kept separately)
    hs = df.groupby("class", group_keys=False).apply(lambda d: d.sample(min(50, len(d)), random_state=1)).reset_index(drop=True)
    flip = [random.Random(100 + k).random() < 0.5 for k in range(len(hs))]
    pd.DataFrame({"item": range(len(hs)), "sentence_1": [p if f else s for s, p, f in zip(hs.src, hs.pred, flip)],
                  "sentence_2": [s if f else p for s, p, f in zip(hs.src, hs.pred, flip)],
                  "better (1/2/same)": "", "meaning_same (yes/no)": ""}).to_csv(OUT / "human_check_sheet.csv", index=False)
    pd.DataFrame({"item": range(len(hs)), "output_is": ["sentence_1" if f else "sentence_2" for f in flip],
                  "class": hs["class"], "bea_dev_index": hs.bea_dev_index}).to_csv(OUT / "human_check_KEY.csv", index=False)
    del m; gc.collect(); torch.cuda.empty_cache()
'''

B_INTRO = md("""# Round 2 - Notebook B: the final system under the reviewers' protocol

| Phase | Reviewer | What it produces |
|---|---|---|
| P2 | R2.4 | **blind BEA-2019 test** predictions (`bea19_test_submission.zip`, upload to CodaBench) + reproduction check on BEA dev |
| P3 | R1.4, R2.3 | length penalty x refinement x threshold, **on the full-budget checkpoints** |
| P4 | R1.4, R2.3 | no-edit pairs at full budget (stage 1 decoded like run 2; compared offline with run 2) |
| P5 | R1.5, R2.2 | **GEC-specific baselines** (gec-t5-small/base-clang8, GECToR-RoBERTa-base) on identical splits + official scorers |
| P6 | R1.6 | Qwen2.5-1.5B **fair re-run**: training-set exemplars by a stated rule, pinned revision, retokenized outputs |
| P7 | - | JFLEG with the **official** GLEU (the paper used NLTK's MT GLEU) |
| P8 | R2.5 | latency + memory for every system on the **same T4** |
| P9 | R2.6 | blind LLM-judge check of "harm" (reference-free) + an optional blinded sheet for a human check |

**Setup (Kaggle):** GPU **T4 x2** - Internet **on** - Add Input -> *Your Work* -> the **`t5v3_final_system`** version that produced the
paper's numbers (it contains `t5v3_model/`, `t5v3_model_stage2/`, `test.parquet`). Without it, P2-P4 and our own P7/P8 rows are skipped.

1. `QUICK_TEST = True`, *Save & Run All* (~30 min) -> check the last cell. 2. `QUICK_TEST = False`, *Save & Run All* (~5 h).
3. Download **`r2B_results.zip`** and **`bea19_test_submission.zip`**, send `r2B_results.zip` back.""")

nbB = [B_INTRO, code(B_CFG), code(C.INSTALL), code(C.IMPORTS), code(C.DATA), code(C.SCORE), code(C.T5DECODE),
       code(B_STORED), code(B_LOCATE), code(B_P2), code(B_P3), code(B_P4), code(B_P5), code(B_P6), code(B_P7),
       code(B_P8), code(B_P9), code(C.EXPORT)]
write("R2_B_final_system_eval.ipynb", nbB)

# ---------------------------------------------------------------- syntax check of every code cell
import ast
for name in ("R2_A_cleaning_full_budget.ipynb", "R2_B_final_system_eval.ipynb"):
    for i, c in enumerate(json.load(open(f"{P}/{name}", encoding="utf-8"))["cells"]):
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))
    print("syntax OK:", name)
