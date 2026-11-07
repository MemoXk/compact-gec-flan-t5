"""Build R2_E_round3.ipynb -- the corrected experiments requested in round 3 (IJIES 20266919).

* Essay-level stage-2 holdout: 5% of the ESSAYS of each proficiency band (A, B, C) is held out; no essay and no source
  sentence occurs in both parts (round-2 holdout: 99.8% of its sentences had sibling sentences in training).
* Stage 2 with three seeds for both stage-1 models (with / without no-edit pairs)   -> seed uncertainty
* An extra stage-1 seed (1337) for the no-edit comparison, paired with Notebook A's clean seed-1337 model
* A data-budget curve: stage 2 on 25% / 50% / 100% of the training essays (same backbone, same stage 1)
* Length penalty re-tested and latency re-measured on the corrected final model; CoEdIT-large as an instruction-tuned
  editor baseline under the same protocol.

Sessions (set JOBS in the first code cell):
  S1  final_s42 final_s1337 final_s2024 coedit latency      inputs: R2_C 'final' output
  S2  nogate_s42 nogate_s1337 nogate_s2024 curve25 curve50  inputs: R2_C 'final' + R2_C 'nogate' outputs
  S3  s1_noedit_1337 s2_noedit_1337 s2_clean_1337           inputs: R2_C 'final' + R2_A 'clean' outputs
"""
import ast, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); P = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import build_r2C as BC          # embeds the ported m2scorer into nb_common.DATA; rebuilds R2_C identically
C = BC.C

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}

CFG = r'''# ============================== CONFIG ==============================
import os; os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
JOBS = ["final_s42", "final_s1337", "final_s2024", "coedit", "latency"]          # SESSION 1 (see the table above)
# JOBS = ["nogate_s42", "nogate_s1337", "nogate_s2024", "curve25", "curve50"]     # SESSION 2
# JOBS = ["s1_noedit_1337", "s2_noedit_1337", "s2_clean_1337"]                     # SESSION 3
QUICK_TEST = False               # True: ~30-minute smoke test of the chosen jobs on small subsets
SESSION_BUDGET_H = 11.6
RESULTS_NAME, EXPORT_NAME = "r2E_results.json", "r2E_results.zip"
SEED = 42
TEST_N, C4_EVAL_N, SUBSET_SEED = 25_000, (5_000 if not QUICK_TEST else 300), 2026
PER_DEVICE_BATCH, GRAD_ACCUM, EPOCHS = 16, 4, 1
LEARNING_RATE, WARMUP_FRAC, WEIGHT_DECAY = 3e-4, 0.06, 0.01
LABEL_SMOOTHING, SRC_DROPOUT_P = 0.1, 0.05
EVAL_SAVE_STEPS = 2500 if not QUICK_TEST else 50
STAGE2_EPOCHS, STAGE2_LR, STAGE2_GA = 3, 1e-4, 2
HOLDOUT_FRAC = 0.05                                       # of the ESSAYS of each band
GATE_INIT_BIAS = 2.0
TAUS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
MAX_TRAIN_EVAL_RATIO = 2.0
LLM_NAME, LLM_MAX_NEW, LLM_BATCH = "Qwen/Qwen2.5-1.5B-Instruct", 96, 32
COEDIT_NAME = "grammarly/coedit-large"
GEC_T5 = ["gotutiyan/gec-t5-small-clang8", "gotutiyan/gec-t5-base-clang8"]
LAT_N, LAT_WARM = (300, 10) if not QUICK_TEST else (30, 3)
RUN2_CLEAN_TEST_SHA256 = "%s"
''' % BC.CLEAN_TEST_SHA

LOCATE = r'''# ---------------------------------------------------------------- inputs (attached Kaggle outputs) + clean C4 test split
import hashlib
def find_dirs(name):
    hits = []
    for root, dirs, files in os.walk("/kaggle/input", followlinks=True):
        if Path(root).name == name and "config.json" in files and any(f.endswith((".safetensors", ".bin")) for f in files):
            hits.append(Path(root))
    return hits[0] if hits else None
def find_file(name):
    for root, dirs, files in os.walk("/kaggle/input", followlinks=True):
        if name in files: return Path(root) / name
    return None
S1 = {"final": find_dirs("t5v3_model"), "nogate": find_dirs("model_nogate"), "clean_s1337": find_dirs("model_clean_s1337")}
DATA_IN = {n: find_file(n) for n in ("clean_test.parquet", "clean_val.parquet", "clean_train_noedit.parquet")}
print("stage-1 models:", {k: str(v) if v else "NOT FOUND" for k, v in S1.items()})
print("data files:", {k: str(v) if v else "NOT FOUND" for k, v in DATA_IN.items()})
if DATA_IN["clean_test.parquet"] is None or not any(S1.values()):
    print("\n!! What is attached under /kaggle/input (top three levels):")
    base = Path("/kaggle/input")
    for p in sorted(base.glob("*")) + sorted(base.glob("*/*")) + sorted(base.glob("*/*/*")):
        print("   ", p.relative_to(base), "/" if p.is_dir() else "")
    if not any(base.iterdir()): print("    (nothing attached)")
assert DATA_IN["clean_test.parquet"] is not None, "attach the R2_C 'final' output (it contains clean_test.parquet)"
te = pq.read_table(DATA_IN["clean_test.parquet"]).to_pandas()
assert hashlib.sha256("\n".join(f"{s}\t{t}" for s, t in zip(te.src, te.tgt)).encode("utf-8")).hexdigest() == RUN2_CLEAN_TEST_SHA256
SUB_IDX = np.sort(np.random.default_rng(SUBSET_SEED).permutation(TEST_N)[:C4_EVAL_N])
C4_SRC = [te.src[i] for i in SUB_IDX]; C4_TGT = [te.tgt[i] for i in SUB_IDX]; del te
NEED = {"final_s42": ["final"], "final_s1337": ["final"], "final_s2024": ["final"], "nogate_s42": ["nogate"],
        "nogate_s1337": ["nogate"], "nogate_s2024": ["nogate"], "curve25": ["final"], "curve50": ["final"],
        "coedit": [], "latency": ["final"], "s1_noedit_1337": [], "s2_noedit_1337": [], "s2_clean_1337": ["clean_s1337"]}
missing = {j: [m for m in NEED[j] if S1.get(m) is None] for j in JOBS}
if "s1_noedit_1337" in JOBS:
    assert DATA_IN["clean_train_noedit.parquet"] and DATA_IN["clean_val.parquet"], "attach the R2_C 'final' output"
JOBS = [j for j in JOBS if not missing[j]]
print("jobs to run:", JOBS, "| skipped (missing inputs):", {j: m for j, m in missing.items() if m})
'''

WI = r'''# ---------------------------------------------------------------- W&I+LOCNESS: sentence -> essay, essay-level split
# The training M2 files were made from the JSON essays in file order, so a sequential alignment on letters and digits
# recovers each sentence's essay (verified locally: all 34,308 sentences -> all 3,000 essays, contiguous, 0 unmatched).
def _norm_ad(s): return re.sub(r"[^0-9a-z]", "", s.lower())
def essay_ids(json_path, sents):
    essays = [json.loads(l) for l in open(json_path, encoding="utf-8") if l.strip()]
    texts = [_norm_ad(e["text"]) for e in essays]; out, e, cur, unmatched = [], 0, 0, 0
    for s in sents:
        key = _norm_ad(s)
        pos = texts[e].find(key, cur) if key else cur
        if pos >= 0: cur = pos + len(key); out.append(essays[e]["id"]); continue
        nxt = texts[e + 1].find(key) if e + 1 < len(essays) and key else -1
        if 0 <= nxt < 400: e += 1; cur = nxt + len(key); out.append(essays[e]["id"]); continue
        unmatched += 1; out.append(essays[e]["id"])
    return out, unmatched, {x["id"]: x.get("cefr") for x in essays}
pairs, essay, band = [], [], []
for lv in "ABC":
    blocks = read_m2(EV / "wi+locness" / "m2" / f"{lv}.train.gold.bea19.m2")
    ids, unm, _ = essay_ids(EV / "wi+locness" / "json" / f"{lv}.train.json", [s for s, _ in blocks])
    assert unm == 0, f"{lv}: {unm} sentences not aligned to an essay"
    for (s, ed), eid in zip(blocks, ids): pairs.append((s, apply_m2_edits(s, ed.get(0, [])))); essay.append(eid); band.append(lv)
assert len(pairs) == 34_308
rng = np.random.default_rng(SEED); hold_essays, train_order = set(), {}
for lv in "ABC":
    es = sorted({e for e, b in zip(essay, band) if b == lv}); perm = rng.permutation(len(es))
    k = int(round(len(es) * HOLDOUT_FRAC)); hold_essays |= {es[i] for i in perm[:k]}; train_order[lv] = [es[i] for i in perm[k:]]
ho_idx = [i for i, e in enumerate(essay) if e in hold_essays]
hold_src = {pairs[i][0] for i in ho_idx}
tr_all = [i for i, e in enumerate(essay) if e not in hold_essays]
tr_idx = [i for i in tr_all if pairs[i][0] not in hold_src]            # also source-disjoint
assert not ({essay[i] for i in tr_idx} & hold_essays) and not ({pairs[i][0] for i in tr_idx} & hold_src)
if QUICK_TEST: tr_idx, ho_idx = tr_idx[:3000], ho_idx[:200]
def write_pairs(idx, name): pq.write_table(pa.table({"src": [pairs[i][0] for i in idx], "tgt": [pairs[i][1] for i in idx]}), OUT / name)
write_pairs(tr_idx, "wi_train_essay.parquet"); write_pairs(ho_idx, "wi_holdout_essay.parquet")
# data-budget subsets: within each band, the first 25% / 50% of its shuffled training essays (nested, stratified)
for frac in (25, 50):
    keep = {e for lv in "ABC" for e in train_order[lv][:round(len(train_order[lv]) * frac / 100)]}
    write_pairs([i for i in tr_idx if essay[i] in keep], f"wi_train_essay_{frac}.parquet")
HOLD_SRC, HOLD_TGT = [pairs[i][0] for i in ho_idx], [pairs[i][1] for i in ho_idx]
(OUT / "wi_holdout_essay_index.txt").write_text("\n".join(map(str, ho_idx)) + "\n")
(OUT / "wi_sentence_essay.tsv").write_text("index\tband\tessay_id\n" + "\n".join(f"{i}\t{band[i]}\t{essay[i]}" for i in range(len(pairs))) + "\n")
RESULTS["wi_essay_split"] = {
    "pairs": len(pairs), "essays": len(set(essay)), "holdout_essays": len(hold_essays), "holdout_pairs": len(ho_idx),
    "train_pairs": len(tr_idx), "train_pairs_dropped_as_source_duplicates": len(tr_all) - len(tr_idx),
    "holdout_by_band": {b: {"essays": len({e for e in hold_essays if band[essay.index(e)] == b}),
                            "pairs": sum(1 for i in ho_idx if band[i] == b)} for b in "ABC"},
    "budget_train_pairs": {f: pq.read_metadata(OUT / f"wi_train_essay_{f}.parquet").num_rows for f in (25, 50)},
    "rule": "5% of the essays of each band (seed 42); holdout essays and holdout source sentences excluded from training"}
save_results(); print(json.dumps(RESULTS["wi_essay_split"], indent=1))
'''

JOBS = r'''# ---------------------------------------------------------------- the jobs
NEED_H = {"final": 1.9, "nogate": 1.4, "curve": 1.3, "coedit": 1.0, "latency": 0.8, "s1": 5.0, "s2": 1.6}
def need(job): return NEED_H[next(k for k in NEED_H if job.startswith(k))] if not QUICK_TEST else 0.3
def run_stage2(tag, init, train_file, seed, full_variants, with_test, lp_extra=False):
    save = OUT / f"model_{tag}"
    train(tag, OUT / train_file, OUT / "wi_holdout_essay.parquet", init, save, stage=2, seed=seed)
    tok, m = load_model(save)
    tau, curve = tune_tau(tok, m, HOLD_SRC, HOLD_TGT); RESULTS[f"tau|{tag}"] = {"tau": tau, "curve": curve}; save_results()
    if full_variants:
        score_all(tag, tok, m, taus=tuple(dict.fromkeys([tau, None, 0.05, 0.1, 0.2])), refines=(False, True),
                  with_test=with_test, test_tau=tau)
    else:
        score_all(tag, tok, m, taus=tuple(dict.fromkeys([tau, None])), refines=(False, True) if with_test else (True,),
                  with_test=with_test, test_tau=tau)
    if lp_extra:                                           # length penalty 1.0 vs the default 0.7, same checkpoint and tau
        r = {}
        for name, srcs, rt in (("c4", C4_SRC, None), ("bea_dev", BEA_SRC[:BEA_N], "bea"), ("conll", CONLL_SRC, "conll")):
            p = decode_variants(tok, m, srcs, (1.0,), (True,), (tau,), desc=f"{tag}/lp1.0/{name}")[(1.0, True, tau)]
            if name == "c4": mm, _ = single_ref_errant(C4_SRC, C4_TGT, p); r[name] = mm
            else:
                p = [RETOK[rt](x) for x in p]
                r[name] = errant_official(srcs, p, BEA_DEV_M2, f"{tag}_lp1_{name}") if name == "bea_dev" else official_m2(p, f"{tag}_lp1_{name}")
            pd.DataFrame({"src": srcs, "lp1.0": p}).to_csv(OUT / f"pred_{name}_{tag}_lp1.0.csv", index=False)
        RESULTS[f"lp1.0|{tag}"] = r; save_results()
    del m; gc.collect(); torch.cuda.empty_cache()

for job in JOBS:
    if RESULTS.get(f"done|{job}"): print("done already:", job); continue
    if not budget_ok(need(job), job): continue
    print("=" * 30, job, "=" * 30, flush=True)
    if job.startswith(("final_s", "nogate_s")):
        arm, seed = job.split("_s")[0], int(job.split("_s")[1])
        run_stage2(f"{arm}_e{seed}", S1[arm], "wi_train_essay.parquet", seed,
                   full_variants=(job == "final_s42"), with_test=(arm == "final"), lp_extra=(job == "final_s42"))
    elif job.startswith("curve"):
        frac = int(job[5:]); run_stage2(f"final_curve{frac}", S1["final"], f"wi_train_essay_{frac}.parquet", 42, False, False)
    elif job == "s1_noedit_1337":
        save = OUT / "model_noedit_s1337"; trf, vaf = DATA_IN["clean_train_noedit.parquet"], DATA_IN["clean_val.parquet"]
        if QUICK_TEST:
            pq.write_table(pq.read_table(trf).slice(0, 3000), OUT / "_quick_s1_train.parquet"); trf = OUT / "_quick_s1_train.parquet"
        train("noedit_s1337", trf, vaf, MODEL_NAME_S1, save, stage=1, seed=1337)
        tok, m = load_model(save); score_all("noedit_s1337_stage1", tok, m, taus=(None, 0.0), refines=(True,))
        del m; gc.collect(); torch.cuda.empty_cache()
    elif job == "s2_noedit_1337":
        init = OUT / "model_noedit_s1337"
        assert (init / "config.json").exists(), "run s1_noedit_1337 first (same session)"
        run_stage2("noedit_s1337_e1337", init, "wi_train_essay.parquet", 1337, False, False)
    elif job == "s2_clean_1337":
        run_stage2("clean_s1337_e1337", S1["clean_s1337"], "wi_train_essay.parquet", 1337, False, False)
    elif job == "coedit": run_coedit()
    elif job == "latency": run_latency()
    RESULTS[f"done|{job}"] = True; save_results()
'''

BASELINES = r'''# ---------------------------------------------------------------- CoEdIT baseline + latency on one GPU
MODEL_NAME_S1 = "google/flan-t5-small"
from huggingface_hub import model_info
@torch.no_grad()
def coedit_gen(tok, m, srcs, bs=32):
    out = []
    for i in range(0, len(srcs), bs):
        x = tok(["Fix grammatical errors in this sentence: " + s for s in srcs[i:i + bs]], return_tensors="pt",
                padding=True, truncation=True, max_length=256).to(m.device)
        out += tok.batch_decode(m.generate(**x, max_length=256), skip_special_tokens=True)   # model-card decoding
    return [norm_ws(o) for o in out]
def run_coedit():
    try:
        rev = model_info(COEDIT_NAME).sha
        tok = AutoTokenizer.from_pretrained(COEDIT_NAME, revision=rev)
        m = AutoModelForSeq2SeqLM.from_pretrained(COEDIT_NAME, revision=rev).to(DEV).eval()
        r = {"model": COEDIT_NAME, "revision": rev, "prompt": "Fix grammatical errors in this sentence: <source>",
             "decoding": "greedy, max_length=256 (model card)", "params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 1)}
        for name, srcs in (("conll", CONLL_SRC), ("bea_dev", BEA_SRC[:BEA_N]), ("jfleg", JF_SRC[:JF_N])):
            p = coedit_gen(tok, m, srcs); pr = [RETOK["bea" if name == "bea_dev" else name](x) for x in p]
            s = (official_m2(pr, "coedit") if name == "conll" else errant_official(srcs, pr, BEA_DEV_M2, "coedit")
                 if name == "bea_dev" else jfleg_gleu(pr, JF_N)) or {"error": "scorer returned nothing"}
            s["edited_pct"] = outcome_table(srcs, srcs, pr)["edited_pct"]; r[name] = s
            pd.DataFrame({"src": srcs, "pred": p, "pred_retok": pr}).to_csv(OUT / f"pred_{name}_coedit-large.csv", index=False)
            print("CoEdIT", name, s, flush=True)
        del m; gc.collect(); torch.cuda.empty_cache()
    except Exception as ex:
        r = {"error": repr(ex)}
    RESULTS["baseline|coedit-large"] = r; save_results()

SYSTEM = ("You are a grammatical error correction system. Return ONLY the corrected sentence. "
          "Make the minimum number of edits. If the sentence is already correct, repeat it unchanged.")
def pick_exemplars():          # identical to Notebook B, so the same four exemplars are drawn
    pool = []
    for p in sorted((EV / "wi+locness" / "m2").glob("*.train.gold.bea19.m2")):
        for s, ed in read_m2(p):
            e0 = ed.get(0, []); t = apply_m2_edits(s, e0); n = len(s.split())
            if 8 <= n <= 20 and 1 <= len(e0) <= 3 and s != t and s.isascii(): pool.append((s, t))
    return random.Random(2024).sample(pool, 4)
@torch.no_grad()
def llm_gen(m, tok, srcs, shots):
    P = []
    for s in srcs:
        msgs = [{"role": "system", "content": SYSTEM}]
        for a, b in shots: msgs += [{"role": "user", "content": a}, {"role": "assistant", "content": b}]
        msgs.append({"role": "user", "content": s}); P.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    out = []
    for i in range(0, len(P), LLM_BATCH):
        enc = tok(P[i:i + LLM_BATCH], return_tensors="pt", padding=True).to(m.device)
        g = m.generate(**enc, max_new_tokens=LLM_MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
        out += [tok.decode(x[enc.input_ids.shape[1]:], skip_special_tokens=True).split("\n")[0] for x in g]
    return out
def measure(name, fn_batch, params, n=LAT_N, warm=LAT_WARM):
    srcs = CONLL_SRC[:n]; torch.cuda.synchronize(); fn_batch(srcs[:warm])
    torch.cuda.reset_peak_memory_stats(0); lat = []
    for s in srcs:
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn_batch([s]); torch.cuda.synchronize(); lat.append(1000 * (time.perf_counter() - t0))
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for i in range(0, n, 32): fn_batch(srcs[i:i + 32])
    torch.cuda.synchronize(); thr = n / (time.perf_counter() - t0)
    return {"system": name, "params_M": round(params / 1e6, 1), "lat_median_ms": round(float(np.median(lat)), 1),
            "lat_p90_ms": round(float(np.percentile(lat, 90)), 1), "throughput_sent_s_b32": round(thr, 2),
            "peak_mem_MB": round(torch.cuda.max_memory_allocated(0) / 2**20, 0)}
def nparams(m): return sum(p.numel() for p in m.parameters())
def run_latency():
    rows = []
    final = OUT / "model_final_e42"
    assert (final / "config.json").exists(), "run final_s42 first (same session): latency is measured on that checkpoint"
    tok, m = load_model(final); tau = RESULTS["tau|final_e42"]["tau"]
    rows.append(measure("ours, final pipeline (beam 5 + refinement + threshold)", lambda b: decode_variants(tok, m, b, (0.7,), (True,), (tau,)), nparams(m)))
    rows.append(measure("ours, greedy single pass", lambda b: t5_gen(tok, m, b, {"num_beams": 1, "length_penalty": 1.0}), nparams(m)))
    del m; gc.collect(); torch.cuda.empty_cache()
    for mid in GEC_T5:
        try:
            tk = AutoTokenizer.from_pretrained(mid); mm = AutoModelForSeq2SeqLM.from_pretrained(mid).to("cuda:0").eval()
            def f(b, tk=tk, mm=mm):
                with torch.no_grad():
                    x = tk(b, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda:0")
                    return mm.generate(**x, num_beams=5, max_length=128)
            rows.append(measure(mid + " (beam 5)", f, nparams(mm))); del mm
        except Exception as ex: rows.append({"system": mid, "error": repr(ex)})
        gc.collect(); torch.cuda.empty_cache()
    try:
        tk = AutoTokenizer.from_pretrained(COEDIT_NAME); mm = AutoModelForSeq2SeqLM.from_pretrained(COEDIT_NAME).to("cuda:0").eval()
        rows.append(measure(COEDIT_NAME + " (greedy)", lambda b, tk=tk, mm=mm: coedit_gen(tk, mm, b), nparams(mm))); del mm
    except Exception as ex: rows.append({"system": COEDIT_NAME, "error": repr(ex)})
    gc.collect(); torch.cuda.empty_cache()
    try:
        rev = model_info(LLM_NAME).sha; tk = AutoTokenizer.from_pretrained(LLM_NAME, revision=rev); tk.padding_side = "left"
        if tk.pad_token is None: tk.pad_token = tk.eos_token
        mm = AutoModelForCausalLM.from_pretrained(LLM_NAME, revision=rev, torch_dtype=torch.float16, device_map={"": 0}).eval()
        shots = pick_exemplars()
        rows.append(measure(f"{LLM_NAME}, 4-shot greedy", lambda b, tk=tk, mm=mm, shots=shots: llm_gen(mm, tk, b, shots), nparams(mm))); del mm
    except Exception as ex: rows.append({"system": LLM_NAME, "error": repr(ex)})
    gc.collect(); torch.cuda.empty_cache()
    RESULTS["latency"] = {"gpu": torch.cuda.get_device_name(0), "rows": rows}; save_results()
    print(json.dumps(RESULTS["latency"], indent=1))
'''

EVAL = BC.EVAL
for a, b in (('(OUT / "bea19_test_hyp.txt").write_text(', '(OUT / f"bea19_test_hyp_{tag}.txt").write_text('),
             ('z.write(OUT / "bea19_test_hyp.txt", "bea19_test_hyp.txt")', 'z.write(OUT / f"bea19_test_hyp_{tag}.txt", "bea19_test_hyp.txt")'),
             ('OUT / "bea19_test_submission_retrained.zip"', 'OUT / f"bea19_test_submission_{tag}.zip"'),
             ('"file": "bea19_test_submission_retrained.zip"', '"file": f"bea19_test_submission_{tag}.zip"')):
    assert EVAL.count(a) == 1, a
    EVAL = EVAL.replace(a, b)

INSTALL = C.INSTALL + '\npip("--no-deps", "git+https://github.com/gotutiyan/gec-t5")\n'

INTRO = md("""# Round 3 - Notebook E: corrected experiments requested by the reviewers

**Setup (Kaggle):** GPU **T4 x2**, Internet **on**. Add Input -> Your Work:
* `R2_C_retrain_core`, the version that ran `JOBS = ["final"]` (output has `t5v3_model/`, `clean_test.parquet`, `clean_train_noedit.parquet`) - **every session**
* `R2_C_retrain_core`, the version that ran `JOBS = ["nogate"]` (output has `model_nogate/`) - **session 2**
* `R2_A_cleaning_full_budget`, the **clean** version (output has `model_clean_s1337/`) - **session 3**

| Session | `JOBS` (first code cell) | ~Time |
|---|---|---|
| 1 | `final_s42 final_s1337 final_s2024 coedit latency` | 6-7 h |
| 2 | `nogate_s42 nogate_s1337 nogate_s2024 curve25 curve50` | 6 h |
| 3 | `s1_noedit_1337 s2_noedit_1337 s2_clean_1337` | 7-8 h |

Run `QUICK_TEST = True` once first (about 30 min) to catch problems. Each session writes **`r2E_results.zip`**; session 1
also writes three blind-test files `bea19_test_submission_final_e{42,1337,2024}.zip` for CodaBench.""")

cells = [INTRO, code(CFG), code(INSTALL), code(C.IMPORTS), code(C.DATA), code(C.SCORE), code(C.T5DECODE),
         code(LOCATE), code(WI), code(BC.GATE), code(BC.TRAIN), code(EVAL), code(BASELINES), code(JOBS), code(C.EXPORT)]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = f"{P}/R2_E_round3.ipynb"
json.dump(nb, open(out, "w", encoding="utf-8"), indent=1)
for c in cells:
    if c["cell_type"] == "code": ast.parse("".join(c["source"]))
print("wrote", out, "-", len(cells), "cells | syntax OK")
