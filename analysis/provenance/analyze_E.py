"""Round-3 analysis of Notebook E (essay-level stage-2 holdout, 3 seeds per arm, extra stage-1 seed, data-budget
curve, CoEdIT-large) plus the corrected cleaning comparison on uncontaminated C4 pairs.

Every score is recomputed locally with the released verifier's own scorer code (analysis/verify.py, executed up to
its scoring loop), and each Notebook E system is asserted to reproduce the score Kaggle reported.

    python analyze_E.py --part 0 --nparts 5     # optional: fill the shared score cache in parallel
    python analyze_E.py                          # statistics -> round2_results/E_analysis.json

Statistics: paired bootstrap over sentences (2,000 resamples, generator seeded by the comparison name). When an arm
has several seeds, each resample averages the paired delta over the seed pairs (test-set uncertainty of the
seed-averaged effect); seed spread is reported separately as sd. 95% CI and two-sided p for differences; a 90% CI
and TOST p against an equivalence margin of +/-0.01 F0.5 for the null-result claims; Holm correction within each table.
"""
import argparse, csv, gzip, hashlib, json, os, sys
from pathlib import Path
import numpy as np

P = Path(r"<PROJECT>")
PKG = P / "release/compact-gec-flan-t5"; R = P / "round2_results"; E = R / "E"
ap = argparse.ArgumentParser(); ap.add_argument("--part", type=int); ap.add_argument("--nparts", type=int, default=1)
args = ap.parse_args()

# ---------------------------------------------------------------- the released verifier's scorers, verbatim
src = (PKG / "analysis/verify.py").read_text(encoding="utf-8")
head = src[:src.index("# ---------------------------------------------------------------- score every released output")]
V = {"__file__": str(PKG / "analysis/verify.py"), "__name__": "verify_lib"}
argv, sys.argv = sys.argv, ["verify.py"]; exec(compile(head, "verify.py", "exec"), V); sys.argv = argv
f05, prf, cached, CACHE = V["f05"], V["prf"], V["cached"], V["CACHE"]

csv.field_size_limit(10 ** 9)
def col(sess, bench, tag, c):
    return [r[c] for r in csv.DictReader(open(E / f"session{sess}/pred_{bench}_{tag}.csv", encoding="utf-8"))]
BENCH4 = ("c4", "bea_dev", "conll", "jfleg")
SYS = {}
def add(name, sess, tag, c, benches=BENCH4):
    SYS[name] = {b: (sess, tag, c) for b in benches}
for s in (42, 1337, 2024):
    add(f"E/final_e{s}", 1, f"final_e{s}", "ref1_tau0.0")
    add(f"E/final_e{s}_no_threshold", 1, f"final_e{s}", "ref1_taunone")
    add(f"E/final_e{s}_no_refinement", 1, f"final_e{s}", "ref0_tau0.0")
    add(f"E/nogate_e{s}", 2, f"nogate_e{s}", "ref1_tau0.0")
    add(f"E/nogate_e{s}_no_threshold", 2, f"nogate_e{s}", "ref1_taunone")
for t in ("0.05", "0.1", "0.2"): add(f"E/final_e42_tau{t}", 1, "final_e42", f"ref1_tau{t}")
add("E/final_e42_lp1.0", 1, "final_e42_lp1.0", "lp1.0", ("c4", "bea_dev", "conll"))
for f in (25, 50): add(f"E/final_curve{f}", 2, f"final_curve{f}", "ref1_tau0.0")
add("E/noedit_s1337_stage1", 3, "noedit_s1337_stage1", "ref1_tau0.0")
add("E/noedit_s1337_stage1_no_threshold", 3, "noedit_s1337_stage1", "ref1_taunone")
add("E/noedit_s1337_e1337", 3, "noedit_s1337_e1337", "ref1_tau0.0")
add("E/clean_s1337_e1337", 3, "clean_s1337_e1337", "ref1_tau0.0")
add("baselines/coedit-large", 1, "coedit-large", "pred_retok", ("bea_dev", "conll", "jfleg"))
PKG_SYS = ["final_stage1", "final_stage1_no_threshold", "nogate_stage1", "clean_s1337", "noisy_s1337", "noisy_s42",
           "baselines/gec-t5-small-clang8", "baselines/gec-t5-base-clang8", "baselines/qwen2.5-1.5b_fourshot"]
manifest = json.load(open(PKG / "outputs/manifest.json"))

def hyps(system, bench):
    if system in SYS: sess, tag, c = SYS[system][bench]; return col(sess, bench, tag, c)
    return V["read_hyps"](system, bench)
def benches(system):
    return list(SYS[system]) if system in SYS else [b for b in manifest[system] if b in BENCH4]

# alignment checks: every E prediction file is in the verifier's sentence order
C4SRC = [r["src"] for r in V["C4"]]; BEASRC, _ = V["bea_gold"]()
for name, bs in SYS.items():
    for b, (sess, tag, c) in bs.items():
        s_ = col(sess, b, tag, "src")
        if b == "c4": assert s_ == C4SRC, f"{name}: C4 order differs"
        if b == "bea_dev": assert s_ == BEASRC, f"{name}: BEA dev order differs"

def jfleg(h, tag):
    p = CACHE / ("jf_" + hashlib.sha1("\n".join(h).encode()).hexdigest() + ".json")
    if p.exists(): return json.load(open(p))
    v = V["jfleg_gleu"](h, tag); json.dump(v, open(p, "w")); return v
def counts(system, b):
    h = hyps(system, b); key = f"{b}|{system}|{hashlib.sha1(chr(10).join(h).encode()).hexdigest()}"
    if b == "conll": return cached(key, lambda: V["conll_counts"](h, system.replace("/", "_")))
    if b == "bea_dev": return cached(key, lambda: V["bea_counts"](h))
    if b == "c4": return cached(key, lambda: V["c4_counts"](h))
    raise ValueError(b)

jobs = [(s, b) for s in list(SYS) + PKG_SYS for b in benches(s)]
if args.part is not None:                                   # cache-filling worker
    for i, (s, b) in enumerate(jobs):
        if i % args.nparts == args.part:
            (jfleg(hyps(s, b), s.replace("/", "_")) if b == "jfleg" else counts(s, b)); print("done", s, b, flush=True)
    sys.exit(0)

CNT, SC = {}, {}
for s, b in jobs:
    if b == "jfleg": SC.setdefault(s, {})[b] = jfleg(hyps(s, b), s.replace("/", "_"))
    else: CNT[(s, b)] = np.asarray(counts(s, b)); SC.setdefault(s, {})[b] = prf(CNT[(s, b)])

# ---------------------------------------------------------------- local scores must reproduce Kaggle's
RE = {k: json.load(open(E / f"session{k}/r2E_results.json")) for k in (1, 2, 3)}
def kaggle(system, b):
    if system == "baselines/coedit-large": return RE[1]["baseline|coedit-large"][b]
    if system == "E/final_e42_lp1.0": return RE[1]["lp1.0|final_e42"][b]
    sess, tag, c = SYS[system][b]
    tag2 = "noedit_s1337_stage1" if tag == "noedit_s1337_stage1" else tag
    return RE[sess][f"eval|{tag2}"][f"{b}|{c}"]
mismatch = []
for s in SYS:
    for b in SYS[s]:
        k = "GLEU" if b == "jfleg" else "F05"; a_, l_ = kaggle(s, b)[k], SC[s][b][k]
        if abs(a_ - l_) > 6e-4: mismatch.append((s, b, a_, l_))
print("Kaggle vs local:", "all", sum(len(v) for v in SYS.values()), "scores reproduce" if not mismatch else mismatch)
assert not mismatch

# ---------------------------------------------------------------- statistics
B_ = 2000; MARGIN = 0.01
def weights(name, n):
    rng = np.random.default_rng(int(hashlib.sha1(name.encode()).hexdigest()[:8], 16))
    idx = rng.integers(0, n, size=(B_, n))
    return np.stack([np.bincount(r, minlength=n) for r in idx]).astype(np.int32)
def f05v(t):
    tp, fp, fn = t[:, 0].astype(float), t[:, 1].astype(float), t[:, 2].astype(float)
    p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 1.0); r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 1.0)
    return np.where(p + r > 0, 1.25 * p * r / np.maximum(0.25 * p + r, 1e-12), 0.0)
def compare(name, pairs, b, mask=None):
    """pairs: list of (system_a, system_b); delta = mean over pairs of F0.5(a) - F0.5(b)."""
    ca = [CNT[(a, b)] for a, _ in pairs]; cb = [CNT[(x, b)] for _, x in pairs]
    if mask is not None: ca, cb = [c[mask] for c in ca], [c[mask] for c in cb]
    W = weights(f"{name}|{b}", len(ca[0]))
    d = np.mean([f05v(W @ x) - f05v(W @ y) for x, y in zip(ca, cb)], axis=0)
    obs = float(np.mean([f05(*x.sum(0)) - f05(*y.sum(0)) for x, y in zip(ca, cb)]))
    per = [round(f05(*x.sum(0)) - f05(*y.sum(0)), 6) for x, y in zip(ca, cb)]
    return {"benchmark": b, "pairs": [list(p) for p in pairs], "n": int(len(ca[0])), "delta": round(obs, 6),
            "per_pair_delta": per, "ci95": [round(float(np.percentile(d, 2.5)), 6), round(float(np.percentile(d, 97.5)), 6)],
            "p_two_sided": round(float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))), 4),
            "ci90": [round(float(np.percentile(d, 5)), 6), round(float(np.percentile(d, 95)), 6)],
            "tost_p_margin_0.01": round(float(max((d <= -MARGIN).mean(), (d >= MARGIN).mean())), 4)}
def holm(rows):
    ps = sorted(range(len(rows)), key=lambda i: rows[i]["p_two_sided"]); m = len(rows); run = 0.0
    for rank, i in enumerate(ps):
        run = max(run, min(1.0, (m - rank) * rows[i]["p_two_sided"])); rows[i]["p_holm"] = round(run, 4)
    return rows

SEEDS = (42, 1337, 2024)
T4 = [  # ablations on the corrected study (all stage-2 runs use the essay-level holdout)
    ("noedit_two_stage", [(f"E/final_e{s}", f"E/nogate_e{s}") for s in SEEDS] + [("E/noedit_s1337_e1337", "E/clean_s1337_e1337")]),
    ("noedit_stage1", [("final_stage1_no_threshold", "nogate_stage1"), ("E/noedit_s1337_stage1_no_threshold", "clean_s1337")]),
    ("human_data_stage", [(f"E/final_e{s}", "final_stage1") for s in SEEDS] + [("E/noedit_s1337_e1337", "E/noedit_s1337_stage1")]),
    ("threshold_tau0", [(f"E/{a}_e{s}", f"E/{a}_e{s}_no_threshold") for a in ("final", "nogate") for s in SEEDS]),
    ("refinement", [(f"E/final_e{s}", f"E/final_e{s}_no_refinement") for s in SEEDS]),
    ("length_penalty_0.7_vs_1.0", [("E/final_e42", "E/final_e42_lp1.0")]),
    ("budget_50_vs_25", [("E/final_curve50", "E/final_curve25")]),
    ("budget_100_vs_50", [("E/final_e42", "E/final_curve50")]),
]
T6 = [("final_vs_coedit_large", [(f"E/final_e{s}", "baselines/coedit-large") for s in SEEDS]),
      ("final_vs_gec_t5_small", [(f"E/final_e{s}", "baselines/gec-t5-small-clang8") for s in SEEDS]),
      ("final_vs_gec_t5_base", [(f"E/final_e{s}", "baselines/gec-t5-base-clang8") for s in SEEDS]),
      ("final_vs_qwen_fourshot", [(f"E/final_e{s}", "baselines/qwen2.5-1.5b_fourshot") for s in SEEDS])]
out = {"scores": SC, "tables": {}}
for tab, spec, bl in (("T4", T4, ("c4", "bea_dev", "conll")), ("T6", T6, ("bea_dev", "conll"))):
    rows = []
    for name, pairs in spec:
        for b in bl:
            if all((a, b) in CNT and (x, b) in CNT for a, x in pairs): rows.append({"name": name, **compare(name, pairs, b)})
    out["tables"][tab] = holm(rows)

# seed spread of each multi-seed arm (stage 2, final decoding)
def spread(names):
    r = {}
    for b in BENCH4:
        v = [SC[n][b]["GLEU" if b == "jfleg" else "F05"] for n in names if b in SC[n]]
        if v: r[b] = {"values": v, "mean": round(float(np.mean(v)), 4), "sd": round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else None}
    return r
out["seed_spread"] = {"final (with no-edit pairs)": spread([f"E/final_e{s}" for s in SEEDS]),
                      "nogate (without no-edit pairs)": spread([f"E/nogate_e{s}" for s in SEEDS])}
out["data_budget_curve_seed42"] = {f"{k}%": {b: SC[n][b].get("F05", SC[n][b].get("GLEU")) for b in BENCH4}
                                   for k, n in ((25, "E/final_curve25"), (50, "E/final_curve50"), (100, "E/final_e42"))}

# ---------------------------------------------------------------- cleaning comparison on uncontaminated C4 pairs
def split_rows(path):
    rows = {"train": [], "val": [], "test": []}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        next(f)
        for ln in f: sh, li, sp = ln.rstrip("\n").split("\t"); rows[sp].append((int(sh), int(li)))
    return rows
clean, noisy = split_rows(R / "A_clean/split_index_clean.tsv.gz"), split_rows(R / "A_noisy/split_index_noisy.tsv.gz")
sub = [int(r["test_index"]) for r in csv.DictReader(open(R / "A_noisy/c4_eval_subset_index.csv"))]
assert sub == [int(r["test_index"]) for r in V["C4"]], "subset order differs from the verifier's"
ntr, ctr = set(noisy["train"]), set(clean["train"])
mask = np.array([clean["test"][i] not in ntr and clean["test"][i] not in ctr for i in sub])
rows = [{"name": n, **compare(n, pr, "c4", mask)} for n, pr in
        (("cleaning_seed42", [("nogate_stage1", "noisy_s42")]), ("cleaning_seed1337", [("clean_s1337", "noisy_s1337")]),
         ("cleaning_pooled", [("nogate_stage1", "noisy_s42"), ("clean_s1337", "noisy_s1337")]))]
full = [{"name": n, **compare(n + "_all5k", pr, "c4")} for n, pr in
        (("cleaning_seed42", [("nogate_stage1", "noisy_s42")]), ("cleaning_seed1337", [("clean_s1337", "noisy_s1337")]))]
out["cleaning_c4_uncontaminated"] = {"pairs_kept": int(mask.sum()), "pairs_dropped_in_noisy_train": int((~mask).sum()),
                                     "absolute": {s: prf(CNT[(s, "c4")][mask]) for s in ("nogate_stage1", "noisy_s42", "clean_s1337", "noisy_s1337")},
                                     "comparisons": rows, "for_reference_all_5000": full}
# cleaning (composite filtering policy) on the learner benchmarks: stage-1 models, pooled over the two seeds
CLEAN_PAIRS = [("nogate_stage1", "noisy_s42"), ("clean_s1337", "noisy_s1337")]
out["cleaning_learner_benchmarks"] = {
    "comparisons": holm([{"name": "cleaning_pooled", **compare("cleaning_pooled_learner", CLEAN_PAIRS, b)} for b in ("bea_dev", "conll")]),
    "per_seed": [{"name": f"cleaning_{a}", **compare(f"cleaning_{a}_learner", [(a, x)], b)} for a, x in CLEAN_PAIRS for b in ("bea_dev", "conll")],
    "absolute": {s: {b: SC[s][b].get("F05", SC[s][b].get("GLEU")) for b in BENCH4 if b in SC[s]} for s in ("nogate_stage1", "noisy_s42", "clean_s1337", "noisy_s1337")}}
# BEA-2019 dev by CEFR level (A 1,037 / B 1,290 / C 1,069 / N 988 sentences, in file order) for the final arm
LEVELS = (("A", 1037), ("B", 1290), ("C", 1069), ("N", 988))
def by_level(system):
    c, o, r = CNT[(system, "bea_dev")], 0, {}
    for lv, n in LEVELS: r[lv] = prf(c[o:o + n]); o += n
    return r
out["bea_dev_by_level_final"] = {s: by_level(f"E/final_e{s}") for s in SEEDS}
out["bea_dev_by_level_final_mean"] = {lv: {"F05_mean": round(float(np.mean([out["bea_dev_by_level_final"][s][lv]["F05"] for s in SEEDS])), 4),
                                           "F05_sd": round(float(np.std([out["bea_dev_by_level_final"][s][lv]["F05"] for s in SEEDS], ddof=1)), 4)}
                                      for lv, _ in LEVELS}
out["notes"] = {"margin": MARGIN, "bootstrap": B_, "pooling": "mean of per-pair F0.5 deltas, same resample for all pairs",
                "decoding": "ours: beam 5, length penalty 0.7, one refinement pass, threshold tau tuned on the essay-level "
                            "holdout (0.0 in every run); clean_s1337 / noisy_* are single-pass outputs from Notebook A",
                "coedit": "grammarly/coedit-large, prompt 'Fix grammatical errors in this sentence: ', greedy, max_length 256"}
json.dump(out, open(R / "E_analysis.json", "w"), indent=1)

def show(rows):
    for r in rows:
        print(f"  {r['name']:28s} {r['benchmark']:8s} d={r['delta']:+.4f} per-pair {r['per_pair_delta']} CI95 {r['ci95']} "
              f"p={r['p_two_sided']:.3f} holm={r.get('p_holm', float('nan')):.3f} CI90 {r['ci90']} TOST={r['tost_p_margin_0.01']:.3f}")
for t in ("T4", "T6"): print(t); show(out["tables"][t])
print("seed spread", json.dumps(out["seed_spread"]))
print("budget curve", json.dumps(out["data_budget_curve_seed42"]))
print("cleaning on", int(mask.sum()), "uncontaminated C4 pairs"); show(rows)
print("  absolute:", {k: v["F05"] for k, v in out["cleaning_c4_uncontaminated"]["absolute"].items()})
print("wrote", R / "E_analysis.json")
