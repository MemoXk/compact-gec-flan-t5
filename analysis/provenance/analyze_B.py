"""Round-2 Notebook B analysis: the final system under the reviewers' protocol.

Every comparison is scored with the official scorer of its benchmark (CoNLL-2014: MaxMatch M2; BEA-2019 dev:
errant_compare) after retokenizing EVERY system's output with the tokenizer matched to that benchmark.
Significance = paired bootstrap over sentences (2,000 resamples) on per-sentence counts that reproduce the
official totals (asserted / printed)."""
import csv, json, os, re, subprocess, sys
import numpy as np
import errant

P = r"<PROJECT>"
S = r"<SCRATCH>/eval"
sys.path.insert(0, S)
from retok import regex_tok, spacy_tok
B = f"{P}/round2_results/B"
RES = json.load(open(f"{B}/r2B_results.json", encoding="utf-8"))
def rd(p): return list(csv.DictReader(open(p, encoding="utf-8", errors="replace")))
def f05(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0; r = tp / (tp + fn) if tp + fn else 1.0
    return 1.25 * p * r / (0.25 * p + r) if (p + r) else 0.0
def prf(c):
    tp, fp, fn = c.sum(0); return {"P": round(tp / max(tp + fp, 1), 4), "R": round(tp / max(tp + fn, 1), 4), "F05": round(f05(tp, fp, fn), 4)}
ANN = errant.load("en")

def rejoin(pred, src):        # undo spaCy hyphen splits that the CoNLL source does not have (June pipeline artefact)
    toks = set(src.split())
    return re.sub(r"(\S+) - (\S+)", lambda m: m.group(1) + "-" + m.group(2) if m.group(1) + "-" + m.group(2) in toks else m.group(0), pred)

# ---------------------------------------------------------------- systems
cv, bv = rd(f"{B}/pred_conll_stage2_variants.csv"), rd(f"{B}/pred_bea_dev_stage2_variants.csv")
run2_conll, run2_bea = rd(f"{P}/run2_results/t5max_conll14_examples.csv"), rd(f"{P}/run2_results/t5max_bea19_examples.csv")
CONLL = {"ours_final": [r["lp0.7_ref1_tau0.0"] for r in cv], "ours_no_threshold": [r["lp0.7_ref1_taunone"] for r in cv],
         "ours_no_refinement": [r["lp0.7_ref0_tau0.0"] for r in cv], "ours_lp1.0": [r["lp1.0_ref1_tau0.0"] for r in cv],
         "qwen_fourshot": [r["pred_retok"] for r in rd(f"{B}/pred_conll_qwen_fourshot.csv")],
         "qwen_zeroshot": [r["pred_retok"] for r in rd(f"{B}/pred_conll_qwen_zeroshot.csv")],
         "gec_t5_small": [r["pred_retok"] for r in rd(f"{B}/pred_conll_gec-t5-small-clang8.csv")],
         "gec_t5_base": [r["pred_retok"] for r in rd(f"{B}/pred_conll_gec-t5-base-clang8.csv")],
         "stage1_run2decode": [r["pred_retok"] for r in rd(f"{B}/pred_conll_stage1_run2decode.csv")],
         "run2_stored_fixed": [regex_tok(rejoin(r["pred"], r["src"])) for r in run2_conll],
         "paper_stored_fixed": [regex_tok(rejoin(r["pred"], r["src"])) for r in rd(f"{P}/v3_results/t5v3_conll14_examples.csv")]}
BEA = {"ours_final": [r["lp0.7_ref1_tau0.0"] for r in bv], "ours_no_threshold": [r["lp0.7_ref1_taunone"] for r in bv],
       "ours_no_refinement": [r["lp0.7_ref0_tau0.0"] for r in bv], "ours_lp1.0": [r["lp1.0_ref1_tau0.0"] for r in bv],
       "qwen_fourshot": [r["pred_retok"] for r in rd(f"{B}/pred_bea_dev_qwen_fourshot.csv")],
       "qwen_zeroshot": [r["pred_retok"] for r in rd(f"{B}/pred_bea_dev_qwen_zeroshot.csv")],
       "gec_t5_small": [r["pred_retok"] for r in rd(f"{B}/pred_bea_dev_gec-t5-small-clang8.csv")],
       "gec_t5_base": [r["pred_retok"] for r in rd(f"{B}/pred_bea_dev_gec-t5-base-clang8.csv")],
       "stage1_run2decode": [r["pred_retok"] for r in rd(f"{B}/pred_bea_dev_stage1_run2decode.csv")],
       "run2_stored": [r["pred"] for r in run2_bea]}
csrc = [l[2:].strip() for l in open(f"{S}/conll14st-test-data/noalt/official-2014.combined.m2", encoding="utf-8") if l.startswith("S ")]
bsrc = [r["src"] for r in bv]

# ---------------------------------------------------------------- CoNLL: official M2, per-sentence via -v
GOLD_C = f"{S}/conll14st-test-data/noalt/official-2014.combined.m2"
def m2_counts(name, hyps):
    cache = f"{S}/_B_m2v_{name}.json"
    if os.path.exists(cache): d = json.load(open(cache)); return np.array(d["per"]), d["off"]
    hp = f"{S}/_B_{name}.txt"; open(hp, "w", encoding="utf-8").write("\n".join(hyps) + "\n")
    out = subprocess.run([sys.executable, f"{S}/m2scorer/scripts/m2scorer.py", "-v", hp, GOLD_C], capture_output=True, text=True,
                         env={**os.environ, "PYTHONUTF8": "1"}).stdout
    cum, cur, ann = [], {}, None
    for ln in out.splitlines():
        m = re.match(r">> Annotator: (\d+)", ln)
        if m: ann = int(m.group(1)); cur[ann] = [0, 0, 0]; continue
        m = re.match(r"# (correct|proposed|gold)\s*:\s*(\d+)", ln)
        if m and ann is not None: cur[ann][("correct", "proposed", "gold").index(m.group(1))] = int(m.group(2)); continue
        m = re.match(r">> Chosen Annotator for line (\d+) : (\d+)", ln)
        if m: c, p, g = cur[int(m.group(2))]; cum.append((c, p - c, g - c)); cur = {}; ann = None
    off = {k: float(v) for k, v in re.findall(r"(Precision|Recall|F_0\.5)\s*:\s*([0-9.]+)", out)}
    per = np.diff(np.vstack([np.zeros((1, 3), int), np.array(cum)]), axis=0)
    json.dump({"per": per.tolist(), "off": off}, open(cache, "w")); return per, off

# ---------------------------------------------------------------- BEA dev: ERRANT per sentence vs the gold M2
gold = []
for b in open(f"{S}/wi+locness/m2/ABCN.dev.gold.bea19.m2", encoding="utf-8").read().strip().split("\n\n"):
    g = {}
    for ln in b.split("\n")[1:]:
        f = ln[2:].split("|||"); s, e = map(int, f[0].split()); a = int(f[-1]); g.setdefault(a, set())
        if f[1] not in ("noop", "UNK"): g[a].add((s, e, "" if f[2] == "-NONE-" else f[2]))
    gold.append(g or {0: set()})
def bea_counts(hyps):
    out = []
    for s, h, g in zip(bsrc, hyps, gold):
        o = ANN.parse(s)
        he = {(e.o_start, e.o_end, e.c_str) for e in ANN.annotate(o, ANN.parse(h)) if e.type != "noop"}
        out.append(max(((len(he & ge), len(he - ge), len(ge - he)) for ge in g.values()), key=lambda c: (f05(*c), c[0], -c[1] - c[2])))
    return np.array(out)

CC = {k: m2_counts(k, v) for k, v in CONLL.items()}
BC = {k: bea_counts(v) for k, v in BEA.items()}
out = {"conll_m2": {k: {**prf(v[0]), "official": v[1]} for k, v in CC.items()},
       "bea_dev_errant": {k: prf(v) for k, v in BC.items()}}
for k, (per, off) in CC.items():
    assert abs(f05(*per.sum(0)) - off["F_0.5"]) < 6e-4, k                    # per-sentence == official M2
print("BEA-dev check vs notebook official errant_compare:",
      {"ours_final": (out["bea_dev_errant"]["ours_final"]["F05"], RES["factors_bench"]["bea_dev|lp0.7_ref1_tau0.0"]["F05"]),
       "gec_t5_small": (out["bea_dev_errant"]["gec_t5_small"]["F05"], RES["baseline|gotutiyan/gec-t5-small-clang8"]["bea_dev"]["F05"]),
       "qwen_fourshot": (out["bea_dev_errant"]["qwen_fourshot"]["F05"], RES["qwen"]["bea_dev|fourshot"]["F05"])})

# ---------------------------------------------------------------- C4 decode factors (stage 1) + no-edit factor
c4 = rd(f"{B}/pred_c4_stage1_variants.csv")
sub = [int(r["test_index"]) for r in c4]; r2c4 = rd(f"{P}/run2_results/t5max_examples.csv")
def c4_counts(preds):
    o_ = []
    for r, p in zip(c4, preds):
        s, t = r["src"], r["tgt"]; o = ANN.parse(s)
        g = set() if t == s else {(e.o_start, e.o_end, e.c_str) for e in ANN.annotate(o, ANN.parse(t)) if e.type != "noop"}
        h = set() if p == s else {(e.o_start, e.o_end, e.c_str) for e in ANN.annotate(o, ANN.parse(p)) if e.type != "noop"}
        o_.append((len(g & h), len(h - g), len(g - h)))
    return np.array(o_)
C4 = {"final": c4_counts([r["lp0.7_ref1_tau0.0"] for r in c4]), "no_threshold": c4_counts([r["lp0.7_ref1_taunone"] for r in c4]),
      "no_refinement": c4_counts([r["lp0.7_ref0_tau0.0"] for r in c4]), "lp1.0": c4_counts([r["lp1.0_ref1_tau0.0"] for r in c4]),
      "lp1.2": c4_counts([r["lp1.2_ref1_tau0.0"] for r in c4]),
      "stage1_run2decode": c4_counts([r["lp0.7_ref1_taunone"] for r in c4]),
      "run2_stored": c4_counts([r2c4[i]["pred"] for i in sub])}
out["c4_errant"] = {k: prf(v) for k, v in C4.items()}

# ---------------------------------------------------------------- paired bootstraps
rng = np.random.default_rng(0)
def boot(ca, cb, B_=2000):
    n = len(ca); idx = rng.integers(0, n, size=(B_, n))
    d = np.array([f05(*ca[ix].sum(0)) - f05(*cb[ix].sum(0)) for ix in idx])
    return {"delta": round(f05(*ca.sum(0)) - f05(*cb.sum(0)), 4),
            "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
            "p_two_sided": round(float(2 * min((d <= 0).mean(), (d >= 0).mean())), 4)}
T = {}
for other in ("qwen_fourshot", "qwen_zeroshot", "gec_t5_small", "gec_t5_base"):
    T[f"conll|ours_final-{other}"] = boot(CC["ours_final"][0], CC[other][0])
    T[f"bea|ours_final-{other}"] = boot(BC["ours_final"], BC[other])
for v in ("ours_no_threshold", "ours_no_refinement", "ours_lp1.0"):
    T[f"conll|ours_final-{v}"] = boot(CC["ours_final"][0], CC[v][0])
    T[f"bea|ours_final-{v}"] = boot(BC["ours_final"], BC[v])
for v in ("no_threshold", "no_refinement", "lp1.0", "lp1.2"):
    T[f"c4|final-{v}"] = boot(C4["final"], C4[v])
T["noedit|c4|with-without"] = boot(C4["stage1_run2decode"], C4["run2_stored"])
T["noedit|bea|with-without"] = boot(BC["stage1_run2decode"], BC["run2_stored"])
T["noedit|conll|with-without"] = boot(CC["stage1_run2decode"][0], CC["run2_stored_fixed"][0])
out["bootstrap"] = T
json.dump(out, open(f"{P}/round2_results/B_analysis.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
