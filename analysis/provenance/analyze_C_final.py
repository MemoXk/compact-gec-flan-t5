"""Round-2 Notebook C (session 1) analysis: the RETRAINED final system vs the June system and the baselines.

Same protocol as analyze_B.py: official CoNLL-2014 M2 (per-sentence counts via m2scorer -v, asserted to reproduce the
official totals), BEA-2019 dev ERRANT per sentence vs the gold M2, paired bootstrap (2,000 resamples).
Retrained predictions in C_final/*.csv are already retokenized by the notebook (RETOK[bench])."""
import csv, json, os, re, subprocess, sys
import numpy as np
import errant

P = r"<PROJECT>"
S = r"<SCRATCH>/eval"
B, C = f"{P}/round2_results/B", f"{P}/round2_results/C_final"
RES_C = json.load(open(f"{C}/r2C_results.json", encoding="utf-8"))
def rd(p): return list(csv.DictReader(open(p, encoding="utf-8", errors="replace")))
def f05(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0; r = tp / (tp + fn) if tp + fn else 1.0
    return 1.25 * p * r / (0.25 * p + r) if (p + r) else 0.0
def prf(c):
    tp, fp, fn = c.sum(0); return {"P": round(tp / max(tp + fp, 1), 4), "R": round(tp / max(tp + fn, 1), 4), "F05": round(f05(tp, fp, fn), 4)}
ANN = errant.load("en")

cs2, bs2 = rd(f"{C}/pred_conll_final_s2.csv"), rd(f"{C}/pred_bea_dev_final_s2.csv")
cs1, bs1 = rd(f"{C}/pred_conll_final_s1.csv"), rd(f"{C}/pred_bea_dev_final_s1.csv")
cvB, bvB = rd(f"{B}/pred_conll_stage2_variants.csv"), rd(f"{B}/pred_bea_dev_stage2_variants.csv")
CONLL = {"retrained_final": [r["ref1_tau0.0"] for r in cs2], "retrained_no_threshold": [r["ref1_taunone"] for r in cs2],
         "retrained_no_refinement": [r["ref0_tau0.0"] for r in cs2], "retrained_stage1": [r["ref1_tau0.0"] for r in cs1],
         "june_final": [r["lp0.7_ref1_tau0.0"] for r in cvB],
         "qwen_fourshot": [r["pred_retok"] for r in rd(f"{B}/pred_conll_qwen_fourshot.csv")],
         "gec_t5_small": [r["pred_retok"] for r in rd(f"{B}/pred_conll_gec-t5-small-clang8.csv")],
         "gec_t5_base": [r["pred_retok"] for r in rd(f"{B}/pred_conll_gec-t5-base-clang8.csv")]}
BEA = {"retrained_final": [r["ref1_tau0.0"] for r in bs2], "retrained_no_threshold": [r["ref1_taunone"] for r in bs2],
       "retrained_no_refinement": [r["ref0_tau0.0"] for r in bs2], "retrained_stage1": [r["ref1_tau0.0"] for r in bs1],
       "june_final": [r["lp0.7_ref1_tau0.0"] for r in bvB],
       "qwen_fourshot": [r["pred_retok"] for r in rd(f"{B}/pred_bea_dev_qwen_fourshot.csv")],
       "gec_t5_small": [r["pred_retok"] for r in rd(f"{B}/pred_bea_dev_gec-t5-small-clang8.csv")],
       "gec_t5_base": [r["pred_retok"] for r in rd(f"{B}/pred_bea_dev_gec-t5-base-clang8.csv")]}
assert [r["src"] for r in bs2] == [r["src"] for r in bvB], "BEA dev source order differs between notebooks"
bsrc = [r["src"] for r in bs2]

GOLD_C = f"{S}/conll14st-test-data/noalt/official-2014.combined.m2"
def m2_counts(name, hyps):
    cache = f"{S}/_C_m2v_{name}.json"
    if os.path.exists(cache): d = json.load(open(cache)); return np.array(d["per"]), d["off"]
    hp = f"{S}/_C_{name}.txt"; open(hp, "w", encoding="utf-8").write("\n".join(hyps) + "\n")
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

gold = []
for b in open(f"{S}/wi+locness/m2/ABCN.dev.gold.bea19.m2", encoding="utf-8").read().strip().split("\n\n"):
    g = {}
    for ln in b.split("\n")[1:]:
        f = ln[2:].split("|||"); s, e = map(int, f[0].split()); a = int(f[-1]); g.setdefault(a, set())
        if f[1] not in ("noop", "UNK"): g[a].add((s, e, "" if f[2] == "-NONE-" else f[2]))
    gold.append(g or {0: set()})
def bea_counts(name, hyps):
    cache = f"{S}/_C_bea_{name}.json"
    if os.path.exists(cache): return np.array(json.load(open(cache)))
    out = []
    for s, h, g in zip(bsrc, hyps, gold):
        o = ANN.parse(s)
        he = {(e.o_start, e.o_end, e.c_str) for e in ANN.annotate(o, ANN.parse(h)) if e.type != "noop"}
        out.append(max(((len(he & ge), len(he - ge), len(ge - he)) for ge in g.values()), key=lambda c: (f05(*c), c[0], -c[1] - c[2])))
    json.dump(out, open(cache, "w")); return np.array(out)

CC = {k: m2_counts(k, v) for k, v in CONLL.items()}
BC = {k: bea_counts(k, v) for k, v in BEA.items()}
for k, (per, off) in CC.items():
    assert abs(f05(*per.sum(0)) - off["F_0.5"]) < 6e-4, k
out = {"conll_m2": {k: {**prf(v[0]), "official": v[1]} for k, v in CC.items()},
       "bea_dev_errant": {k: prf(v) for k, v in BC.items()}}
ev = RES_C["eval|final_s2"]
print("check vs notebook official scorers:",
      {"conll": (out["conll_m2"]["retrained_final"]["F05"], ev["conll|ref1_tau0.0"]["F05"]),
       "bea_dev": (out["bea_dev_errant"]["retrained_final"]["F05"], ev["bea_dev|ref1_tau0.0"]["F05"])})

rng = np.random.default_rng(0)
def boot(ca, cb, B_=2000):
    n = len(ca); idx = rng.integers(0, n, size=(B_, n))
    d0 = f05(*ca.sum(0)) - f05(*cb.sum(0))
    ds = np.array([f05(*ca[i].sum(0)) - f05(*cb[i].sum(0)) for i in idx])
    return {"delta": round(d0, 4), "ci95": [round(float(np.percentile(ds, 2.5)), 4), round(float(np.percentile(ds, 97.5)), 4)],
            "p_two_sided": round(float(min(1.0, 2 * min((ds <= 0).mean(), (ds >= 0).mean()))), 3)}
out["bootstrap"] = {}
for other in ["june_final", "qwen_fourshot", "gec_t5_small", "gec_t5_base", "retrained_no_threshold", "retrained_no_refinement", "retrained_stage1"]:
    out["bootstrap"][f"conll|retrained_final-{other}"] = boot(CC["retrained_final"][0], CC[other][0])
    out["bootstrap"][f"bea|retrained_final-{other}"] = boot(BC["retrained_final"], BC[other])
json.dump(out, open(f"{P}/round2_results/C_final_analysis.json", "w"), indent=1)
print(json.dumps(out, indent=1))
