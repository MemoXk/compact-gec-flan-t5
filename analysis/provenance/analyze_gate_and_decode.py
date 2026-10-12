"""Two remaining rows of the full-budget ablation table, on the same endpoints and scorers as analyze_C_nogate.py:
  1. copy gate (June configuration, both arms identical apart from the gate): run 1 (gate) vs run 2 (no gate),
     on the C4 clean 5k subset, BEA-2019 dev (ERRANT, spaCy retokenization) and CoNLL-2014 (official M2, regex
     retokenization + the hyphen re-join used in analyze_B for the June pipeline);
  2. decode-time factors of the retrained final system on C4: threshold tau=0 vs none, refinement on vs off."""
import csv, json, os, re, subprocess, sys
import numpy as np
import errant

P = r"<PROJECT>"
S = r"<SCRATCH>/eval"
sys.path.insert(0, S)
from retok import regex_tok, spacy_tok
R = f"{P}/round2_results"
def rd(p): return list(csv.DictReader(open(p, encoding="utf-8", errors="replace")))
def f05(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0; r = tp / (tp + fn) if tp + fn else 1.0
    return 1.25 * p * r / (0.25 * p + r) if (p + r) else 0.0
def prf(c):
    tp, fp, fn = c.sum(0); return {"P": round(tp / max(tp + fp, 1), 4), "R": round(tp / max(tp + fn, 1), 4), "F05": round(f05(tp, fp, fn), 4)}
ANN = errant.load("en")
def rejoin(pred, src):
    toks = set(src.split())
    return re.sub(r"(\S+) - (\S+)", lambda m: m.group(1) + "-" + m.group(2) if m.group(1) + "-" + m.group(2) in toks else m.group(0), pred)

c4n = rd(f"{R}/C_nogate/pred_c4_nogate_s1.csv"); C4_SRC, C4_TGT = [r["src"] for r in c4n], [r["tgt"] for r in c4n]
sub = [int(r["test_index"]) for r in c4n]
r1c4, r2c4 = rd(f"{P}/run1_results/t5max_examples.csv"), rd(f"{P}/run2_results/t5max_examples.csv")
assert [r1c4[i]["src"] for i in sub] == C4_SRC == [r2c4[i]["src"] for i in sub]
fc4 = rd(f"{R}/C_final/pred_c4_final_s2.csv"); assert [r["src"] for r in fc4] == C4_SRC

def c4_counts(preds):
    out = []
    for s, t, p in zip(C4_SRC, C4_TGT, preds):
        o = ANN.parse(s)
        g = set() if t == s else {(e.o_start, e.o_end, e.c_str) for e in ANN.annotate(o, ANN.parse(t)) if e.type != "noop"}
        h = set() if p == s else {(e.o_start, e.o_end, e.c_str) for e in ANN.annotate(o, ANN.parse(p)) if e.type != "noop"}
        out.append((len(g & h), len(h - g), len(g - h)))
    return np.array(out)
gold = []
for b in open(f"{S}/wi+locness/m2/ABCN.dev.gold.bea19.m2", encoding="utf-8").read().strip().split("\n\n"):
    g = {}
    for ln in b.split("\n")[1:]:
        f = ln[2:].split("|||"); s, e = map(int, f[0].split()); a = int(f[-1]); g.setdefault(a, set())
        if f[1] not in ("noop", "UNK"): g[a].add((s, e, "" if f[2] == "-NONE-" else f[2]))
    gold.append(g or {0: set()})
def bea_counts(srcs, hyps):
    out = []
    for s, h, g in zip(srcs, hyps, gold):
        o = ANN.parse(s)
        he = {(e.o_start, e.o_end, e.c_str) for e in ANN.annotate(o, ANN.parse(h)) if e.type != "noop"}
        out.append(max(((len(he & ge), len(he - ge), len(ge - he)) for ge in g.values()), key=lambda c: (f05(*c), c[0], -c[1] - c[2])))
    return np.array(out)
GOLD_C = f"{S}/conll14st-test-data/noalt/official-2014.combined.m2"
def m2_counts(name, hyps):
    hp = f"{S}/_G_{name}.txt"; open(hp, "w", encoding="utf-8").write("\n".join(hyps) + "\n")
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
    off = float(re.search(r"F_0\.5\s*:\s*([0-9.]+)", out).group(1))
    per = np.diff(np.vstack([np.zeros((1, 3), int), np.array(cum)]), axis=0)
    assert abs(f05(*per.sum(0)) - off) < 6e-4, name
    return per

CNT = {}
for run, c4rows in (("run1_gate", r1c4), ("run2_nogate", r2c4)):
    d = run.split("_")[0]
    CNT[(run, "c4")] = c4_counts([c4rows[i]["pred"] for i in sub])
    b = rd(f"{P}/{d}_results/t5max_bea19_examples.csv")
    CNT[(run, "bea")] = bea_counts([r["src"] for r in b], [spacy_tok(r["pred"]) for r in b])
    c = rd(f"{P}/{d}_results/t5max_conll14_examples.csv")
    CNT[(run, "conll")] = m2_counts(run, [regex_tok(rejoin(r["pred"], r["src"])) for r in c])
    print(run, {ep: prf(CNT[(run, ep)]) for ep in ("c4", "bea", "conll")}, flush=True)
for k, col in (("final_tau0", "ref1_tau0.0"), ("final_notau", "ref1_taunone"), ("final_noref", "ref0_tau0.0")):
    CNT[(k, "c4")] = c4_counts([r[col] for r in fc4]); print(k, prf(CNT[(k, "c4")]), flush=True)

rng = np.random.default_rng(0)
def boot(ca, cb, B_=2000):
    n = len(ca); idx = rng.integers(0, n, size=(B_, n))
    d = np.array([f05(*ca[ix].sum(0)) - f05(*cb[ix].sum(0)) for ix in idx])
    return {"delta": round(f05(*ca.sum(0)) - f05(*cb.sum(0)), 4), "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
            "p_two_sided": round(float(2 * min((d <= 0).mean(), (d >= 0).mean())), 4)}
out = {"scores": {f"{a}|{ep}": prf(v) for (a, ep), v in CNT.items()}, "bootstrap": {}}
for ep in ("c4", "bea", "conll"):
    out["bootstrap"][f"gate_minus_nogate|{ep}"] = boot(CNT[("run1_gate", ep)], CNT[("run2_nogate", ep)])
out["bootstrap"]["final|c4|tau0_minus_none"] = boot(CNT[("final_tau0", "c4")], CNT[("final_notau", "c4")])
out["bootstrap"]["final|c4|refine_minus_norefine"] = boot(CNT[("final_tau0", "c4")], CNT[("final_noref", "c4")])
json.dump(out, open(f"{R}/gate_decode_analysis.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
