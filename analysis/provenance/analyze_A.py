"""Round-2 Notebook A analysis: reference-side cleaning at the FINAL training budget (1.15M pairs, run-2 recipe).

Arms:   noisy_s42, noisy_s1337, clean_s1337  -- trained on Kaggle (round2_results/A_*)
        clean_s42                            -- the paper's run 2 (stored predictions; NOTE: its selected checkpoint
                                                was step 7,500 under an older transformers, the new runs ended at 8,985)
Endpoints: C4 clean test subset (5,000, single reference, ERRANT), BEA-2019 dev (official ERRANT compare),
           CoNLL-2014 test (official M2), JFLEG test (official GLEU, from the notebook).
Significance: paired bootstrap over sentences (2,000 resamples) on per-sentence edit counts that reproduce the
official totals exactly (asserted)."""
import csv, json, os, re, subprocess, sys
import numpy as np
import errant

P = r"<PROJECT>"
S = r"<SCRATCH>/eval"
sys.path.insert(0, S)
from retok import regex_tok, spacy_tok

OUT = f"{P}/round2_results/A_analysis.json"
A_N, A_C = f"{P}/round2_results/A_noisy", f"{P}/round2_results/A_clean"
def rd(p): return list(csv.DictReader(open(p, encoding="utf-8", errors="replace")))
def f05(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0; r = tp / (tp + fn) if tp + fn else 1.0
    return 1.25 * p * r / (0.25 * p + r) if (p + r) else 0.0

sub = [int(x["test_index"]) for x in rd(f"{A_N}/c4_eval_subset_index.csv")]
run2_c4 = rd(f"{P}/run2_results/t5max_examples.csv")
SYS = {
    "noisy_s42":   {"c4": rd(f"{A_N}/pred_c4_noisy_s42.csv"),   "bea": rd(f"{A_N}/pred_bea_dev_noisy_s42.csv"),   "conll": rd(f"{A_N}/pred_conll_noisy_s42.csv")},
    "noisy_s1337": {"c4": rd(f"{A_N}/pred_c4_noisy_s1337.csv"), "bea": rd(f"{A_N}/pred_bea_dev_noisy_s1337.csv"), "conll": rd(f"{A_N}/pred_conll_noisy_s1337.csv")},
    "clean_s1337": {"c4": rd(f"{A_C}/pred_c4_clean_s1337.csv"), "bea": rd(f"{A_C}/pred_bea_dev_clean_s1337.csv"), "conll": rd(f"{A_C}/pred_conll_clean_s1337.csv")},
}
c4_ref = SYS["noisy_s42"]["c4"]
SYS["clean_s42"] = {"c4": [{"src": run2_c4[i]["src"], "tgt": run2_c4[i]["tgt"], "pred": run2_c4[i]["pred"]} for i in sub],
                    "bea": [{"src": r["src"], "pred_retok": spacy_tok(r["pred"])} for r in rd(f"{P}/run2_results/t5max_bea19_examples.csv")],
                    "conll": [{"src": r["src"], "pred_retok": regex_tok(r["pred"])} for r in rd(f"{P}/run2_results/t5max_conll14_examples.csv")]}
ANN = errant.load("en")

# ---------------------------------------------------------------- per-sentence counts
def c4_counts(rows):
    out = []
    for r in rows:
        s, t, p = r["src"], r["tgt"], r["pred"]; o = ANN.parse(s)
        g = set() if t == s else {(e.o_start, e.o_end, e.c_str) for e in ANN.annotate(o, ANN.parse(t)) if e.type != "noop"}
        h = set() if p == s else {(e.o_start, e.o_end, e.c_str) for e in ANN.annotate(o, ANN.parse(p)) if e.type != "noop"}
        out.append((len(g & h), len(h - g), len(g - h)))
    return np.array(out)

def read_gold(path):
    gold = []
    for b in open(path, encoding="utf-8").read().strip().split("\n\n"):
        ls = b.split("\n"); g = {}
        for ln in ls[1:]:
            f = ln[2:].split("|||"); s, e = map(int, f[0].split()); a = int(f[-1]); g.setdefault(a, set())
            if f[1] not in ("noop", "UNK"):                      # errant_compare discards both in correction mode
                g[a].add((s, e, "" if f[2] == "-NONE-" else f[2]))
        gold.append(g or {0: set()})
    return gold
BEA_GOLD = read_gold(f"{S}/wi+locness/m2/ABCN.dev.gold.bea19.m2")

def bea_counts(rows):
    out = []
    for r, g in zip(rows, BEA_GOLD):
        o = ANN.parse(r["src"])
        h = {(e.o_start, e.o_end, e.c_str) for e in ANN.annotate(o, ANN.parse(r["pred_retok"])) if e.type != "noop"}
        out.append(max(((len(h & ge), len(h - ge), len(ge - h)) for ge in g.values()), key=lambda c: (f05(*c), c[0], -c[1] - c[2])))
    return np.array(out)

GOLD_C = f"{S}/conll14st-test-data/noalt/official-2014.combined.m2"
def conll_counts(name, rows):
    hp = f"{S}/_A_{name}.txt"; open(hp, "w", encoding="utf-8").write("\n".join(r["pred_retok"] for r in rows) + "\n")
    outp = subprocess.run([sys.executable, f"{S}/m2scorer/scripts/m2scorer.py", "-v", hp, GOLD_C], capture_output=True,
                          text=True, env={**os.environ, "PYTHONUTF8": "1"}).stdout
    rows_cum, cur, ann = [], {}, None
    for ln in outp.splitlines():
        m = re.match(r">> Annotator: (\d+)", ln)
        if m: ann = int(m.group(1)); cur[ann] = [0, 0, 0]; continue
        m = re.match(r"# (correct|proposed|gold)\s*:\s*(\d+)", ln)
        if m and ann is not None: cur[ann][("correct", "proposed", "gold").index(m.group(1))] = int(m.group(2)); continue
        m = re.match(r">> Chosen Annotator for line (\d+) : (\d+)", ln)
        if m: c, p, g = cur[int(m.group(2))]; rows_cum.append((c, p - c, g - c)); cur = {}; ann = None
    tot = dict(re.findall(r"(Precision|Recall|F_0\.5)\s*:\s*([0-9.]+)", outp))
    per = np.diff(np.vstack([np.zeros((1, 3), int), np.array(rows_cum)]), axis=0)   # running totals -> per sentence
    return per, {"P": float(tot["Precision"]), "R": float(tot["Recall"]), "F05": float(tot["F_0.5"])}

C, OFFICIAL_CONLL = {}, {}
for name, d in SYS.items():
    C[(name, "c4")] = c4_counts(d["c4"])
    C[(name, "bea")] = bea_counts(d["bea"])
    C[(name, "conll")], OFFICIAL_CONLL[name] = conll_counts(name, d["conll"])
    print(name, "done", flush=True)

# validation: per-sentence totals must reproduce the official notebook numbers
nb = {**json.load(open(f"{A_N}/r2A_results.json")), **json.load(open(f"{A_C}/r2A_results.json"))}
for name in ("noisy_s42", "noisy_s1337", "clean_s1337"):
    e = nb[f"eval|{name}"]
    assert tuple(C[(name, "c4")].sum(0)) == (e["c4_clean_subset"]["TP"], e["c4_clean_subset"]["FP"], e["c4_clean_subset"]["FN"]), name
    b = e["bea_dev_errant"]; got = tuple(int(x) for x in C[(name, "bea")].sum(0))
    print(f"  BEA-dev totals {name}: per-sentence {got} vs official {(b['TP'], b['FP'], b['FN'])}")
for name in SYS:
    assert abs(f05(*C[(name, "conll")].sum(0)) - OFFICIAL_CONLL[name]["F05"]) < 6e-4, name

# ---------------------------------------------------------------- scores + paired bootstrap
def F(c): return f05(*c.sum(0))
res = {"scores": {name: {ep: round(F(C[(name, ep)]), 4) for ep in ("c4", "bea", "conll")} for name in SYS},
       "conll_official_m2": OFFICIAL_CONLL,
       "jfleg_gleu_official": {n: nb[f"eval|{n}"]["jfleg_gleu_official"]["GLEU"] for n in ("noisy_s42", "noisy_s1337", "clean_s1337")},
       "jfleg_do_nothing": nb["jfleg_do_nothing"]["GLEU"], "train": {n: nb[f"train|{n}"] for n in ("noisy_s42", "noisy_s1337", "clean_s1337")}}
rng = np.random.default_rng(0); B = 2000
for ep in ("c4", "bea", "conll"):
    n = len(C[("noisy_s42", ep)]); idx = rng.integers(0, n, size=(B, n))
    def boot(fn):
        d = np.array([fn(ix) for ix in idx]); pt = fn(np.arange(n))
        return {"delta": round(pt, 4), "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
                "p_two_sided": round(float(2 * min((d <= 0).mean(), (d >= 0).mean())), 4)}
    res[f"clean_minus_noisy|{ep}|seed1337"] = boot(lambda ix: F(C[("clean_s1337", ep)][ix]) - F(C[("noisy_s1337", ep)][ix]))
    res[f"clean_minus_noisy|{ep}|seed42_run2"] = boot(lambda ix: F(C[("clean_s42", ep)][ix]) - F(C[("noisy_s42", ep)][ix]))
    res[f"clean_minus_noisy|{ep}|mean_of_seeds"] = boot(lambda ix: 0.5 * (F(C[("clean_s1337", ep)][ix]) - F(C[("noisy_s1337", ep)][ix])
                                                                      + F(C[("clean_s42", ep)][ix]) - F(C[("noisy_s42", ep)][ix])))
    res[f"seed_noise|{ep}|noisy_s1337_minus_s42"] = boot(lambda ix: F(C[("noisy_s1337", ep)][ix]) - F(C[("noisy_s42", ep)][ix]))
json.dump(res, open(OUT, "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
