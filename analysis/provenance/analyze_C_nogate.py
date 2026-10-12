"""Round-2 Notebook C (session 3: JOBS=["nogate"]) analysis. All arms trained in the current, correctly normalised image.

  1. no-edit pairs, end to end:  final_s2 (stage 1 WITH 15% no-edit pairs -> W&I)  vs  nogate_s2 (WITHOUT -> W&I)
                                 and at stage 1: final_s1 vs nogate_s1 (same decoding: lp 0.7 + refinement, no threshold)
  2. cleaning, second seed:      nogate_s1 (clean, seed 42)  vs  Notebook A noisy_s42 (noisy, seed 42) -- same recipe,
                                 same decoding (run-2 decoding); pooled with Notebook A's seed-1337 pair
  3. stage-2 effect without no-edit pairs: nogate_s2 vs nogate_s1 (same decoding, no threshold)
Endpoints and scorers as analyze_B / analyze_A: C4 clean 5k subset (single-reference ERRANT), BEA-2019 dev (ERRANT vs
gold M2, per sentence), CoNLL-2014 (official M2 via -v, asserted == official). Paired bootstrap, 2,000 resamples."""
import csv, json, os, re, subprocess, sys
import numpy as np
import errant

P = r"<PROJECT>"
S = r"<SCRATCH>/eval"
R = f"{P}/round2_results"
CF, CN, AN, AC = f"{R}/C_final", f"{R}/C_nogate", f"{R}/A_noisy", f"{R}/A_clean"
def rd(p): return list(csv.DictReader(open(p, encoding="utf-8", errors="replace")))
def f05(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0; r = tp / (tp + fn) if tp + fn else 1.0
    return 1.25 * p * r / (0.25 * p + r) if (p + r) else 0.0
def prf(c):
    tp, fp, fn = c.sum(0); return {"P": round(tp / max(tp + fp, 1), 4), "R": round(tp / max(tp + fn, 1), 4), "F05": round(f05(tp, fp, fn), 4)}
ANN = errant.load("en")

# ---------------------------------------------------------------- systems: (c4 rows, c4 pred col), bea preds, conll preds
def col(path, c): return [r[c] for r in rd(path)]
SYS = {
    "final_s2":    {"c4": col(f"{CF}/pred_c4_final_s2.csv", "ref1_tau0.0"), "bea": col(f"{CF}/pred_bea_dev_final_s2.csv", "ref1_tau0.0"),
                    "conll": col(f"{CF}/pred_conll_final_s2.csv", "ref1_tau0.0")},
    "nogate_s2":   {"c4": col(f"{CN}/pred_c4_nogate_s2.csv", "ref1_tau0.0"), "bea": col(f"{CN}/pred_bea_dev_nogate_s2.csv", "ref1_tau0.0"),
                    "conll": col(f"{CN}/pred_conll_nogate_s2.csv", "ref1_tau0.0")},
    "nogate_s2_notau": {"c4": col(f"{CN}/pred_c4_nogate_s2.csv", "ref1_taunone"), "bea": col(f"{CN}/pred_bea_dev_nogate_s2.csv", "ref1_taunone"),
                    "conll": col(f"{CN}/pred_conll_nogate_s2.csv", "ref1_taunone")},
    "final_s1_notau":  {"c4": col(f"{CF}/pred_c4_final_s1.csv", "ref1_taunone"), "bea": col(f"{CF}/pred_bea_dev_final_s1.csv", "ref1_taunone"),
                    "conll": col(f"{CF}/pred_conll_final_s1.csv", "ref1_taunone")},
    "nogate_s1":   {"c4": col(f"{CN}/pred_c4_nogate_s1.csv", "ref1_taunone"), "bea": col(f"{CN}/pred_bea_dev_nogate_s1.csv", "ref1_taunone"),
                    "conll": col(f"{CN}/pred_conll_nogate_s1.csv", "ref1_taunone")},
}
for arm, d in (("noisy_s42", AN), ("noisy_s1337", AN), ("clean_s1337", AC)):
    SYS[arm] = {"c4": col(f"{d}/pred_c4_{arm}.csv", "pred"), "bea": col(f"{d}/pred_bea_dev_{arm}.csv", "pred_retok"),
                "conll": col(f"{d}/pred_conll_{arm}.csv", "pred_retok")}

# every system must be on identical sentences in identical order
c4n = rd(f"{CN}/pred_c4_nogate_s1.csv"); C4_SRC, C4_TGT = [r["src"] for r in c4n], [r["tgt"] for r in c4n]
assert [r["src"] for r in rd(f"{CF}/pred_c4_final_s1.csv")] == C4_SRC
assert [r["src"] for r in rd(f"{AN}/pred_c4_noisy_s42.csv")] == C4_SRC, "C4 subset differs from notebook A"
assert [r["test_index"] for r in c4n] == [r["test_index"] for r in rd(f"{AN}/c4_eval_subset_index.csv")]
BEA_SRC = [r["src"] for r in rd(f"{CN}/pred_bea_dev_nogate_s1.csv")]
assert BEA_SRC == [r["src"] for r in rd(f"{CF}/pred_bea_dev_final_s2.csv")] == [r["src"] for r in rd(f"{AN}/pred_bea_dev_noisy_s42.csv")]
for k, v in SYS.items(): assert (len(v["c4"]), len(v["bea"]), len(v["conll"])) == (5000, 4384, 1312), k

# ---------------------------------------------------------------- per-sentence counts (cached)
def cached(name, fn):
    p = f"{S}/_N_{name}.json"
    if os.path.exists(p): return np.array(json.load(open(p)))
    a = fn(); json.dump(np.asarray(a).tolist(), open(p, "w")); return np.asarray(a)
def c4_counts(preds):
    out = []
    for s, t, p in zip(C4_SRC, C4_TGT, preds):
        o = ANN.parse(s)
        g = set() if t == s else {(e.o_start, e.o_end, e.c_str) for e in ANN.annotate(o, ANN.parse(t)) if e.type != "noop"}
        h = set() if p == s else {(e.o_start, e.o_end, e.c_str) for e in ANN.annotate(o, ANN.parse(p)) if e.type != "noop"}
        out.append((len(g & h), len(h - g), len(g - h)))
    return out
gold = []
for b in open(f"{S}/wi+locness/m2/ABCN.dev.gold.bea19.m2", encoding="utf-8").read().strip().split("\n\n"):
    g = {}
    for ln in b.split("\n")[1:]:
        f = ln[2:].split("|||"); s, e = map(int, f[0].split()); a = int(f[-1]); g.setdefault(a, set())
        if f[1] not in ("noop", "UNK"): g[a].add((s, e, "" if f[2] == "-NONE-" else f[2]))
    gold.append(g or {0: set()})
def bea_counts(hyps):
    out = []
    for s, h, g in zip(BEA_SRC, hyps, gold):
        o = ANN.parse(s)
        he = {(e.o_start, e.o_end, e.c_str) for e in ANN.annotate(o, ANN.parse(h)) if e.type != "noop"}
        out.append(max(((len(he & ge), len(he - ge), len(ge - he)) for ge in g.values()), key=lambda c: (f05(*c), c[0], -c[1] - c[2])))
    return out
GOLD_C = f"{S}/conll14st-test-data/noalt/official-2014.combined.m2"
def m2_counts(name, hyps):
    hp = f"{S}/_N_{name}.txt"; open(hp, "w", encoding="utf-8").write("\n".join(hyps) + "\n")
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
    assert abs(f05(*per.sum(0)) - off) < 6e-4, name                          # per-sentence == official M2
    return per

CNT = {}
for name, d in SYS.items():
    CNT[(name, "c4")] = cached(f"c4_{name}", lambda: c4_counts(d["c4"]))
    CNT[(name, "bea")] = cached(f"bea_{name}", lambda: bea_counts(d["bea"]))
    CNT[(name, "conll")] = cached(f"conll_{name}", lambda: m2_counts(name, d["conll"]))
    print(name, {ep: prf(CNT[(name, ep)])["F05"] for ep in ("c4", "bea", "conll")}, flush=True)

# reproduce the notebooks' own totals
RN = json.load(open(f"{CN}/r2C_results.json")); RF = json.load(open(f"{CF}/r2C_results.json"))
RA = {**json.load(open(f"{AN}/r2A_results.json")), **json.load(open(f"{AC}/r2A_results.json"))}
chk = {"nogate_s1": (RN["eval|nogate_s1"], "ref1_taunone"), "nogate_s2": (RN["eval|nogate_s2"], "ref1_tau0.0"),
       "final_s2": (RF["eval|final_s2"], "ref1_tau0.0"), "final_s1_notau": (RF["eval|final_s1"], "ref1_taunone")}
for name, (ev, k) in chk.items():
    c4 = ev[f"c4|{k}"]; assert tuple(int(x) for x in CNT[(name, "c4")].sum(0)) == (c4["TP"], c4["FP"], c4["FN"]), name
    print(f"  {name}: BEA per-sentence F05 {prf(CNT[(name, 'bea')])['F05']} vs notebook {ev[f'bea_dev|{k}']['F05']} | "
          f"CoNLL {prf(CNT[(name, 'conll')])['F05']} vs {ev[f'conll|{k}']['F05']}")
for arm in ("noisy_s42", "noisy_s1337", "clean_s1337"):
    c4 = RA[f"eval|{arm}"]["c4_clean_subset"]; assert tuple(int(x) for x in CNT[(arm, "c4")].sum(0)) == (c4["TP"], c4["FP"], c4["FN"]), arm

# ---------------------------------------------------------------- paired bootstraps
def F(c): return f05(*c.sum(0))
out = {"scores": {n: {ep: prf(CNT[(n, ep)]) for ep in ("c4", "bea", "conll")} for n in SYS}, "bootstrap": {}}
rng = np.random.default_rng(0); B = 2000
def boot(ep, fn):
    n = len(CNT[("nogate_s1", ep)]); idx = rng.integers(0, n, size=(B, n))
    d = np.array([fn(ix) for ix in idx]); pt = fn(np.arange(n))
    return {"delta": round(pt, 4), "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
            "p_two_sided": round(float(2 * min((d <= 0).mean(), (d >= 0).mean())), 4)}
def diff(a, b, ep): return lambda ix: F(CNT[(a, ep)][ix]) - F(CNT[(b, ep)][ix])
for ep in ("c4", "bea", "conll"):
    T = out["bootstrap"]
    T[f"noedit_end_to_end|{ep}|with_minus_without"] = boot(ep, diff("final_s2", "nogate_s2", ep))
    T[f"noedit_stage1|{ep}|with_minus_without"] = boot(ep, diff("final_s1_notau", "nogate_s1", ep))
    T[f"clean_minus_noisy|{ep}|seed42"] = boot(ep, diff("nogate_s1", "noisy_s42", ep))
    T[f"clean_minus_noisy|{ep}|seed1337"] = boot(ep, diff("clean_s1337", "noisy_s1337", ep))
    T[f"clean_minus_noisy|{ep}|mean_of_seeds"] = boot(ep, lambda ix: 0.5 * (diff("nogate_s1", "noisy_s42", ep)(ix) + diff("clean_s1337", "noisy_s1337", ep)(ix)))
    T[f"seed_noise_clean|{ep}|s1337_minus_s42"] = boot(ep, diff("clean_s1337", "nogate_s1", ep))
    T[f"stage2_effect_no_noedit|{ep}|s2_minus_s1_notau"] = boot(ep, diff("nogate_s2_notau", "nogate_s1", ep))
    T[f"threshold|{ep}|nogate_s2_tau0_minus_none"] = boot(ep, diff("nogate_s2", "nogate_s2_notau", ep))
out["jfleg_gleu_official"] = {"final_s2": RF["eval|final_s2"]["jfleg|ref1_tau0.0"]["GLEU"], "nogate_s2": RN["eval|nogate_s2"]["jfleg|ref1_tau0.0"]["GLEU"],
                              "nogate_s1": RN["eval|nogate_s1"]["jfleg|ref1_taunone"]["GLEU"], "final_s1_notau": RF["eval|final_s1"]["jfleg|ref1_taunone"]["GLEU"],
                              **{a: RA[f"eval|{a}"]["jfleg_gleu_official"]["GLEU"] for a in ("noisy_s42", "noisy_s1337", "clean_s1337")}}
# copy gate, corrected training, matched step (only the gate's first evaluation exists: the run timed out)
gs = json.load(open(f"{R}/C_session2_gate/trainer_state_gate_ck2500.json"))["log_history"]
ng = json.load(open(f"{CN}/trainer_log_nogate_s1.json"))
out["gate_matched_step_2500"] = {"gate": {"train_loss": next(r["loss"] for r in gs if r.get("step") == 2500 and "loss" in r),
                                          "eval_loss": next(r["eval_loss"] for r in gs if "eval_loss" in r), "eval_runtime_s": 2244.2},
                                 "nogate": {"train_loss": next(r["loss"] for r in ng if r.get("step") == 2500 and "loss" in r),
                                            "eval_loss": next(r["eval_loss"] for r in ng if r.get("step") == 2500 and "eval_loss" in r),
                                            "eval_runtime_s": next(r["eval_runtime"] for r in ng if r.get("step") == 2500 and "eval_runtime" in r)},
                                 "note": "label-smoothed CE; the gate's output is log(mixture + 1e-9), so the two losses are not strictly comparable"}
json.dump(out, open(f"{R}/C_nogate_analysis.json", "w"), indent=1, default=float)
print(json.dumps(out["bootstrap"], indent=1)); print(json.dumps(out["jfleg_gleu_official"])); print(json.dumps(out["gate_matched_step_2500"]))
