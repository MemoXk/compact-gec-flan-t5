"""Re-score every released system output with the official scorers and recompute the paper's paired bootstraps.

    python analysis/get_data.py            # once: CoNLL-2014 test, W&I+LOCNESS v2.1, JFLEG test + official GLEU
    python analysis/verify.py              # writes results/verified/verified_scores.json
    python analysis/verify.py --check      # compares with the committed file and prints PASS / FAIL

Scorers (Section 4.7 of the paper):
  conll   CoNLL-2014 test, official MaxMatch (M2) scorer (third_party/m2scorer_py3, a Python-3 port of the original);
          per-sentence counts are recovered from `m2scorer -v` and asserted to reproduce the official totals
  bea_dev BEA-2019 development, ERRANT edits per sentence against the gold M2 (best annotator per sentence, as
          errant_compare); totals equal errant_compare's
  c4      in-domain 5,000-pair C4-200M test subset (splits/c4_test_subset_5k.tsv), single-reference ERRANT
  jfleg   JFLEG test, official GLEU (keisks/jfleg eval/gleu.py, 500 iterations); no paired test
Paired bootstrap: 2,000 resamples of sentences; each comparison uses its own seeded generator.
"""
import argparse, csv, hashlib, json, os, re, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "data"), help="directory filled by analysis/get_data.py")
ap.add_argument("--check", action="store_true", help="compare with results/verified/verified_scores.json")
ap.add_argument("--only", nargs="*", help="restrict to these systems")
args = ap.parse_args()
DATA = Path(args.data); CACHE = ROOT / ".cache"; CACHE.mkdir(exist_ok=True)
OUTF = ROOT / "results" / "verified" / "verified_scores.json"

def f05(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0; r = tp / (tp + fn) if tp + fn else 1.0
    return 1.25 * p * r / (0.25 * p + r) if (p + r) else 0.0
def prf(c):
    tp, fp, fn = (int(x) for x in np.asarray(c).sum(0))
    return {"P": round(tp / max(tp + fp, 1), 4), "R": round(tp / max(tp + fn, 1), 4), "F05": round(f05(tp, fp, fn), 4), "TP": tp, "FP": fp, "FN": fn}
def read_hyps(system, bench):
    p = ROOT / "outputs" / bench / f"{system}.txt"
    return p.read_text(encoding="utf-8").split("\n")[:-1] if p.exists() else None
def cached(key, fn):
    p = CACHE / (hashlib.sha1(key.encode()).hexdigest() + ".json")
    if p.exists(): return np.array(json.load(open(p)))
    v = np.asarray(fn()); tmp = p.with_suffix(f".{os.getpid()}.tmp")
    json.dump(v.tolist(), open(tmp, "w")); os.replace(tmp, p); return v          # atomic: parallel runs share the cache

_ANN = None
def ann():
    global _ANN
    if _ANN is None:
        import errant; _ANN = errant.load("en")
    return _ANN

# ---------------------------------------------------------------- CoNLL-2014 (official M2, per sentence)
CONLL_GOLD = DATA / "conll14st-test-data" / "noalt" / "official-2014.combined.m2"
M2 = ROOT / "third_party" / "m2scorer_py3" / "scripts" / "m2scorer.py"
def conll_counts(hyps, tag):
    hp = CACHE / f"_hyp_{tag}.txt"; hp.write_text("\n".join(hyps) + "\n", encoding="utf-8")
    out = subprocess.run([sys.executable, str(M2), "-v", str(hp), str(CONLL_GOLD)], capture_output=True, text=True,
                         env={**os.environ, "PYTHONUTF8": "1"}).stdout
    cum, cur, a = [], {}, None
    for ln in out.splitlines():
        m = re.match(r">> Annotator: (\d+)", ln)
        if m: a = int(m.group(1)); cur[a] = [0, 0, 0]; continue
        m = re.match(r"# (correct|proposed|gold)\s*:\s*(\d+)", ln)
        if m and a is not None: cur[a][("correct", "proposed", "gold").index(m.group(1))] = int(m.group(2)); continue
        m = re.match(r">> Chosen Annotator for line (\d+) : (\d+)", ln)
        if m: c, p, g = cur[int(m.group(2))]; cum.append((c, p - c, g - c)); cur = {}; a = None
    off = float(re.search(r"F_0\.5\s*:\s*([0-9.]+)", out).group(1))
    per = np.diff(np.vstack([np.zeros((1, 3), int), np.array(cum)]), axis=0)
    assert abs(f05(*per.sum(0)) - off) < 6e-4, f"{tag}: per-sentence counts do not reproduce the official M2 score"
    return per

# ---------------------------------------------------------------- BEA-2019 dev (ERRANT, per sentence)
BEA_GOLD = DATA / "wi+locness" / "m2" / "ABCN.dev.gold.bea19.m2"
def bea_gold():
    src, gold = [], []
    for b in BEA_GOLD.read_text(encoding="utf-8").strip().split("\n\n"):
        ls = b.split("\n"); src.append(ls[0][2:].strip()); g = {}
        for ln in ls[1:]:
            f = ln[2:].split("|||"); s, e = map(int, f[0].split()); a = int(f[-1]); g.setdefault(a, set())
            if f[1] not in ("noop", "UNK"): g[a].add((s, e, "" if f[2] == "-NONE-" else f[2]))
        gold.append(g or {0: set()})
    return src, gold
def bea_counts(hyps):
    src, gold = bea_gold(); a = ann(); out = []
    for s, h, g in zip(src, hyps, gold):
        o = a.parse(s); he = {(e.o_start, e.o_end, e.c_str) for e in a.annotate(o, a.parse(h)) if e.type != "noop"}
        out.append(max(((len(he & ge), len(he - ge), len(ge - he)) for ge in g.values()), key=lambda c: (f05(*c), c[0], -c[1] - c[2])))
    return out

# ---------------------------------------------------------------- in-domain C4 subset (single reference)
C4 = list(csv.DictReader(open(ROOT / "splits" / "c4_test_subset_5k.tsv", encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))
def c4_counts(hyps):
    a = ann(); out = []
    for r, p in zip(C4, hyps):
        s, t = r["src"], r["tgt"]; o = a.parse(s)
        g = set() if t == s else {(e.o_start, e.o_end, e.c_str) for e in a.annotate(o, a.parse(t)) if e.type != "noop"}
        h = set() if p == s else {(e.o_start, e.o_end, e.c_str) for e in a.annotate(o, a.parse(p)) if e.type != "noop"}
        out.append((len(g & h), len(h - g), len(g - h)))
    return out

# ---------------------------------------------------------------- JFLEG (official GLEU)
JF = DATA / "jfleg"
def jfleg_gleu(hyps, tag):
    sys.path.insert(0, str(JF)); from gleu import GLEU
    hp = CACHE / f"_jf_{tag}.txt"; hp.write_text("\n".join(hyps) + "\n", encoding="utf-8")
    g = GLEU(4); g.load_sources(str(JF / "test.src")); g.load_references([str(JF / f"test.ref{k}") for k in range(4)])
    mean, std, ci = list(g.run_iterations(num_iterations=500, source=str(JF / "test.src"), hypothesis=str(hp), per_sent=False))[0]
    return {"GLEU": round(float(mean), 4), "std": round(float(std), 4)}

# ---------------------------------------------------------------- score every released output
manifest = json.load(open(ROOT / "outputs" / "manifest.json"))
have = {"conll": CONLL_GOLD.exists(), "bea_dev": BEA_GOLD.exists(), "c4": True, "jfleg": (JF / "gleu.py").exists()}
print("benchmarks available:", {k: v for k, v in have.items()})
CNT, scores = {}, {}
for system, benches in manifest.items():
    if args.only and system not in args.only: continue
    for b in benches:
        if b == "bea_test" or not have.get(b): continue
        h = read_hyps(system, b); key = f"{b}|{system}|{hashlib.sha1(chr(10).join(h).encode()).hexdigest()}"
        if b == "conll": c = cached(key, lambda: conll_counts(h, system.replace("/", "_")))
        elif b == "bea_dev": c = cached(key, lambda: bea_counts(h))
        elif b == "c4": c = cached(key, lambda: c4_counts(h))
        else:
            scores.setdefault(system, {})[b] = jfleg_gleu(h, system.replace("/", "_")); continue
        CNT[(system, b)] = c; scores.setdefault(system, {})[b] = prf(c)
        if b == "bea_dev":     # CEFR levels: ABCN.dev is A (1,037), B (1,290), C (1,069), N (988) sentences, in that order
            o = 0
            for lev, n in (("A", 1037), ("B", 1290), ("C", 1069), ("N", 988)):
                scores[system][f"bea_dev_{lev}"] = prf(c[o:o + n]); o += n
    print(system, {b: v.get("F05", v.get("GLEU")) for b, v in scores.get(system, {}).items() if "_" not in b or b == "bea_dev"}, flush=True)

def boot(name, ca, cb, B=2000):
    rng = np.random.default_rng(int(hashlib.sha1(name.encode()).hexdigest()[:8], 16))
    n = len(ca); idx = rng.integers(0, n, size=(B, n))
    d = np.array([f05(*ca[ix].sum(0)) - f05(*cb[ix].sum(0)) for ix in idx])
    return {"delta": round(f05(*ca.sum(0)) - f05(*cb.sum(0)), 4), "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
            "p_two_sided": round(float(2 * min((d <= 0).mean(), (d >= 0).mean())), 4)}
comparisons = {}
for row in json.load(open(ROOT / "analysis" / "table_spec.json")):
    for b in row["benchmarks"]:
        a_, b_ = (row["a"], row["b"]) if isinstance(row["a"], str) else (row["a"][b], row["b"][b])
        if (a_, b) in CNT and (b_, b) in CNT:
            comparisons[f"{row['table']}|{row['name']}|{b}"] = {"a": a_, "b": b_, **boot(f"{row['name']}|{b}", CNT[(a_, b)], CNT[(b_, b)])}
result = {"scores": scores, "comparisons": comparisons}
if args.check:
    ref = json.load(open(OUTF)); bad = []
    for s, d in ref["scores"].items():
        for b, v in d.items():
            got = scores.get(s, {}).get(b)
            if got is None: continue
            k = "F05" if "F05" in v else "GLEU"
            if abs(got[k] - v[k]) > 6e-4: bad.append((s, b, v[k], got[k]))
    for k, v in ref["comparisons"].items():
        if k in comparisons and abs(comparisons[k]["delta"] - v["delta"]) > 6e-4: bad.append((k, v["delta"], comparisons[k]["delta"]))
    print("PASS" if not bad else f"FAIL: {bad}")
else:
    OUTF.parent.mkdir(parents=True, exist_ok=True); json.dump(result, open(OUTF, "w"), indent=1)
    print("wrote", OUTF)
