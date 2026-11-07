"""Re-score every released system output with the official scorers and recompute every paired comparison of the paper.

    python analysis/get_data.py            # once: CoNLL-2014 test, W&I+LOCNESS v2.1, JFLEG test + official GLEU
    python analysis/verify.py              # writes results/verified/verified_scores.json and results/per_sentence_counts/
    python analysis/verify.py --check      # compares with the committed verified_scores.json and prints PASS / FAIL

Scorers (Section 3.3 of the paper):
  conll   CoNLL-2014 test, official MaxMatch (M2) scorer (third_party/m2scorer_py3, a Python-3 port of the original);
          per-sentence counts are recovered from `m2scorer -v` and asserted to reproduce the official totals
  bea_dev BEA-2019 development, ERRANT edits per sentence against the gold M2 (best annotator per sentence, as
          errant_compare); totals equal errant_compare's
  c4      in-domain 5,000-pair C4-200M test subset (splits/c4_test_subset_5k.tsv), single-reference ERRANT
  jfleg   JFLEG test, official GLEU (keisks/jfleg eval/gleu.py, 4 references, 500 iterations); no paired test
Comparisons (analysis/table_spec.json): paired bootstrap over sentences, 2,000 resamples, generator seeded by the
comparison name and benchmark. A comparison with several seed pairs reports the mean paired F0.5 difference; every
resample uses the same sentences for all pairs. Reported: difference, 95% and 90% percentile intervals, two-sided p,
Holm-adjusted p within each table, and the TOST p for an equivalence margin of +/-0.01 F0.5.
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
MARGIN, B_ = 0.01, 2000

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
SUBSETS = {"c4_uncontaminated": ROOT / "splits" / "c4_test_subset_uncontaminated.txt"}     # row numbers into the 5k subset

# ---------------------------------------------------------------- JFLEG (official GLEU)
JF = DATA / "jfleg"
def jfleg_gleu(hyps, tag):
    p = CACHE / ("jf_" + hashlib.sha1("\n".join(hyps).encode()).hexdigest() + ".json")
    if p.exists(): return json.load(open(p))
    sys.path.insert(0, str(JF)); from gleu import GLEU
    hp = CACHE / f"_jf_{tag}.txt"; hp.write_text("\n".join(hyps) + "\n", encoding="utf-8")
    g = GLEU(4); g.load_sources(str(JF / "test.src")); g.load_references([str(JF / f"test.ref{k}") for k in range(4)])
    mean, std, ci = list(g.run_iterations(num_iterations=500, source=str(JF / "test.src"), hypothesis=str(hp), per_sent=False))[0]
    v = {"GLEU": round(float(mean), 4), "std": round(float(std), 4)}; json.dump(v, open(p, "w")); return v

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
        CNT[(system, b)] = np.asarray(c); scores.setdefault(system, {})[b] = prf(c)
        if b == "bea_dev":     # CEFR levels: ABCN.dev is A (1,037), B (1,290), C (1,069), N (988) sentences, in that order
            o = 0
            for lev, n in (("A", 1037), ("B", 1290), ("C", 1069), ("N", 988)):
                scores[system][f"bea_dev_{lev}"] = prf(c[o:o + n]); o += n
    print(system, {b: v.get("F05", v.get("GLEU")) for b, v in scores.get(system, {}).items() if "_" not in b or b == "bea_dev"}, flush=True)

# ---------------------------------------------------------------- paired comparisons
def weights(name, n):
    rng = np.random.default_rng(int(hashlib.sha1(name.encode()).hexdigest()[:8], 16))
    idx = rng.integers(0, n, size=(B_, n))
    return np.stack([np.bincount(r, minlength=n) for r in idx]).astype(np.int32)
def f05v(t):
    tp, fp, fn = t[:, 0].astype(float), t[:, 1].astype(float), t[:, 2].astype(float)
    p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 1.0); r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 1.0)
    return np.where(p + r > 0, 1.25 * p * r / np.maximum(0.25 * p + r, 1e-12), 0.0)
def compare(name, pairs, b, mask=None):
    ca = [CNT[(a, b)] for a, _ in pairs]; cb = [CNT[(x, b)] for _, x in pairs]
    if mask is not None: ca, cb = [c[mask] for c in ca], [c[mask] for c in cb]
    W = weights(f"{name}|{b}", len(ca[0]))
    d = np.mean([f05v(W @ x) - f05v(W @ y) for x, y in zip(ca, cb)], axis=0)
    obs = float(np.mean([f05(*x.sum(0)) - f05(*y.sum(0)) for x, y in zip(ca, cb)]))
    return {"pairs": [list(p) for p in pairs], "n": int(len(ca[0])), "delta": round(obs, 6),
            "per_pair_delta": [round(f05(*x.sum(0)) - f05(*y.sum(0)), 6) for x, y in zip(ca, cb)],
            "ci95": [round(float(np.percentile(d, 2.5)), 6), round(float(np.percentile(d, 97.5)), 6)],
            "ci90": [round(float(np.percentile(d, 5)), 6), round(float(np.percentile(d, 95)), 6)],
            "p_two_sided": round(float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))), 4),
            "tost_p_margin_0.01": round(float(max((d <= -MARGIN).mean(), (d >= MARGIN).mean())), 4)}
comparisons = {}
for row in json.load(open(ROOT / "analysis" / "table_spec.json")):
    for b in row["benchmarks"]:
        if "pairs" in row: pairs = row["pairs"]
        else: pairs = [[row["a"], row["b"]]] if isinstance(row["a"], str) else [[row["a"][b], row["b"][b]]]
        if not all((a, b) in CNT and (x, b) in CNT for a, x in pairs): continue
        mask = None
        if row.get("subset"):
            rows = [int(x) for x in open(SUBSETS[row["subset"]]).read().split()]
            mask = np.zeros(len(CNT[(pairs[0][0], b)]), bool); mask[rows] = True
        comparisons[f"{row['table']}|{row['name']}|{b}"] = {"table": row["table"], **({"subset": row["subset"]} if mask is not None else {}),
                                                           **compare(row["name"], pairs, b, mask)}
for tab in sorted({v["table"] for v in comparisons.values()}):              # Holm within each table
    ks = sorted((k for k, v in comparisons.items() if v["table"] == tab), key=lambda k: comparisons[k]["p_two_sided"]); run = 0.0
    for rank, k in enumerate(ks):
        run = max(run, min(1.0, (len(ks) - rank) * comparisons[k]["p_two_sided"])); comparisons[k]["p_holm"] = round(run, 4)
result = {"scores": scores, "comparisons": comparisons}
if args.check:
    ref = json.load(open(OUTF)); bad = []
    for s, d_ in ref["scores"].items():
        for b, v in d_.items():
            got = scores.get(s, {}).get(b)
            if got is None: continue
            k = "F05" if "F05" in v else "GLEU"
            if abs(got[k] - v[k]) > 6e-4: bad.append((s, b, v[k], got[k]))
    for k, v in ref["comparisons"].items():
        if k in comparisons and (abs(comparisons[k]["delta"] - v["delta"]) > 6e-4 or
                                 any(abs(x - y) > 1e-6 for x, y in zip(comparisons[k]["ci95"], v["ci95"]))): bad.append((k, v["delta"], comparisons[k]["delta"]))
    print("PASS" if not bad else f"FAIL: {bad}")
else:
    OUTF.parent.mkdir(parents=True, exist_ok=True); json.dump(result, open(OUTF, "w"), indent=1)
    for (s, b), c in CNT.items():                     # per-sentence edit counts behind every score (benchmark order)
        p = ROOT / "results" / "per_sentence_counts" / b / f"{s}.tsv"; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("tp\tfp\tfn\n" + "".join(f"{int(x)}\t{int(y)}\t{int(z)}\n" for x, y, z in c), encoding="utf-8")
    print("wrote", OUTF, "and per-sentence counts for", len(CNT), "system-benchmark pairs")
