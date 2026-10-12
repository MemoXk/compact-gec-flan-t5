"""Code cells shared by the two round-2 notebooks (kept as strings; assembled by build_r2.py)."""

INSTALL = r'''# ---------------------------------------------------------------- installs
import subprocess, sys, importlib
def pip(*a):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *a], capture_output=True, text=True)
    if r.returncode: print("pip warning:", " ".join(a), r.stderr[-300:])
pip("errant==3.0.2", "sentencepiece", "pyarrow")
import spacy
try:
    spacy.load("en_core_web_sm")
except OSError:
    v = ".".join(spacy.__version__.split(".")[:2]) + ".0"
    pip(f"https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-{v}/en_core_web_sm-{v}-py3-none-any.whl")
spacy.load("en_core_web_sm")
print("spaCy", spacy.__version__, "| en_core_web_sm OK")
'''

IMPORTS = r'''# ---------------------------------------------------------------- imports + run bookkeeping
import os, re, io, csv, json, gc, math, time, random, shutil, tarfile, zipfile, urllib.request, contextlib
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments)
os.environ["WANDB_DISABLED"] = "true"
OUT = Path("/kaggle/working"); EV = OUT / "eval_data"; EV.mkdir(exist_ok=True)
T0 = time.time()
def hours_left(): return SESSION_BUDGET_H - (time.time() - T0) / 3600
def budget_ok(need_h, what=""):
    ok = hours_left() > need_h
    if not ok: print(f"!! budget guard: {hours_left():.2f} h left < {need_h} h needed -> skipping {what}")
    return ok
RES_PATH = OUT / RESULTS_NAME
RESULTS = json.loads(RES_PATH.read_text()) if RES_PATH.exists() else {}
def save_results(): RES_PATH.write_text(json.dumps(RESULTS, indent=1, default=str))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={DEV} | GPUs={torch.cuda.device_count()} | budget={SESSION_BUDGET_H} h | QUICK_TEST={QUICK_TEST}")
'''

DATA = r'''# ---------------------------------------------------------------- official evaluation data + scorers
CONLL_URL = "https://www.comp.nus.edu.sg/~nlp/conll14st/conll14st-test-data.tar.gz"
M2_URL    = "https://www.comp.nus.edu.sg/~nlp/sw/m2scorer.tar.gz"
BEA_URL   = "https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz"
JF_BASE   = "https://raw.githubusercontent.com/keisks/jfleg/master/"

def fetch(url, dest):
    if not Path(dest).exists(): urllib.request.urlretrieve(url, dest)
    return Path(dest)

def untar(tgz, into):
    with tarfile.open(tgz) as t: t.extractall(into)

# CoNLL-2014 test (official, 1,312 sentences, 2 annotators)
if not (EV / "conll14st-test-data").exists(): untar(fetch(CONLL_URL, EV / "conll.tgz"), EV)
CONLL_M2 = EV / "conll14st-test-data" / "noalt" / "official-2014.combined.m2"
# official MaxMatch scorer, ported to Python 3 OFFLINE and embedded: Kaggle now runs Python 3.13, which removed
# lib2to3, so converting it at run time is no longer possible. (Port verified: reproduces 0.4852 / 0.3096 exactly.)
import base64 as _b64, gzip as _gz
with tarfile.open(fileobj=io.BytesIO(_gz.decompress(_b64.b64decode("__M2_PORTED_B64__")))) as _t:
    _t.extractall(EV)                                   # always overwrite with the ported copy
M2_SCRIPT = EV / "m2scorer" / "scripts" / "m2scorer.py"
# W&I+LOCNESS v2.1 (BEA-2019): dev gold M2, blind test input, train M2
if not (EV / "wi+locness").exists(): untar(fetch(BEA_URL, EV / "wi.tgz"), EV)
BEA_DEV_M2 = EV / "wi+locness" / "m2" / "ABCN.dev.gold.bea19.m2"
BEA_TEST_ORIG = EV / "wi+locness" / "test" / "ABCN.test.bea19.orig"
# JFLEG test (747 sentences, 4 references) + the OFFICIAL GLEU script (Napoles et al., 2016 version)
JF = EV / "jfleg"; JF.mkdir(exist_ok=True)
for n in ["test/test.src"] + [f"test/test.ref{k}" for k in range(4)] + ["eval/gleu.py"]:
    fetch(JF_BASE + n, JF / Path(n).name)
sys.path.insert(0, str(JF))
from gleu import GLEU

def m2_sources(path): return [l[2:].strip() for l in open(path, encoding="utf-8") if l.startswith("S ")]
def read_m2(path):
    out = []
    for block in Path(path).read_text(encoding="utf-8").strip().split("\n\n"):
        lines = block.split("\n"); src = lines[0][2:].strip(); ed = {}
        for ln in lines[1:]:
            if not ln.startswith("A "): continue
            f = ln[2:].split("|||"); s, e = map(int, f[0].split()); ann = int(f[-1])
            ed.setdefault(ann, [])
            if f[1] != "noop": ed[ann].append((s, e, f[2]))
        out.append((src, ed))
    return out
def apply_m2_edits(src, edits):
    toks = src.split(); off = 0
    for s, e, cor in sorted(edits, key=lambda x: (x[0], x[1])):
        c = cor.split() if cor and cor != "-NONE-" else []
        toks[s + off:e + off] = c; off += len(c) - (e - s)
    return " ".join(toks)

CONLL_SRC = m2_sources(CONLL_M2)
if QUICK_TEST:                       # smoke test: first 200 CoNLL sentences with the matching gold
    _b = Path(CONLL_M2).read_text(encoding="utf-8").strip().split("\n\n")[:200]
    CONLL_M2 = EV / "conll_quick.m2"; CONLL_M2.write_text("\n\n".join(_b) + "\n\n", encoding="utf-8")
    CONLL_SRC = m2_sources(CONLL_M2)
BEA_DEV = read_m2(BEA_DEV_M2)
BEA_SRC = [s for s, _ in BEA_DEV]
BEA_TGT = [apply_m2_edits(s, ed.get(0, [])) for s, ed in BEA_DEV]      # first annotator (as in the paper)
BEA_TEST_SRC = [l.rstrip("\n") for l in open(BEA_TEST_ORIG, encoding="utf-8")]
JF_SRC = [l.rstrip("\n") for l in open(JF / "test.src", encoding="utf-8")]
JF_REFS = [JF / f"test.ref{k}" for k in range(4)]
if QUICK_TEST:
    BEA_N, JF_N = 200, 120
else:
    BEA_N, JF_N = len(BEA_SRC), len(JF_SRC)
print(f"CoNLL-2014 {len(CONLL_SRC)} | BEA dev {len(BEA_SRC)} | BEA test {len(BEA_TEST_SRC)} | JFLEG {len(JF_SRC)}")
'''

SCORE = r'''# ---------------------------------------------------------------- retokenizers + official scorers
# Both benchmarks ship PRE-TOKENIZED sources and both scorers compare whitespace tokens, so every
# system output is retokenized with the tokenizer that best reproduces the benchmark's own convention
# (round-trip test on the sources: regex 99.3% on CoNLL-2014, spaCy 98.7% on BEA-2019).
import spacy
_NLP = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer", "tagger", "attribute_ruler"])
def spacy_tok(s): return " ".join(t.text for t in _NLP.make_doc(str(s)) if not t.is_space)
_CLITIC = re.compile(r"(?i)(\w)(n't|'s|'re|'ve|'ll|'d|'m)\b")
_PA = re.compile(r"(\S)([.,!?;:%)\]\"](?:\s|$))"); _PB = re.compile(r"(^|\s)([(\[\"])(\S)")
def regex_tok(s):
    s = _CLITIC.sub(r"\1 \2", str(s))
    for _ in range(3): s = _PA.sub(r"\1 \2", s)
    return re.sub(r"\s+", " ", _PB.sub(r"\1\2 \3", s)).strip()
RETOK = {"conll": regex_tok, "jfleg": regex_tok, "bea": spacy_tok}
def norm_ws(s): return re.sub(r"\s+", " ", str(s)).strip()

def official_m2(hyps, tag, gold=None):
    gold = gold or CONLL_M2
    n = sum(1 for l in open(gold, encoding="utf-8") if l.startswith("S "))
    assert len(hyps) == n, f"{tag}: {len(hyps)} hyps vs {n} gold"
    hp = OUT / f"_m2_{tag}.txt"; hp.write_text("\n".join(hyps) + "\n", encoding="utf-8")
    r = subprocess.run([sys.executable, str(M2_SCRIPT), str(hp), str(gold)], capture_output=True, text=True,
                       timeout=7200, env={**os.environ, "PYTHONUTF8": "1"})
    g = dict(re.findall(r"(Precision|Recall|F_0\.5)\s*:\s*([0-9.]+)", r.stdout))
    if not g: print("m2scorer failed:", r.stderr[-400:]); return {"error": r.stderr[-400:]}
    return {"P": float(g["Precision"]), "R": float(g["Recall"]), "F05": float(g["F_0.5"])}

import errant
from errant.commands import compare_m2
_ANN = {"a": None}
def ann():
    if _ANN["a"] is None: _ANN["a"] = errant.load("en")
    return _ANN["a"]
def errant_official(srcs, hyps, gold_m2, tag):
    """Hypothesis M2 via ERRANT (whitespace tokens, as errant_parallel), then the official errant_compare."""
    a = ann(); hp = OUT / f"_errant_{tag}.m2"
    with open(hp, "w", encoding="utf-8") as f:
        for s, h in zip(srcs, hyps):
            o = a.parse(s); ed = a.annotate(o, a.parse(h))
            f.write("S " + s + "\n")
            if not ed: f.write("A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\n")
            for e in ed: f.write(e.to_m2() + "\n")
            f.write("\n")
    gold_use = gold_m2
    if len(srcs) < len(m2_sources(gold_m2)):                       # QUICK_TEST subsets
        blocks = Path(gold_m2).read_text(encoding="utf-8").strip().split("\n\n")[:len(srcs)]
        gold_use = OUT / f"_gold_{tag}.m2"; gold_use.write_text("\n\n".join(blocks) + "\n\n", encoding="utf-8")
    buf = io.StringIO(); old = sys.argv
    sys.argv = ["compare_m2", "-hyp", str(hp), "-ref", str(gold_use)]
    try:
        with contextlib.redirect_stdout(buf): compare_m2.main()
    finally: sys.argv = old
    found = re.findall(r"^(\d+)\s+(\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$", buf.getvalue(), re.M)
    if not found: print("errant_compare output not parsed"); return {"error": buf.getvalue()[-400:]}
    tp, fp, fn, p, r, f = found[-1]
    return {"TP": int(tp), "FP": int(fp), "FN": int(fn), "P": float(p), "R": float(r), "F05": float(f)}

def jfleg_gleu(hyps, n=None):
    """OFFICIAL JFLEG GLEU (keisks/jfleg eval/gleu.py, 500 random-reference iterations), source-aware."""
    n = n or len(hyps)
    sp, hp = OUT / "_jf_src.txt", OUT / "_jf_hyp.txt"
    sp.write_text("\n".join(JF_SRC[:n]) + "\n", encoding="utf-8"); hp.write_text("\n".join(hyps[:n]) + "\n", encoding="utf-8")
    refs = []
    for k, rp in enumerate(JF_REFS):
        p = OUT / f"_jf_ref{k}.txt"; p.write_text("\n".join(open(rp, encoding="utf-8").read().split("\n")[:n]) + "\n", encoding="utf-8"); refs.append(str(p))
    g = GLEU(4); g.load_sources(str(sp)); g.load_references(refs)
    mean, std, ci = list(g.run_iterations(num_iterations=500, source=str(sp), hypothesis=str(hp), per_sent=False))[0]
    return {"GLEU": float(mean), "std": float(std), "ci95": ci}

def lev(a, b):
    a, b = str(a).split(), str(b).split(); prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1): cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]
def outcome_table(srcs, tgts, preds):
    c = Counter(); n = len(srcs)
    for s, t, p in zip(srcs, tgts, preds):
        if p.strip() == t.strip(): c["perfect"] += 1
        elif p.strip() == s.strip(): c["unchanged"] += 1
        else:
            ds, dp = lev(s, t), lev(p, t)
            c["improved" if dp < ds else ("lateral" if dp == ds else "harm")] += 1
    o = {k: round(100 * c[k] / n, 2) for k in ("perfect", "improved", "unchanged", "lateral", "harm")}
    o["edited_pct"] = round(100 * sum(a.strip() != b.strip() for a, b in zip(srcs, preds)) / n, 2)
    return o
def single_ref_errant(srcs, tgts, preds):
    """Edit-level P/R/F0.5 against ONE reference (the in-domain C4 test), plus per-sentence counts."""
    a = ann(); tp = fp = fn = 0; per = []
    for s, t, p in zip(srcs, tgts, preds):
        o = a.parse(s)
        gold = set() if t == s else {(e.o_start, e.o_end, e.c_str) for e in a.annotate(o, a.parse(t)) if e.type != "noop"}
        hyp = set() if p == s else {(e.o_start, e.o_end, e.c_str) for e in a.annotate(o, a.parse(p)) if e.type != "noop"}
        x = (len(gold & hyp), len(hyp - gold), len(gold - hyp)); per.append(x)
        tp += x[0]; fp += x[1]; fn += x[2]
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return {"P": P, "R": R, "F05": (1.25 * P * R / (0.25 * P + R)) if (P + R) else 0.0, "TP": tp, "FP": fp, "FN": fn}, per
print("scorers ready")
'''

T5DECODE = r'''# ---------------------------------------------------------------- FLAN-T5 decoding, exactly as in the paper
PREFIX = "Correct the grammar in this sentence: "
MAX_INPUT_LEN = MAX_TARGET_LEN = 128
GEN_KWARGS = dict(max_length=128, num_beams=5, length_penalty=0.7, no_repeat_ngram_size=3,
                  early_stopping=True, min_length=1)
REFINE_MAX_EDITS = 3
GEN_BATCH = 64
def token_edit(a, b):
    sm = SequenceMatcher(None, str(a).split(), str(b).split())
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")
def load_t5(path):
    tok = AutoTokenizer.from_pretrained(str(path)); m = AutoModelForSeq2SeqLM.from_pretrained(str(path)).to(DEV).eval()
    return tok, m
@torch.no_grad()
def t5_gen(tok, m, texts, gen=None, bs=GEN_BATCH, desc=""):
    kw = dict(GEN_KWARGS); kw.update(gen or {}); out = []
    for i in range(0, len(texts), bs):
        enc = tok([PREFIX + str(s) for s in texts[i:i + bs]], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_INPUT_LEN).to(m.device)
        out += tok.batch_decode(m.generate(**enc, **kw), skip_special_tokens=True)
        if desc and (i // bs) % 25 == 0: print(f"   {desc} {i}/{len(texts)}", flush=True)
    return out
def refine(tok, m, srcs, pass1, gen=None):
    need = [i for i, p in enumerate(pass1) if p.strip() != str(srcs[i]).strip()]
    out = list(pass1)
    if need:
        r = t5_gen(tok, m, [pass1[k] for k in need], gen)
        for k, i in enumerate(need):
            if token_edit(pass1[i], r[k]) <= REFINE_MAX_EDITS: out[i] = r[k]
    return out
@torch.no_grad()
def mean_nll(tok, m, ins, outs, bs=GEN_BATCH):
    res = []
    for i in range(0, len(ins), bs):
        enc = tok([PREFIX + str(s) for s in ins[i:i + bs]], return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_INPUT_LEN).to(m.device)
        lab = tok(text_target=[str(t) for t in outs[i:i + bs]], return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_TARGET_LEN).to(m.device)
        labels = lab["input_ids"].masked_fill(lab["attention_mask"] == 0, -100)
        lp = torch.log_softmax(m(**enc, labels=labels).logits.float(), dim=-1)
        nll = -lp.gather(2, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1); mask = (labels != -100).float()
        res += ((nll * mask).sum(1) / mask.sum(1).clamp(min=1)).tolist()
    return res
def margins(tok, m, srcs, preds):
    idx = [i for i, (s, p) in enumerate(zip(srcs, preds)) if str(p).strip() != str(s).strip()]
    mg = [None] * len(srcs)
    if idx:
        a = mean_nll(tok, m, [srcs[i] for i in idx], [preds[i] for i in idx])
        b = mean_nll(tok, m, [srcs[i] for i in idx], [srcs[i] for i in idx])
        for k, i in enumerate(idx): mg[i] = b[k] - a[k]
    return mg
def revert(srcs, preds, mg, tau):
    if tau is None: return list(preds)
    return [s if (x is not None and x < tau) else p for s, p, x in zip(srcs, preds, mg)]
def decode_variants(tok, m, srcs, lps=(0.7,), refines=(True,), taus=(0.0,), desc=""):
    """All (length-penalty x refinement x threshold) variants from ONE beam decode per length penalty."""
    out = {}
    for lp in lps:
        p1 = t5_gen(tok, m, srcs, {"length_penalty": lp}, desc=f"{desc} lp={lp}")
        for rf in refines:
            p = refine(tok, m, srcs, p1, {"length_penalty": lp}) if rf else list(p1)
            mg = margins(tok, m, srcs, p) if any(t is not None for t in taus) else None
            for tau in taus:
                out[(lp, rf, tau)] = [norm_ws(x) for x in revert(srcs, p, mg, tau)]
    return out
def vkey(lp, rf, tau): return f"lp{lp}_ref{int(rf)}_tau{'none' if tau is None else tau}"
'''

EXPORT = r'''# ---------------------------------------------------------------- export everything except model weights
zp = OUT / EXPORT_NAME
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()):
        if p.is_file() and not p.name.startswith("_") and p.suffix in (".json", ".csv", ".txt", ".zip", ".png", ".tsv", ".gz") \
                and p.name != EXPORT_NAME:
            z.write(p, p.name)
print(f"wrote {zp} ({zp.stat().st_size/1e6:.1f} MB) -- download this file and send it back")
print(json.dumps({k: v for k, v in RESULTS.items() if not isinstance(v, (list,))}, indent=1, default=str)[:4000])
'''
