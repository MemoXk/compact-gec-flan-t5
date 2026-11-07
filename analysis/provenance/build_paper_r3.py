"""Round-3 paper: rebuild inass_paper.docx as one corrected study (Notebook E) from the verified analysis files.

Input : _bk_round3_start.docx  (the paper as submitted in round 3; never modified) -- supplies the IJIES masthead,
        title block, styles, section layout, back matter and reference formatting
Output: inass_paper.docx
Every number is read from round2_results/*.json (E_analysis.json is the locally re-scored, Kaggle-reproducing analysis).
Text markup: **bold**, *italic*, _{sub}, ^{sup}, [[highlight]] (marks text still pending), "\n" (line break in a cell).
Cross-references: <<T:key>> table, <<F:key>> figure, <<E:key>> equation, <<S:key>> section; citations [@a; @b]
(numbered in order of first appearance, IJIES style)."""
import copy, csv, io, json, math, os, re, statistics as st, tempfile
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import docx, pypandoc
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from PIL import Image

P_ = Path(r"<PROJECT>"); R_ = P_ / "round2_results"
SRC, OUT = P_ / "_bk_round3_start.docx", P_ / "inass_paper.docx"
J = lambda f: json.load(open(R_ / f, encoding="utf-8"))
A = J("E_analysis.json"); S = A["scores"]
RE = {k: J(f"E/session{k}/r2E_results.json") for k in (1, 2, 3)}
T = J("E/bea19_test_codabench_E.json")
OV = J("overlap_audit.json"); DOV = J("D_holdout_essay_overlap.json")
RC, RN = J("C_final/r2C_results.json"), J("C_nogate/r2C_results.json")
RA, RAN = J("A_clean/r2A_results.json"), J("A_noisy/r2A_results.json")
MC, MN = J("A_clean/c4_meta_clean.json")["filter_stats"], J("A_noisy/c4_meta_noisy.json")["filter_stats"]
RB = J("B/r2B_results.json")
HE_PATH = P_ / "human_eval_round3/human_eval_results.json"
HE = json.load(open(HE_PATH)) if HE_PATH.exists() else None
HE_MAN = json.load(open(P_ / "human_eval_round3/sampling_manifest.json"))
SEEDS = (42, 1337, 2024); FIN = [f"E/final_e{s}" for s in SEEDS]; NOG = [f"E/nogate_e{s}" for s in SEEDS]
WI = RE[1]["wi_essay_split"]

# ------------------------------------------------------------------------------------------------ formatting
MINUS = "\u2212"
def f3(x):
    v = float(Decimal(str(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)); return f"{v:.3f}".replace("-", MINUS) if v != 0 else "0.000"
def _r(x, nd=3): return float(Decimal(str(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))
def sg(x, nd=3):
    v = _r(x, nd); return ("+" if v > 0 else MINUS if v < 0 else "") + f"{abs(v):.{nd}f}"     # 0.000 carries no sign
def short(x):
    v = _r(x); s = f"{abs(v):.3f}"; return (MINUS if v < 0 else "") + (s[1:] if s.startswith("0") else s)
def ci(r, k="ci95"): return f"{f3(r[k][0])} to {f3(r[k][1])}"
def pv(p): return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
def pct(x, nd=3): return str((Decimal(str(x)) / 100).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))
def F(sys, b): v = S[sys][b]; return v["F05"] if "F05" in v else v["GLEU"]
def msd(names, b): v = [F(n, b) for n in names]; return st.mean(v), st.stdev(v)
def pm(names, b): m, s = msd(names, b); return f"{m:.3f} \u00b1 {s:.3f}"
def rows_named(tab, name): return {r["benchmark"]: r for r in A["tables"][tab] if r["name"] == name}
def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i])); adj[i] = run
    return adj
MARGIN = A["notes"]["margin"]; assert MARGIN == 0.01
def negligible(r): return -MARGIN < r["ci90"][0] and r["ci90"][1] < MARGIN

# ------------------------------------------------------------------------------------------------ key numbers
TEST = {k: T["submissions"][f"final_e{s}"] for k, s in zip(("a", "b", "c"), SEEDS)}
TEST_F = [TEST[k]["F05"] for k in "abc"]; TEST_P = [TEST[k]["P"] for k in "abc"]; TEST_R = [TEST[k]["R"] for k in "abc"]
TESTM, TESTSD = st.mean(TEST_F), st.stdev(TEST_F)
assert abs(TESTM - T["summary_F05"]["mean"]) < 0.006 and abs(TESTSD - T["summary_F05"]["sd"]) < 0.006
TEST_STR = f"{pct(round(TESTM, 2))} \u00b1 {pct(round(TESTSD, 2))}"
IDS = [T["submissions"][f"final_e{s}"]["codabench_id"] for s in SEEDS]
PRIM = {b: msd(FIN, b) for b in ("c4", "bea_dev", "conll", "jfleg")}
SD_MAX = max(max(s for _, s in PRIM.values()), TESTSD / 100)
JF_DN = RA["jfleg_do_nothing"]["GLEU"]
CO = RE[1]["baseline|coedit-large"]; COEDIT_REV = CO["revision"]
QWEN = "baselines/qwen2.5-1.5b_fourshot"; G5S, G5B, CED = "baselines/gec-t5-small-clang8", "baselines/gec-t5-base-clang8", "baselines/coedit-large"
assert abs(F(CED, "conll") - CO["conll"]["F05"]) < 6e-4
HUMAN_RATIO = 2.37e6 / WI["train_pairs"]
def edited_share(system, bench):
    src = [r["src"] for r in csv.DictReader(open(R_ / "E/session1/pred_conll_final_e42.csv", encoding="utf-8"))]
    hyp = (P_ / f"release/compact-gec-flan-t5/outputs/{bench}/{system}.txt").read_text(encoding="utf-8").split("\n")[:-1]
    n = lambda s: re.sub(r"\s+", " ", s).strip(); return 100 * sum(n(a) != n(b) for a, b in zip(src, hyp)) / len(src)
csv.field_size_limit(10 ** 9)
def _ed(seed):
    rows = list(csv.DictReader(open(R_ / f"E/session1/pred_conll_final_e{seed}.csv", encoding="utf-8")))
    return 100 * sum(re.sub(r"\s+", " ", r["ref1_tau0.0"]).strip() != re.sub(r"\s+", " ", r["src"]).strip() for r in rows) / len(rows)
ED_OURS = st.mean(_ed(s) for s in SEEDS)                      # mean over the three primary seeds
_ess = list(csv.DictReader(open(R_ / "E/session1/wi_sentence_essay.tsv", encoding="utf-8"), delimiter="\t"))
assert {b: len({r["essay_id"] for r in _ess if r["band"] == b}) for b in "ABC"} == {"A": 1300, "B": 1000, "C": 700}
ED_QWEN = edited_share("baselines/qwen2.5-1.5b_fourshot", "conll"); ED_COEDIT = CO["conll"]["edited_pct"]
T6 = {r["name"] + "|" + r["benchmark"]: r for r in A["tables"]["T6"]}
# training diagnostics
S1 = RC["train|final_s1"]; S2 = RE[1]["train|final_e42"]
RATIOS = [v["train_eval_loss_ratio"] for k in (1, 2, 3) for kk, v in RE[k].items() if kk.startswith("train|")] + \
         [RC["train|final_s1"]["train_eval_loss_ratio"], RN["train|nogate_s1"]["train_eval_loss_ratio"]] + \
         [d[f"train|{t}"]["train_loss"] / d[f"train|{t}"]["best_eval_loss"] for d, t in ((RA, "clean_s1337"), (RAN, "noisy_s42"), (RAN, "noisy_s1337"))]
S2_SECONDS = [v["train_seconds"] for k in (1, 2, 3) for kk, v in RE[k].items() if kk.startswith("train|") and v["stage"] == 2 and v["train_rows"] == WI["train_pairs"]]
LAT = {r["system"]: r for r in RE[1]["latency"]["rows"] if "error" not in r}
assert len(LAT) == 6, LAT.keys()
# blind test by CEFR level, three seeds
TL = {s: J(f"E/codabench_final_e{s}/bea19_test_scores_final_e{s}.json")[f"final_e{s}"] for s in SEEDS}
LEVELS = [("A (beginner)", "A", 1037), ("B (intermediate)", "B", 1290), ("C (advanced)", "C", 1069), ("N (native)", "N", 988)]
ET = T["per_level_and_error_type"]["error_types_F05_3seeds"]
STABLE = {k: v for k, v in ET.items() if v["sd"] <= 2.5 and not re.match(r"^[MRU]:", k)}     # main types, stable across seeds
# essay split, overlap audit
OVA, OVW = OV["cleaning_ablation"], OV["wi_train_vs_evaluation_sources"]
UNCONT = A["cleaning_c4_uncontaminated"]; assert UNCONT["pairs_kept"] + UNCONT["pairs_dropped_in_noisy_train"] == 5000
# cross-check: the released verifier (analysis/verify.py on the released outputs) reproduces every comparison used here
VER = json.load(open(P_ / "release/compact-gec-flan-t5/results/verified/verified_scores.json"))["comparisons"]
def _same(ver_key, rec):
    v = VER[ver_key]; return abs(v["delta"] - rec["delta"]) < 1e-6 and all(abs(x - y) < 1e-6 for x, y in zip(v["ci95"] + v["ci90"], rec["ci95"] + rec["ci90"]))
for tab, vtab in (("T4", "T4"), ("T6", "T3")):
    for r in A["tables"][tab]:
        vt = "T5" if r["name"].startswith("budget") else vtab
        assert _same(f"{vt}|{r['name']}|{r['benchmark']}", r), (tab, r["name"], r["benchmark"])
for r in A["cleaning_learner_benchmarks"]["comparisons"]: assert _same(f"T4|cleaning_pooled_learner|{r['benchmark']}", r)
assert _same("T4|cleaning_pooled|c4", next(x for x in UNCONT["comparisons"] if x["name"] == "cleaning_pooled"))
print("released verifier reproduces all", len(VER), "comparisons used in the paper")
assert UNCONT["pairs_dropped_in_noisy_train"] == OVA["c4_test_subset_5k_in_noisy_train"]

# ------------------------------------------------------------------------------------------------ docx plumbing
d = docx.Document(str(SRC)); BODY = d.element.body; ELS = list(BODY.iterchildren())
def text_of(el): return "".join(t.text or "" for t in el.iter(qn("w:t")))
assert text_of(ELS[12]).startswith("1. Introduction") and text_of(ELS[116]).startswith("Conflicts of interest")
assert text_of(ELS[122]) == "References" and ELS[-1].tag == qn("w:sectPr")
TPL = {k: copy.deepcopy(ELS[i]) for k, i in (("h1", 12), ("h2", 23), ("pf", 13), ("pb", 14), ("tcap", 25), ("eq", 65),
                                             ("wide2", 37), ("fig", 38), ("figcap", 39), ("ref", 123), ("empty", 74))}
assert TPL["fig"].find(".//" + qn("a:blip")) is not None and TPL["figcap"].find(".//" + qn("w:sectPr")) is not None
ONECOL_SECT = copy.deepcopy(TPL["figcap"].find(".//" + qn("w:sectPr")))
for el in ELS[11:116] + ELS[123:-1]: BODY.remove(el)          # old body and old reference list
ANCHOR, REF_HEAD = ELS[10], ELS[122]
TOK = re.compile(r"(\*\*.+?\*\*|\*[^*\s][^*]*?\*|_\{[^}]*\}|\^\{[^}]*\}|\[\[.+?\]\]|\n)")
def base_rpr(el):
    for r in el.iter(qn("w:r")):
        rp = r.find(qn("w:rPr")); rp = copy.deepcopy(rp) if rp is not None else OxmlElement("w:rPr")
        for t in ("w:b", "w:bCs", "w:i", "w:iCs", "w:vertAlign", "w:highlight"):
            for e in rp.findall(qn(t)): rp.remove(e)
        return rp
    return OxmlElement("w:rPr")
def _run(text, rpr, b=False, i=False, va=None, hl=False, br=False):
    r = OxmlElement("w:r"); rp = copy.deepcopy(rpr)
    for flag, tag, val in ((b, "w:b", None), (i, "w:i", None), (hl, "w:highlight", "yellow"), (va, "w:vertAlign", va)):
        if flag:
            e = OxmlElement(tag)
            if val: e.set(qn("w:val"), val)
            rp.append(e)
    r.append(rp)
    if br: r.append(OxmlElement("w:br")); return r
    t = OxmlElement("w:t"); t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve"); t.text = text; r.append(t)
    return r
def runs(markup, rpr, bold=False):
    out = []
    for seg in TOK.split(markup):
        if not seg: continue
        if seg == "\n": out.append(_run("", rpr, br=True))
        elif seg.startswith("**"): out.append(_run(seg[2:-2], rpr, b=True))
        elif seg.startswith("_{"): out.append(_run(seg[2:-1], rpr, b=bold, va="subscript"))
        elif seg.startswith("^{"): out.append(_run(seg[2:-1], rpr, b=bold, va="superscript"))
        elif seg.startswith("[["): out.append(_run(seg[2:-2], rpr, b=bold, hl=True))
        elif seg.startswith("*") and seg.endswith("*") and len(seg) > 2: out.append(_run(seg[1:-1], rpr, b=bold, i=True))
        else: out.append(_run(seg, rpr, b=bold))
    return out
def para(tpl, markup):
    el = copy.deepcopy(TPL[tpl]); rpr = base_rpr(el)
    for ch in list(el):
        if ch.tag != qn("w:pPr"): el.remove(ch)
        else:
            for s_ in ch.findall(qn("w:sectPr")): ch.remove(s_)
    for r in runs(markup, rpr): el.append(r)
    return el
def cell_rpr(sz):
    rp = OxmlElement("w:rPr")
    for tag, attrs in (("w:rFonts", {"w:ascii": "Times New Roman", "w:hAnsi": "Times New Roman", "w:cs": "Times New Roman"}),
                       ("w:sz", {"w:val": str(sz)}), ("w:szCs", {"w:val": str(sz)})):
        e = OxmlElement(tag)
        for k, v in attrs.items(): e.set(qn(k), v)
        rp.append(e)
    return rp
def make_table(widths, rows, left_cols=1, sz=16, header_rows=1):
    assert all(len(r) == len(widths) for r in rows), [len(r) for r in rows]
    tbl = OxmlElement("w:tbl"); pr = OxmlElement("w:tblPr")
    jc = OxmlElement("w:jc"); jc.set(qn("w:val"), "center"); pr.append(jc)
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); pr.append(lay)
    bd = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{side}")
        for k, v in (("val", "single"), ("sz", "4"), ("space", "0"), ("color", "auto")): e.set(qn(f"w:{k}"), v)
        bd.append(e)
    pr.append(bd)
    mar = OxmlElement("w:tblCellMar")
    for side, w in (("top", 0), ("left", 30), ("bottom", 0), ("right", 30)):
        e = OxmlElement(f"w:{side}"); e.set(qn("w:w"), str(w)); e.set(qn("w:type"), "dxa"); mar.append(e)
    pr.append(mar)
    tw = OxmlElement("w:tblW"); tw.set(qn("w:type"), "dxa"); tw.set(qn("w:w"), str(sum(widths))); pr.append(tw)
    tbl.append(pr); grid = OxmlElement("w:tblGrid")
    for w in widths:
        g = OxmlElement("w:gridCol"); g.set(qn("w:w"), str(w)); grid.append(g)
    tbl.append(grid); rpr = cell_rpr(sz)
    for ri, row in enumerate(rows):
        tr = OxmlElement("w:tr"); trp = OxmlElement("w:trPr"); trp.append(OxmlElement("w:cantSplit"))
        if ri < header_rows: trp.append(OxmlElement("w:tblHeader"))
        tr.append(trp)
        for ci_, (w, txt) in enumerate(zip(widths, row)):
            tc = OxmlElement("w:tc"); tcp = OxmlElement("w:tcPr"); cw = OxmlElement("w:tcW")
            cw.set(qn("w:type"), "dxa"); cw.set(qn("w:w"), str(w)); tcp.append(cw)
            va = OxmlElement("w:vAlign"); va.set(qn("w:val"), "center"); tcp.append(va); tc.append(tcp)
            p = OxmlElement("w:p"); pp = OxmlElement("w:pPr")
            if ri < len(rows) - 1: pp.append(OxmlElement("w:keepNext"))
            sp = OxmlElement("w:spacing")
            for k, v in (("before", "0"), ("after", "0"), ("line", "228"), ("lineRule", "auto")): sp.set(qn(f"w:{k}"), v)
            pp.append(sp); ind = OxmlElement("w:ind"); ind.set(qn("w:firstLine"), "0"); pp.append(ind)
            if ci_ >= left_cols:
                j = OxmlElement("w:jc"); j.set(qn("w:val"), "center"); pp.append(j)
            p.append(pp)
            for r in runs(txt, rpr, bold=(ri < header_rows)): p.append(r)
            tc.append(p); tr.append(tc)
        tbl.append(tr)
    return tbl
def section_end(one_col):
    """Tiny paragraph that ends the current section: two-column (before a full-width block) or one-column (after it)."""
    el = copy.deepcopy(TPL["wide2"])
    if one_col:
        old = el.find(".//" + qn("w:sectPr")); old.getparent().replace(old, copy.deepcopy(ONECOL_SECT))
    return el
# equations via pandoc (LaTeX -> OMML)
def omml(latex_list):
    md = "\n\n".join(f"$${x}$$" for x in latex_list)
    tmp = Path(tempfile.gettempdir()) / "r3_eqs.docx"; pypandoc.convert_text(md, "docx", format="markdown", outputfile=str(tmp))
    ms = docx.Document(str(tmp)).element.body.findall(".//" + qn("m:oMath")); assert len(ms) == len(latex_list), len(ms)
    return [copy.deepcopy(m) for m in ms]
def eq_par(math_el, num):
    el = copy.deepcopy(TPL["eq"]); old = el.find(qn("m:oMath")); old.getparent().replace(old, math_el)
    ts = [t for t in el.iter(qn("w:t")) if re.fullmatch(r"\s*\(\d+\)\s*", t.text or "")]
    assert len(ts) == 1; ts[0].text = f"({num})"; return el

# ------------------------------------------------------------------------------------------------ content model
BLOCKS = []
def H1(t, key): BLOCKS.append(("h1", t, key))
def H2(t, key): BLOCKS.append(("h2", t, key))
def PF(t): BLOCKS.append(("pf", t))
def PB(t): BLOCKS.append(("pb", t))
def EQ(latex, key): BLOCKS.append(("eq", latex, key))
def TAB(key, caption, widths, rows, wide=False, left_cols=1, sz=16, header_rows=1):
    BLOCKS.append(("tab", key, caption, widths, rows, wide, left_cols, sz, header_rows))
def FIG(key, caption, image): BLOCKS.append(("fig", key, caption, image))
COL_W, FULL_W = 4657, 9746

# ================================================================================================ TEXT
F05 = "F_{0.5}"
HS, ID2, ID1 = rows_named("T4", "human_data_stage"), rows_named("T4", "noedit_two_stage"), rows_named("T4", "noedit_stage1")
TH, RF, LP = rows_named("T4", "threshold_tau0"), rows_named("T4", "refinement"), rows_named("T4", "length_penalty_0.7_vs_1.0")
B50, B100 = rows_named("T4", "budget_50_vs_25"), rows_named("T4", "budget_100_vs_50")
CLC4 = {r["name"]: r for r in UNCONT["comparisons"]}
CLL = {r["benchmark"]: r for r in A["cleaning_learner_benchmarks"]["comparisons"]}
CLS = {(r["name"], r["benchmark"]): r for r in A["cleaning_learner_benchmarks"]["per_seed"]}
assert all(x < 0 for b in ("c4", "bea_dev", "conll") for x in ID2[b]["per_pair_delta"])
assert all(x > 0 for b in ("c4", "bea_dev", "conll") for x in TH[b]["per_pair_delta"])
assert all(negligible(r[b]) for r in (RF, LP) for b in ("c4", "bea_dev", "conll"))
assert negligible(ID2["bea_dev"]) and not negligible(ID2["conll"]) and ID2["c4"]["p_holm"] < 0.05
assert negligible(CLL["bea_dev"]) and negligible(CLC4["cleaning_pooled"]) and not negligible(CLL["conll"])
TAU_SEL = [RE[k][kk]["tau"] for k in (1, 2, 3) for kk in RE[k] if kk.startswith("tau|")]
assert len(TAU_SEL) == 10 and set(TAU_SEL) == {0.0}      # 3 primary + 3 no-identity + 2 second-seed + 2 budget runs
# JFLEG: the threshold lowers GLEU slightly in every pair (no paired test for GLEU)
TH_JF = [F(a, "jfleg") - F(b, "jfleg") for a, b in TH["bea_dev"]["pairs"]]
if HE:
    HEA = HE["agreement"]["verdict"]; HEP = HE["population_estimate_all_edited"]
    HE_ABS = (f"Blinded ratings by two annotators (Cohen's kappa {HEA['cohen_kappa']:.2f}) judge an estimated "
              f"{100 * HEP['better']:.0f}% of its edits better than the input and {100 * HEP['worse']:.0f}% worse.")
else:
    HE_ABS = "[[Human-evaluation sentence pending.]]"

TITLE = ("What Matters for a Compact Grammatical Error Corrector? Seed-Replicated Ablations of FLAN-T5-small with "
         "Little Human Data")
ABSTRACT = (
    "Grammatical error correction (GEC) systems usually rely on large models and millions of human-corrected sentences. "
    "We ask which design choices matter for a compact 77-million-parameter FLAN-T5-small model trained on synthetic data "
    f"and {WI['train_pairs']:,} human-annotated pairs. With one corrected pipeline, essay-level held-out selection, three "
    "seeds per arm and paired bootstrap tests, the learner-data training stage dominates "
    f"({sg(HS['bea_dev']['delta'])} F0.5 on BEA-2019 development) and a confidence threshold adds a small gain "
    f"({sg(TH['bea_dev']['delta'])}); identity training pairs, iterative refinement, length-penalty tuning and the tested "
    f"corpus filtering bring no benefit, and halving the learner data costs {abs(B100['bea_dev']['delta']):.3f}. The system "
    f"reaches F0.5 = {TEST_STR} on the blind BEA-2019 test set, above a 1.5-billion-parameter instruction-tuned language "
    "model but below a same-size model trained on 2.4 million human-corrected pairs. " + HE_ABS)
KEYWORDS = ("**Keywords:** Grammatical error correction, FLAN-T5, Synthetic data, Low-resource fine-tuning, Ablation study, "
            "Human evaluation.")

# ---------------------------------------------------------------- 1. Introduction
H1("Introduction", "intro")
PF("Grammatical error correction (GEC) maps a possibly ungrammatical sentence to a grammatical one. Most tokens should be "
   "copied unchanged, and the standard edit-level metric weights precision above recall, so needless edits cost more than "
   "missed ones. Strong systems combine large pre-trained models with millions of human-corrected sentence pairs "
   "[@rothe2021], and the best published results come from combining several such systems [@omelianchuk2024]. Many "
   "applications, such as writing support in a classroom or a low-cost correction service, instead need a model that runs "
   "on one commodity graphics processing unit (GPU) and is adapted with tens of thousands of annotated sentences. For such compact, low-human-data "
   "systems a list of additions is commonly proposed: cleaning the synthetic corpus, adding identity (no-edit) training "
   "pairs, re-decoding the output, tuning the length penalty, thresholding on model confidence and adding a second "
   "training stage on learner data. Which of them change accuracy is rarely tested under one controlled protocol.")
PB(f"We study this question for FLAN-T5-small (77 million parameters) trained on 1.15 million synthetic pairs from C4-200M "
   f"[@stahlberg2021] and {WI['train_pairs']:,} learner pairs from the Write & Improve and LOCNESS corpus (W&I+LOCNESS) [@bryant2019]. All runs use one corrected "
   "training pipeline (Section <<S:valid>>), and the configuration of the primary system and its primary endpoint, the "
   "blind BEA-2019 test set, were fixed before that test set was first used. Our contributions are: (i) one-factor "
   "ablations at the full training budget that report both arms' absolute scores, three stage-2 training seeds per arm, "
   "paired bootstrap intervals, Holm-corrected tests and an equivalence margin; (ii) an essay-level split of the learner "
   "data for checkpoint and threshold selection, with overlap audits against every evaluation set; (iii) a learner-data "
   "budget curve on the same backbone and synthetic stage; (iv) a comparison under one protocol with an instruction-tuned "
   "language model, an instruction-tuned text editor and two GEC-specific T5 models, with inference cost on the same "
   "GPU; and (v) a blinded human evaluation by two annotators with agreement statistics and frequency-weighted estimates.")
PB("The learner-data stage dominates every other factor, and the confidence threshold gives a small but consistent gain. "
   "Identity pairs lower in-domain accuracy and do not help on the learner benchmarks; refinement and length-penalty "
   "tuning have effects within ±0.01 F0.5 on every benchmark; and the tested filtering policy is within ±0.01 on "
   "BEA-2019 development and in domain and inconclusive on CoNLL-2014. Section <<S:rel>> reviews related work, Section "
   "<<S:data>> describes the data, Section <<S:method>> the system and the analysis plan, Section <<S:res>> the results, "
   "Section <<S:valid>> the validity checks and released artifacts, and Section <<S:concl>> concludes; Figure <<F:pipe>> "
   "gives an overview of the system.")

# ---------------------------------------------------------------- 2. Related work
H1("Related work", "rel")
PF("A recent survey [@bryant2023] covers the field; we organise the work most relevant to this study below.")
PB("**Synthetic data and two-stage training.** Neural GEC is usually framed as monolingual translation [@junczys2018]. "
   "Because annotated learner data are scarce, large synthetic corpora are made by corrupting clean text [@lichtarge2019; "
   "@grundkiewicz2019; @kiyono2019]; C4-200M, which we sample, was produced with tagged corruption models "
   "[@stahlberg2021]. The dominant recipe pre-trains on synthetic pairs and then fine-tunes on human-annotated learner "
   "data [@rothe2021], whose cLang-8 corpus of about 2.4 million English pairs is now the usual fine-tuning set for T5 "
   "models. Data augmentation remains active: MixEdit [@ye2023], for example, characterises augmented data by their "
   "affinity to real errors and their diversity, and finds that high affinity with appropriate diversity helps most. "
   "Low-resource GEC usually refers to languages other than English with little annotated data, such as Czech, German "
   "and Russian [@naplava2019]. Our setting differs: English, for which large synthetic corpora and pre-trained models "
   "exist, but with deliberately little human-annotated data.")
PB("**Models and editors.** Pre-trained Transformer encoder-decoders [@vaswani2017] such as T5 [@raffel2020] and its "
   "instruction-tuned variant FLAN-T5 [@chung2024] are strong GEC backbones. Sequence taggers such as GECToR "
   "[@omelianchuk2020] predict token-level edit operations and are fast and precise. Instruction-tuned editors such as "
   "CoEdIT [@raheja2023] are fine-tuned on several editing tasks; their GEC data come from the NUS Corpus of Learner "
   "English (NUCLE) [@dahlmeier2013], Lang-8 and W&I+LOCNESS. Current state-of-the-art results combine large systems by ensembling or ranking [@omelianchuk2024], at "
   "a compute cost far beyond the compact regime studied here.")
PB("**Large language models.** General-purpose language models such as GPT-3.5 and GPT-4 are fluent correctors but tend "
   "to over-correct, which lowers precision on minimal-edit benchmarks [@coyne2023]. A study of seven open-source and "
   "three commercial models prompted to correct English learner text [@davis2024] found that they do not consistently "
   "outperform supervised GEC systems, and that commercial models fare better on fluency-oriented than on minimal-edit "
   "benchmarks. We therefore compare against a small instruction-tuned language model under exactly our input and "
   "scoring protocol rather than against published numbers.")
PB("**Evaluation.** Corrections are scored at the edit level with " + F05 + ", using the MaxMatch (M²) scorer "
   "[@dahlmeier2012] of the CoNLL-2014 shared task [@ng2014] or the ERRor ANnotation Toolkit (ERRANT) [@bryant2017] of "
   "BEA-2019 [@bryant2019]; fluency is scored on JFLEG [@napoles2017] with GLEU [@napoles2015; @napoles2016] (Section "
   "<<S:metrics>>). The scorers extract and match edits differently and are applied to different splits, so numbers "
   "from different protocols cannot be ranked against each other; we score every system ourselves. Meta-evaluation shows "
   "that agreement between automatic metrics and human judgments depends on the systems compared: on the SEEDA dataset "
   "most metrics correlate less well with human ratings for modern neural systems than for older ones, and traditional "
   "metrics assess fluent rewrites poorly [@kobayashi2024]. Edit-level gains therefore need not match perceived "
   "quality, which motivates our human evaluation (Section <<S:human>>).")

# ---------------------------------------------------------------- 3. Data
H1("Data", "data")
H2("Synthetic corpus and filtering policy", "synth")
PF("We sample Google's C4-200M corpus [@stahlberg2021], ten shards of (source, target) pairs in which the source is an "
   "automatically corrupted Common Crawl sentence and the target the original. From each shard, pairs are scanned in "
   f"file order until {MC['kept'] // 10:,} are kept ({MC['kept']:,} in total); the kept set is permuted (seed 42) into "
   "1,150,000 training, 25,000 validation and 25,000 "
   "test pairs, and a fixed random subset of 5,000 test pairs is the in-domain evaluation set. The primary system uses "
   "the filtered sample of Table <<T:sample>>: after whitespace normalisation a pair is discarded if a side is empty or "
   "the two sides are identical, if either side is outside 4 to 50 words, if the source duplicates one already kept, if "
   "either side contains markup (an HTML or XML tag or entity, or a URL marker) or mojibake, if the target contains a "
   "punctuation run (two or more '!' or '?', four or more periods, two or more commas, '&&' or ';;') or an immediately "
   "repeated word, if more than 25% of the target characters are non-ASCII, if the lengths differ by more than 10 "
   "tokens, or if the token-multiset overlap (intersection over the longer side) is below 0.5. Each discarded pair is "
   "counted under the first rule it fails. The unfiltered comparison sample applies only the identity, length and "
   "duplicate rules. Because filtering changes which pairs are drawn, the two samples differ on both their source and "
   "target sides; we therefore treat the filtered sample as a composite filtering policy, not as an intervention on "
   f"reference noise alone. The two training splits share {100 * OVA['train_overlap_share_of_clean']:.1f}% of their pairs. "
   f"Because the unfiltered sample was split independently, {OVA['c4_test_subset_5k_in_noisy_train']:,} of the 5,000 "
   "in-domain test pairs occur in its training split, and none in the filtered training split; comparisons between the "
   f"two samples are therefore scored in domain on the {UNCONT['pairs_kept']} test pairs absent from both.")
s1 = lambda k: f"{MC[k]:,}"; s0 = lambda k: f"{MN.get(k, 0):,}"
TAB("sample", "Sampling statistics of the two C4-200M samples (pairs discarded per rule)", [2457, 1100, 1100], [
    ["Step", "Filtered (primary)", "Unfiltered"],
    ["Pairs scanned (10 shards)", s1("scanned"), s0("scanned")],
    ["Empty, or source identical to target", s1("drop_identical"), s0("drop_identical")],
    ["Outside 4 to 50 words", s1("drop_length"), s0("drop_length")],
    ["Duplicate source", f"{MC.get('drop_duplicate', 0):,}", s0("drop_duplicate")],
    ["Markup, source / target", f"{MC['drop_markup_src']:,} / {MC['drop_markup_tgt']:,}", "–"],
    ["Mojibake, source / target", f"{MC['drop_mojibake_src']:,} / {MC['drop_mojibake_tgt']:,}", "–"],
    ["Punctuation run in target", s1("drop_punct_run_tgt"), "–"],
    ["Repeated word in target", s1("drop_repeat_word_tgt"), "–"],
    ["Non-ASCII share above 0.25 in target", s1("drop_nonascii_tgt"), "–"],
    ["Length difference above 10 tokens", s1("drop_len_diff"), "–"],
    ["Token overlap below 0.5", s1("drop_low_overlap"), "–"],
    ["Kept", s1("kept"), s0("kept")]])
assert MC["scanned"] - sum(v for k, v in MC.items() if k.startswith("drop_")) == MC["kept"]
assert MN["scanned"] - sum(v for k, v in MN.items() if k.startswith("drop_")) == MN["kept"]
H2("Learner data and essay-level split", "wi")
PF(f"The second stage uses the W&I+LOCNESS training data of the BEA-2019 shared task [@bryant2019]. Applying all edits of "
   f"the first annotator in the gold M² files of the three proficiency bands (Common European Framework of Reference, "
   f"CEFR, levels A, B and C) gives {WI['pairs']:,} sentence "
   "pairs. Sentences from one essay share topic, vocabulary and error habits, so the held-out set is formed from whole "
   "essays: we align every sentence with its essay in the corpus's JSON release (all sentences align, to "
   f"{WI['essays']:,} essays: 1,300 A, 1,000 B and 700 C) and hold out 5% of the essays of each band (seed 42), i.e. "
   f"{WI['holdout_essays']} essays with {WI['holdout_pairs']:,} pairs (A: {WI['holdout_by_band']['A']['essays']} essays, "
   f"{WI['holdout_by_band']['A']['pairs']} pairs; B: {WI['holdout_by_band']['B']['essays']}, "
   f"{WI['holdout_by_band']['B']['pairs']}; C: {WI['holdout_by_band']['C']['essays']}, {WI['holdout_by_band']['C']['pairs']}). "
   f"Training pairs whose source sentence also occurs in a held-out essay are removed "
   f"({WI['train_pairs_dropped_as_source_duplicates']} pairs), leaving {WI['train_pairs']:,} training pairs, so that no "
   "essay and no source sentence occurs in both parts. The holdout selects the stage-2 checkpoint and the threshold "
   "τ (Section <<S:method>>); the BEA-2019 development and test sets are never used for training or selection.")
ow = OVW
PB(f"We also audited exact source-sentence overlap between the W&I+LOCNESS training pairs and every evaluation set. "
   f"{ow['bea2019_dev']['exact_matches']} of the {ow['bea2019_dev']['sentences']:,} BEA-2019 development sources and "
   f"{ow['bea2019_test']['exact_matches']} of the {ow['bea2019_test']['sentences']:,} test sources also occur verbatim in "
   "training; among sentences with at least 20 letters and digits, compared case-insensitively without punctuation, the "
   f"counts are {ow['bea2019_dev']['normalised_matches_(>=20_chars)']} and "
   f"{ow['bea2019_test']['normalised_matches_(>=20_chars)']}. No CoNLL-2014, JFLEG or in-domain test source occurs in "
   "training. These overlaps are inherent in the official splits; we report them for completeness and did not test "
   "their effect on any score.")
assert all(ow[k]["exact_matches"] == 0 for k in ("conll2014_test", "jfleg_test", "c4_test_subset_5k"))
H2("Endpoints and metrics", "metrics")
PF("Edit-level scores compare the edits that turn the source into the output with gold edits. With TP correct edits, FP "
   "proposed edits not in the gold set and FN gold edits missed, precision is P = TP/(TP + FP), recall R = TP/(TP + FN) and")
EQ(r"F_{0.5}=\frac{(1+0.5^{2})\,PR}{0.5^{2}\,P+R}=\frac{1.25\,\mathrm{TP}}{1.25\,\mathrm{TP}+\mathrm{FP}+0.25\,\mathrm{FN}}", "f05")
PF("In the count form of Eq. (<<E:f05>>) a false positive increases the denominator four times as much as a false "
   "negative; this weighting applies to the corpus-level counts and is not a fixed cost per sentence. The primary "
   "endpoint is the blind BEA-2019 test set, scored by the shared task's ERRANT-based scorer on the CodaBench platform. "
   "Secondary endpoints are CoNLL-2014 test with the official M² scorer, BEA-2019 development with ERRANT's "
   "errant_compare, JFLEG test with GLEU, and the 5,000-pair in-domain subset with single-reference ERRANT. GLEU "
   "[@napoles2015; @napoles2016] is an n-gram measure for GEC: like the machine-translation metric BLEU it counts output "
   "n-grams that match a reference, but it also penalises n-grams that the output shares with the source and not with "
   "the reference, i.e. errors left uncorrected. We use the official JFLEG implementation with its four references and "
   "500 resampling iterations. CoNLL-2014 and BEA-2019 are pre-tokenised and both scorers compare whitespace-separated "
   "tokens, so every system's output is re-tokenised with one tokeniser per benchmark, chosen by how faithfully it "
   "reproduces the references: a rule-based tokeniser reproduces 99.3% of the CoNLL-2014 references (also used for "
   "JFLEG) and spaCy 98.7% of the BEA-2019 references.")

# ---------------------------------------------------------------- 4. System and analysis plan
H1("System and analysis plan", "method")
PF("Figure <<F:pipe>> summarises the system. Its training and decoding components are described in Sections "
   "<<S:train>> and <<S:decode>>, and Section <<S:plan>> states which configuration is primary, how the ablations are "
   "built and how they are tested.")
FIG("pipe", "Overview of the primary system: filtered synthetic data with identity pairs, two training stages, decoding "
            "with a confidence threshold, and evaluation", P_ / "figures/figure1_r3.png")
H2("Training", "train")
PF("We fine-tune flan-t5-small [@chung2024; @flant5card] (77M parameters), prefixing every input with “Correct the "
   "grammar in this sentence: ”. Inputs and targets are truncated to 128 sub-word tokens, and each source word of an "
   "edited training pair is dropped with probability 0.05. Stage 1 trains for one epoch with AdamW (learning rate "
   "3×10^{−4}, weight decay 0.01, label smoothing 0.1, cosine schedule with 6% warm-up) at a global batch of 128 "
   "(16 per device, 4 accumulation steps, 2 GPUs) in fp16 and keeps the checkpoint with the lowest loss on the "
   "25,000-pair validation split, evaluated every 2,500 steps and at the last step. The primary system adds identity pairs: 15% of the "
   f"training split (172,500 targets) is added as target-to-target pairs without source dropout, giving "
   f"{S1['train_rows']:,} stage-1 examples and {S1['steps']:,} optimiser steps (8,985 without identity pairs). Stage 2 "
   f"fine-tunes the stage-1 model on the {WI['train_pairs']:,} learner pairs for three epochs at learning rate "
   f"1×10^{{−4}} and a global batch of 64 (16 per device, 2 accumulation steps, 2 GPUs), i.e. "
   f"{S2['steps'] // 3} updates per epoch and {S2['steps']:,} in total, and keeps the checkpoint with the lowest holdout "
   "loss among the three epoch ends. Each stage-2 configuration is trained with seeds 42, 1337 and 2024. Table "
   "<<T:hyper>> lists all settings.")
assert S1["train_rows"] == 1_322_500 and S1["steps"] == math.ceil(1_322_500 / 128) and S2["steps"] == 3 * math.ceil(WI["train_pairs"] / 64)
H2("Decoding and confidence threshold", "decode")
PF("Outputs are decoded by beam search (5 beams, length penalty 0.7, no repeated trigram). A guarded refinement pass "
   "re-decodes every edited output and accepts the second output only if it differs from the first by at most three "
   "word-level changes. A confidence threshold then reverts edits the model is unsure of. For an input x and an output "
   "y = (y_{1}, …, y_{|y|}), whose last token is the end-of-sequence token that the T5 tokeniser appends and whose "
   "padding positions are excluded, the length-normalised log-likelihood under teacher forcing is")
EQ(r"\ell(y\mid x)=\frac{1}{|y|}\sum_{t=1}^{|y|}\log p_{\theta}\left(y_{t}\mid y_{<t},x\right)", "ell")
PF("where the instruction prefix is prepended to x. For an output ŷ ≠ x, the confidence margin compares the "
   "output with copying the input unchanged, and the final output ỹ keeps the edit only if the margin reaches τ:")
EQ(r"m(x,\hat{y})=\ell(\hat{y}\mid x)-\ell(x\mid x)", "margin")
EQ(r"\tilde{y}=\begin{cases}\hat{y}, & m(x,\hat{y})\ge\tau\\ x, & \text{otherwise}\end{cases}", "revert")
PF("Reversion acts on whole sentences, so no alignment between input and output is needed. τ is chosen on the "
   "essay-level holdout from {none, 0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5} by edit-level " + F05 + "; τ = 0 was "
   f"selected in all {len(TAU_SEL)} stage-2 runs, so an edit is kept only if the model finds it at least as likely as "
   "the unchanged input.")
H2("Primary system, ablations and statistics", "plan")
PF("**Primary system.** The configuration with identity pairs, both training stages, length penalty 0.7, refinement and "
   "the holdout-tuned threshold was fixed before the blind BEA-2019 test set was first used, and each of its three "
   f"stage-2 seeds was submitted once (CodaBench submissions {IDS[0]}, {IDS[1]} and {IDS[2]}). Every other configuration "
   "is an ablation of it. Replacing the primary system by a better-scoring ablation would select it on test data, so we "
   "keep it even where an ablation favours the alternative (Section <<S:abl>>).")
PB("**Ablations.** Each contrast changes one factor (Table <<T:runs>>). Training factors are retrained: the learner-data "
   "stage (each stage-2 model against its own stage-1 initialisation), identity pairs (stage 1 with and without them, "
   "each followed by the same stage 2) and the filtering policy (stage 1 on each sample). Each training contrast is "
   "replicated with two independently trained stage-1 seeds, and stage-2 contrasts additionally with three stage-2 "
   "seeds. Decoding factors (threshold, refinement and length penalty) reuse one checkpoint and are exactly paired.")
PB("**Statistics.** For systems A and B scored on the same n sentences we draw 2,000 bootstrap resamples of the "
   "sentences [@koehn2004], recompute corpus " + F05 + " from resampled per-sentence edit counts, which reproduce the "
   "official totals exactly, and report the observed difference with its 95% percentile interval; the two-sided p is "
   "twice the smaller share of resampled differences on either side of zero. When a contrast has k seed pairs "
   "(A_{j}, B_{j}), every resample uses the same sentences for all pairs and the statistic is the mean paired difference")
EQ(r"\Delta=\frac{1}{k}\sum_{j=1}^{k}\left[F_{0.5}(A_{j})-F_{0.5}(B_{j})\right]", "delta")
PF("so the interval reflects test-set variability around the seed-averaged effect, and training variability is "
   "reported separately as the standard deviation across seeds. The comparisons of each table form one family, and "
   "their p-values are adjusted with Holm's step-down procedure. We regard differences below 0.01 " + F05 + " as "
   f"practically negligible, about six times the seed standard deviation of the primary system on BEA-2019 development "
   f"({PRIM['bea_dev'][1]:.4f}), and call an effect negligible only when its 90% interval lies inside (−0.01, 0.01), which is the "
   "two one-sided tests procedure at α = 0.05 [@schuirmann1987]; an interval that neither excludes zero nor lies "
   "inside the margin is reported as inconclusive. The official JFLEG GLEU is not decomposable by sentence, so for "
   "JFLEG we report means only.")
assert 5.5 < MARGIN / PRIM["bea_dev"][1] < 7
TAB("runs", "Runs behind every contrast. B: beam search (5 beams, length penalty 0.7); R: refinement; τ: confidence "
    "threshold (0 unless stated). Stage-1 checkpoints: lowest validation loss; stage-2 checkpoints: lowest holdout loss, "
    "the third epoch in every run except P seed 2024 and P25 (second epoch)",
    [700, 2350, 1500, 2700, 2496], [
    ["Run", "Stage 1: sample, identity pairs, seed", "Stage-1 checkpoint", "Stage 2: learner data, seeds", "Decoding (contrasts)"],
    ["P (primary)", "filtered, 172,500, 42", f"step {S1['steps']:,} (last)", f"essay split, {WI['train_pairs']:,} pairs; 42, 1337, 2024",
     "B+R+τ (all); also no τ, no R, τ grid and length penalty 1.0 for seed 42"],
    ["N", "filtered, none, 42", "step 7,500 of 8,985", "as P; 42, 1337, 2024", "B+R+τ, B+R (identity, threshold, filtering)"],
    ["P′", "filtered, 172,500, 1337", f"step {RE[3]['train|noedit_s1337']['steps']:,} (last)", "as P; 1337",
     "B+R+τ, B+R (learner stage, identity)"],
    ["N′", "filtered, none, 1337", "step 8,985 (last)", "as P; 1337", "B+R+τ, B+R (identity, filtering)"],
    ["U, U′", "unfiltered, none, 42 / 1337", "step 8,985 (last)", "none", "B+R (filtering)"],
    ["P25, P50", "stage 1 of P", "–", f"25% / 50% of the training essays ({WI['budget_train_pairs']['25']:,} / "
     f"{WI['budget_train_pairs']['50']:,} pairs); 42", "B+R+τ (learner-data budget)"]], wide=True, sz=15)
assert RN["train|nogate_s1"]["best_checkpoint"].endswith("checkpoint-7500") and RA["train|clean_s1337"]["best_checkpoint"].endswith("-8985")
assert all(RAN[f"train|noisy_{s}"]["best_checkpoint"].endswith("-8985") for s in ("s42", "s1337"))
for tag, ck in (("final_e42", 1515), ("final_e1337", 1515), ("final_e2024", 1010), ("nogate_e42", 1515), ("nogate_e1337", 1515),
                ("nogate_e2024", 1515), ("noedit_s1337_e1337", 1515), ("clean_s1337_e1337", 1515), ("final_curve25", 254), ("final_curve50", 765)):
    rec = next(RE[k][f"train|{tag}"] for k in (1, 2, 3) if f"train|{tag}" in RE[k])
    assert rec["best_checkpoint"].endswith(f"checkpoint-{ck}"), (tag, rec["best_checkpoint"])

# ---------------------------------------------------------------- 5. Results
H1("Results and discussion", "res")
H2("Main results", "main")
qa, qb = T6["final_vs_qwen_fourshot|bea_dev"], T6["final_vs_qwen_fourshot|conll"]
ca, cb = T6["final_vs_coedit_large|bea_dev"], T6["final_vs_coedit_large|conll"]
ga, gb = T6["final_vs_gec_t5_small|bea_dev"], T6["final_vs_gec_t5_small|conll"]
ba, bb = T6["final_vs_gec_t5_base|bea_dev"], T6["final_vs_gec_t5_base|conll"]
assert max(r["p_holm"] for r in (qa, qb, ga, gb, ba, bb)) < 0.001 and ca["p_holm"] > 0.05 and cb["p_holm"] > 0.05
OURS_CONLL_P = st.mean(S[n]["conll"]["P"] for n in FIN); OURS_CONLL_R = st.mean(S[n]["conll"]["R"] for n in FIN)
assert S[QWEN]["conll"]["R"] > OURS_CONLL_R and S[QWEN]["conll"]["P"] < OURS_CONLL_P     # the gap is precision, not recall
PF(f"Table <<T:main>> summarises the primary system and the systems scored under the same protocol. Over three stage-2 "
   f"seeds the primary system reaches {F05} = {TEST_STR} on the blind BEA-2019 test set (P {pct(round(st.mean(TEST_P), 2))}, "
   f"R {pct(round(st.mean(TEST_R), 2))}), {pm(FIN, 'bea_dev')} on BEA-2019 development and {pm(FIN, 'conll')} on "
   f"CoNLL-2014; every seed standard deviation is below {math.ceil(SD_MAX * 1000) / 1000:.3f}. It outperforms "
   f"Qwen2.5-1.5B-Instruct [@qwen2024] with four in-context examples by {sg(qa['delta'])} on BEA-2019 development (95% CI "
   f"{ci(qa)}) and {sg(qb['delta'])} on CoNLL-2014 ({ci(qb)}), both Holm-adjusted p < 0.001. The language model edits "
   f"{ED_QWEN:.0f}% of the CoNLL-2014 sentences against {ED_OURS:.0f}% for our system, with precision "
   f"{S[QWEN]['conll']['P']:.3f} against {OURS_CONLL_P:.3f}: over-correction, not missed errors, explains the gap.")
PB(f"CoEdIT-large [@raheja2023], an instruction-tuned FLAN-T5-large editor ten times larger, is not distinguishable from "
   f"our system on BEA-2019 development ({sg(ca['delta'])} in our favour, {ci(ca)}) and scores higher on CoNLL-2014 "
   f"({sg(cb['delta'])}, {ci(cb)}; Holm-adjusted p = {cb['p_holm']:.2f}, not significant after correction) and on JFLEG ({F(CED, 'jfleg'):.3f} against "
   f"{PRIM['jfleg'][0]:.3f}). Its training data include NUCLE, the training corpus of the CoNLL-2014 shared task "
   "[@dahlmeier2013], and its model card lists JFLEG among its training datasets, so its CoNLL-2014 and JFLEG scores may "
   f"not be held-out estimates in the same sense as ours. Conversely, gec-t5-small-clang8 [@gect5card], a model of the "
   f"same size trained on about 2.4 million human-corrected cLang-8 pairs [@rothe2021], is ahead by {abs(ga['delta']):.3f} "
   f"on BEA-2019 development and {abs(gb['delta']):.3f} on CoNLL-2014, and its 248M-parameter sibling by "
   f"{abs(ba['delta']):.3f} and {abs(bb['delta']):.3f} (all Holm-adjusted p < 0.001). Both gec-t5 checkpoints were "
   "selected by their authors on BEA-2019 development loss, so their development scores are not held-out either. The "
   f"gaps are associations, not causal estimates: besides about {HUMAN_RATIO:.0f} times more human training data, the "
   "gec-t5 models start from T5 v1.1 rather than FLAN-T5, have no synthetic stage and use another schedule and decoder. "
   "Section <<S:budget>> varies the amount of human data inside our own pipeline.")
def row_sys(label, params, hd, sysname, test="–", c4=True, jf=True):
    g = lambda b: f"{F(sysname, b):.3f}" if b in S[sysname] else "–"
    return [label, params, hd, test, g("bea_dev"), g("conll"), g("jfleg") if jf else "–", g("c4") if c4 else "–"]
TAB("main", "Official-scorer " + F05 + " (JFLEG: GLEU) on identical splits and tokenisation. Test: blind BEA-2019 test; "
    "Dev: BEA-2019 development; C4: in-domain subset; Human data: human-annotated GEC training pairs. Primary system: "
    "mean ± standard deviation over three stage-2 seeds. † includes NUCLE, Lang-8 and W&I+LOCNESS [@raheja2023]; "
    "‡ cLang-8 [@rothe2021]", [2246, 800, 1100, 1100, 1100, 1100, 1150, 1150], [
    ["System", "Params", "Human data", "Test", "Dev", "CoNLL", "JFLEG", "C4"],
    ["Do nothing", "–", "–", "–", "0.000", "0.000", f"{JF_DN:.3f}", "0.000"],
    row_sys("Qwen2.5-1.5B-Instruct, 4-shot", "1.5B", "0", QWEN),
    row_sys("CoEdIT-large", "783M", "mixture†", CED),
    row_sys("gec-t5-small-clang8", "77M", "2.4M‡", G5S),
    row_sys("gec-t5-base-clang8", "248M", "2.4M‡", G5B),
    row_sys("Ours, stage 1 only", "77M", "0", "final_stage1"),
    ["**Ours, primary**", "77M", f"{WI['train_pairs'] / 1000:.0f}k", TEST_STR, pm(FIN, "bea_dev"), pm(FIN, "conll"),
     pm(FIN, "jfleg"), pm(FIN, "c4")]], wide=True, sz=16)
assert all(F(n, "c4") > 0 for n in FIN) and "c4" not in S[QWEN]

H2("Which design choices matter?", "abl")
def mean_arm(pairs, b, side, mask=False):
    if mask: return st.mean(UNCONT["absolute"][p[side]]["F05"] for p in pairs)     # in-domain, uncontaminated pairs only
    return st.mean(F(p[side], b) for p in pairs)
ABL = [("Learner-data stage (4)", HS), ("Identity pairs, after stage 2 (4)", ID2), ("Identity pairs, stage 1 only (2)", ID1),
       ("Threshold τ = 0 vs. none (6)", TH), ("Refinement vs. none (3)", RF), ("Length penalty 0.7 vs. 1.0 (1)", LP),
       ("Filtering policy, stage 1 (2)", {"c4": CLC4["cleaning_pooled"], "bea_dev": CLL["bea_dev"], "conll": CLL["conll"]})]
cells = [(i, b, rec[b]) for i, (_, rec) in enumerate(ABL) for b in ("c4", "bea_dev", "conll")]
HADJ = dict(zip([(i, b) for i, b, _ in cells], holm([r["p_two_sided"] for _, _, r in cells])))
def abl_cell(i, b, r):
    mask = (b == "c4" and i == 6)
    a_, b_ = mean_arm(r["pairs"], b, 0, mask), mean_arm(r["pairs"], b, 1, mask)
    assert abs((a_ - b_) - r["delta"]) < 6e-4, (i, b, a_ - b_, r["delta"])
    mk = "*" if HADJ[(i, b)] < 0.05 else ("≈" if negligible(r) else "")
    return f"{a_:.3f} / {b_:.3f}\n{sg(r['delta'])}{mk} [{short(r['ci95'][0])}, {short(r['ci95'][1])}]"
def jf_cell(rec):
    pairs = rec["bea_dev"]["pairs"]
    if not all("jfleg" in S[a] and "jfleg" in S[x] for a, x in pairs): return "–"
    a_, b_ = st.mean(F(a, "jfleg") for a, _ in pairs), st.mean(F(x, "jfleg") for _, x in pairs)
    return f"{a_:.3f} / {b_:.3f}\n{sg(a_ - b_)}"
abl_rows = [["Contrast (seed pairs)", "In-domain C4", "BEA-2019 dev", "CoNLL-2014", "JFLEG (GLEU)"]]
for i, (label, rec) in enumerate(ABL):
    abl_rows.append([label] + [abl_cell(i, b, rec[b]) for b in ("c4", "bea_dev", "conll")] + [jf_cell(rec)])
TAB("abl", "One-factor ablations. Each cell gives the score with the factor / without it (mean over seed pairs) and, "
    "below, the mean paired difference with its 95% bootstrap interval. * Holm-adjusted p < 0.05 within the table; "
    "≈ 90% interval inside ±0.01 (negligible); unmarked: inconclusive. JFLEG: means only. The filtering "
    f"contrast is scored in domain on the {UNCONT['pairs_kept']} test pairs absent from both training splits. Runs: "
    "Table <<T:runs>>", [2246, 1875, 1875, 1875, 1875], abl_rows, wide=True, sz=15)
n_sig_id = sum(HADJ[(1, b)] < 0.05 for b in ("c4", "bea_dev", "conll"))
PF(f"Table <<T:abl>> lists every contrast with the absolute scores of both arms. The learner-data stage dominates: "
   f"averaged over four seed pairs it raises BEA-2019 development {F05} by {sg(HS['bea_dev']['delta'])} ({ci(HS['bea_dev'])}) "
   f"and CoNLL-2014 by {sg(HS['conll']['delta'])} ({ci(HS['conll'])}), with per-pair differences between "
   f"{min(HS['bea_dev']['per_pair_delta']):.3f} and {max(HS['bea_dev']['per_pair_delta']):.3f} on BEA-2019 development, "
   f"while the in-domain score falls by {abs(HS['c4']['delta']):.3f}, a forgetting effect that makes the best checkpoint "
   "domain-specific. Each stage-2 model is compared with its own stage-1 initialisation under the same decoder.")
PB(f"Identity pairs, which the primary system contains, do not help. After stage 2 all twelve paired differences are "
   f"negative: in domain {sg(ID2['c4']['delta'])} (Holm-adjusted {pv(HADJ[(1, 'c4')])}), on BEA-2019 development "
   f"{sg(ID2['bea_dev']['delta'])}, inside the ±0.01 margin, and on CoNLL-2014 {sg(ID2['conll']['delta'])}, which is "
   f"inconclusive (90% interval {ci(ID2['conll'], 'ci90')}). At stage 1 they cost {abs(ID1['c4']['delta']):.3f} in domain, "
   f"{abs(ID1['bea_dev']['delta']):.3f} on BEA-2019 development and {abs(ID1['conll']['delta']):.3f} on CoNLL-2014. We keep "
   "them in the primary system because its configuration was fixed in advance, but they should be omitted in practice; "
   "they cost nothing at inference.")
PB(f"The threshold is the only decoding factor that matters: it improves {F05} in all 18 paired comparisons, by "
   f"{sg(TH['bea_dev']['delta'])} on BEA-2019 development ({ci(TH['bea_dev'])}), {sg(TH['conll']['delta'])} on CoNLL-2014 "
   f"and {sg(TH['c4']['delta'])} in domain, while on JFLEG, whose fluency references reward more extensive rewriting, it "
   f"lowers GLEU slightly in every pair (mean {sg(st.mean(TH_JF))}). Refinement and the length penalty are negligible on "
   "every benchmark (largest absolute difference "
   f"{max(abs(r[b]['delta']) for r in (RF, LP) for b in ('c4', 'bea_dev', 'conll')):.3f}); refinement only adds latency "
   "(Section <<S:cost>>).")
cs42, cs1337 = CLS[("cleaning_nogate_stage1", "conll")], CLS[("cleaning_clean_s1337", "conll")]
PB(f"The filtering policy of Table <<T:sample>> leaves the in-domain score ({sg(CLC4['cleaning_pooled']['delta'])}, 90% "
   f"interval {ci(CLC4['cleaning_pooled'], 'ci90')}) and BEA-2019 development ({sg(CLL['bea_dev']['delta'])}, "
   f"{ci(CLL['bea_dev'], 'ci90')}) unchanged within the margin; on CoNLL-2014 it is inconclusive and seed-dependent "
   f"({sg(cs1337['delta'])} with stage-1 seed 1337, {sg(cs42['delta'])} with seed 42). Because the policy removes "
   "source-side and alignment problems as well as reference noise, and changes which pairs are drawn, this shows only "
   "that this composite policy brings no detectable benefit at our budget, not that reference quality is unimportant.")
assert all(x < 0 for b in ("c4", "bea_dev", "conll") for x in ID2[b]["per_pair_delta"]) and len(ID2["c4"]["per_pair_delta"]) == 4
assert len(TH["c4"]["per_pair_delta"]) == 6 and all(x < 0 for x in TH_JF)

H2("Learner-data budget", "budget")
BUD = [("25%", WI["budget_train_pairs"]["25"], "E/final_curve25"), ("50%", WI["budget_train_pairs"]["50"], "E/final_curve50"),
       ("100%", WI["train_pairs"], "E/final_e42")]
BH = holm([r["p_two_sided"] for r in (B50["c4"], B50["bea_dev"], B50["conll"], B100["c4"], B100["bea_dev"], B100["conll"])])
TAB("budget", "Learner-data budget: " + F05 + " (JFLEG: GLEU) of the primary configuration (seed 42) trained on nested "
    "subsets of the training essays, stratified by band", [800, 857, 750, 750, 750, 750], [
    ["Learner data", "Pairs", "Dev", "CoNLL", "JFLEG", "C4"]] +
    [[lab, f"{n:,}"] + [f"{F(sy, b):.3f}" for b in ("bea_dev", "conll", "jfleg", "c4")] for lab, n, sy in BUD])
PF(f"Table <<T:budget>> trains the primary configuration on 25%, 50% and 100% of the training essays. Halving the learner "
   f"data from {WI['train_pairs']:,} to {WI['budget_train_pairs']['50']:,} pairs lowers BEA-2019 development {F05} by "
   f"{abs(B100['bea_dev']['delta']):.3f} (95% CI {ci(B100['bea_dev'])}) and halving it again costs a further "
   f"{abs(B50['bea_dev']['delta']):.3f} ({ci(B50['bea_dev'])}); on CoNLL-2014 the first halving costs "
   f"{abs(B100['conll']['delta']):.3f} (Holm-adjusted p = {BH[5]:.2f}, inconclusive) and the second "
   f"{abs(B50['conll']['delta']):.3f} (p = {BH[2]:.3f}). The gains diminish with each doubling but "
   "have not levelled off at the full budget, and the in-domain score falls as more learner data is used, so forgetting "
   "grows with the amount of learner data. The curve uses one seed; at the full budget the seed standard deviation is "
   f"{PRIM['bea_dev'][1]:.3f} on BEA-2019 development. It is consistent with, but does not establish, the amount of human "
   "data as the cause of the gec-t5 advantage of Section <<S:main>>.")
assert F("E/final_curve25", "c4") > F("E/final_curve50", "c4") > F("E/final_e42", "c4")
assert B50["bea_dev"]["delta"] > B100["bea_dev"]["delta"] > 0 and BH[0] < 0.05 and BH[1] < 0.05 and BH[3] < 0.05 and BH[4] < 0.05

H2("Over-correction and the threshold", "dial")
EV42 = RE[1]["eval|final_e42"]
DIAL = [("none", "E/final_e42_no_threshold", "ref1_taunone"), ("0", "E/final_e42", "ref1_tau0.0"),
        ("0.05", "E/final_e42_tau0.05", "ref1_tau0.05"), ("0.1", "E/final_e42_tau0.1", "ref1_tau0.1"),
        ("0.2", "E/final_e42_tau0.2", "ref1_tau0.2")]
TAB("dial", "Effect of the threshold τ on the primary system (seed 42): BEA-2019 development P, R and " + F05 + "; "
    "CoNLL-2014 " + F05 + "; in-domain shares of unchanged and harmful outputs (%)", [560, 682, 682, 682, 682, 682, 687], [
    ["τ", "P", "R", F05, "CoNLL", "Unch.", "Harm"]] +
    [[t, f"{S[sy]['bea_dev']['P']:.3f}", f"{S[sy]['bea_dev']['R']:.3f}", f"{F(sy, 'bea_dev'):.3f}", f"{F(sy, 'conll'):.3f}",
      f"{EV42['c4|' + k]['unchanged']:.1f}", f"{EV42['c4|' + k]['harm']:.1f}"] for t, sy, k in DIAL])
best = {b: max(DIAL, key=lambda x: F(x[1], b))[0] for b in ("bea_dev", "conll", "c4")}
assert set(best.values()) == {"0"}
PF(f"Raising τ trades recall for precision and safety (Table <<T:dial>>): from no threshold to τ = 0.2 the "
   f"in-domain share of harmful outputs falls from {EV42['c4|ref1_taunone']['harm']:.1f}% to "
   f"{EV42['c4|ref1_tau0.2']['harm']:.1f}% and the share of unchanged outputs rises from "
   f"{EV42['c4|ref1_taunone']['unchanged']:.1f}% to {EV42['c4|ref1_tau0.2']['unchanged']:.1f}%, while BEA-2019 "
   "development precision rises only up to τ = 0.1. The holdout-selected τ = 0 is the " + F05 + " optimum on all "
   "three " + F05 + " endpoints; larger values suit applications that must avoid wrong edits. Here an edited output is "
   "harmful when its word-level edit distance to the single in-domain reference exceeds that of the input, and lateral "
   "when the two distances are equal.")

H2("Human evaluation", "human")
POP = HE_MAN["population_by_stratum"]; NPOP = HE_MAN["population_edited_sentences"]; SAMP = HE_MAN["sample_by_stratum"]
assert NPOP == sum(POP.values()) and sum(SAMP.values()) == 175
PF("Reference-based harm is a weak proxy, because a valid edit that differs from the single reference counts as harmful. "
   f"Both authors therefore judged a stratified random sample of {sum(SAMP.values())} of the {NPOP:,} BEA-2019 "
   f"development sentences that the primary system (seed 42) edits: {SAMP['perfect']} whose output equals the reference "
   f"and {SAMP['improved']} each of improved, lateral and harmful outputs, defined as in Section <<S:dial>> against the "
   "first reference annotation. Each item showed the original and the output in random order, with no reference and no indication of which was which, "
   "and with the differing words highlighted in both. For each item the annotators chose which sentence is better "
   "English (either one, or neither) and whether the meaning is the same, independently and before the key was opened. "
   f"Population estimates weight each stratum by its share of the {NPOP:,} edited sentences.")
order = ["perfect", "improved", "lateral", "harm"]; names = {"perfect": "Equals reference", "improved": "Improved",
                                                              "lateral": "Lateral", "harm": "Harmful"}
if HE:
    PS = HE["per_stratum"]; PE = HE["population_estimate_all_edited"]; MAg = HE["agreement"]["meaning"]
    hrows = [[names[s], f"{POP[s]:,}", f"{SAMP[s]}", f"{100 * PS[s]['mean']['better']:.0f}", f"{100 * PS[s]['mean']['same']:.0f}",
              f"{100 * PS[s]['mean']['worse']:.0f}", f"{100 * PS[s]['meaning_kept_mean']:.0f}"] for s in order]
    hrows.append(["**Weighted, all edits**", f"{NPOP:,}", "175", f"{100 * PE['better']:.0f}", f"{100 * PE['same']:.0f}",
                  f"{100 * PE['worse']:.0f}", f"{100 * PE['meaning_kept']:.0f}"])
    PCI = HE["population_estimate_ci95"]; HCI = HE["automatic_harm_judged_not_worse_ci95"]
    pc = lambda x: f"{100 * x:.0f}%"; rng_ = lambda k: f"{100 * PCI[k][0]:.0f} to {100 * PCI[k][1]:.0f}%"
    PB(f"The annotators agreed on {HEA['percent_agreement']:.0f}% of the verdicts (Cohen's kappa {HEA['cohen_kappa']:.2f}, "
       f"95% CI {HEA['kappa_ci95'][0]:.2f} to {HEA['kappa_ci95'][1]:.2f}; kappa is lower than raw agreement because most "
       f"outputs are judged better, which raises chance agreement) and on {MAg['percent_agreement']:.0f}% of the meaning "
       f"judgments. Weighted over all edited sentences, an estimated {pc(PE['better'])} of the outputs are better than the "
       f"input (95% CI {rng_('better')}), {pc(PE['same'])} neither better nor worse and {pc(PE['worse'])} worse "
       f"({rng_('worse')}), and {pc(PE['meaning_kept'])} keep the meaning (Table <<T:human>>). Reference distance "
       f"therefore overstates damage on learner text: the automatic harm class covers "
       f"{100 * HE['automatic_harm_share_of_edited']:.0f}% of the edited outputs, but "
       f"{100 * HE['automatic_harm_judged_not_worse']:.0f}% of them ({100 * HCI[0]:.0f} to {100 * HCI[1]:.0f}%) are "
       "judged no worse than the input. Intervals reflect the sampling of the 175 items only.")
    assert PE["worse"] < PE["same"] < PE["better"] and HE["automatic_harm_judged_not_worse"] > 0.5
else:
    hrows = [[names[s], f"{POP[s]:,}", f"{SAMP[s]}", "[[…]]", "[[…]]", "[[…]]", "[[…]]"] for s in order]
    hrows.append(["**Weighted, all edits**", f"{NPOP:,}", "175", "[[…]]", "[[…]]", "[[…]]", "[[…]]"])
    PB("[[Human-evaluation results pending: agreement (Cohen's kappa with 95% CI), per-stratum verdicts and the "
       "frequency-weighted estimates are filled in from human_eval_round3/human_eval_results.json.]]")
TAB("human", "Blinded human judgments of edited BEA-2019 development outputs (primary system, seed 42): verdict on the "
    "output relative to the input (%, mean of the two annotators) and share with meaning kept. Weighted row: strata "
    "weighted by their share of all edited sentences", [1157, 560, 380, 610, 610, 610, 730], [
    ["Stratum", "Edited", "n", "Better", "Same", "Worse", "Meaning kept"]] + hrows)

H2("Proficiency levels and error types", "levels")
LV = A["bea_dev_by_level_final_mean"]
lrows = [["Level", "n (dev)", "Dev " + F05, "Test P", "Test R", "Test " + F05]]
for name, lv, n in LEVELS:
    fs = [TL[s][name]["F05"] for s in SEEDS]
    lrows.append([name, f"{n:,}", f"{LV[lv]['F05_mean']:.3f}", pct(round(st.mean(TL[s][name]["P"] for s in SEEDS), 2)),
                  pct(round(st.mean(TL[s][name]["R"] for s in SEEDS), 2)), f"{pct(round(st.mean(fs), 2))} ± {pct(round(st.stdev(fs), 2))}"])
lrows.append(["All", "4,384", f"{PRIM['bea_dev'][0]:.3f}", pct(round(st.mean(TEST_P), 2)), pct(round(st.mean(TEST_R), 2)), TEST_STR])
TAB("levels", "Results by CEFR level: BEA-2019 development " + F05 + " and blind-test P, R and " + F05 + " (means over "
    "three seeds; test " + F05 + " ± standard deviation)", [1100, 560, 640, 600, 600, 1157], lrows)
srt = sorted(STABLE, key=lambda k: -STABLE[k]["mean"]); NAMES = {"PUNCT": "punctuation", "VERB:SVA": "subject–verb agreement",
    "VERB:FORM": "verb form", "NOUN:NUM": "noun number", "DET": "determiners", "WO": "word order", "VERB": "verb choice",
    "ADV": "adverbs", "OTHER": "other edits", "NOUN": "noun choice", "PREP": "prepositions", "SPELL": "spelling",
    "ORTH": "orthography", "PRON": "pronouns", "MORPH": "morphology", "NOUN:INFL": "noun inflection", "PART": "particles",
    "VERB:TENSE": "tense", "NOUN:POSS": "possessives", "CONTR": "contractions", "VERB:INFL": "verb inflection",
    "ADJ": "adjectives", "CONJ": "conjunctions", "ADJ:FORM": "adjective form"}
strong = [k for k in ("PUNCT", "VERB:SVA", "VERB:FORM", "NOUN:NUM", "DET") if k in STABLE]
weak = [k for k in ("WO", "VERB", "ADV", "OTHER", "NOUN") if k in STABLE]
assert len(strong) == 5 and len(weak) == 5, (strong, weak)
assert min(STABLE[k]["mean"] for k in strong) > max(STABLE[k]["mean"] for k in weak)
lvA, lvN = [st.mean(TL[s][n]["R"] for s in SEEDS) for n in ("A (beginner)", "N (native)")]
PF(f"On the blind test the system scores lowest on beginner (A) essays and highest on advanced and native essays "
   f"(Table <<T:levels>>), with recall rising from {pct(round(lvA, 2))} (A) to {pct(round(lvN, 2))} (N), whereas on the "
   "development set the A and B levels score highest; because the ordering differs between splits we draw no "
   "conclusion about proficiency. By ERRANT error type (blind test, mean of three seeds, types whose seed standard "
   "deviation is at most 0.025), the system is strongest on " +
   ", ".join(f"{NAMES[k]} ({STABLE[k]['mean'] / 100:.2f})" for k in strong[:-1]) +
   f" and {NAMES[strong[-1]]} ({STABLE[strong[-1]]['mean'] / 100:.2f}), which are mostly local edits, and weakest on " +
   ", ".join(f"{NAMES[k]} ({STABLE[k]['mean'] / 100:.2f})" for k in weak[:-1]) +
   f" and {NAMES[weak[-1]]} ({STABLE[weak[-1]]['mean'] / 100:.2f}), which typically require meaning-level or "
   "reordering decisions.")

H2("Inference and training cost", "cost")
LP_, LG, L5, LB, LC, LQ = (LAT[k] for k in ("ours, final pipeline (beam 5 + refinement + threshold)", "ours, greedy single pass",
                                             "gotutiyan/gec-t5-small-clang8 (beam 5)", "gotutiyan/gec-t5-base-clang8 (beam 5)",
                                             "grammarly/coedit-large (greedy)", "Qwen/Qwen2.5-1.5B-Instruct, 4-shot greedy"))
TAB("cost", "Inference cost on one NVIDIA T4 GPU (first 300 CoNLL-2014 sentences): batch-1 latency (median and 90th "
    "percentile, ms), batch-32 throughput (sentences/s) and peak memory (MB). Ours: primary checkpoint, seed 42",
    [1557, 600, 600, 600, 650, 650], [["System", "Params", "Median", "p90", "Sent./s", "MB"]] +
    [[lab, prm, f"{r['lat_median_ms']:.0f}", f"{r['lat_p90_ms']:.0f}", f"{r['throughput_sent_s_b32']:.1f}", f"{r['peak_mem_MB']:,.0f}"]
     for lab, prm, r in (("Ours, full pipeline", "77M", LP_), ("Ours, greedy, one pass", "77M", LG), ("gec-t5-small (beam 5)", "77M", L5),
                         ("gec-t5-base (beam 5)", "248M", LB), ("CoEdIT-large (greedy)", "783M", LC), ("Qwen2.5-1.5B, 4-shot", "1.5B", LQ))])
PF(f"Table <<T:cost>> compares inference cost on one GPU, measured on the primary checkpoint. Our full pipeline costs "
   f"{LP_['lat_median_ms'] / L5['lat_median_ms']:.1f} times the median latency of gec-t5-small with beam search, because "
   "refinement and the confidence margin add model passes; a single greedy pass is the cheapest configuration "
   f"({LG['lat_median_ms']:.0f} ms, {LG['throughput_sent_s_b32']:.1f} sentences/s, {LG['peak_mem_MB']:.0f} MB). CoEdIT-large "
   f"needs {LC['peak_mem_MB'] / LP_['peak_mem_MB']:.1f} times the memory of our full pipeline and "
   f"{LC['lat_median_ms'] / LP_['lat_median_ms']:.1f} times its latency, and the 1.5B language model "
   f"{LQ['peak_mem_MB'] / LP_['peak_mem_MB']:.1f} and {LQ['lat_median_ms'] / LP_['lat_median_ms']:.1f} times. Because "
   "refinement brings no measurable accuracy (Section <<S:abl>>), a deployment can drop it while keeping the threshold. "
   f"Training took {S1['train_seconds'] / 3600:.1f} h for stage 1 and {min(S2_SECONDS) / 60:.0f} to "
   f"{max(S2_SECONDS) / 60:.0f} min per stage-2 run on two T4 GPUs, so the study is reproducible on free-tier hardware.")

# ---------------------------------------------------------------- 6. Validity and reproducibility
H1("Validity checks and reproducibility", "valid")
PF("While building this study we found and corrected four problems in our own earlier pipeline (Table <<T:defects>>). "
   "All results in Section <<S:res>> come from corrected runs, and experiments that existed only in the earlier "
   "pipeline are not reported. The ratio of training to validation loss exposes the accumulation defect; in every "
   f"reported run it lies between {min(RATIOS):.2f} and {max(RATIOS):.2f}. It is logged by an automatic check that stops "
   "any run whose ratio exceeds 2, except for the three stage-1 runs N′, U and U′, trained before the check "
   "existed, for which it is computed from their logged training and validation losses. Table <<T:hyper>> lists the training and decoding settings.")
TAB("defects", "Problems found in the earlier pipeline and their handling in this study", [1500, 1357, 1800], [
    ["Problem", "Affected", "Handling here"],
    ["Training loss not divided by the gradient-accumulation steps (earlier framework version); clipping active at almost every step",
     "all earlier training runs", "all reported runs retrained with corrected normalisation; loss-ratio check"],
    ["Stage-2 loader read the per-band and the combined W&I+LOCNESS files, so every pair appeared twice",
     "earlier stage-2 runs", "A, B and C files only (34,308 pairs)"],
    [f"Stage-2 holdout drawn by source sentence: {DOV['holdout_pairs_whose_essay_also_in_training']:,} of its "
     f"{DOV['round2_holdout_pairs']:,} pairs ({100 * DOV['share']:.1f}%) came from essays that also supplied training pairs",
     "checkpoint and τ selection", "essay-level split (Section <<S:wi>>)"],
    [f"Unfiltered sample split independently: {OVA['c4_test_subset_5k_in_noisy_train']:,} of the 5,000 in-domain test "
     "pairs are in its training split", "filtering contrast, in domain", f"scored on the {UNCONT['pairs_kept']} "
     "uncontaminated pairs"]], sz=15)
TAB("hyper", "Training and decoding settings", [1250, 1080, 1250, 1077], [
    ["Setting", "Value", "Setting", "Value"],
    ["Backbone", "google/flan-t5-small, 77M", "Max input / target", "128 / 128 tokens"],
    ["Stage-1 data", "1,150,000 pairs + 172,500 identity", "Precision / hardware", "fp16 / 2 × T4"],
    ["Batch per device / accum. / GPUs", "16 / 4 / 2 (stage 2: 16 / 2 / 2)", "Stage-2 data / epochs / lr / steps",
     f"{WI['train_pairs']:,} / 3 / 1e-4 / {S2['steps']:,}"],
    ["Global batch, stage 1 / 2", "128 / 64", "Selection, stage 1 / 2", "val. loss, every 2,500 steps / holdout loss, each epoch"],
    ["Stage-1 steps", f"{S1['steps']:,} (8,985 without identity pairs)", "Beam / no-repeat n-gram / length penalty", "5 / 3 / 0.7"],
    ["Learning rate / schedule", "3e-4 (stage 2: 1e-4), cosine, 6% warm-up", "Threshold grid / selected", "none, 0 to 0.5 (9 values) / 0"],
    ["Label smoothing / weight decay", "0.1 / 0.01", "Refinement acceptance", "at most 3 changed words"],
    ["Source word dropout", "0.05 (edit pairs only)", "Seeds", "stage 1: 42, 1337; stage 2: 42, 1337, 2024"]], left_cols=4, sz=15)
assert RE[1]["train|final_e42"]["warmup_steps"] == round(0.06 * S2["steps"]) or abs(RE[1]["train|final_e42"]["warmup_steps"] - 0.06 * S2["steps"]) < 1
PB("**Release.** The notebooks, configurations and split indices (including the essay of every W&I+LOCNESS training "
   "sentence), every system output scored in this paper with its per-sentence edit counts, a verification script that "
   "re-scores all outputs with the official scorers, and a run manifest that links every table cell to its runs, "
   "configuration, checkpoint, output files and scorer versions are available at "
   "https://github.com/MemoXk/compact-gec-flan-t5 (archived on Zenodo, https://doi.org/10.5281/zenodo.23326147); the "
   "checkpoints of all runs in Table <<T:runs>> except the stage-1 models of N′, U and U′ are archived at "
   "https://doi.org/10.5281/zenodo.23328228. All scores except the blind-test ones, "
   "which only the shared-task server can compute, were recomputed from the released outputs with ERRANT 3.0.2, spaCy 3.8 (en_core_web_sm 3.8.0), the official M² scorer ported "
   "to Python 3 and the JFLEG GLEU script, and match the values logged during training. Baselines: Qwen2.5-1.5B-Instruct "
   "(revision 989aa79) in float16 with the system prompt “You are a grammatical error correction system. Return ONLY "
   "the corrected sentence. Make the minimum number of edits. If the sentence is already correct, repeat it "
   "unchanged.”, greedy decoding of at most 96 new tokens and four exemplars drawn with seed 2024 from the 20,734 "
   "W&I+LOCNESS training pairs of 8 to 20 tokens with one to three edits; CoEdIT-large (revision "
   f"{COEDIT_REV[:7]}) with the prompt “Fix grammatical errors in this sentence: ” and greedy decoding of at "
   "most 256 tokens; gec-t5 models with their authors' generate function (beam 5, maximum length 128, length penalty "
   "1.0, no prefix).")
assert RB["qwen"]["revision"].startswith("989aa79") and RB["qwen"]["exemplar_pool_size"] == 20734

# ---------------------------------------------------------------- 7. Conclusion
H1("Conclusion", "concl")
PF("We asked which design choices matter for a compact FLAN-T5-small corrector trained on synthetic data and about "
   f"{WI['train_pairs'] / 1000:.0f},000 annotated learner sentences. In one corrected pipeline with three seeds per arm, "
   "essay-level selection and paired tests, the learner-data stage dominates every other factor "
   f"({sg(HS['bea_dev']['delta'])} {F05} on BEA-2019 development), a confidence threshold adds a small consistent gain, "
   "and identity pairs, refinement, length-penalty tuning and the tested filtering policy bring no benefit; halving the "
   f"learner data costs {abs(B100['bea_dev']['delta']):.3f}. The primary system reaches {F05} = {TEST_STR} on the blind "
   f"BEA-2019 test set and {PRIM['conll'][0]:.3f} on CoNLL-2014, runs on one commodity GPU, outperforms a "
   "1.5B-parameter instruction-tuned language model, does not differ significantly from a ten-times-larger "
   "instruction-tuned editor on BEA-2019 development, and trails a same-size model trained on "
   f"{HUMAN_RATIO:.0f} times more human data.")
PB("**Limitations and future work.** Our conclusions concern one compact backbone, one synthetic corpus and one learner "
   "corpus, in English. Stage-2 contrasts use three stage-2 seeds and two stage-1 seeds, so their intervals describe "
   "test-set variability around seed-averaged effects, and the budget curve uses one seed. The identity-pair, filtering "
   "and threshold findings are specific to the policies and budget tested. Both annotators are authors; the blinded "
   "design limits but does not remove this source of bias, and 175 items give wide intervals for rare outcomes. The "
   "CoEdIT and gec-t5 scores may benefit from training or model selection on the evaluation corpora. The clearest "
   "next steps are more learner data within the same pipeline, such as cLang-8 [@rothe2021] or NUCLE [@dahlmeier2013], "
   "an evaluation on a second learner domain, and larger backbones.")

# ================================================================================================ REFERENCES
def markup_of(el):
    merged = []
    for r in el.iter(qn("w:r")):
        t = "".join(x.text or "" for x in r.iter(qn("w:t")))
        if not t: continue
        rp = r.find(qn("w:rPr")); i_ = rp is not None and rp.find(qn("w:i")) is not None
        if merged and merged[-1][1] == i_: merged[-1] = (merged[-1][0] + t, i_)
        else: merged.append((t, i_))
    return "".join(f"*{t}*" if i_ else t for t, i_ in merged)
OLD_REF = {"bryant2023": 123, "junczys2018": 124, "lichtarge2019": 125, "grundkiewicz2019": 126, "kiyono2019": 127,
           "stahlberg2021": 128, "rothe2021": 129, "vaswani2017": 130, "raffel2020": 131, "chung2024": 132,
           "omelianchuk2020": 136, "raheja2023": 137, "coyne2023": 138, "napoles2015": 139, "bryant2017": 140, "ng2014": 141,
           "bryant2019": 142, "napoles2017": 143, "dahlmeier2013": 144, "flant5card": 145, "qwen2024": 146}
REFS = {k: markup_of(ELS[i]) for k, i in OLD_REF.items()}
for k, frag in (("bryant2023", "Survey of the State"), ("junczys2018", "Low-Resource Machine"), ("rothe2021", "Simple Recipe"),
                ("raheja2023", "CoEdIT"), ("coyne2023", "GPT-3.5"), ("napoles2015", "Ground Truth"), ("ng2014", "CoNLL-2014"),
                ("bryant2019", "BEA-2019"), ("napoles2017", "JFLEG"), ("dahlmeier2013", "NUS Corpus"), ("qwen2024", "Qwen2.5"),
                ("stahlberg2021", "Tagged Corruption"), ("chung2024", "Instruction-Finetuned"), ("omelianchuk2020", "GECToR"),
                ("bryant2017", "Error Types"), ("flant5card", "FLAN-T5-small")):
    assert frag in REFS[k], (k, REFS[k][:80])
NB = "\u00a0"
REFS.update({
    "omelianchuk2024": f"K.{NB}Omelianchuk, A.{NB}Liubonko, O.{NB}Skurzhanskyi, A.{NB}Chernodub, O.{NB}Korniienko, and I.{NB}Samokhin, "
                       "\u201cPillars of Grammatical Error Correction: Comprehensive Inspection Of Contemporary Approaches In The Era of "
                       "Large Language Models\u201d, In: *Proc. of the 19th Workshop on Innovative Use of NLP for Building Educational "
                       f"Applications (BEA)*, pp.{NB}17\u201333, 2024.",
    "ye2023": f"J.{NB}Ye, Y.{NB}Li, Y.{NB}Li, and H.-T.{NB}Zheng, \u201cMixEdit: Revisiting Data Augmentation and Beyond for Grammatical "
              f"Error Correction\u201d, In: *Findings of the Association for Computational Linguistics: EMNLP*, pp.{NB}10161\u201310175, 2023.",
    "naplava2019": f"J.{NB}N\u00e1plava and M.{NB}Straka, \u201cGrammatical Error Correction in Low-Resource Scenarios\u201d, In: *Proc. of "
                   f"the 5th Workshop on Noisy User-generated Text (W-NUT)*, pp.{NB}346\u2013356, 2019.",
    "davis2024": f"C.{NB}Davis, A.{NB}Caines, \u00d8.{NB}E.{NB}Andersen, S.{NB}Taslimipoor, H.{NB}Yannakoudakis, Z.{NB}Yuan, C.{NB}Bryant, "
                 f"M.{NB}Rei, and P.{NB}Buttery, \u201cPrompting Open-Source and Commercial Language Models for Grammatical Error "
                 "Correction of English Learner Text\u201d, In: *Findings of the Association for Computational Linguistics: ACL*, "
                 f"pp.{NB}11952\u201311967, 2024.",
    "dahlmeier2012": f"D.{NB}Dahlmeier and H.{NB}T.{NB}Ng, \u201cBetter Evaluation for Grammatical Error Correction\u201d, In: *Proc. of "
                     f"NAACL-HLT*, pp.{NB}568\u2013572, 2012.",
    "napoles2016": f"C.{NB}Napoles, K.{NB}Sakaguchi, M.{NB}Post, and J.{NB}Tetreault, \u201cGLEU Without Tuning\u201d, arXiv:1605.02592, 2016.",
    "kobayashi2024": f"M.{NB}Kobayashi, M.{NB}Mita, and M.{NB}Komachi, \u201cRevisiting Meta-evaluation for Grammatical Error "
                     f"Correction\u201d, *Transactions of the Association for Computational Linguistics*, Vol.{NB}12, pp.{NB}837\u2013855, 2024.",
    "koehn2004": f"P.{NB}Koehn, \u201cStatistical Significance Tests for Machine Translation Evaluation\u201d, In: *Proc. of EMNLP*, "
                 f"pp.{NB}388\u2013395, 2004.",
    "schuirmann1987": f"D.{NB}J.{NB}Schuirmann, \u201cA Comparison of the Two One-Sided Tests Procedure and the Power Approach for "
                      "Assessing the Equivalence of Average Bioavailability\u201d, *Journal of Pharmacokinetics and Biopharmaceutics*, "
                      f"Vol.{NB}15, No.{NB}6, pp.{NB}657\u2013680, 1987.",
    "gect5card": "gotutiyan, \u201cgec-t5-small-clang8 and gec-t5-base-clang8 model cards and gec-t5 code\u201d, huggingface.co/gotutiyan "
                 "and github.com/gotutiyan/gec-t5, accessed 2026.",
})

# ================================================================================================ RENDER
# layout: Figure 1 is a full-width block; FIG_BEFORE names the heading it is placed before (page-filling choice)
FIG_BEFORE = os.environ.get("FIG_BEFORE", "data")      # chosen by rendering six positions: fills pages 2 to 4 completely
_fi = next(i for i, b in enumerate(BLOCKS) if b[0] == "fig"); _fb = BLOCKS.pop(_fi)
BLOCKS.insert(next(i for i, b in enumerate(BLOCKS) if b[0] in ("h1", "h2") and b[2] == FIG_BEFORE), _fb)
num = {"S": {}, "T": {}, "F": {}, "E": {}}; h1 = h2 = 0
for b in BLOCKS:
    if b[0] == "h1": h1 += 1; h2 = 0; num["S"][b[2]] = str(h1)
    elif b[0] == "h2": h2 += 1; num["S"][b[2]] = f"{h1}.{h2}"
    elif b[0] == "tab": num["T"][b[1]] = len(num["T"]) + 1
    elif b[0] == "fig": num["F"][b[1]] = len(num["F"]) + 1
    elif b[0] == "eq": num["E"][b[2]] = len(num["E"]) + 1
def strings_of(b):
    if b[0] in ("h1", "h2", "pf", "pb"): return [b[1]]
    if b[0] == "tab": return [b[2]] + [c for r in b[4] for c in r]
    if b[0] == "fig": return [b[2]]
    return []
CITE = []
for b in BLOCKS:
    for s_ in strings_of(b):
        for m in re.finditer(r"\[@([^\]]+)\]", s_):
            for k in (x.strip().lstrip("@") for x in m.group(1).split(";")):
                assert k in REFS, f"no reference entry for {k}"
                if k not in CITE: CITE.append(k)
unused = sorted(set(REFS) - set(CITE)); print("unused reference entries:", unused)
def resolve(s_):
    s_ = re.sub(r"<<([TFES]):(\w+)>>", lambda m: str(num[m.group(1)][m.group(2)]), s_)
    return re.sub(r"\[@([^\]]+)\]", lambda m: "[" + ", ".join(str(n) for n in sorted(
        CITE.index(x.strip().lstrip("@")) + 1 for x in m.group(1).split(";"))) + "]", s_)
def set_text(el, markup):
    rpr = base_rpr(el)
    for ch in list(el):
        if ch.tag != qn("w:pPr"): el.remove(ch)
    for r in runs(markup, rpr): el.append(r)
set_text(ELS[1], TITLE); set_text(ELS[8], ABSTRACT); set_text(ELS[9], KEYWORDS)
# author contributions: both authors carried out the human evaluation (confirmed by the corresponding author, 7 Nov 2026)
_contrib = text_of(ELS[119]); assert text_of(ELS[118]) == "Author contributions" and "investigation, Nermin Negied;" in _contrib
set_text(ELS[119], _contrib.replace("investigation, Nermin Negied;", "investigation, Mohammed A. ElTaweel and Nermin Negied;"))
assert len(ABSTRACT.split()) <= 150 + (0 if HE else 5), len(ABSTRACT.split())
EQS = omml([b[1] for b in BLOCKS if b[0] == "eq"]); eq_i = 0
cur = ANCHOR
def put(*els):
    global cur
    for e in els: cur.addnext(e); cur = e
for b in BLOCKS:
    k = b[0]
    if k == "h1": put(para("h1", f"{num['S'][b[2]]}. {resolve(b[1])}"))
    elif k == "h2": put(para("h2", f"{num['S'][b[2]]} {resolve(b[1])}"))
    elif k in ("pf", "pb"): put(para(k, resolve(b[1])))
    elif k == "eq": put(eq_par(EQS[eq_i], num["E"][b[2]])); eq_i += 1
    elif k == "tab":
        _, key, cap, widths, rows, wide, lc, sz, hr = b
        assert sum(widths) <= (FULL_W if wide else COL_W), (key, sum(widths))
        c = para("tcap", f"Table {num['T'][key]}. {resolve(cap)}")
        tb = make_table(widths, [[resolve(x) for x in r] for r in rows], left_cols=lc, sz=sz, header_rows=hr)
        put(*( [section_end(False), c, tb, section_end(True)] if wide else [c, tb, para("empty", "")] ))
    elif k == "fig":
        _, key, cap, img = b
        fp = copy.deepcopy(TPL["fig"]); blip = fp.find(".//" + qn("a:blip"))
        part = d.part.related_parts[blip.get(qn("r:embed"))]
        im = Image.open(img).convert("RGB"); buf = io.BytesIO(); im.save(buf, "PNG", optimize=True); part._blob = buf.getvalue()
        cx = int(6.7 * 914400); cy = int(cx * im.size[1] / im.size[0])
        for e in list(fp.iter(qn("wp:extent"))) + list(fp.iter(qn("a:ext"))): e.set("cx", str(cx)); e.set("cy", str(cy))
        fc = para("figcap", f"Figure. {num['F'][key]} {resolve(cap)}"); fc.find(qn("w:pPr")).append(copy.deepcopy(ONECOL_SECT))
        put(section_end(False), fp, fc)
assert eq_i == len(EQS)
cur = REF_HEAD
for k in CITE: put(para("ref", REFS[k]))
d.save(str(OUT))
full = "\n".join(p.text for p in docx.Document(str(OUT)).paragraphs)
assert "<<" not in full and "[@" not in full and "**" not in full, "unresolved markup"
pending = len(re.findall(r"pending|\u2026", full)); print(f"wrote {OUT} | {len(num['T'])} tables, {len(num['F'])} figure, "
      f"{len(num['E'])} equations, {len(CITE)} references | abstract {len(ABSTRACT.split())} words | pending markers: {pending}")
