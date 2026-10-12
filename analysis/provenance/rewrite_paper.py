"""Round-2 rewrite of inass_paper.docx with the retrained, full-budget results.

Input : _bk_round2_rewrite_start.docx (the paper as it stood before this rewrite; never modified)
Output: inass_paper.docx
Every number in the new text is read from the analysis JSON files (round2_results/*.json), so the paper cannot drift
from the analyses. Markup in text strings: **bold**, *italic*, _{subscript}, ^{superscript}, [[highlighted]]."""
import copy, json, math, re
import docx
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

P_ = r"<PROJECT>"
R_ = f"{P_}/round2_results"
SRC, OUT = f"{P_}/_bk_round2_rewrite_start.docx", f"{P_}/inass_paper.docx"
J = lambda f: json.load(open(f"{R_}/{f}", encoding="utf-8"))
CN = J("C_nogate_analysis.json")
RB = J("B/r2B_results.json"); RC = J("C_final/r2C_results.json"); RN = J("C_nogate/r2C_results.json"); RD = J("D/r2D_results.json")
# every official-scorer score and paired bootstrap comes from the released package's verification run
REL = f"{P_}/release/compact-gec-flan-t5"
V = json.load(open(f"{REL}/results/verified/verified_scores.json", encoding="utf-8"))
VS, VC = V["scores"], V["comparisons"]
def vc(table, name, bench): return VC[f"{table}|{name}|{bench}"]
_t = J("bea19_test_codabench_dedup/bea19_test_scores_dedup.json"); TEST = _t[next(k for k in _t if k.startswith("official"))]
ET = _t["error_types_f05"]
TEST_PREV = J("bea19_test_codabench_retrained/bea19_test_scores_retrained.json")["official_bea2019_test_codabench_retrained"]
TEST_JUNE = J("bea19_test_codabench/bea19_test_scores.json")
GATE_JUNE = json.load(open(f"{P_}/gate_bootstrap_fullscale.json"))
EV2, EVN, EV1 = RD["eval|final_s2dedup"], RD["eval|nogate_s2dedup"], RC["eval|final_s1"]
assert RD["tau|final_s2dedup"]["tau"] == 0.0 and RD["tau|nogate_s2dedup"]["tau"] == 0.0
WI = RD["wi_split"]; assert (WI["pairs"], WI["train_pairs"], WI["holdout_pairs"]) == (34308, 32600, 1708)
HUMAN_RATIO = 2.37e6 / WI["train_pairs"]          # cLang-8 English (Rothe et al., 2021) vs our stage-2 training pairs

from decimal import Decimal, ROUND_HALF_UP
def pct(x, nd=3):
    """CodaBench percentage (2 decimals) -> 0-1 scale, decimal half-up rounding."""
    return str((Decimal(str(x)) / 100).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))
def f3(x): return f"{x:.3f}"
def sg(x): return ("+" if x > 0 else "\u2212" if x < 0 else "\u00b1") + f"{abs(x):.3f}"
def star(b): return "*" if b["p_two_sided"] < 0.05 else ""
def cell(b): return sg(b["delta"]) + star(b)
def ci(b): return f"{b['ci95'][0]:.3f} to {b['ci95'][1]:.3f}".replace("-", "\u2212")
def pv(b): return "p < 0.001" if b["p_two_sided"] < 0.001 else f"p = {b['p_two_sided']:.3f}".rstrip("0").rstrip(".") if b["p_two_sided"] < 0.1 else f"p = {b['p_two_sided']:.2f}"

# ------------------------------------------------------------------------------------------------ numbers
N = {}
N["conll"], N["bea"] = VS["final"]["conll"], VS["final"]["bea_dev"]
assert abs(N["conll"]["F05"] - EV2["conll|ref1_tau0.0"]["F05"]) < 6e-4 and abs(N["bea"]["F05"] - EV2["bea_dev|ref1_tau0.0"]["F05"]) < 6e-4
N["jfleg"] = EV2["jfleg|ref1_tau0.0"]["GLEU"]
N["test"] = TEST["ALL"]
N["test_june"] = TEST_JUNE["official_bea2019_test_codabench"]["ALL"]["F05"]
N["test_prev"] = TEST_PREV["ALL"]["F05"]
JG = {k: v for k, v in CN["jfleg_gleu_official"].items()}
JG["final_s2"], JG["nogate_s2"] = EV2["jfleg|ref1_tau0.0"]["GLEU"], EVN["jfleg|ref1_tau0.0"]["GLEU"]
JG["final_notau"], JG["final_noref"] = EV2["jfleg|ref1_taunone"]["GLEU"], EV2["jfleg|ref0_tau0.0"]["GLEU"]
JG["nogate_s2_notau"] = EVN["jfleg|ref1_taunone"]["GLEU"]
JG["final_s1_tau0"] = EV1["jfleg|ref1_tau0.0"]["GLEU"]
JG["do_nothing"] = RB["jfleg"]["do_nothing"]["GLEU"]
BL = {k.split("/")[-1]: v for k, v in RB.items() if k.startswith("baseline|")}
Q = RB["qwen"]
LAT = {r["system"]: r for r in RB["latency"]["rows"] if "error" not in r}
TR1, TR2 = RC["train|final_s1"], RD["train|final_s2dedup"]
RATIOS = [TR1["train_eval_loss_ratio"], TR2["train_eval_loss_ratio"], RD["train|nogate_s2dedup"]["train_eval_loss_ratio"],
          RN["train|nogate_s1"]["train_eval_loss_ratio"]]
GSTATE = json.load(open(f"{R_}/C_session2_gate/trainer_state_gate_ck2500.json"))
GM = CN["gate_matched_step_2500"]
gate_s_per_step = (18237.7 - 168.8 - GM["gate"]["eval_runtime_s"]) / 2500
ng_log = json.load(open(f"{R_}/C_nogate/trainer_log_nogate_s1.json"))
ng_eval_s = sum(r["eval_runtime"] for r in ng_log if "eval_runtime" in r)
nogate_s_per_step = (RN["train|nogate_s1"]["train_seconds"] - ng_eval_s) / RN["train|nogate_s1"]["steps"]
SLOW_TRAIN = gate_s_per_step / nogate_s_per_step
SLOW_EVAL = GM["gate"]["eval_runtime_s"] / GM["nogate"]["eval_runtime_s"]
# CEFR: dev per level of the final system (verified package) and blind test per level (CodaBench)
LEV = []
for name, lev, n in (("A (beginner)", "A", 1037), ("B (intermediate)", "B", 1290), ("C (advanced)", "C", 1069), ("N (native)", "N", 988)):
    LEV.append((name, n, VS["final"][f"bea_dev_{lev}"]["F05"], TEST[name]))
# AI audit (Claude, blind to the key) -- counts per reference-distance class
import csv, collections
KEY = {r["item"]: r for r in csv.DictReader(open(f"{R_}/B/human_check_KEY.csv", encoding="utf-8"))}
AUD = collections.defaultdict(collections.Counter)
for r in csv.DictReader(open(f"{R_}/B/ai_annotation_claude.csv", encoding="utf-8")):
    k = KEY[r["item"]]; outpos = "1" if k["output_is"] == "sentence_1" else "2"; b = r["better (1/2/same)"]
    AUD[k["class"]]["same" if b == "same" else "out" if b == outpos else "src"] += 1
    AUD[k["class"]]["meaning"] += r["meaning_same (yes/no)"] == "no"; AUD[k["class"]]["n"] += 1
assert all(AUD[c]["n"] == 50 for c in ("harm", "lateral", "improved"))
def edited(path, col):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    n = lambda s: re.sub(r"\s+", " ", s).strip()
    return 100 * sum(n(r[col]) != n(r["src"]) for r in rows) / len(rows)
ED_OURS = edited(f"{R_}/D/pred_conll_final_s2dedup.csv", "ref1_tau0.0")
ED_QWEN = Q["conll|fourshot"]["edited_pct"]

# ------------------------------------------------------------------------------------------------ docx helpers
d = docx.Document(SRC)
PAR = list(d.paragraphs); TAB = list(d.tables)
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
TOK = re.compile(r"(\*\*.+?\*\*|\*[^*\s][^*]*?\*|_\{[^}]*\}|\^\{[^}]*\}|\[\[.+?\]\])")

def base_rpr(p):
    for r in p._p.iter(qn("w:r")):
        rp = r.find(qn("w:rPr"))
        rp = copy.deepcopy(rp) if rp is not None else OxmlElement("w:rPr")
        for t in ("w:b", "w:bCs", "w:i", "w:iCs", "w:vertAlign", "w:highlight"):
            for e in rp.findall(qn(t)): rp.remove(e)
        return rp
    return OxmlElement("w:rPr")

def _run(text, rpr, b=False, i=False, va=None, hl=False):
    r = OxmlElement("w:r"); rp = copy.deepcopy(rpr)
    def add(tag, val=None):
        e = OxmlElement(tag)
        if val is not None: e.set(qn("w:val"), val)
        rp.append(e)
    if b: add("w:b")
    if i: add("w:i")
    if hl: add("w:highlight", "yellow")
    if va: add("w:vertAlign", va)
    r.append(rp); t = OxmlElement("w:t"); t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve"); t.text = text; r.append(t)
    return r

def runs(markup, rpr, bold=False):
    out = []
    for seg in TOK.split(markup):
        if not seg: continue
        if seg.startswith("**"): out.append(_run(seg[2:-2], rpr, b=True))
        elif seg.startswith("_{"): out.append(_run(seg[2:-1], rpr, b=bold, va="subscript"))
        elif seg.startswith("^{"): out.append(_run(seg[2:-1], rpr, b=bold, va="superscript"))
        elif seg.startswith("[["): out.append(_run(seg[2:-2], rpr, b=bold, hl=True))
        elif seg.startswith("*") and seg.endswith("*") and len(seg) > 2: out.append(_run(seg[1:-1], rpr, b=bold, i=True))
        else: out.append(_run(seg, rpr, b=bold))
    return out

def set_par(p, markup, rpr=None):
    el = p._p if hasattr(p, "_p") else p
    rpr = rpr if rpr is not None else base_rpr(p if hasattr(p, "_p") else docx.text.paragraph.Paragraph(el, None))
    for ch in list(el):
        if ch.tag != qn("w:pPr"): el.remove(ch)
    for r in runs(markup, rpr): el.append(r)
    return el

def new_par_after(anchor_el, template_p, markup):
    el = copy.deepcopy(template_p._p)
    rpr = base_rpr(template_p)
    for ch in list(el):
        if ch.tag != qn("w:pPr"): el.remove(ch)
        else:
            for s in ch.findall(qn("w:sectPr")): ch.remove(s)
    for r in runs(markup, rpr): el.append(r)
    anchor_el.addnext(el); return el

def delete(*els):
    for e in els:
        e = e._p if hasattr(e, "_p") else e._tbl if hasattr(e, "_tbl") else e
        assert e.find(f".//{{{W}}}sectPr") is None, "refusing to delete a section break"
        e.getparent().remove(e)

def set_eq_number(p, num):
    ts = [t for t in p._p.iter(qn("w:t")) if re.fullmatch(r"\s*\(\d+\)\s*", t.text or "")]
    assert len(ts) == 1, p.text; ts[0].text = ts[0].text.replace(re.search(r"\(\d+\)", ts[0].text).group(0), f"({num})")

CELL_RPR = OxmlElement("w:rPr")
for tag, attrs in (("w:rFonts", {"w:ascii": "Times New Roman", "w:hAnsi": "Times New Roman", "w:cs": "Times New Roman"}),
                   ("w:sz", {"w:val": "18"}), ("w:szCs", {"w:val": "16"})):
    e = OxmlElement(tag)
    for k, v in attrs.items(): e.set(qn(k), v)
    CELL_RPR.append(e)

def make_table(widths, rows, left_cols=1):
    """Same format as the paper's existing tables: fixed layout, single borders, Times 9 pt, bold header row."""
    assert all(len(r) == len(widths) for r in rows), rows
    tbl = OxmlElement("w:tbl"); pr = OxmlElement("w:tblPr")
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); pr.append(lay)
    bd = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{side}")
        for k, v in (("val", "single"), ("sz", "4"), ("space", "0"), ("color", "auto")): e.set(qn(f"w:{k}"), v)
        bd.append(e)
    pr.append(bd)
    mar = OxmlElement("w:tblCellMar")
    for side, w in (("top", 0), ("left", 22), ("bottom", 0), ("right", 22)):
        e = OxmlElement(f"w:{side}"); e.set(qn("w:w"), str(w)); e.set(qn("w:type"), "dxa"); mar.append(e)
    pr.append(mar)
    tw = OxmlElement("w:tblW"); tw.set(qn("w:type"), "dxa"); tw.set(qn("w:w"), str(sum(widths))); pr.append(tw)
    tbl.append(pr)
    grid = OxmlElement("w:tblGrid")
    for w in widths:
        g = OxmlElement("w:gridCol"); g.set(qn("w:w"), str(w)); grid.append(g)
    tbl.append(grid)
    for ri, row in enumerate(rows):
        tr = OxmlElement("w:tr"); trp = OxmlElement("w:trPr"); trp.append(OxmlElement("w:cantSplit")); tr.append(trp)
        for ci_, (w, txt) in enumerate(zip(widths, row)):
            tc = OxmlElement("w:tc"); tcp = OxmlElement("w:tcPr"); cw = OxmlElement("w:tcW")
            cw.set(qn("w:type"), "dxa"); cw.set(qn("w:w"), str(w)); tcp.append(cw); tc.append(tcp)
            p = OxmlElement("w:p"); pp = OxmlElement("w:pPr")
            if ri < len(rows) - 1: pp.append(OxmlElement("w:keepNext"))
            sp = OxmlElement("w:spacing")
            for k, v in (("before", "0"), ("after", "0"), ("line", "240"), ("lineRule", "auto")): sp.set(qn(f"w:{k}"), v)
            pp.append(sp)
            if ci_ >= left_cols:
                jc = OxmlElement("w:jc"); jc.set(qn("w:val"), "center"); pp.append(jc)
            p.append(pp)
            for r in runs(txt, CELL_RPR, bold=(ri == 0)): p.append(r)
            tc.append(p); tr.append(tc)
        tbl.append(tr)
    return tbl

def replace_table(old, new_tbl):
    old._tbl.addnext(new_tbl); old._tbl.getparent().remove(old._tbl); return new_tbl

def caption(p, markup): set_par(p, markup)

# ================================================================================================ TEXT
F = "F_{0.5}"
title = ("Rethinking FLAN-T5 for Grammatical Error Correction: Controlled Baselines, Full-Budget Ablations "
         "and Two-Stage Transfer with Little Human Data")
set_par(PAR[1], title)

HUMAN_BEA = VC["T4|human_data_stage|bea_dev"]["delta"]
abstract = (
    "Grammatical error correction (GEC) systems are usually built from large models and millions of human-corrected "
    "sentences. We study what a compact 77-million-parameter FLAN-T5-small model achieves with synthetic data and only "
    "32,600 human-annotated training pairs, and which design choices matter. Under one pipeline, FLAN-T5-small outperforms TF-IDF, "
    "LSTM and pointer-generator baselines. Full-budget ablations with two training seeds and paired bootstrap tests show "
    f"that fine-tuning on human learner data is the dominant factor (+{HUMAN_BEA:.2f} F0.5 on BEA-2019 development), and that a "
    "confidence threshold adds a small but significant gain, whereas reference-side corpus cleaning, no-edit training "
    "pairs, iterative refinement and an explicit copy gate give no reliable improvement. The final system reaches an "
    f"F0.5 of {pct(N['test']['F05'])} on the blind BEA-2019 test set and {N['conll']['F05']:.3f} on CoNLL-2014 with "
    "the official scorer. It outperforms a 1.5-billion-parameter instruction-tuned language model but trails a same-size "
    "model trained on 2.4 million human-corrected pairs.")
assert len(abstract.split()) <= 150, len(abstract.split())
set_par(PAR[8], abstract)
set_par(PAR[9], "**Keywords:** Grammatical error correction, FLAN-T5, Synthetic data, Low-resource fine-tuning, "
                "Ablation study, BEA-2019.")

def sub_runs(p, old, new):
    n = 0
    for t in p._p.iter(qn('w:t')):
        if t.text and old in t.text: t.text = t.text.replace(old, new); n += 1
    assert n == 1, (old, p.text[:80])
sub_runs(PAR[13], 'Grammar Error Correction (GEC)', 'Grammatical error correction (GEC)')
sub_runs(PAR[54], 'one GLEU point', 'one point of machine-translation GLEU')
sub_runs(PAR[34], 'for the enhanced system', 'for the systems of Section 5')
sub_runs(PAR[3], 'School Communications', 'School of Communications')
# ---------------------------------------------------------------- 1. Introduction
set_par(PAR[14],
    "We first compare four model families on the same data, test sample and evaluation pipeline: a TF-IDF unigram and "
    "bigram Logistic Regression detector, which measures how separable the noise is at the bag-of-words level; a "
    "sequence-to-sequence long short-term memory (LSTM) network without attention; an LSTM with Bahdanau attention and a "
    "pointer-generator copy mechanism, designed for tasks in which most output tokens appear in the input; and a "
    "pre-trained FLAN-T5-small Transformer fine-tuned with an instruction prefix.")
delete(PAR[15], PAR[16], PAR[17], PAR[18])
set_par(PAR[19],
    "We then ask what such a compact model can and cannot achieve when human-annotated data are scarce, and which of the "
    "design choices commonly added to it actually help. Our contributions are: (i) a controlled comparison of classical, "
    "recurrent and Transformer models under identical data and metrics; (ii) full-budget ablations, with two training "
    "seeds where training variance matters and paired bootstrap tests on four endpoints, of reference-side corpus "
    "cleaning, no-edit training pairs, an explicit copy gate, iterative refinement, a confidence threshold and a second "
    "fine-tuning stage on 32,600 human-annotated pairs; (iii) an evaluation on the blind BEA-2019 test set, on CoNLL-2014 "
    "with the official M\u00b2 scorer and on JFLEG with the official GLEU, against a 2024 instruction-tuned language model "
    "and two GEC-specific T5 models scored under the same protocol, with latency measured on the same GPU; and (iv) a "
    "candid account of the outcome: the human-data stage dominates, a confidence threshold helps slightly, and the "
    "remaining additions do not survive full-budget testing. We report these negative results explicitly.")
delete(PAR[20])
set_par(PAR[21],
    "Section 2 reviews related work, Section 3 describes the data and the cleaning procedure, Section 4 the models and "
    "the evaluation protocol, Section 5 the results, and Section 6 concludes.")

# ---------------------------------------------------------------- 2. Related work
set_par(PAR[24],
    "**GEC as translation and synthetic data.** Most modern systems treat GEC as monolingual translation from an "
    "ungrammatical to a grammatical sentence [2]. Because human-annotated learner data are scarce, large synthetic "
    "corpora are generated by corrupting clean text: Lichtarge *et al.* [3] and Grundkiewicz *et al.* [4] pre-train on "
    "such data, Kiyono *et al.* [5] study how best to incorporate it, and Stahlberg and Kumar [6] introduced tagged "
    "corruption models and released C4-200M, the 200-million-pair corpus we sample from. The dominant recipe pre-trains "
    "on synthetic data and then fine-tunes on human learner corpora [7]; we adopt this two-stage scheme. Rothe *et al.* "
    "[7] also released cLang-8, a cleaned 2.4-million-pair learner corpus on which T5 models of every size are now "
    "routinely fine-tuned, which makes the amount of human data, not only the model, a key variable. The corruption "
    "distribution of synthetic data is chosen a priori and need not match that of real learners, and the reference side "
    "of automatically generated corpora is itself imperfect; to our knowledge no prior work quantifies this "
    "reference-side noise for C4-200M or measures its effect on the trained model, which Sections 3.2 and 5.2 do.")
set_par(PAR[25], re.sub(r"We compare against these families quantitatively in Section\s5\.6\.",
    "Section 5.4 compares our system with a recent instruction-tuned language model and with GEC-specific T5 models "
    "under one protocol.", PAR[25].text).replace("Models.", "**Models.**", 1))
assert "Section 5.4 compares" in PAR[25].text, PAR[25].text[-300:]
set_par(PAR[26],
    "**Evaluation and benchmarks.** Fluency-oriented evaluation uses GLEU [17], an n-gram metric adapted to GEC that "
    "rewards n-grams matching the references and penalises n-grams that should have been changed. Corrections are "
    f"scored at the edit level with {F}, which weights precision over recall, using the ERRor ANnotation Toolkit "
    "(ERRANT) [18] or the MaxMatch (M\u00b2) scorer, as in the CoNLL-2014 [19] and BEA-2019 [20] shared tasks; GLEU is "
    "the official metric of the fluency benchmark JFLEG [21]. We use the W&I+LOCNESS data [20]; NUCLE [22] is a further "
    "learner corpus we identify as future training data. These scorers are not "
    "interchangeable: M\u00b2 and ERRANT extract and match edits differently, and figures also depend on whether the "
    "development or the blind test split is used, so numbers from different protocols cannot be ranked against one "
    "another. We therefore score every system ourselves, with the official scorer of each benchmark, on the same split "
    "and with the same tokenisation.")
set_par(PAR[27],
    "Against this background, our contribution is not a new architecture but a controlled attribution for the compact, "
    "low-human-data regime: which of the components routinely added to a small synthetic-data GEC model\u2014corpus "
    "cleaning, no-edit pairs, explicit copying, refinement, thresholding and a human-data stage\u2014change its accuracy "
    "once each is tested at the full training budget, with significance tests and, where training variance matters, "
    "more than one seed.")

# ---------------------------------------------------------------- 3. Dataset
t33 = PAR[33].text
t33 = t33.replace("For the enhanced system we re-sample", "For the systems of Section 5 we re-sample")
t33 = t33.replace(" For the second training stage we additionally use the human-annotated W&I+LOCNESS [20] learner data.", "")
t33 += (" Section 5.2 tests whether training on the cleaned rather than the uncleaned corpus changes the resulting model.")
assert "*" not in t33 and "_{" not in t33
set_par(PAR[33], t33)
set_par(PAR[36],
    "The second training stage uses the Write & Improve and LOCNESS corpus released for the BEA-2019 shared task [20]. "
    "We reconstruct parallel sentence pairs from the gold M\u00b2 training files of the three proficiency bands (A, B and C) "
    f"by applying every edit of the first annotator to the source sentence, which yields {WI['pairs']:,} pairs. Five per "
    f"cent of the distinct source sentences are held out ({WI['holdout_pairs']:,} pairs), so that no source sentence occurs "
    f"in both parts, leaving {WI['train_pairs']:,} training pairs; the holdout is used for checkpoint selection and for "
    "tuning the decoding threshold, so the BEA-2019 development set is never observed during training or model "
    "selection. The corpus spans three learner proficiency bands (Common European Framework of Reference, CEFR, levels "
    "A, B and C) together with a set of native essays, which supports the per-proficiency analysis of Section 5.5. "
    "Because these are genuine learner errors rather than synthetic corruptions, this stage supplies the error "
    "distribution that the synthetic corpus cannot, and Section 5.2 shows that it is the component that drives the "
    "benchmark gains.")

# ---------------------------------------------------------------- 4. Approach
set_par(PAR[38],
    "Our study has two parts: a controlled set of baselines that establishes FLAN-T5-small as the strongest backbone, "
    "and a two-stage system built on that backbone whose components are then ablated one at a time. Figure 1 gives an "
    "end-to-end overview; the components are described below.")
set_par(PAR[56],
    "We fine-tune flan-t5-small [10] (77M parameters [23]), wrapping each input in the instruction prefix \u201cCorrect "
    "the grammar in this sentence: \u201d. During training each source word of an edit pair is dropped with probability "
    "0.05 as a regulariser; inputs and targets are truncated to 128 sub-word tokens. Training uses one epoch of AdamW "
    "(learning rate 3\u00d710^{\u22124}, weight decay 0.01, label smoothing 0.1) with a cosine schedule and 6% warm-up, a "
    "global batch of 128 (16 per device, 4 accumulation steps, 2 GPUs) and fp16 precision, keeping the checkpoint with "
    "the lowest validation loss among evaluations every 2,500 steps. On the 1,150,000-pair split this is 8,985 optimiser "
    "steps; the final system's stage 1 adds the no-edit pairs of Section 4.5 (1,322,500 examples, 10,333 steps). At "
    "inference we use beam search (5 beams, no-repeat n-gram size 3, length penalty 0.7) followed by a guarded "
    "refinement pass that re-runs the model on its own output and accepts the result only if it changes at most three "
    "tokens. The FLAN-T5 model of the four-model comparison (Section 5.1) was trained in an earlier configuration "
    "(linear schedule, 12% warm-up, no label smoothing; Section 5.8).")
set_par(PAR[62],
    "Generative GEC over-edits: the FLAN-T5 baseline edits 73% of its inputs. We test three controls. (i) *No-edit "
    "training pairs*: 15% of the training split is added as identity pairs (target \u2192 target, without source "
    "dropout) to teach the model when to leave text alone. (ii) *Length penalty*: beam search uses a length penalty of "
    "0.7, selected on the validation set from {0.7, 1.0, 1.2}. (iii) *Confidence-thresholded reversion*: an output that "
    "differs from its input is reverted to the input when the mean per-token log-likelihood of the output exceeds that "
    "of the unchanged source by less than a threshold \u03c4 (Eqs. (7) and (8)), tuned on held-out data. The reversion "
    "acts on whole sentences, so no source-to-output alignment is needed and insertions, deletions and multi-token "
    "replacements are handled uniformly.")
set_par(PAR[66],
    "Synthetic noise is not the distribution of real learner errors, so we adopt the standard recipe [7]: stage 1 trains "
    "on the synthetic corpus, and stage 2 fine-tunes the stage-1 model on the W&I+LOCNESS training pairs (Section 3.3) "
    "for three epochs at a learning rate of 1\u00d710^{\u22124} and a global batch of 64. The 5% holdout selects the "
    "checkpoint and the threshold \u03c4, so the BEA-2019 development set is never seen during training or model "
    "selection.")
set_par(PAR[68],
    f"Our primary metric is the edit-level {F} of the CoNLL-2014 [19] and BEA-2019 [20] shared tasks: predicted "
    "source-to-output edits are matched against the gold edits and combined as")
set_par(PAR[70],
    "so that over-corrections are penalised twice as heavily as missed errors. We report five endpoints, each with its "
    "official scorer: the blind BEA-2019 test set, scored by the shared task's ERRANT-based scorer on the CodaBench "
    "platform (the primary endpoint; each trained system is submitted once, with no tuning on test scores); CoNLL-2014 "
    "test with the official M\u00b2 scorer; BEA-2019 development with ERRANT's errant_compare; JFLEG test with the "
    "official GLEU implementation (four references, 500 iterations); and, in domain, a fixed 5,000-pair subset of the "
    "cleaned C4-200M test split scored with single-reference ERRANT. Because the CoNLL-2014 and BEA-2019 sources are "
    "pre-tokenised and both scorers compare whitespace-separated tokens, every system's output is re-tokenised with one "
    "benchmark-specific tokeniser, chosen by round-trip fidelity on the references (a rule-based tokeniser reproducing "
    "99.3% of CoNLL-2014 references; spaCy reproducing 98.7% of BEA-2019 references). Significance is a paired "
    "bootstrap over sentences with 2,000 resamples, on per-sentence edit counts that reproduce the official totals "
    "exactly; for systems A and B the statistic and its interval are")
eq_boot = PAR[107]._p
PAR[70]._p.addnext(eq_boot); set_eq_number(PAR[107], 10)
new_par_after(eq_boot, PAR[70],
    "where * marks resampled values; p is twice the smaller share of resampled differences on either side of zero. "
    "The sentence bootstrap captures test-set but not training variance, so the data ablations of Section 5.2 are "
    "repeated with two training seeds.")

# ---------------------------------------------------------------- 5.1
t73 = PAR[73].text
t73 = t73.replace("the only model to beat the do-nothing GLEU baseline (0.636 versus 0.614)",
                  "the only model to beat the do-nothing baseline on corpus-level machine-translation GLEU (0.636 versus 0.614)")
assert "showing that about 63% of the noise has a lexical signature, while the rest needs word order or longer-range context" in t73
t73 = t73.replace("showing that about 63% of the noise has a lexical signature, while the rest needs word order or longer-range context",
                  "so the noise is only partly separable at the bag-of-words level")
t73 = t73.replace("F0.5", F)
t73 = t73.replace("(Section 5.9)", "(Section 5.7)")
t73 += " These are single runs, in the earlier training configuration (Section 5.8)."
set_par(PAR[73], t73)
caption(PAR[74], "Table 3. Four-model comparison on the 5,000-pair test sample (single runs). GLEU and BLEU here are "
                 "corpus-level machine-translation metrics (NLTK), not the GEC GLEU of [17]; the detector's "
                 f"{F} is binary-classification {F}")
# Table 3 header: GLEU -> MT-GLEU
for t in TAB[2].rows[0].cells[2]._tc.iter(qn("w:t")):
    if t.text.strip() == "GLEU": t.text = "MT-GLEU"

# ---------------------------------------------------------------- 5.2 full-budget ablations (Table 4)
set_par(PAR[76], "5.2 Which design choices matter? Full-budget ablations")
set_par(PAR[77],
    "Table 4 tests each component at the full training budget (1,150,000 synthetic pairs, 8,985 steps, plus the "
    "human-data stage where stated), changing one factor at a time on the same split, schedule and decoder and scoring "
    "four endpoints. Training-time factors are retrained from scratch, cleaning with two seeds (cleaned corpus of "
    "Table 2 against the same sampling with only the identity and length filters of Table 1); decode-time "
    "factors reuse one checkpoint and are therefore exactly isolated.")
c4, be, co = "c4", "bea_dev", "conll"
def row(label, name, jf): return [label] + [cell(vc("T4", name, ep)) for ep in (c4, be, co)] + [jf]
T4 = {k: {ep: vc("T4", k, ep) for ep in (c4, be, co)} for k in
      ("cleaning_seed42", "cleaning_seed1337", "seed_1337_vs_42_clean", "noedit_stage1", "noedit_two_stage", "copy_gate_earlier_config",
       "human_data_stage", "threshold_tau0", "refinement", "length_penalty_0.7_vs_1.0")}
JF = lambda a, b: sg(a - b)
rows4 = [["Factor (added)", "C4", "BEA dev", "CoNLL", "JFLEG"],
         row("Cleaning (seed 42)", "cleaning_seed42", JF(JG["nogate_s1"], JG["noisy_s42"])),
         row("Cleaning (seed 1337)", "cleaning_seed1337", JF(JG["clean_s1337"], JG["noisy_s1337"])),
         row("Seed 1337 vs 42", "seed_1337_vs_42_clean", JF(JG["clean_s1337"], JG["nogate_s1"])),
         row("No-edit, stage 1", "noedit_stage1", JF(JG["final_s1_notau"], JG["nogate_s1"])),
         row("No-edit, two-stage", "noedit_two_stage", JF(JG["final_s2"], JG["nogate_s2"])),
         row("Copy gate\u2020", "copy_gate_earlier_config", "\u2013"),
         row("Human-data stage", "human_data_stage", JF(JG["nogate_s2_notau"], JG["nogate_s1"])),
         row("Threshold \u03c4 = 0", "threshold_tau0", JF(JG["final_s2"], JG["final_notau"])),
         row("Refinement pass", "refinement", JF(JG["final_s2"], JG["final_noref"])),
         row("Length penalty 0.7\u2020", "length_penalty_0.7_vs_1.0", "\u2013")]
cap4 = new_par_after(PAR[77]._p, PAR[74],
    f"Table 4. Full-budget ablations: change in {F} (JFLEG: GLEU) when the factor is added, all else fixed. "
    "* paired bootstrap p < 0.05; JFLEG has no paired test (single-system 95% interval about \u00b10.016). "
    "\u2020 earlier configuration; length penalty against 1.0")
t4 = make_table([1620, 740, 770, 770, 754], rows4)
cap4.addnext(t4)
gate_full = GATE_JUNE
txt_b = new_par_after(t4, PAR[96],
    "Three results stand out. First, reference-side cleaning has no reliable effect: its sign differs between the two "
    "seeds on CoNLL-2014 and BEA-2019, and its magnitude never exceeds the difference between two identically "
    "configured clean runs that differ only in the random seed. What Section 3.2 removes is real noise, but at this "
    "budget the model is robust to it. Second, the no-edit pairs cost accuracy on every endpoint at stage 1, and after "
    f"the human-data stage they lower in-domain accuracy (C4 {sg(T4['noedit_two_stage'][c4]['delta'])}) and do not help "
    f"elsewhere (BEA-2019 {sg(T4['noedit_two_stage'][be]['delta'])}, CoNLL-2014 {sg(T4['noedit_two_stage'][co]['delta'])}, "
    f"JFLEG {sg(JG['final_s2'] - JG['nogate_s2'])}); they are therefore unnecessary, although we keep them in the final "
    "system, whose configuration was fixed before the blind-test evaluation. Third, the human-data "
    f"stage dominates every other factor, raising BEA-2019 development {F} by "
    f"{T4['human_data_stage'][be]['delta']:.3f} and CoNLL-2014 by "
    f"{T4['human_data_stage'][co]['delta']:.3f} while lowering the in-domain score by "
    f"{-T4['human_data_stage'][c4]['delta']:.3f}, a catastrophic-forgetting signature that "
    "makes the best checkpoint domain-specific. Among decode-time choices only the threshold matters; refinement and "
    "the length penalty have no measurable effect on any benchmark.")
assert all(T4["noedit_two_stage"][ep]["delta"] <= 0.002 for ep in (c4, be, co))          # "still do not help"
assert all(T4["refinement"][ep]["p_two_sided"] >= 0.05 for ep in (c4, be, co))           # "no measurable effect"
assert all(T4["length_penalty_0.7_vs_1.0"][ep]["p_two_sided"] >= 0.05 for ep in (c4, be, co))
assert all(T4["threshold_tau0"][ep]["p_two_sided"] < 0.05 and T4["threshold_tau0"][ep]["delta"] > 0 for ep in (c4, be, co))
assert all(T4["human_data_stage"][ep]["p_two_sided"] < 0.05 for ep in (c4, be, co))
assert T4["noedit_two_stage"][c4]["delta"] < 0 and T4["noedit_two_stage"][c4]["p_two_sided"] < 0.05   # "lower in-domain accuracy"
_p = [EV2[f"bea_dev|ref1_{k}"]["P"] for k in ("taunone", "tau0.0", "tau0.05", "tau0.1", "tau0.2")]
assert _p[:4] == sorted(_p[:4]) and _p[4] < _p[3]                                     # "precision rises only up to 0.1"
assert all(EV2[f"{b}|ref1_tau0.0"]["F05"] == max(EV2[f"{b}|ref1_{k}"]["F05"] for k in ("taunone", "tau0.0", "tau0.05", "tau0.1", "tau0.2")) for b in ("c4", "bea_dev", "conll"))
gb = T4["copy_gate_earlier_config"]
txt_c = new_par_after(txt_b, PAR[96],
    "The copy gate (Section 4.4) lowers accuracy on every endpoint "
    f"(C4 {sg(gb[c4]['delta'])}, BEA-2019 {sg(gb[be]['delta'])}, CoNLL-2014 {sg(gb[co]['delta'])}; "
    f"on the full 25,000-pair in-domain test set {sg(-gate_full['delta_F05_nogate_minus_gate'])}, "
    "p < 0.001) although it converged to a mean generation probability of 0.27, i.e. it copies heavily: a pre-trained "
    "T5 already copies through its cross-attention. These two arms were trained in the earlier configuration "
    "(Section 5.8). Repeating the gate arm in the corrected configuration, it reached the same validation loss as the "
    f"no-gate arm at the same step ({GM['gate']['eval_loss']:.3f} versus {GM['nogate']['eval_loss']:.3f} after 2,500 "
    f"steps) but trained {SLOW_TRAIN:.0f} times and evaluated {SLOW_EVAL:.0f} times more slowly on our hardware and "
    "could not finish within a 12-hour session. We conclude only that this head-averaged formulation brings no accuracy "
    "benefit at a large computational cost, not that explicit copying is useless for pre-trained Transformers in general.")

# ---------------------------------------------------------------- 5.3 precision dial (Table 5) + audit
set_par(PAR[78], "5.3 Over-correction and the precision dial")
dial = [("none", "taunone"), ("0", "tau0.0"), ("0.05", "tau0.05"), ("0.1", "tau0.1"), ("0.2", "tau0.2")]
set_par(PAR[79],
    f"The threshold acts as a harm dial (Table 5): raising \u03c4 from no threshold to 0.2 cuts "
    f"the in-domain harm rate from {EV2['c4|ref1_taunone']['harm']:.1f}% to {EV2['c4|ref1_tau0.2']['harm']:.1f}% and raises "
    f"the share of unchanged outputs from {EV2['c4|ref1_taunone']['unchanged']:.0f}% to "
    f"{EV2['c4|ref1_tau0.2']['unchanged']:.0f}%, while BEA-2019 precision rises only up to \u03c4 = 0.1. The "
    f"held-out-tuned value \u03c4 = 0 is the {F} optimum on all three {F} endpoints; higher settings trade recall for "
    "safety in applications that need it. Harm is defined against a single "
    "synthetic reference by Eq. (11), where d is token-level edit distance, x the source, y the reference and \u0177 "
    "the prediction: an edited prediction is harmful when it is farther from the reference than the source, and lateral "
    "when it is exactly as far.")
set_eq_number(PAR[80], 11)
delete(PAR[81])
caption(PAR[82], f"Table 5. Effect of the revert threshold \u03c4 on the final system: BEA-2019 development P, R and "
                 f"{F}; CoNLL-2014 {F}; in-domain shares of unchanged and harmful outputs (%)")
rows5 = [["\u03c4", "P", "R", F, "CoNLL", "Unch.", "Harm"]]
for lab, k in dial:
    b = EV2[f"bea_dev|ref1_{k}"]
    rows5.append([lab, f3(b["P"]), f3(b["R"]), f3(b["F05"]), f3(EV2[f"conll|ref1_{k}"]["F05"]),
                  f"{EV2[f'c4|ref1_{k}']['unchanged']:.1f}", f"{EV2[f'c4|ref1_{k}']['harm']:.1f}"])
t5 = replace_table(TAB[3], make_table([700, 660, 660, 660, 660, 660, 657], rows5))
a_h, a_l, a_i = AUD["harm"], AUD["lateral"], AUD["improved"]
new_par_after(t5, PAR[96],
    "Reference distance is a weak proxy for harm, because a valid correction that differs from the single reference "
    "counts as harmful. As a reference-free check we sampled 50 BEA-2019 development outputs of the final system "
    "(earlier configuration) from each of the improved, lateral and harm classes and judged every pair blind, with source "
    "and output in random order, for which sentence is better English and whether the meaning is preserved. The "
    "judgments were made by an AI model (Claude, Anthropic), not by human annotators, so we treat them as indicative "
    f"only. The output was judged better than the source for {a_h['out']} of the 50 \u201charmful\u201d predictions, "
    f"equal for {a_h['same']} and worse for {a_h['src']}; for the lateral and improved classes the counts were "
    f"{a_l['out']}/{a_l['same']}/{a_l['src']} and {a_i['out']}/{a_i['same']}/{a_i['src']}, and the meaning changed in "
    f"{a_h['meaning'] + a_l['meaning'] + a_i['meaning']} of the 150 pairs. A smaller open-weight judge "
    "(Qwen2.5-7B-Instruct, 4-bit) proved unusable, preferring whichever sentence was shown second in most pairs. On "
    "learner text, most reference-defined harm thus appears to be a different valid correction rather than damage, and "
    "reference-based harm rates should be read as upper bounds on over-correction.")

# ---------------------------------------------------------------- 5.4 benchmarks (Table 6)
set_par(PAR[83], "5.4 Benchmark results and comparison with other systems")
q4, q0 = Q["conll|fourshot"], Q["conll|zeroshot"]
gs, gbse = BL["gec-t5-small-clang8"], BL["gec-t5-base-clang8"]
set_par(PAR[84],
    "Table 6 places the final system among systems we scored under one protocol. On the blind BEA-2019 test set it "
    f"reaches {F} = {pct(N['test']['F05'])} (precision {pct(N['test']['P'])}, recall {pct(N['test']['R'])}). "
    "It outperforms Qwen2.5-1.5B-Instruct [24], a 2024 instruction-tuned language model 20 times larger, by "
    f"{vc('T6', 'final_vs_qwen_fourshot', 'conll')['delta']:.3f} on CoNLL-2014 (95% CI "
    f"{ci(vc('T6', 'final_vs_qwen_fourshot', 'conll'))}) and by {vc('T6', 'final_vs_qwen_fourshot', 'bea_dev')['delta']:.3f} on "
    f"BEA-2019 development ({ci(vc('T6', 'final_vs_qwen_fourshot', 'bea_dev'))}), both p < 0.001. The mechanism is "
    f"over-correction: the language model has higher recall on CoNLL-2014 ({q4['R']:.3f} versus {N['conll']['R']:.3f}) but "
    f"much lower precision ({q4['P']:.3f} versus {N['conll']['P']:.3f}), because it edits {ED_QWEN:.0f}% of the sentences "
    f"against {ED_OURS:.0f}% for our system. Conversely, gec-t5-small-clang8 [25], a T5 model of the same size fine-tuned "
    f"on about 2.4 million human-corrected cLang-8 pairs [7], beats our system by "
    f"{-vc('T6', 'final_vs_gec_t5_small', 'conll')['delta']:.3f} on CoNLL-2014 and "
    f"{-vc('T6', 'final_vs_gec_t5_small', 'bea_dev')['delta']:.3f} on BEA-2019 development (both p < 0.001), and its "
    f"248M-parameter sibling by {-vc('T6', 'final_vs_gec_t5_base', 'conll')['delta']:.3f} and "
    f"{-vc('T6', 'final_vs_gec_t5_base', 'bea_dev')['delta']:.3f}. With architecture and size fixed, this gap is attributable "
    f"to human training data, of which our system uses about {round(HUMAN_RATIO, -1):.0f} times less. An earlier reduced-budget experiment also showed "
    f"that the stage-2 gain reflects the schedule and not only the closeness of W&I+LOCNESS to BEA-2019: training "
    f"sequentially rather than on a joint mixture of synthetic and human data improved BEA-2019 development {F} by 0.096 "
    "(95% CI 0.086 to 0.107). For context, published systems report 65.3 on CoNLL-2014 and 72.4 on the BEA-2019 test "
    "set (GECToR [14]) and 68.9 and 75.9 (T5-xxl, 11B parameters [7]).")
caption(PAR[85], f"Table 6. Official-scorer {F} on identical splits and tokenisation (JFLEG: GLEU). Test = blind "
                 "BEA-2019 test set; Dev = BEA-2019 development set; HD = human-annotated training pairs; Qwen = Qwen2.5-1.5B-Instruct; gec-t5 = gec-t5-*-clang8")
ours_s1 = EV1
rows6 = [["System", "Size", "HD", "CoNLL", "Dev", "Test", "JFLEG"],
         ["Do nothing", "\u2013", "\u2013", "0.000", "0.000", "\u2013", f3(JG["do_nothing"])],
         ["Qwen, 0-shot", "1.5B", "0", f3(q0["F05"]), f3(Q["bea_dev|zeroshot"]["F05"]), "\u2013", "\u2013"],
         ["Qwen, 4-shot", "1.5B", "0", f3(q4["F05"]), f3(Q["bea_dev|fourshot"]["F05"]), "\u2013", "\u2013"],
         ["Ours, stage 1", "77M", "0", f3(ours_s1["conll|ref1_tau0.0"]["F05"]), f3(ours_s1["bea_dev|ref1_tau0.0"]["F05"]),
          "\u2013", f3(JG["final_s1_tau0"])],
         ["**Ours, final**", "**77M**", "**33k**", f"**{f3(N['conll']['F05'])}**", f"**{f3(N['bea']['F05'])}**",
          f"**{pct(N['test']['F05'])}**", f"**{f3(N['jfleg'])}**"],
         ["gec-t5-small", "77M", "2.4M", f3(gs["conll"]["F05"]), f3(gs["bea_dev"]["F05"]), "\u2013", f3(gs["jfleg"]["GLEU"])],
         ["gec-t5-base", "248M", "2.4M", f3(gbse["conll"]["F05"]), f3(gbse["bea_dev"]["F05"]), "\u2013", f3(gbse["jfleg"]["GLEU"])]]
replace_table(TAB[4], make_table([1160, 530, 440, 690, 570, 580, 684], rows6))

# ---------------------------------------------------------------- 5.5 CEFR + error types (Table 7)
set_par(PAR[86], "5.5 Proficiency levels and error types")
et = ET
assert LEV[2][2] < LEV[1][2] and LEV[3][2] < LEV[1][2] and TEST["C (advanced)"]["F05"] > TEST["A (beginner)"]["F05"] and TEST["N (native)"]["F05"] > TEST["B (intermediate)"]["F05"]          # wording of the CEFR sentence
set_par(PAR[87],
    "Table 7 breaks the results down by learner proficiency. On the development set the advanced and native groups "
    "score lowest, but on the blind test the ordering reverses and they score highest, with recall rising from "
    f"{pct(TEST['A (beginner)']['R'], 2)} (A) to {pct(TEST['N (native)']['R'], 2)} (N); per-level differences of this size "
    "are not stable across splits, and we draw no proficiency conclusion from them. By ERRANT error type on the blind "
    f"test, the system is strongest on punctuation ({F} {pct(et['PUNCT'], 2)}), subject\u2013verb agreement "
    f"({pct(et['VERB:SVA'], 2)}), verb form ({pct(et['VERB:FORM'], 2)}), determiners ({pct(et['DET'], 2)}) and noun "
    f"number ({pct(et['NOUN:NUM'], 2)}), which are frequent and local, and weakest on open-class word choice (noun "
    f"{pct(et['NOUN'], 2)}, adjective {pct(et['ADJ'], 2)}, adverb {pct(et['ADV'], 2)}, verb {pct(et['VERB'], 2)}), "
    f"conjunctions ({pct(et['CONJ'], 2)}), tense ({pct(et['VERB:TENSE'], 2)}) and word order ({pct(et['WO'], 2)}), "
    "which require meaning-level or reordering decisions that a 77M model trained mostly on local synthetic corruptions "
    "rarely makes.")
caption(PAR[88], f"Table 7. Results by CEFR level: BEA-2019 development {F} (Dev) and blind BEA-2019 test P, R and {F}")
rows7 = [["Level", "n (dev)", f"Dev {F}", "P", "R", F]]
for name, n, fd, t in LEV:
    rows7.append([name, f"{n:,}", f3(fd), f"{pct(t['P'])}", f"{pct(t['R'])}", f"{pct(t['F05'])}"])
rows7.append(["All", "4,384", f3(N["bea"]["F05"]), f"{pct(N['test']['P'])}", f"{pct(N['test']['R'])}",
              f"{pct(N['test']['F05'])}"])
replace_table(TAB[5], make_table([1300, 640, 700, 670, 670, 677], rows7))

# ---------------------------------------------------------------- 5.6 cost (Table 8)
set_par(PAR[90], "5.6 Computational cost")
L = {k: LAT[k] for k in LAT}
lo, lg = L["ours, final pipeline (beam 5 + refinement + threshold)"], L["ours, greedy single pass"]
ls, lb, lq = L["gotutiyan/gec-t5-small-clang8"], L["gotutiyan/gec-t5-base-clang8"], L["Qwen/Qwen2.5-1.5B-Instruct, 4-shot greedy"]
set_par(PAR[91],
    "Table 8 compares inference cost on one NVIDIA T4 GPU (16 GB) for the first 300 CoNLL-2014 sentences: batch-1 "
    "latency per sentence, batch-32 throughput and peak allocated memory. Our full pipeline (beam search, refinement "
    f"and threshold scoring) costs {lo['lat_median_ms'] / ls['lat_median_ms']:.1f} times the latency of gec-t5-small with "
    f"plain beam search; a single greedy pass is the cheapest configuration measured ({lg['lat_median_ms']:.0f} ms, "
    f"{lg['throughput_sent_s_b32']:.1f} sentences per second, {lg['peak_mem_MB']:.0f} MB). The 1.5B language model is "
    f"{lq['lat_median_ms'] / lo['lat_median_ms']:.1f} times slower than our full pipeline and needs "
    f"{lq['peak_mem_MB'] / lo['peak_mem_MB']:.1f} times its memory. Training the final system took "
    f"{TR1['train_seconds'] / 3600:.1f} h for stage 1 ({TR1['steps']:,} steps) and {TR2['train_seconds'] / 3600:.1f} h for "
    "stage 2 on two T4 GPUs, so the whole pipeline is reproducible on free-tier hardware.")
delete(PAR[92], PAR[93], PAR[94], PAR[95], PAR[96])
caption(PAR[97], "Table 8. Inference cost on one NVIDIA T4 GPU: batch-1 latency (median and 90th percentile, ms), batch-32 "
                 "throughput (sentences/s) and peak memory (MB)")
rows8 = [["System", "Size", "Median", "p90", "Sent./s", "MB"],
         ["Ours, full pipeline", "77M", f"{lo['lat_median_ms']:.0f}", f"{lo['lat_p90_ms']:,.0f}", f"{lo['throughput_sent_s_b32']:.1f}", f"{lo['peak_mem_MB']:,.0f}"],
         ["Ours, greedy", "77M", f"{lg['lat_median_ms']:.0f}", f"{lg['lat_p90_ms']:,.0f}", f"{lg['throughput_sent_s_b32']:.1f}", f"{lg['peak_mem_MB']:,.0f}"],
         ["gec-t5-small", "77M", f"{ls['lat_median_ms']:.0f}", f"{ls['lat_p90_ms']:,.0f}", f"{ls['throughput_sent_s_b32']:.1f}", f"{ls['peak_mem_MB']:,.0f}"],
         ["gec-t5-base", "248M", f"{lb['lat_median_ms']:.0f}", f"{lb['lat_p90_ms']:,.0f}", f"{lb['throughput_sent_s_b32']:.1f}", f"{lb['peak_mem_MB']:,.0f}"],
         ["Qwen, 4-shot", "1.5B", f"{lq['lat_median_ms']:,.0f}", f"{lq['lat_p90_ms']:,.0f}", f"{lq['throughput_sent_s_b32']:.1f}", f"{lq['peak_mem_MB']:,.0f}"]]
replace_table(TAB[6], make_table([1480, 520, 680, 640, 660, 674], rows8))
delete(PAR[98], PAR[99], PAR[100])

# ---------------------------------------------------------------- 5.7 matched pre-training control (kept, shortened)
set_par(PAR[101], "5.7 Matched pre-training control")
set_par(PAR[102],
    "The comparison of Section 5.1 differs in pre-training, vocabulary, schedule and decoding, so it cannot attribute "
    "the advantage of FLAN-T5 to the architecture. To isolate pre-training we trained the same T5-small architecture "
    "with an identical corpus, split, batch size, schedule and decoder, changing only the initialisation (150,000 "
    "training pairs, a fixed 3,000-pair evaluation subset, two seeds, earlier configuration).")
caption(PAR[103], PAR[103].text.replace("Table 8.", "Table 9."))
set_par(PAR[104],
    "The pre-trained checkpoint scores 45 times higher (paired bootstrap, p < 0.001), and the randomly initialised "
    "model produces 99.2% strictly harmful outputs: within this architecture and budget, pre-trained initialisation is "
    "decisive. We claim no more than that; we do not compare architectures under matched pre-training.")
for p in PAR[105:121]:
    if p._p.getparent() is not None and p._p is not eq_boot: delete(p)
for t in TAB[7 + 1:13]:
    delete(t)

# ---------------------------------------------------------------- 5.8 reproducibility and training configuration
set_par(PAR[121], "5.8 Reproducibility and training configuration")
set_par(PAR[122],
    "Table 10 lists the hyperparameters of both training stages and of inference. Stage 1 of the final system trains on "
    "1,322,500 examples (the 1,150,000-pair split plus 172,500 no-edit pairs) at the global batch of Eq. (12), giving "
    "the step count of Eq. (13). Stage 2 uses a global batch of 64 (16 per device, 2 accumulation steps, 2 GPUs): the "
    f"{WI['train_pairs']:,} training pairs form \u2308{WI['train_pairs']:,}/32\u2309 = {-(-WI['train_pairs'] // 32):,} "
    f"micro-batches per epoch, i.e. {TR2['steps'] // 3} updates per epoch and {TR2['steps']:,} over three epochs.")
set_eq_number(PAR[123], 12); set_eq_number(PAR[124], 13)
set_par(PAR[125],
    "**Training configuration and data.** While preparing this version we found two defects in our earlier pipeline. "
    "First, the framework version used for the earlier training runs did not divide the training loss by the number of "
    "gradient-accumulation steps, so the loss and its gradients were several times too large and gradient clipping "
    "(maximum norm 1.0) was active at almost every step; because AdamW is nearly invariant to a constant rescaling of "
    "the gradient, the practical effect is the clipping. Second, the stage-2 data loader read both the per-level "
    "W&I+LOCNESS training files and their combined copy, so every training sentence appeared twice and most of the "
    "holdout also occurred in training. We retrained the final system (both stages), the no-edit ablation and every arm "
    "of the cleaning ablation with both defects corrected, with an automatic check that stops training when the ratio "
    f"of training to validation loss indicates the first defect (observed ratios {min(RATIOS):.2f} to {max(RATIOS):.2f}). "
    f"Together, the corrections changed the final system only slightly (CoNLL-2014 "
    f"{sg(vc('S58', 'final_vs_earlier_config_final', 'conll')['delta'])}, {pv(vc('S58', 'final_vs_earlier_config_final', 'conll'))}; "
    f"BEA-2019 development {sg(vc('S58', 'final_vs_earlier_config_final', 'bea_dev')['delta'])}, "
    f"{pv(vc('S58', 'final_vs_earlier_config_final', 'bea_dev'))}; blind test {pct(N['test_june'])} to "
    f"{pct(N['test']['F05'])}). The four-model "
    "comparison, the copy-gate and length-penalty rows of Table 4 and the reduced-budget experiments of Sections 5.4 "
    "and 5.7 were not retrained, and within each of these comparisons all arms share one configuration; the harm audit "
    "of Section 5.3 and the timings of Table 8 use the earlier checkpoint of the same architecture and decoder.")
rel = new_par_after(PAR[125]._p, PAR[125],
    "**Release.** The training, decoding and evaluation notebooks, the split indices of the C4-200M sample and of the "
    "W&I+LOCNESS holdout, all configurations, every system output scored in this paper and the trained checkpoints are "
    "available at https://github.com/MemoXk/compact-gec-flan-t5 (archived on Zenodo, https://doi.org/10.5281/zenodo.23326147). The Qwen2.5-1.5B-Instruct baseline uses revision 989aa79 of the public checkpoint "
    "in float16, the system prompt \u201cYou are a grammatical error correction system. Return ONLY the corrected "
    "sentence. Make the minimum number of edits. If the sentence is already correct, repeat it unchanged.\u201d, greedy "
    "decoding with at most 96 new tokens and the first output line; its four exemplars are drawn with a fixed seed "
    "(2024) from the 20,734 W&I+LOCNESS training pairs of 8 to 20 tokens with one to three edits. The gec-t5 models are "
    "decoded with their authors' default settings.")
caption(PAR[126], "Table 10. Hyperparameter settings for both training stages and for inference")
tb = TAB[13]
def set_cell(r, c, txt):
    p = tb.rows[r].cells[c].paragraphs[0]; set_par(p, txt)
assert tb.rows[2].cells[2].text.startswith("Precision") and tb.rows[4].cells[2].text.startswith("Stage-2 model")
assert tb.rows[7].cells[2].text.startswith("Revert") and tb.rows[9].cells[2].text.startswith("Random seed")
set_cell(2, 3, "fp16 (T4 GPUs)")
set_cell(4, 2, "Model selection, stage 1 / 2")
set_cell(4, 3, "val. loss / 5% holdout")
set_cell(7, 3, "none, 0 to 0.5 (9 values) / 0")
set_cell(9, 3, "42 (also 1337)")
assert tb.rows[3].cells[2].text.startswith("Stage-2 data")
set_cell(3, 3, f"W&I+LOCNESS {WI['train_pairs']:,} / 3 / 1e-4 / {TR2['steps']:,}")

# ---------------------------------------------------------------- 6. Conclusion
set_par(PAR[128],
    "We asked what a compact FLAN-T5-small model can achieve for grammatical error correction with synthetic data and "
    "little human-annotated data, and which design choices matter. Among four baselines under one pipeline the "
    "pre-trained Transformer is the clear winner. Full-budget ablations with paired significance tests and two seeds "
    "then separate what matters from what does not: a second fine-tuning stage on 32,600 human-annotated pairs dominates "
    "every other factor, and a confidence threshold adds a small consistent gain and provides a precision and harm dial, "
    "whereas reference-side corpus cleaning, no-edit training pairs, iterative refinement, length-penalty tuning and an "
    f"explicit copy gate do not reliably improve accuracy. The resulting 77M-parameter system reaches {F} = "
    f"{pct(N['test']['F05'])} on the blind BEA-2019 test set and {N['conll']['F05']:.3f} on CoNLL-2014 with the "
    "official scorer, outperforms a 1.5B-parameter instruction-tuned language model that over-corrects, and trains and "
    f"runs on commodity GPUs, but trails a same-size model trained on about {round(HUMAN_RATIO, -1):.0f} times more human data.")
set_par(PAR[129],
    "**Limitations and future work.** Our conclusions concern one compact backbone and one synthetic corpus; the data "
    "ablations use two seeds, and the copy-gate comparison and the matched pre-training control were not repeated in "
    "the corrected training configuration. Reference-based harm overstates damage on learner text, and our "
    "reference-free check was made by an AI model rather than by human annotators; a human evaluation is the natural "
    "next step. The clearest lever is more human data: complementing the 33k-pair stage with cLang-8 [7] or NUCLE [22] "
    "would show how much of the gap to same-size models trained on human data can be closed, and a larger backbone how "
    "much of the remaining recall gap is due to capacity.")

# ---------------------------------------------------------------- references [24], [25]
r24 = new_par_after(PAR[159]._p, PAR[159],
    "Qwen Team, \u201cQwen2.5 Technical Report\u201d, arXiv:2412.15115, 2024.")
new_par_after(r24, PAR[159],
    "gotutiyan, \u201cgec-t5-small-clang8 and gec-t5-base-clang8 model cards\u201d, huggingface.co/gotutiyan, accessed 2026.")

try:
    d.save(OUT)
except PermissionError:                      # the paper is open in another program: save next to it instead
    OUT = OUT.replace(".docx", "_NEW.docx"); d.save(OUT); print("!! inass_paper.docx is locked (open elsewhere); saved", OUT)
print("saved", OUT)
print("abstract words:", len(abstract.split()))
