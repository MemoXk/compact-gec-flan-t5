"""Round-3 human evaluation: blinded, stratified sample of the corrected final system's BEA-2019 dev edits.

System: Notebook E final_e42 (stage 1 clean + no-edit pairs, stage 2 on the essay-level split, seed 42; beam 5,
length penalty 0.7, one refinement pass, tau 0.0), i.e. exactly the outputs scored in the paper.
Population: the dev sentences the system EDITED (output != source). Strata by the automatic outcome against the
reference (annotator 0 of ABCN.dev.gold.bea19.m2, the same definition as the paper's outcome table):
    perfect  output == reference            improved  token edit distance to the reference decreases
    lateral  distance unchanged             harm      distance increases
Sample: 25 perfect + 50 improved + 50 lateral + 50 harm (random.Random(2026) within each stratum), shuffled together;
the output is shown as sentence 1 or 2 at random (random.Random(31)); differing tokens are bold in BOTH sentences.
Outputs (human_eval_round3/): two identical annotator workbooks and a key file that must stay closed until both
annotators have finished. Analysis afterwards: analyze_human_eval.py (Cohen's kappa, stratum-weighted estimates).
"""
import csv, difflib, json, random
from collections import Counter
from pathlib import Path
from openpyxl import Workbook
from openpyxl.cell.rich_text import CellRichText, TextBlock
from openpyxl.cell.text import InlineFont
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

P = Path(r"<PROJECT>"); OUT = P / "human_eval_round3"; OUT.mkdir(exist_ok=True)
GOLD = P / "release/compact-gec-flan-t5/data/wi+locness/m2/ABCN.dev.gold.bea19.m2"
PRED = P / "round2_results/E/session1/pred_bea_dev_final_e42.csv"; COL = "ref1_tau0.0"
N = {"perfect": 25, "improved": 50, "lateral": 50, "harm": 50}

def m2_targets(path):
    src, tgt = [], []
    for block in path.read_text(encoding="utf-8").strip().split("\n\n"):
        ls = block.split("\n"); toks = ls[0][2:].split(); edits = []
        for ln in ls[1:]:
            f = ln[2:].split("|||"); s, e = map(int, f[0].split())
            if int(f[-1]) == 0 and f[1] not in ("noop", "UNK"): edits.append((s, e, [] if f[2] in ("", "-NONE-") else f[2].split()))
        out, off = list(toks), 0
        for s, e, rep in sorted(edits):
            out[s + off:e + off] = rep; off += len(rep) - (e - s)
        src.append(" ".join(toks)); tgt.append(" ".join(out))
    return src, tgt
def lev(a, b):
    a, b = a.split(), b.split(); prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1): cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

src, tgt = m2_targets(GOLD)
rows = list(csv.DictReader(open(PRED, encoding="utf-8"))); assert [r["src"] for r in rows] == src
pred = [r[COL] for r in rows]
cls = {}
for i, (s, t, p) in enumerate(zip(src, tgt, pred)):
    if p.strip() == s.strip(): continue                       # not edited: outside the population
    if p.strip() == t.strip(): cls[i] = "perfect"
    else:
        ds, dp = lev(s, t), lev(p, t); cls[i] = "improved" if dp < ds else "lateral" if dp == ds else "harm"
pop = Counter(cls.values())
rng = random.Random(2026); items = []
for c, n in N.items():
    pool = sorted(i for i, x in cls.items() if x == c); items += [(i, c) for i in rng.sample(pool, min(n, len(pool)))]
random.Random(7).shuffle(items)
flip = random.Random(31); order = [flip.random() < 0.5 for _ in items]       # True: output shown as sentence 1

def rich(a, b):
    """Bold + dark red for the tokens of `a` that differ from `b`."""
    ta, tb = a.split(), b.split(); out = CellRichText(); bold = InlineFont(b=True, color="9C0006", rFont="Arial", sz=11)
    plain = InlineFont(rFont="Arial", sz=11)
    for op, i1, i2, _, _ in difflib.SequenceMatcher(a=ta, b=tb, autojunk=False).get_opcodes():
        if i2 > i1:
            seg = " ".join(ta[i1:i2]) + " "
            out.append(TextBlock(plain if op == "equal" else bold, seg))
    return out

THIN = Side(style="thin", color="BFBFBF"); BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
YELLOW = PatternFill("solid", fgColor="FFF2CC"); HEAD = PatternFill("solid", fgColor="D9E1F2")
F = lambda **k: Font(**{"name": "Arial", "size": 11, **k})
def workbook(label):
    wb = Workbook(); ins = wb.active; ins.title = "Instructions"
    lines = [
        ("Human evaluation of grammatical error correction outputs", F(bold=True, size=14)),
        (f"Annotator: {label}   Name: ______________________", F(bold=True)),
        ("", F()),
        ("What to do", F(bold=True)),
        ("Each row on the 'Annotate' sheet shows two versions of one sentence written by an English learner. One is the "
         "original sentence and the other is an automatically corrected version; the order is random and is not "
         "shown. Words that differ between the two versions are printed in bold red in both.", F()),
        ("1. Column D, 'Which is better?': choose 1 if sentence 1 is better English, 2 if sentence 2 is better, or same "
         "if neither is better (both equally correct, or both equally wrong). Judge grammar, spelling, word choice "
         "and fluency.", F()),
        ("2. Column E, 'Same meaning?': choose yes if a reader would understand the same content from both versions, "
         "no if the change alters or loses meaning.", F()),
        ("3. Column F is optional, for a short comment on difficult cases.", F()),
        ("Spaces before punctuation (for example 'day .' or 'do n't') come from tokenisation. Ignore them.", F()),
        ("Work independently: do not discuss items with the other annotator, and do not open the KEY file in the "
         "folder, until both of you have finished. Only the yellow cells (columns D, E and F) should be edited.", F(bold=True)),
        ("", F()),
        ("Example (invented sentence, not part of the items)", F(bold=True)),
    ]
    for r, (t, f) in enumerate(lines, 1):
        c = ins.cell(row=r, column=1, value=t); c.font = f; c.alignment = Alignment(wrap_text=True, vertical="top")
    ex = r + 1
    for j, h in enumerate(("item", "Sentence 1", "Sentence 2", "Which is better? (1 / 2 / same)", "Same meaning? (yes / no)", "Comment (optional)"), 1):
        c = ins.cell(row=ex, column=j, value=h); c.font = F(bold=True); c.fill = HEAD; c.border = BOX; c.alignment = Alignment(wrap_text=True)
    for j, v in enumerate(("example", rich("She go to school every days .", "She goes to school every day ."),
                           rich("She goes to school every day .", "She go to school every days ."), "2", "yes", ""), 1):
        c = ins.cell(row=ex + 1, column=j, value=v); c.border = BOX; c.alignment = Alignment(wrap_text=True, vertical="top")
        if j > 3: c.fill = YELLOW; c.font = F()
    ins.cell(row=ex + 3, column=1, value="Progress").font = F(bold=True)
    ins.cell(row=ex + 4, column=1, value="Items with column D answered").font = F()
    ins.cell(row=ex + 4, column=2, value=f"=COUNTA(Annotate!D2:D{len(items) + 1})").font = F()
    ins.cell(row=ex + 5, column=1, value="Items with column E answered").font = F()
    ins.cell(row=ex + 5, column=2, value=f"=COUNTA(Annotate!E2:E{len(items) + 1})").font = F()
    ins.cell(row=ex + 6, column=1, value="Total items").font = F()
    ins.cell(row=ex + 6, column=2, value=len(items)).font = F()
    for r_ in range(1, ex):
        ins.merge_cells(start_row=r_, start_column=1, end_row=r_, end_column=6)
        ins.row_dimensions[r_].height = 15 * max(1, len(str(ins.cell(row=r_, column=1).value)) // 120 + 1)
    for col, w in zip("ABCDEF", (30, 45, 45, 18, 16, 24)): ins.column_dimensions[col].width = w

    ws = wb.create_sheet("Annotate")
    for j, h in enumerate(("item", "Sentence 1", "Sentence 2", "Which is better? (1 / 2 / same)", "Same meaning? (yes / no)", "Comment (optional)"), 1):
        c = ws.cell(row=1, column=j, value=h); c.font = F(bold=True); c.fill = HEAD; c.border = BOX
        c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
    for k, ((i, _), first) in enumerate(zip(items, order), 2):
        s1, s2 = (pred[i], src[i]) if first else (src[i], pred[i])
        vals = (k - 1, rich(s1, s2), rich(s2, s1), None, None, None)
        for j, v in enumerate(vals, 1):
            c = ws.cell(row=k, column=j, value=v); c.border = BOX
            c.alignment = Alignment(wrap_text=True, vertical="top", horizontal="center" if j in (1, 4, 5) else "left")
            if j == 1: c.font = F()
            if j > 3: c.fill = YELLOW; c.font = F()
        ws.row_dimensions[k].height = 15 * max(1, max(len(s1), len(s2)) // 55 + 1)
    for col, w in zip("ABCDEF", (6, 62, 62, 18, 16, 28)): ws.column_dimensions[col].width = w
    ws.row_dimensions[1].height = 48; ws.freeze_panes = "B2"
    last = len(items) + 1
    for rng_, opts in ((f"D2:D{last}", '"1,2,same"'), (f"E2:E{last}", '"yes,no"')):
        dv = DataValidation(type="list", formula1=opts, allow_blank=True, showErrorMessage=True,
                            errorTitle="Invalid answer", error="Please choose a value from the list.")
        ws.add_data_validation(dv); dv.add(rng_)
    wb.active = 1
    return wb

for label in ("A", "B"):
    workbook(label).save(OUT / f"HumanEval_annotator_{label}.xlsx")
with open(OUT / "KEY_do_not_open_until_both_annotators_finish.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["item", "output_is", "stratum", "bea_dev_index"])
    for k, ((i, c), first) in enumerate(zip(items, order), 1): w.writerow([k, "sentence_1" if first else "sentence_2", c, i])
json.dump({"system": "Notebook E final_e42, column " + COL, "population_edited_sentences": sum(pop.values()),
           "population_by_stratum": dict(pop), "sample_by_stratum": dict(Counter(c for _, c in items)),
           "dev_sentences": len(src), "seeds": {"stratum_sampling": 2026, "shuffle": 7, "order": 31}},
          open(OUT / "sampling_manifest.json", "w"), indent=1)
print("population (edited):", sum(pop.values()), dict(pop)); print("sample:", dict(Counter(c for _, c in items)))
print("output shown first in", sum(order), "of", len(order), "items")
