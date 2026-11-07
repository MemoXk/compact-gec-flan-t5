"""Score the round-3 human evaluation once BOTH annotator workbooks are complete.

Reads human_eval_round3/HumanEval_annotator_{A,B}.xlsx and the key, converts each answer into a verdict about the
system output (better / same / worse than the original sentence), and reports
  * Cohen's kappa between the annotators (verdict: 3 classes; meaning: yes/no) with a 2,000-item bootstrap 95% CI;
  * per-stratum verdict rates for each annotator and their mean;
  * population estimates for all 1,732 edited dev sentences, weighting each stratum by its frequency
    (stratified estimator; sampling rates differ by stratum, so unweighted pooled rates would be biased).
Output: human_eval_round3/human_eval_results.json
"""
import csv, json
from collections import Counter
from pathlib import Path
import numpy as np
from openpyxl import load_workbook

H = Path(r"<PROJECT>/human_eval_round3")
key = {int(r["item"]): r for r in csv.DictReader(open(H / "KEY_do_not_open_until_both_annotators_finish.csv", encoding="utf-8"))}
man = json.load(open(H / "sampling_manifest.json")); POP = man["population_by_stratum"]; NPOP = sum(POP.values())

def read(label):
    ws = load_workbook(H / f"HumanEval_annotator_{label}.xlsx", data_only=True)["Annotate"]; out = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None: continue
        b = str(row[3]).strip().lower() if row[3] is not None else ""; m = str(row[4]).strip().lower() if row[4] is not None else ""
        b = {"1.0": "1", "2.0": "2"}.get(b, b)
        out[int(row[0])] = (b, m)
    return out
ann = {l: read(l) for l in ("A", "B")}
missing = {l: [i for i in key if ann[l].get(i, ("", ""))[0] not in ("1", "2", "same") or ann[l].get(i, ("", ""))[1] not in ("yes", "no")] for l in ann}
if any(missing.values()):
    raise SystemExit(f"incomplete or invalid answers (item numbers): {missing}")

def verdict(i, b):
    if b == "same": return "same"
    out_pos = "1" if key[i]["output_is"] == "sentence_1" else "2"
    return "better" if b == out_pos else "worse"
items = sorted(key); strata = ["perfect", "improved", "lateral", "harm"]
V = {l: [verdict(i, ann[l][i][0]) for i in items] for l in ann}; M = {l: [ann[l][i][1] for i in items] for l in ann}
S = [key[i]["stratum"] for i in items]

def kappa(a, b):
    cats = sorted(set(a) | set(b)); n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b); pe = sum(ca[c] * cb[c] for c in cats) / n ** 2
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
def kappa_ci(a, b, B=2000):
    rng = np.random.default_rng(20261105); n = len(a); ks = []
    for _ in range(B):
        ix = rng.integers(0, n, n); ks.append(kappa([a[j] for j in ix], [b[j] for j in ix]))
    return [round(float(np.percentile(ks, 2.5)), 3), round(float(np.percentile(ks, 97.5)), 3)]
res = {"n_items": len(items), "agreement": {}}
for name, (a, b) in (("verdict", (V["A"], V["B"])), ("meaning", (M["A"], M["B"]))):
    res["agreement"][name] = {"percent_agreement": round(100 * sum(x == y for x, y in zip(a, b)) / len(a), 1),
                              "cohen_kappa": round(kappa(a, b), 3), "kappa_ci95": kappa_ci(a, b),
                              "confusion": {f"{x}|{y}": c for (x, y), c in sorted(Counter(zip(a, b)).items())}}
def rates(labels, values):
    c = Counter(v for s, v in zip(S, labels) if s == values); n = sum(c.values())
    return {k: round(c[k] / n, 3) for k in ("better", "same", "worse")}, n
res["per_stratum"] = {}
for s in strata:
    ra, n = rates(V["A"], s); rb, _ = rates(V["B"], s)
    ma = np.mean([m == "yes" for m, st in zip(M["A"], S) if st == s]); mb = np.mean([m == "yes" for m, st in zip(M["B"], S) if st == s])
    res["per_stratum"][s] = {"n": n, "population": POP[s], "A": ra, "B": rb,
                             "mean": {k: round((ra[k] + rb[k]) / 2, 3) for k in ra}, "meaning_kept_mean": round(float((ma + mb) / 2), 3)}
w = {s: POP[s] / NPOP for s in strata}
res["population_estimate_all_edited"] = {k: round(sum(w[s] * res["per_stratum"][s]["mean"][k] for s in strata), 3) for k in ("better", "same", "worse")}
res["population_estimate_all_edited"]["meaning_kept"] = round(sum(w[s] * res["per_stratum"][s]["meaning_kept_mean"] for s in strata), 3)
res["automatic_harm_judged_not_worse"] = round(1 - res["per_stratum"]["harm"]["mean"]["worse"], 3)
# uncertainty: stratified bootstrap (items resampled within each stratum; both annotators' answers kept together)
rng = np.random.default_rng(20261107); NB = 2000
by = {s: [k for k, x in enumerate(S) if x == s] for s in strata}
def share(ids, labels_a, labels_b, val): return (sum(labels_a[i] == val for i in ids) + sum(labels_b[i] == val for i in ids)) / (2 * len(ids))
def est(sel):
    o = {k: sum(w[s] * share(sel[s], V["A"], V["B"], k) for s in strata) for k in ("better", "same", "worse")}
    o["meaning_kept"] = sum(w[s] * share(sel[s], M["A"], M["B"], "yes") for s in strata)
    o["harm_not_worse"] = 1 - share(sel["harm"], V["A"], V["B"], "worse"); return o
boots = [est({s: list(rng.choice(by[s], len(by[s]))) for s in strata}) for _ in range(NB)]
res["population_estimate_ci95"] = {k: [round(float(np.percentile([b[k] for b in boots], 2.5)), 3),
                                       round(float(np.percentile([b[k] for b in boots], 97.5)), 3)]
                                   for k in ("better", "same", "worse", "meaning_kept")}
res["automatic_harm_judged_not_worse_ci95"] = [round(float(np.percentile([b["harm_not_worse"] for b in boots], q)), 3) for q in (2.5, 97.5)]
res["automatic_harm_share_of_edited"] = round(POP["harm"] / NPOP, 3)
res["uncertainty_note"] = ("95% percentile intervals from 2,000 stratified bootstrap resamples of items (within strata, "
                           "annotator pairs kept); sampling variability of the 175 items only")
res["note"] = ("Verdicts are about the system output relative to the original sentence. Population = the 1,732 BEA-2019 "
               "dev sentences that Notebook E final_e42 edited; weights = stratum shares of that population.")
json.dump(res, open(H / "human_eval_results.json", "w"), indent=1); print(json.dumps(res, indent=1))
# per-item judgments for release: indices and answers only (no sentence text, no comments -- they quote W&I+LOCNESS)
with open(H / "judgments_no_text.csv", "w", newline="", encoding="utf-8") as fh:
    wr = csv.writer(fh, lineterminator="\n")
    wr.writerow(["item", "bea_dev_index", "stratum", "output_is", "A_better", "A_meaning_same", "B_better", "B_meaning_same",
                 "A_verdict_for_output", "B_verdict_for_output"])
    for k, i in enumerate(items):
        wr.writerow([i, key[i]["bea_dev_index"], key[i]["stratum"], key[i]["output_is"], ann["A"][i][0], ann["A"][i][1],
                     ann["B"][i][0], ann["B"][i][1], V["A"][k], V["B"][k]])
