"""Round-3 overlap audits (reviewer 1, comments 1.5 and 1.7).

1. Cleaning ablation: does the in-domain C4 test subset occur in the training data of the UNCLEANED arm, and how much
   do the two training corpora share? (identity = shard + line of C4-200M, from Notebook A's split index files; the
   index files list the test rows first, in test-split order, then validation, then training)
2. Cross-corpus: exact source-sentence overlap between the W&I+LOCNESS training data and every evaluation set, and
   between the C4 test subset and the W&I training data (raw and lower-cased letters/digits only).
Output: round2_results/overlap_audit.json"""
import csv, gzip, json, re
from pathlib import Path

P = Path(r"<PROJECT>"); R = P / "round2_results"
D = P / "release/compact-gec-flan-t5/data"
def norm(s): return re.sub(r"[^0-9a-z]", "", s.lower())
def split_rows(path):
    rows = {"train": [], "val": [], "test": []}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        next(f)
        for ln in f:
            sh, li, sp = ln.rstrip("\n").split("\t"); rows[sp].append((int(sh), int(li)))
    return rows
clean, noisy = split_rows(R / "A_clean/split_index_clean.tsv.gz"), split_rows(R / "A_noisy/split_index_noisy.tsv.gz")
assert [len(clean[k]) for k in ("train", "val", "test")] == [1_150_000, 25_000, 25_000]
sub = [int(r["test_index"]) for r in csv.DictReader(open(R / "A_noisy/c4_eval_subset_index.csv"))]
sub_ids = {clean["test"][i] for i in sub}; test_ids = set(clean["test"])
ntr, ctr = set(noisy["train"]), set(clean["train"])
out = {"cleaning_ablation": {
    "c4_test_subset_5k_in_noisy_train": len(sub_ids & ntr),
    "c4_test_25k_in_noisy_train": len(test_ids & ntr),
    "c4_test_25k_in_noisy_val_or_test": len(test_ids & (set(noisy["val"]) | set(noisy["test"]))),
    "c4_test_25k_in_clean_train": len(test_ids & ctr),
    "train_overlap_pairs": len(ctr & ntr), "train_overlap_share_of_clean": round(len(ctr & ntr) / len(ctr), 4),
    "train_overlap_share_of_noisy": round(len(ctr & ntr) / len(ntr), 4),
    "identity": "C4-200M shard and line number (Notebook A split index files)"}}
print(json.dumps(out["cleaning_ablation"], indent=1))

# ---------------------------------------------------------------- cross-corpus source overlap
def m2_src(p): return [l[2:].rstrip("\n") for l in open(p, encoding="utf-8") if l.startswith("S ")]
wi_train = [s for lev in "ABC" for s in m2_src(D / f"wi+locness/m2/{lev}.train.gold.bea19.m2")]
evals = {"bea2019_dev": m2_src(D / "wi+locness/m2/ABCN.dev.gold.bea19.m2"),
         "bea2019_test": [l.rstrip("\n") for l in open(D / "wi+locness/test/ABCN.test.bea19.orig", encoding="utf-8")],
         "conll2014_test": m2_src(D / "conll14st-test-data/noalt/official-2014.combined.m2"),
         "jfleg_test": [l.rstrip("\n") for l in open(D / "jfleg/test.src", encoding="utf-8")],
         "c4_test_subset_5k": [r["src"] for r in csv.DictReader(open(R / "C_nogate/pred_c4_nogate_s1.csv", encoding="utf-8"))]}
raw, nrm = set(wi_train), {norm(s) for s in wi_train if len(norm(s)) >= 20}
out["wi_train_vs_evaluation_sources"] = {}
for k, srcs in evals.items():
    e = sum(s in raw for s in srcs); n = sum(norm(s) in nrm for s in srcs if len(norm(s)) >= 20)
    out["wi_train_vs_evaluation_sources"][k] = {"sentences": len(srcs), "exact_matches": e,
                                                "normalised_matches_(>=20_chars)": n}
out["note"] = ("Exact = identical tokenised source string; normalised = same lower-cased letters and digits, sentences "
               "of at least 20 such characters (short stock phrases excluded).")
print(json.dumps(out["wi_train_vs_evaluation_sources"], indent=1))
json.dump(out, open(R / "overlap_audit.json", "w"), indent=1)
