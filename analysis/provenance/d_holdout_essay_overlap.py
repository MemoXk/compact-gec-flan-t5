"""How many sentences of the ROUND-2 stage-2 holdout (Notebook D: 5% of unique source sentences) come from an essay
that also contributed training sentences? Uses the sentence -> essay map written by Notebook E (A+B+C file order).
Output: round2_results/D_holdout_essay_overlap.json"""
import csv, json
from pathlib import Path
R = Path(r"<PROJECT>/round2_results")
essay = {int(r["index"]): r["essay_id"] for r in csv.DictReader(open(R / "E/session1/wi_sentence_essay.tsv", encoding="utf-8"), delimiter="\t")}
assert len(essay) == 34308
hold = [int(x) for x in open(R / "D/wi_holdout_index.txt", encoding="utf-8").read().split()]
assert len(hold) == 1708, len(hold)
D = json.load(open(R / "D/r2D_results.json")); assert D["wi_split"]["holdout_pairs"] == 1708 and D["wi_split"]["train_pairs"] == 32600
hs = set(hold); train = [i for i in essay if i not in hs]       # Notebook D trained on every non-holdout pair (32,600)
assert len(train) == 32600
train_essays = {essay[i] for i in train}
shared = sum(essay[i] in train_essays for i in hold)
out = {"round2_holdout_pairs": len(hold), "holdout_pairs_whose_essay_also_in_training": shared,
       "share": round(shared / len(hold), 4), "holdout_essays": len({essay[i] for i in hold}),
       "note": "Notebook D held out 5% of unique source sentences; sibling sentences of the same essay stayed in training."}
json.dump(out, open(R / "D_holdout_essay_overlap.json", "w"), indent=1); print(out)
