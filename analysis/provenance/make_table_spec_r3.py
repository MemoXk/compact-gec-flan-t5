"""Write release_src/analysis/table_spec.json for the round-3 paper from the exact pairs used in E_analysis.json.
Table ids follow the paper: T3 baselines (Table 3), T4 ablations incl. the filtering policy (Table 4), T4s per-seed
filtering results quoted in the text, T5 learner-data budget (Table 5). Each table is one Holm family."""
import json
from pathlib import Path
P = Path(r"<PROJECT>")
A = json.load(open(P / "round2_results/E_analysis.json"))
def pairs_of(tab, name):
    ps = {json.dumps(r["pairs"]) for r in A["tables"][tab] if r["name"] == name}; assert len(ps) == 1, (name, ps)
    return json.loads(ps.pop())
spec = []
for name in ("human_data_stage", "noedit_two_stage", "noedit_stage1", "threshold_tau0", "refinement", "length_penalty_0.7_vs_1.0"):
    spec.append({"table": "T4", "name": name, "pairs": pairs_of("T4", name), "benchmarks": ["c4", "bea_dev", "conll"]})
cu = {r["name"]: r for r in A["cleaning_c4_uncontaminated"]["comparisons"]}
spec.append({"table": "T4", "name": "cleaning_pooled", "pairs": cu["cleaning_pooled"]["pairs"], "benchmarks": ["c4"],
             "subset": "c4_uncontaminated"})
cl = A["cleaning_learner_benchmarks"]["comparisons"]
assert len({json.dumps(r["pairs"]) for r in cl}) == 1
spec.append({"table": "T4", "name": "cleaning_pooled_learner", "pairs": cl[0]["pairs"], "benchmarks": ["bea_dev", "conll"]})
for r in A["cleaning_learner_benchmarks"]["per_seed"]:
    e = {"table": "T4s", "name": r["name"] + "_learner", "pairs": r["pairs"], "benchmarks": ["bea_dev", "conll"]}
    if e not in spec: spec.append(e)
for name in ("budget_50_vs_25", "budget_100_vs_50"):
    spec.append({"table": "T5", "name": name, "pairs": pairs_of("T4", name), "benchmarks": ["c4", "bea_dev", "conll"]})
for name in ("final_vs_coedit_large", "final_vs_gec_t5_small", "final_vs_gec_t5_base", "final_vs_qwen_fourshot"):
    spec.append({"table": "T3", "name": name, "pairs": pairs_of("T6", name), "benchmarks": ["bea_dev", "conll"]})
out = P / "round2_rescoring/release_src/analysis/table_spec.json"
json.dump(spec, open(out, "w"), indent=1); print("wrote", out, len(spec), "rows")
