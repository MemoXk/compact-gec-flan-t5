"""Map every sentence of the W&I+LOCNESS v2.1 training M2 files (A, B, C) to the essay it comes from.

The M2 files were produced from the JSON essays by the shared task's json_to_m2.py, essay by essay and in file order,
so a sequential alignment recovers the essay of each sentence: walk the essays in order and consume M2 sentences while
the sentence (whitespace removed) can be found in the essay text (whitespace removed) at or after the current cursor.
The same function is embedded in Notebook E; run this file to check that every sentence is assigned."""
import json, re, sys
from pathlib import Path

def _compact(s): return re.sub(r"\s+", "", s)
def _norm(s):        # letters and digits only: the converter also normalised quotes, accents and dashes
    return re.sub(r"[^0-9a-z]", "", s.lower())

def m2_sources(path): return [l[2:].rstrip("\n") for l in open(path, encoding="utf-8") if l.startswith("S ")]

def essay_ids(json_path, m2_path):
    essays = [json.loads(l) for l in open(json_path, encoding="utf-8") if l.strip()]
    sents = m2_sources(m2_path)
    texts = [_norm(e["text"]) for e in essays]
    out, e, cur, unmatched = [], 0, 0, 0
    for s in sents:
        key = _norm(s)
        while e < len(essays):
            pos = texts[e].find(key, cur) if key else cur
            if pos >= 0: cur = pos + len(key); out.append(essays[e]["id"]); break
            # not in the rest of this essay: try the start of the next essay
            nxt = texts[e + 1].find(key) if e + 1 < len(essays) and key else -1
            if nxt >= 0 and nxt < 400: e += 1; cur = nxt + len(key); out.append(essays[e]["id"]); break
            unmatched += 1; out.append(essays[e]["id"]); break          # keep the current essay (tokeniser residue)
        else:
            raise RuntimeError("ran out of essays")
    return out, len(essays), unmatched, [e["id"] for e in essays], [e.get("cefr") for e in essays]

if __name__ == "__main__":
    W = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"<PROJECT>/release/compact-gec-flan-t5/data/wi+locness")
    total = 0
    for lev in "ABC":
        ids, n_essays, unmatched, all_ids, cefr = essay_ids(W / "json" / f"{lev}.train.json", W / "m2" / f"{lev}.train.gold.bea19.m2")
        used = len(set(ids)); total += len(ids)
        # an essay's sentences must be contiguous in the M2 file
        runs = sum(1 for i in range(1, len(ids)) if ids[i] != ids[i - 1]) + 1
        print(f"{lev}: {len(ids):,} sentences -> {used:,} of {n_essays:,} essays | contiguous runs {runs:,} | unmatched {unmatched} | cefr {sorted(set(map(str, cefr)))[:4]}")
    print("total sentences:", total)
