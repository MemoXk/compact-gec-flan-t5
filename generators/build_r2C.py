"""Build R2_C_retrain_core.ipynb -- retrain the paper's core models in the CURRENT Kaggle environment, where the
training loss is correctly normalised by gradient accumulation (the June image multiplied it by 4).

Jobs (each one Kaggle session or less; the budget guard skips a job that will not fit):
  final  : the paper's final system -- stage 1 (clean C4 + 15% no-edit pairs) -> stage 2 (W&I+LOCNESS) ->
           threshold tuned on held-out data -> full evaluation + decode factors + blind BEA-2019 test file.
           Saves t5v3_model/, t5v3_model_stage2/, test.parquet (same names as the June notebook).
  gate   : copy gate, clean C4, no no-edit pairs   (= the paper's run 1, retrained)
  nogate : no gate,   clean C4, no no-edit pairs   (= the paper's run 2, retrained) + its own stage 2
           -> pairs with 'gate' (copy-gate result), with Notebook A's noisy_s42 (cleaning, seed 42),
              and with 'final' (no-edit pairs, end to end through stage 2).
"""
import base64, gzip, hashlib, csv, io, json, os, sys, tarfile, ast

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.dirname(HERE)
SCR = r"<SCRATCH>"
sys.path.insert(0, HERE)
import nb_common as C

# ported m2scorer (Python 3.13 has no lib2to3)
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w") as t:
    t.add(os.path.join(SCR, "eval", "m2scorer"), arcname="m2scorer")
C.DATA = C.DATA.replace("__M2_PORTED_B64__", base64.b64encode(gzip.compress(buf.getvalue())).decode())
assert "__M2_PORTED_B64__" not in C.DATA

V3 = ["".join(c.get("source", "")) for c in json.load(open(f"{P}/t5v3_final_system.ipynb", encoding="utf-8"))["cells"]]
FILTERS = V3[4].split("# Sanity check on real noise examples")[0]
run2 = list(csv.DictReader(open(f"{P}/run2_results/t5max_examples.csv", encoding="utf-8", errors="replace")))
CLEAN_TEST_SHA = hashlib.sha256("\n".join(f"{r['src']}\t{r['tgt']}" for r in run2).encode("utf-8")).hexdigest()

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}

CFG = r'''# ============================== CONFIG ==============================
import os; os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # before any CUDA use
JOBS = ["gate", "nogate"]        # SESSION 2 (the 'final' job is done).   Re-run 'final' only if asked.
QUICK_TEST = False               # True first: ~40-min smoke test that also checks the loss normalisation
SESSION_BUDGET_H = 11.6
RESULTS_NAME, EXPORT_NAME = "r2C_results.json", "r2C_results.zip"
SEED = 42
MODEL_NAME = "google/flan-t5-small"
PER_SHARD = 120_000 if not QUICK_TEST else 3_000
VAL_N = TEST_N = 25_000 if not QUICK_TEST else 1_000
C4_EVAL_N = 5_000 if not QUICK_TEST else 300          # same fixed subset as notebooks A and B
SPLIT_SEED, SUBSET_SEED = 42, 2026
TUNE_VAL_N = 2_000 if not QUICK_TEST else 200         # stage-1 threshold tuning (first rows of val, as the paper)
# recipe of the paper (verified against the June notebooks and trainer logs)
PER_DEVICE_BATCH, GRAD_ACCUM, EPOCHS = 16, 4, 1         # global batch 128 on 2 GPUs
LEARNING_RATE, WARMUP_FRAC, WEIGHT_DECAY = 3e-4, 0.06, 0.01
LABEL_SMOOTHING, SRC_DROPOUT_P, NOEDIT_FRAC = 0.1, 0.05, 0.15
EVAL_SAVE_STEPS = 2500 if not QUICK_TEST else 50        # best checkpoint by validation loss
STAGE2_EPOCHS, STAGE2_LR, STAGE2_GA = 3, 1e-4, 2         # stage-2 global batch 64
GATE_INIT_BIAS = 2.0                                     # sigmoid(2) ~ 0.88 at step 0
TAUS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]       # threshold grid of the paper (None = no threshold)
MIN_W, MAX_W, MAX_LEN_DIFF, MIN_BAG_OVERLAP, MAX_NONASCII_RATIO = 4, 50, 10, 0.5, 0.25
MAX_TRAIN_EVAL_RATIO = 2.0   # loss-normalisation guard: correct runs ~1.15, the June defect ~4.6
RUN2_CLEAN_TEST_SHA256 = "%s"
''' % CLEAN_TEST_SHA

DATA_C4 = r'''# ---------------------------------------------------------------- the clean C4-200M corpus, exactly as the paper
import hashlib
def require_c4_shards():
    for root in (Path("/kaggle/input/datasets/dariocioni/c4200m"), Path("/kaggle/input")):
        f = sorted(root.rglob("C4_200M.tsv-*")) if root.exists() else []
        if f: return f
    raise FileNotFoundError("Attach the C4-200M dataset (Add Input -> dariocioni/c4200m)")
if not (OUT / "clean_test.parquet").exists():
    files = require_c4_shards(); stats = Counter(); seen = set(); parts = []
    for i, fpath in enumerate(files):
        kept = 0; S, T = [], []
        with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                stats["scanned"] += 1
                cols = line.rstrip("\r\n").split("\t")
                if len(cols) != 2: stats["drop_format"] += 1; continue
                src, tgt = normalize_ws(cols[0]), normalize_ws(cols[1])
                if not src or not tgt or src == tgt: stats["drop_identical"] += 1; continue
                ls, lt = len(src.split()), len(tgt.split())
                if not (MIN_W <= ls <= MAX_W and MIN_W <= lt <= MAX_W): stats["drop_length"] += 1; continue
                h = hashlib.md5(src.encode("utf-8")).digest()
                if h in seen: stats["drop_duplicate"] += 1; continue
                reason = pair_noise_reason(src, tgt)
                if reason is not None: stats["drop_" + reason] += 1; continue
                seen.add(h); S.append(src); T.append(tgt); kept += 1; stats["kept"] += 1
                if kept >= PER_SHARD: break
        parts.append(pa.table({"src": pa.array(S, type=pa.large_string()), "tgt": pa.array(T, type=pa.large_string())}))
        print(f"  shard {i}: kept {kept:,} | total {stats['kept']:,}", flush=True)
    full = pa.concat_tables(parts); perm = np.random.default_rng(SPLIT_SEED).permutation(full.num_rows)
    for name, ix in (("test", perm[:TEST_N]), ("val", perm[TEST_N:TEST_N + VAL_N]), ("train", perm[TEST_N + VAL_N:])):
        pq.write_table(full.take(pa.array(ix)), OUT / f"clean_{name}.parquet")
    # the final system's no-edit pairs: target -> target, mixed into TRAIN only (t5v3_final_system.ipynb, cell 6)
    tr = pq.read_table(OUT / "clean_train.parquet"); n_ne = int(tr.num_rows * NOEDIT_FRAC)
    ne_idx = np.random.default_rng(SEED + 13).choice(tr.num_rows, n_ne, replace=False)
    tg = tr.column("tgt").take(pa.array(ne_idx))
    merged = pa.concat_tables([tr, pa.table({"src": tg, "tgt": tg})])
    merged = merged.take(pa.array(np.random.default_rng(SEED + 7).permutation(merged.num_rows)))
    pq.write_table(merged, OUT / "clean_train_noedit.parquet")
    json.dump({"filter_stats": dict(stats), "train": tr.num_rows, "train_noedit": merged.num_rows},
              open(OUT / "c4_meta_clean.json", "w"), indent=1)
    del full, tr, merged; gc.collect()
shutil.copy(OUT / "clean_test.parquet", OUT / "test.parquet")       # name expected by notebook B
te = pq.read_table(OUT / "clean_test.parquet").to_pandas()
sha = hashlib.sha256("\n".join(f"{s}\t{t}" for s, t in zip(te.src, te.tgt)).encode("utf-8")).hexdigest()
RESULTS["clean_test_reproduces_paper"] = (sha == RUN2_CLEAN_TEST_SHA256); save_results()
print("clean test set byte-identical to the paper's:", sha == RUN2_CLEAN_TEST_SHA256,
      "| train rows:", pq.read_metadata(OUT / "clean_train.parquet").num_rows,
      "| with no-edit pairs:", pq.read_metadata(OUT / "clean_train_noedit.parquet").num_rows)
SUB_IDX = np.sort(np.random.default_rng(SUBSET_SEED).permutation(TEST_N)[:C4_EVAL_N])
C4_SRC = [te.src[i] for i in SUB_IDX]; C4_TGT = [te.tgt[i] for i in SUB_IDX]
va = pq.read_table(OUT / "clean_val.parquet").to_pandas()
VAL_SRC, VAL_TGT = list(va.src[:TUNE_VAL_N]), list(va.tgt[:TUNE_VAL_N]); del te, va

# W&I+LOCNESS stage-2 data, reconstructed and split exactly as the paper (t5v3_final_system.ipynb, cell 15)
wi_src, wi_tgt = [], []
for p_ in sorted((EV / "wi+locness" / "m2").glob("*.train.gold.bea19.m2")):
    for s, ed in read_m2(p_): wi_src.append(s); wi_tgt.append(apply_m2_edits(s, ed.get(0, [])))
wperm = np.random.default_rng(SEED).permutation(len(wi_src)); hold_n = max(200, int(len(wi_src) * 0.05))
hold, trn = wperm[:hold_n], wperm[hold_n:]
if QUICK_TEST: trn = trn[:3000]; hold = hold[:200]
pq.write_table(pa.table({"src": [wi_src[i] for i in trn], "tgt": [wi_tgt[i] for i in trn]}), OUT / "wi_train.parquet")
pq.write_table(pa.table({"src": [wi_src[i] for i in hold], "tgt": [wi_tgt[i] for i in hold]}), OUT / "wi_holdout.parquet")
HOLD_SRC, HOLD_TGT = [wi_src[i] for i in hold], [wi_tgt[i] for i in hold]
print(f"W&I+LOCNESS: {len(wi_src):,} pairs -> stage-2 train {len(trn):,} | holdout {len(hold):,}")
'''

GATE = r'''# ---------------------------------------------------------------- the copy gate (the paper's module, two fixes)
# (1) num_items_in_batch is dropped before it reaches T5 (newer transformers inject it);
# (2) the model is loaded with EAGER attention -- the gate needs the cross-attention weights, which faster
#     attention backends do not return. Everything else is the June module verbatim.
GATE_SRC = """
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput
class T5WithCopyGate(T5ForConditionalGeneration):
    is_copy_gate = True
    def __init__(self, config):
        super().__init__(config)
        self.copy_gate = nn.Linear(2 * config.d_model, 1)
        nn.init.zeros_(self.copy_gate.weight); nn.init.constant_(self.copy_gate.bias, 2.0)
        self._last_pgen_mean = None
    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None, copy_src_ids=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        # generation passes use_cache / past_key_values; training and evaluation do not. Outside generation the
        # model returns ONLY loss + logits: multi-GPU DataParallel gathers every returned tensor onto GPU 0, and the
        # attentions / hidden states / cache of all layers exhausted its memory at the first step.
        gen_mode = bool(kwargs.get("use_cache")) or kwargs.get("past_key_values") is not None
        if not gen_mode: kwargs["use_cache"] = False
        if labels is not None and decoder_input_ids is None and kwargs.get("decoder_inputs_embeds") is None:
            decoder_input_ids = self._shift_right(labels)
        kwargs["output_attentions"] = True; kwargs["output_hidden_states"] = True; kwargs["return_dict"] = True
        out = super().forward(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                              labels=None, **kwargs)
        if copy_src_ids is None: copy_src_ids = input_ids
        if copy_src_ids is None:
            new_logits = out.logits
        else:
            assert out.cross_attentions is not None and out.cross_attentions[-1] is not None, \\
                "no cross-attention weights returned -- load the model with attn_implementation='eager'"
            attn = out.cross_attentions[-1].mean(dim=1).float()
            if attention_mask is not None: attn = attn * attention_mask.unsqueeze(1).float()
            enc_h = out.encoder_last_hidden_state.float(); dec_h = out.decoder_hidden_states[-1].float()
            context = torch.bmm(attn, enc_h)
            p_gen = torch.sigmoid(self.copy_gate(torch.cat([dec_h, context], dim=-1)).float())
            self._last_pgen_mean = float(p_gen.mean().detach())
            probs = torch.softmax(out.logits.float(), dim=-1) * p_gen
            idx = copy_src_ids.unsqueeze(1).expand(-1, probs.size(1), -1)
            probs.scatter_add_(2, idx, attn * (1.0 - p_gen))
            new_logits = torch.log(probs + 1e-9)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(new_logits.view(-1, new_logits.size(-1)), labels.view(-1).to(new_logits.device), ignore_index=-100)
        if not gen_mode:
            return Seq2SeqLMOutput(loss=loss, logits=new_logits)
        return Seq2SeqLMOutput(loss=loss, logits=new_logits, past_key_values=out.past_key_values,
                               decoder_hidden_states=out.decoder_hidden_states, cross_attentions=out.cross_attentions,
                               encoder_last_hidden_state=out.encoder_last_hidden_state)
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        mi = super().prepare_inputs_for_generation(input_ids, **kwargs)
        if kwargs.get("copy_src_ids") is not None: mi["copy_src_ids"] = kwargs["copy_src_ids"]
        return mi
"""
(OUT / "t5_copy_gate.py").write_text(GATE_SRC)
sys.path.insert(0, str(OUT))
from t5_copy_gate import T5WithCopyGate

# gate-aware generation: replaces t5_gen from the decoding cell (refine / decode_variants call it by name)
@torch.no_grad()
def t5_gen(tok, m, texts, gen=None, bs=GEN_BATCH, desc=""):
    kw = dict(GEN_KWARGS); kw.update(gen or {}); out = []
    for i in range(0, len(texts), bs):
        enc = tok([PREFIX + str(s) for s in texts[i:i + bs]], return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_INPUT_LEN).to(m.device)
        extra = {"copy_src_ids": enc["input_ids"]} if getattr(m, "is_copy_gate", False) else {}
        out += tok.batch_decode(m.generate(**enc, **kw, **extra), skip_special_tokens=True)
        if desc and (i // bs) % 25 == 0: print(f"   {desc} {i}/{len(texts)}", flush=True)
    return out
def load_model(path, gate=False):
    tok = AutoTokenizer.from_pretrained(str(path))
    m = (T5WithCopyGate.from_pretrained(str(path), attn_implementation="eager") if gate
         else AutoModelForSeq2SeqLM.from_pretrained(str(path)))
    return tok, m.to(DEV).eval()
'''

TRAIN = r'''# ---------------------------------------------------------------- training, with the loss-normalisation guard
from transformers import TrainerCallback
import inspect
def _drop(text, p, rng):
    w = str(text).split()
    if p <= 0 or len(w) < 4: return text
    k = [x for x in w if rng.random() > p]
    return " ".join(k) if k else text
class GECDataset(Dataset):
    def __init__(self, path, tok, dp=0.0, seed=42):
        t = pq.read_table(path); self.s, self.t, self.tok, self.dp, self.seed = t.column("src"), t.column("tgt"), tok, dp, seed
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        s, t = str(self.s[i].as_py()), str(self.t[i].as_py())
        if self.dp > 0 and s != t: s = _drop(s, self.dp, random.Random(self.seed * 1_000_003 + i))   # edit pairs only
        e = self.tok(PREFIX + s, max_length=MAX_INPUT_LEN, truncation=True)
        e["labels"] = self.tok(text_target=t, max_length=MAX_TARGET_LEN, truncation=True)["input_ids"]
        return e

class LossScaleGuard(TrainerCallback):
    """At the first evaluation, compare the recent training loss with the validation loss. A ratio near 4 means the
    loss is not divided by gradient accumulation (the June defect): stop at once instead of training for hours."""
    def __init__(self): self.train, self.ratio, self.gnorm = [], None, []
    def on_log(self, args, state, control, logs=None, **kw):
        logs = logs or {}
        if "loss" in logs: self.train.append((state.global_step, logs["loss"]))
        if "grad_norm" in logs: self.gnorm.append(logs["grad_norm"])
        if "eval_loss" in logs and self.ratio is None:
            recent = [l for s, l in self.train if s > state.global_step - max(1000, 4 * args.logging_steps)] or [l for _, l in self.train]
            if recent:
                self.ratio = float(np.mean(recent) / logs["eval_loss"])
                print(f"[guard] train/eval loss ratio at step {state.global_step}: {self.ratio:.2f} (correct ~1.1-1.3)", flush=True)
                if self.ratio > MAX_TRAIN_EVAL_RATIO:
                    print("!! LOSS-SCALING DEFECT: the training loss is not normalised by gradient accumulation. Stopping.")
                    control.should_training_stop = True
        return control

def train(tag, train_path, val_path, init, save_dir, gate=False, stage=1, seed=SEED):
    if (save_dir / "config.json").exists(): print("exists:", save_dir); return
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(str(init))
    if gate:
        m = T5WithCopyGate.from_pretrained(str(init), attn_implementation="eager")
        m.copy_gate.weight.data.zero_(); m.copy_gate.bias.data.fill_(GATE_INIT_BIAS)     # as the paper
    else:
        m = AutoModelForSeq2SeqLM.from_pretrained(str(init))
    tr = GECDataset(train_path, tok, SRC_DROPOUT_P if stage == 1 else 0.0, seed)
    va = GECDataset(val_path, tok, 0.0, seed + 1)
    if QUICK_TEST and stage == 1: va = torch.utils.data.Subset(va, range(200))
    n_gpu = max(1, torch.cuda.device_count())
    # micro-batch x accumulation; the global batch (128 stage 1, 64 stage 2) is fixed. If a configuration runs out
    # of GPU memory (the copy gate keeps several vocabulary-sized fp32 tensors per token), halve the micro-batch and
    # double the accumulation -- the same global batch and the same recipe.
    ga0 = GRAD_ACCUM if stage == 1 else STAGE2_GA
    for pdb, ga in ((PER_DEVICE_BATCH, ga0), (PER_DEVICE_BATCH // 2, ga0 * 2), (PER_DEVICE_BATCH // 4, ga0 * 4)):
        try:
            return _train_once(tag, tok, m, tr, va, save_dir, gate, stage, seed, n_gpu, pdb, ga)
        except torch.OutOfMemoryError:
            pass
        # cleanup OUTSIDE the except block: while an exception is being handled its traceback keeps the failed
        # trainer (optimizer states, activations) alive on the GPU
        print(f"!! {tag}: out of GPU memory at micro-batch {pdb} x accumulation {ga} -- retrying with half the micro-batch")
        m = None; gc.collect(); torch.cuda.empty_cache(); shutil.rmtree(OUT / f"ck_{tag}", ignore_errors=True)
        m = (T5WithCopyGate.from_pretrained(str(init), attn_implementation="eager") if gate
             else AutoModelForSeq2SeqLM.from_pretrained(str(init)))
        if gate: m.copy_gate.weight.data.zero_(); m.copy_gate.bias.data.fill_(GATE_INIT_BIAS)
    raise RuntimeError(f"{tag}: out of memory even at micro-batch {PER_DEVICE_BATCH // 4}")

def _train_once(tag, tok, m, tr, va, save_dir, gate, stage, seed, n_gpu, pdb, ga):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    total = math.ceil(len(tr) / (pdb * ga * n_gpu)) * (EPOCHS if stage == 1 else STAGE2_EPOCHS)
    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8     # T4 -> fp16, as June
    a = dict(output_dir=str(OUT / f"ck_{tag}"), learning_rate=LEARNING_RATE if stage == 1 else STAGE2_LR,
             lr_scheduler_type="cosine", per_device_train_batch_size=pdb, per_device_eval_batch_size=16 if gate else 64,
             gradient_accumulation_steps=ga, num_train_epochs=EPOCHS if stage == 1 else STAGE2_EPOCHS,
             warmup_steps=max(1, int(total * WARMUP_FRAC)), weight_decay=WEIGHT_DECAY, label_smoothing_factor=LABEL_SMOOTHING,
             logging_steps=(500 if stage == 1 else 50) if not QUICK_TEST else 10, save_total_limit=2 if stage == 1 else 1,
             predict_with_generate=False, fp16=torch.cuda.is_available() and not bf16, bf16=bf16, dataloader_num_workers=2,
             report_to=[], seed=seed, load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False)
    if stage == 1: a.update(save_strategy="steps", save_steps=EVAL_SAVE_STEPS, eval_steps=EVAL_SAVE_STEPS); ev = "steps"
    else: a.update(save_strategy="epoch"); ev = "epoch"
    a["eval_strategy" if "eval_strategy" in inspect.signature(Seq2SeqTrainingArguments.__init__).parameters else "evaluation_strategy"] = ev
    guard = LossScaleGuard()
    t = Seq2SeqTrainer(model=m, args=Seq2SeqTrainingArguments(**a), train_dataset=tr, eval_dataset=va,
                       data_collator=DataCollatorForSeq2Seq(tok, model=m, label_pad_token_id=-100), callbacks=[guard])
    t0 = time.time(); o = t.train(); dt = time.time() - t0
    rec = {"stage": stage, "gate": gate, "train_rows": len(tr), "steps": int(o.global_step), "planned_steps": total,
           "global_batch": pdb * ga * n_gpu, "micro_batch": pdb, "grad_accum": ga,
           "warmup_steps": max(1, int(total * WARMUP_FRAC)),
           "train_seconds": round(dt, 1), "examples_per_s": round(len(tr) * a["num_train_epochs"] / dt, 2),
           "train_loss": float(o.training_loss), "best_checkpoint": t.state.best_model_checkpoint,
           "best_eval_loss": t.state.best_metric, "train_eval_loss_ratio": guard.ratio,
           "median_grad_norm": float(np.median(guard.gnorm)) if guard.gnorm else None}
    if gate:
        m.eval(); b = DataCollatorForSeq2Seq(tok, model=m)([va[i] for i in range(min(32, len(va)))])
        with torch.no_grad(): m(**{k: v.to(m.device) for k, v in b.items()})
        rec["mean_p_gen_val_probe"] = m._last_pgen_mean
    RESULTS[f"train|{tag}"] = rec; save_results(); json.dump(t.state.log_history, open(OUT / f"trainer_log_{tag}.json", "w"))
    if guard.ratio is not None and guard.ratio > MAX_TRAIN_EVAL_RATIO:
        raise RuntimeError(f"{tag}: loss-scaling defect detected (ratio {guard.ratio:.2f}) -- send the log to Claude")
    t.save_model(str(save_dir)); tok.save_pretrained(str(save_dir))
    shutil.rmtree(OUT / f"ck_{tag}", ignore_errors=True); del t, m; gc.collect(); torch.cuda.empty_cache()
    print(tag, json.dumps(rec, default=str))
'''

EVAL = r'''# ---------------------------------------------------------------- evaluation
def edit_sets(srcs, tgts, preds):
    a = ann(); out = []
    for s, t, p in zip(srcs, tgts, preds):
        o = a.parse(s)
        g = set() if t == s else {(e.o_start, e.o_end, e.c_str) for e in a.annotate(o, a.parse(t)) if e.type != "noop"}
        h = set() if p == s else {(e.o_start, e.o_end, e.c_str) for e in a.annotate(o, a.parse(p)) if e.type != "noop"}
        out.append((g, h))
    return out
def tune_tau(tok, m, srcs, tgts):
    """Pick the revert threshold on HELD-OUT data (lp 0.7 + refinement), maximising edit-level F0.5 (as the paper)."""
    p = refine(tok, m, srcs, t5_gen(tok, m, srcs, desc="tune")); mg = margins(tok, m, srcs, p); es = edit_sets(srcs, tgts, p)
    best, curve = (None, -1.0), {}
    for tau in [None] + TAUS:
        tp = fp = fn = 0
        for (g, h), x in zip(es, mg):
            use = h if (tau is None or x is None or x >= tau) else set()
            tp += len(g & use); fp += len(use - g); fn += len(g - use)
        P_ = tp / max(tp + fp, 1); R_ = tp / max(tp + fn, 1); F_ = 1.25 * P_ * R_ / (0.25 * P_ + R_) if (P_ + R_) else 0.0
        curve[str(tau)] = round(F_, 4)
        if F_ > best[1]: best = (tau, F_)
    print("   tau curve:", curve, "-> selected", best[0]); return best[0], curve

def score_all(tag, tok, m, taus=(None,), refines=(True,), with_test=False, test_tau=None):
    """lp 0.7 throughout (the length-penalty sweep had no effect, notebook B). Saves every variant's predictions."""
    r = {}
    sets = (("c4", C4_SRC, None), ("bea_dev", BEA_SRC[:BEA_N], "bea"), ("conll", CONLL_SRC, "conll"), ("jfleg", JF_SRC[:JF_N], "jfleg"))
    for name, srcs, rt in sets:
        V = decode_variants(tok, m, srcs, (0.7,), refines, taus, desc=f"{tag}/{name}")
        cols = {}
        for (lp, rf, tau), p in V.items():
            k = f"ref{int(rf)}_tau{'none' if tau is None else tau}"
            if name == "c4":
                mm, _ = single_ref_errant(C4_SRC, C4_TGT, p); r[f"c4|{k}"] = {**mm, **outcome_table(C4_SRC, C4_TGT, p)}; cols[k] = p
            else:
                pr = [RETOK[rt](x) for x in p]; cols[k] = pr
                r[f"{name}|{k}"] = (errant_official(srcs, pr, BEA_DEV_M2, f"{tag}_{name}_{k}") if name == "bea_dev" else
                                    official_m2(pr, f"{tag}_{name}_{k}") if name == "conll" else jfleg_gleu(pr, JF_N))
        base = {"c4": {"test_index": SUB_IDX, "src": C4_SRC, "tgt": C4_TGT}}.get(name, {"src": srcs})
        pd.DataFrame({**base, **cols}).to_csv(OUT / f"pred_{name}_{tag}.csv", index=False)
    if with_test:
        tsrc = BEA_TEST_SRC if not QUICK_TEST else BEA_TEST_SRC[:50]; tau = test_tau     # the held-out-tuned value
        p = decode_variants(tok, m, tsrc, (0.7,), (True,), (tau,), desc=f"{tag}/bea_test")[(0.7, True, tau)]
        (OUT / "bea19_test_hyp.txt").write_text("\n".join(RETOK["bea"](x) for x in p) + "\n", encoding="utf-8")
        with zipfile.ZipFile(OUT / "bea19_test_submission_retrained.zip", "w") as z: z.write(OUT / "bea19_test_hyp.txt", "bea19_test_hyp.txt")
        r["bea_test_file"] = {"n": len(p), "tau": tau, "file": "bea19_test_submission_retrained.zip"}
    RESULTS[f"eval|{tag}"] = r; save_results()
    print(tag, json.dumps({k: (v.get("F05") if "F05" in v else v.get("GLEU")) for k, v in r.items() if isinstance(v, dict)}, default=str))
'''

JOBS = r'''# ---------------------------------------------------------------- run the jobs
NEED_H = {"final": 6.2, "gate": 5.0, "nogate": 5.3} if not QUICK_TEST else {"final": 0.5, "gate": 0.4, "nogate": 0.5}
S1, S2 = OUT / "t5v3_model", OUT / "t5v3_model_stage2"
for job in JOBS:
    if RESULTS.get(f"done|{job}"): print("done already:", job); continue
    if not budget_ok(NEED_H[job], job): continue
    if job == "final":
        train("final_s1", OUT / "clean_train_noedit.parquet", OUT / "clean_val.parquet", MODEL_NAME, S1)
        tok, m = load_model(S1)
        tau1, curve1 = tune_tau(tok, m, VAL_SRC, VAL_TGT); RESULTS["tau|final_s1"] = {"tau": tau1, "curve": curve1}
        score_all("final_s1", tok, m, taus=tuple(dict.fromkeys([None, tau1])), refines=(True,))
        del m; gc.collect(); torch.cuda.empty_cache()
        train("final_s2", OUT / "wi_train.parquet", OUT / "wi_holdout.parquet", S1, S2, stage=2)
        tok, m = load_model(S2)
        tau2, curve2 = tune_tau(tok, m, HOLD_SRC, HOLD_TGT); RESULTS["tau|final_s2"] = {"tau": tau2, "curve": curve2}
        score_all("final_s2", tok, m, taus=tuple(dict.fromkeys([tau2, None, 0.05, 0.1, 0.2])), refines=(False, True),
                  with_test=True, test_tau=tau2)
        del m; gc.collect(); torch.cuda.empty_cache()
    elif job == "gate":
        train("gate_s1", OUT / "clean_train.parquet", OUT / "clean_val.parquet", MODEL_NAME, OUT / "model_gate", gate=True)
        tok, m = load_model(OUT / "model_gate", gate=True)
        score_all("gate_s1", tok, m, taus=(None,), refines=(True,))                 # run-1/run-2 decoding, no threshold
        del m; gc.collect(); torch.cuda.empty_cache()
    elif job == "nogate":
        train("nogate_s1", OUT / "clean_train.parquet", OUT / "clean_val.parquet", MODEL_NAME, OUT / "model_nogate")
        tok, m = load_model(OUT / "model_nogate")
        score_all("nogate_s1", tok, m, taus=(None,), refines=(True,))
        del m; gc.collect(); torch.cuda.empty_cache()
        train("nogate_s2", OUT / "wi_train.parquet", OUT / "wi_holdout.parquet", OUT / "model_nogate", OUT / "model_nogate_stage2", stage=2)
        tok, m = load_model(OUT / "model_nogate_stage2")
        tau, curve = tune_tau(tok, m, HOLD_SRC, HOLD_TGT); RESULTS["tau|nogate_s2"] = {"tau": tau, "curve": curve}
        score_all("nogate_s2", tok, m, taus=tuple(dict.fromkeys([tau, None])), refines=(True,))
        del m; gc.collect(); torch.cuda.empty_cache()
    RESULTS[f"done|{job}"] = True; save_results()
'''

INTRO = md("""# Round 2 - Notebook C: retrain the core models correctly

The June Kaggle image did not divide the training loss by gradient accumulation (x4), so every model in the paper's
enhanced-system section was trained with 4x gradients clipped on every step. This notebook retrains them in the current
image, where the loss is normalised correctly - **with an automatic guard that stops a run if the defect reappears**.

| Job | What | Replaces | ~Time |
|---|---|---|---|
| `final` | stage 1 (clean C4 + 15% no-edit) -> stage 2 (W&I) -> threshold tuned on held-out data -> full evaluation, decode factors, **blind BEA-2019 test file** | the paper's final system | 6 h |
| `gate` | copy gate, clean C4 | the paper's run 1 | 4.5 h |
| `nogate` | no gate, clean C4, + its own stage 2 | the paper's run 2 (+ two-stage without no-edit pairs) | 5 h |

**Setup (Kaggle):** GPU **T4 x2** - Internet **on** - Add Input -> dataset **`dariocioni/c4200m`**.

1. `QUICK_TEST = True`, `JOBS = ["final", "gate", "nogate"]` -> *Save & Run All* (~40 min). The log must show
   `[guard] train/eval loss ratio ... 1.1-1.3` for each run and reach the last cell.
2. `QUICK_TEST = False`, `JOBS = ["final"]` -> *Save & Run All* (~6 h).
3. New version: `JOBS = ["gate", "nogate"]` -> *Save & Run All* (~9.5 h). Can run at the same time as step 2.
4. Download **`r2C_results.zip`** from each version and send them back. Session 1 also produces
   **`bea19_test_submission_retrained.zip`** - upload it to CodaBench exactly as before.""")

cells = [INTRO, code(CFG), code(C.INSTALL), code(C.IMPORTS), code(C.DATA), code(C.SCORE), code(C.T5DECODE),
         code("# ---------------------------------------------------------------- the final system's noise filters (verbatim)\n" + FILTERS),
         code(DATA_C4), code(GATE), code(TRAIN), code(EVAL), code(JOBS), code(C.EXPORT)]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = f"{P}/R2_C_retrain_core.ipynb"
json.dump(nb, open(out, "w", encoding="utf-8"), indent=1)
for c in cells:
    if c["cell_type"] == "code": ast.parse("".join(c["source"]))
ast.parse(GATE.split('GATE_SRC = """')[1].split('"""')[0].replace("\\\\", "\\"))
print("wrote", out, "-", len(cells), "cells | syntax OK (incl. the gate module)")
