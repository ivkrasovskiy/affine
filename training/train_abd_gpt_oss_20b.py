# %env WANDB_API_KEY=ac693fe3f29e36f4213f091e8676869f1932416e

WANDB_PROJECT = 'abd'

import wandb, weave


wandb.init(project=WANDB_PROJECT, reinit=False)
wandb.define_metric("train/global_step")                    # x-axis
wandb.define_metric("eval/*", step_metric="train/global_step")

# --- requirements ---
# transformers >= 4.55, trl == 0.23.1, peft >= 0.11, datasets >= 2.19, bitsandbytes, unsloth

import os, re, torch, importlib.util as iu
from typing import Dict, Any, List
from tqdm.auto import tqdm, trange

from datasets import load_from_disk
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

# ---------------- Config ----------------
OUT_DIR    = os.getenv("OUT_DIR", "/workspace/outputs/abd-sft")
BATCH      = int(os.getenv("BATCH", "1"))
ACCUM      = int(os.getenv("ACCUM", "16"))
LR         = float(os.getenv("LR", "1e-5"))
EPOCHS     = float(os.getenv("EPOCHS", "1.0"))
USE_BF16   = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

# Eval generation params
GEN_MAX_NEW   = int(os.getenv("GEN_MAX_NEW", "1024"))
GEN_TEMP      = float(os.getenv("GEN_TEMP", "0.0"))
GEN_TOP_P     = float(os.getenv("GEN_TOP_P", "0.95"))
GEN_DO_SAMPLE = bool(int(os.getenv("GEN_DO_SAMPLE", "0")))  # 0/1
EVAL_STEPS    = int(os.getenv("EVAL_STEPS", "200"))
LOG_STEPS     = int(os.getenv("LOG_STEPS", "25"))
SAVE_STEPS    = int(os.getenv("SAVE_STEPS", "200"))
EVAL_SAMPLES  = int(os.getenv('EVAL_SAMPLES', '128'))

MAX_SEQ_LEN = 2048

# ---------------- Load your datasets ----------------
# Expect these were saved with columns `prompt` and `completion` (or adapt below)
train_ds = load_from_disk("/workspace/data/abd/abd_train/")
test_ds  = load_from_disk("/workspace/data/abd/abd_test/")

# ---------------- Chat tokens for GPT-OSS ----------------
USER_PREFIX         = "<|start|>user<|message|>"
ASSIST_PREFIX       = "<|start|>assistant<|channel|>final<|message|>"
END                 = "<|end|>"

def formatting_func(batch_or_example):
    """
    Robust formatter for TRL/Unsloth that accepts:
      - single example: {"messages": [ {role, content}, {role, content} ]}
      - batched list:   {"messages": [ [..pair..], [..pair..], ... ]}
      - columnar batch: {"messages": {"role": [[...],[...],...], "content": [[...],[...],...]}}
      - LazyBatch:      datasets' internal mapping (convert to dict first)

    Returns: List[str]
    """
    # --- turn LazyBatch / Mapping into a plain dict ---
    try:
        # LazyBatch supports iteration; dict(..) materializes to plain dict of columns
        obj = dict(batch_or_example)
    except Exception:
        obj = batch_or_example

    # If we still don't have a dict, it might be a direct list[dict] sample
    if not isinstance(obj, dict):
        # e.g. direct list of message dicts
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            msgs = obj
            user = (msgs[0].get("content") or "").rstrip()
            asst = (msgs[1].get("content") or "").lstrip()
            return [f"{USER_PREFIX}{user}{END}{ASSIST_PREFIX}{asst}{END}"]
        raise ValueError(f"Unsupported input type for formatting_func: {type(batch_or_example)}")

    if "messages" not in obj:
        raise ValueError("formatting_func expected key 'messages'")

    msgs_field = obj["messages"]

    def _format_one(msgs):
        # msgs may be list[dict] or columnar dict with keys role/content
        if isinstance(msgs, dict) and "role" in msgs and "content" in msgs:
            roles, contents = msgs["role"], msgs["content"]
            msgs = [{"role": r, "content": c} for r, c in zip(roles, contents)]
        # now expect list[dict] with 2 turns: user then assistant
        user = (msgs[0].get("content") or "").rstrip()
        asst = (msgs[1].get("content") or "").lstrip()
        return f"{USER_PREFIX}{user}{END}{ASSIST_PREFIX}{asst}{END}"

    # --- cases ---
    # 1) single example: list[dict]
    if isinstance(msgs_field, list) and msgs_field and isinstance(msgs_field[0], dict):
        return [_format_one(msgs_field)]

    # 2) batched list: list[list[dict]]
    if isinstance(msgs_field, list) and msgs_field and isinstance(msgs_field[0], list):
        return [_format_one(m) for m in msgs_field]

    # 3) columnar dict: {"role": [...]/[[...]], "content": [...]/[[...]]}
    if isinstance(msgs_field, dict) and "role" in msgs_field and "content" in msgs_field:
        roles, contents = msgs_field["role"], msgs_field["content"]

        # batched columnar: list[list[str]]
        if isinstance(roles, list) and roles and isinstance(roles[0], list):
            texts = []
            for r_list, c_list in zip(roles, contents):
                msgs = [{"role": r, "content": c} for r, c in zip(r_list, c_list)]
                texts.append(_format_one(msgs))
            return texts

        # single columnar: list[str]
        if isinstance(roles, list):
            msgs = [{"role": r, "content": c} for r, c in zip(roles, contents)]
            return [_format_one(msgs)]

    # fallback
    raise ValueError(f"Unsupported format for 'messages': {type(msgs_field)}")



# New: conversational dataset
def to_messages(sample):
    return {
        "messages": [
            {"role": "user", "content": (sample.get("prompt") or "").rstrip()},
            {"role": "assistant", "content": (sample.get("completion") or "").lstrip()},
        ]
    }

train_conv = train_ds.map(to_messages, remove_columns=train_ds.column_names)
test_conv  = test_ds.map(to_messages, remove_columns=test_ds.column_names)

# ---------------- Custom metric: ABD-style Exact Match on <INPUT> block ----------------
def _extract_assistant_span(txt: str) -> str:
    if ASSIST_PREFIX in txt:
        txt = txt.split(ASSIST_PREFIX, 1)[1]
    if END in txt:
        txt = txt.split(END, 1)[0]
    return txt

import re
def extract_input_from_response(response: str) -> str:
        """Pull out the last <INPUT>…</INPUT> block."""
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        response = re.sub(r"<thinking>.*?</thinking>", "", response, flags=re.DOTALL)
        matches = re.findall(r"<INPUT>(.*?)</INPUT>", response, re.IGNORECASE | re.DOTALL)
        if not matches:
            return ""
        lines = [ln.rstrip() for ln in matches[-1].strip().splitlines()]
        while lines and not lines[-1].strip():
            lines.pop()
        extracted_input = "\n".join(lines)
        return extracted_input

# Precompute eval golds in the order of test_ds_gpt
golds: List[str] = [
    extract_input_from_response(row["messages"][1]["content"]) for row in test_conv
]

from contextlib import nullcontext

class MetricCallback(TrainerCallback):
    def __init__(
        self,
        eval_rows,
        tokenizer,
        every_steps=EVAL_STEPS,
        max_samples=None,
        gen_kwargs=None,
        gen_batch_size=1,       # we generate one-by-one below; keep for future batching
        use_bf16=None,
        wandb_prefix="eval",
    ):
        self.eval_rows      = eval_rows
        self.tok            = tokenizer
        self.every          = int(every_steps)
        self.max_samples    = int(max_samples) if max_samples else None
        self.gen_kwargs     = {"max_new_tokens": GEN_MAX_NEW, "do_sample": False, "num_beams": 1}
        if gen_kwargs:
            self.gen_kwargs.update(gen_kwargs)
        self.use_bf16       = bool(use_bf16)
        self.wandb_prefix   = wandb_prefix

    def on_step_end(self, args, state, control, model=None, **kwargs):
        # only run every N steps (and not at step 0)
        if state.global_step == 0 or state.global_step % self.every != 0:
            return control

        model.eval()
        k = len(self.eval_rows) if self.max_samples is None else min(self.max_samples, len(self.eval_rows))

        matches = 0
        total   = 0

        # autocast context (bf16 if available)
        amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if self.use_bf16 else nullcontext()

        try:
            for i in trange(k, desc='Eval:'):
                messages = self.eval_rows[i]["messages"]
                gold     = golds[i]  # precomputed earlier

                # Build inputs (dict with input_ids + attention_mask)
                enc = self.tok.apply_chat_template(
                    messages,
                    add_generation_prompt = True,
                    return_tensors = "pt",
                    return_dict = True,
                    reasoning_effort = "low", # **NEW!** Set reasoning effort to low, medium or high
                ).to(model.device)

                with torch.inference_mode(), amp_ctx:
                    out = model.generate(**enc, **self.gen_kwargs)

                pred = self.tok.decode(out[0], skip_special_tokens=False)
                # Slice assistant span then <INPUT> block
                if ASSIST_PREFIX in pred:
                    pred = pred.split(ASSIST_PREFIX, 1)[1]
                if END in pred:
                    pred = pred.split(END, 1)[0]
                pred_inp = extract_input_from_response(pred).strip()
                gold_inp = extract_input_from_response(gold).strip()

                matches += int(pred_inp == gold_inp)
                total   += 1

                # free per-iteration tensors ASAP
                del enc, out
                if i % 8 == 0:
                    torch.cuda.empty_cache()

            acc = (matches / max(total, 1))

            # ---- log to W&B (and to HF state for completeness) ----
            step = int(state.global_step)
            wandb.log({f"{self.wandb_prefix}/accuracy": float(acc),
                       "train/global_step": step,
                       f"{self.wandb_prefix}/samples": int(total)},
                      step=step,
                     commit=True)
            state.log_history.append({"step": step, f"{self.wandb_prefix}/accuracy": float(acc)})

        finally:
            model.train()

        return control


packing_ok = False
attn_impl = 'eager'
if iu.find_spec("flash_attn") is not None:
    try:
        attn_impl = "flash_attention_2"
        packing_ok = True
    except Exception:
        packing_ok = False  # kernels not actually usable


# ---------------- Load Unsloth model in 4-bit & attach LoRA ----------------
model, tok = FastLanguageModel.from_pretrained(
    model_name      = "unsloth/gpt-oss-20b",   # or "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
    dtype           = None,                    # auto
    max_seq_length  = MAX_SEQ_LEN,
    load_in_4bit    = True,                    # QLoRA memory footprint
    full_finetuning = False,
    attn_implementation = attn_impl
)

#model.config.attn_implementation = attn_impl

# After loading model
# Make absolutely sure every relevant config sees FA2
try:
    model.config.attn_implementation = "flash_attention_2"
    # some HF versions cache this privately:
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "flash_attention_2"
    if hasattr(model.config, "_attn_implementation_internal"):
        model.config._attn_implementation_internal = "flash_attention_2"

    # For nested Unsloth wrappers, also propagate to the inner base model if present
    inner = getattr(model, "base_model", None)
    inner = getattr(inner, "model", inner)
    inner = getattr(inner, "model", inner)
    if hasattr(inner, "config"):
        inner.config.attn_implementation = "flash_attention_2"
        if hasattr(inner.config, "_attn_implementation"):
            inner.config._attn_implementation = "flash_attention_2"
        if hasattr(inner.config, "_attn_implementation_internal"):
            inner.config._attn_implementation_internal = "flash_attention_2"
except Exception as e:
    print("Could not force FA2 on config:", e)

print("attn impl (top):", getattr(model.config, "attn_implementation", None))
if inner and hasattr(inner, "config"):
    print("attn impl (inner):", getattr(inner.config, "attn_implementation", None))


# small tokenizer hygiene
if tok.pad_token is None and tok.eos_token is not None:
    tok.pad_token = tok.eos_token
if hasattr(tok, "add_bos_token"):
    tok.add_bos_token = False

# LoRA (you can tweak ranks etc.)
model = FastLanguageModel.get_peft_model(
    model,
    r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth",
)

# ---------------- TRL SFTConfig (pure TRL) ----------------
cfg = SFTConfig(
    output_dir=OUT_DIR,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=max(1, BATCH),  # safe default
    gradient_accumulation_steps=ACCUM,
    logging_steps=LOG_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    bf16=USE_BF16,
    fp16=not USE_BF16,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    max_grad_norm=1.0,
    report_to="wandb",

    # DATA / LOSS: mask loss to assistant span (replaces train_on_responses_only)
    max_length=MAX_SEQ_LEN,
    # completion_only_loss=True,
    # response_template=ASSIST_PREFIX,    # CRITICAL: must match your assistant prefix exactly
    assistant_only_loss=True,   # ← use this with Unsloth
    
    # EVAL: use built-in generation so compute_metrics gets preds as token-ids
    # evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    # predict_with_generate=True,
    # generation_max_length=GEN_MAX_NEW,
    # generation_num_beams=1,
    # (you can also set temperature/top_p via model.generation_config)

    # PACKING: only if FA2 present
    packing=packing_ok,
)

# compute_metrics = make_em_metrics(tok, golds)

# ---------------- Build Trainer (pure TRL) ----------------
trainer = SFTTrainer(
    model=model,
    processing_class=tok,   # new TRL API (replaces `tokenizer=...`)
    args=cfg,
    train_dataset=train_conv,
    formatting_func=formatting_func,
    # eval_dataset=test_conv,
    # compute_metrics=compute_metrics,
)

# Add OOM-safe metric callback
trainer.add_callback(
    MetricCallback(
        eval_rows=test_conv,
        tokenizer=tok,
        every_steps=EVAL_STEPS,                    # e.g., 200
        max_samples=EVAL_SAMPLES,      # keep it modest
        gen_kwargs={"max_new_tokens": GEN_MAX_NEW, "do_sample": False, "num_beams": 1},
        gen_batch_size=2,                          # start conservative; increase if stable
        use_bf16=USE_BF16,
    )
)


# optional generation defaults used by eval & your own inference later
trainer.model.generation_config.max_new_tokens = GEN_MAX_NEW
trainer.model.generation_config.temperature    = GEN_TEMP
trainer.model.generation_config.top_p          = GEN_TOP_P
trainer.model.generation_config.do_sample      = GEN_DO_SAMPLE

# ---------------- Train ----------------
os.makedirs(OUT_DIR, exist_ok=True)


try:
    trainer.train()
except KeyboardInterrupt:
    print("Training interrupted by user. Saving model...")

trainer.save_model(OUT_DIR)
tok.save_pretrained(OUT_DIR)

