"""QLoRA fine-tune Qwen3-4B-Thinking, with either SFT or DPO objective.

Switch objective by editing MODE at the top of the file:
- MODE = "sft": supervised fine-tune on teacher traces
    Reads:  data/train_with_traces.jsonl  (base-model RSFT traces, keeps thinking style)
            data/train_math.jsonl         (Hendrycks MATH Levels 4-5, contamination-filtered)
    Writes: adapters/qwen3-lora-v3/
- MODE = "dpo": direct preference optimization on (chosen, rejected) pairs
    Reads:  data/preference_pairs.jsonl   (built by build_pref_pairs.py)
    Writes: adapters/qwen3-lora-v4-dpo/

Common QLoRA setup either way: NF4 base + bf16 compute + rank-16 LoRA on the
seven attention / MLP projections, paged 8-bit AdamW, gradient checkpointing.
"""
import json
import os
import time
from pathlib import Path

# ── Mode ──────────────────────────────────────────────────────────────────────
MODE = "sft"   # "sft" or "dpo"

# ── Common configuration ──────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = "0"

# LoRA hyperparams (shared between SFT and DPO so adapters are comparable)
LORA_R         = 16
LORA_ALPHA     = 16        # was 32; lower α/r ratio = less aggressive updates (v6 conservative)
LORA_DROPOUT   = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Shared training hyperparams
EPOCHS         = 2
BATCH_SIZE     = 1         # per-device
GRAD_ACCUM     = 16        # effective batch = 16
MAX_SEQ_LEN    = 6144      # prompt + completion combined; long tails get truncated

# ── Mode-specific configuration ───────────────────────────────────────────────
if MODE == "sft":
    # SFT sources — set to None to skip
    RSFT_TRACES_PATH     = "data/train_with_traces.jsonl"          # native style + MCQ anchor (132 MCQ items)
    MATH_TRACES_PATH     = None                                    # Hendrycks MATH Levels 4-5 (v3 source, disabled)
    DEEPMATH_TRACES_PATH = "data/deepmath_subset_fmtfix.jsonl"     # DeepMath R1 traces with intermediate \boxed{} stripped
    OUTPUT_DIR       = "adapters/qwen3-lora-v6-mixed-fmtfix"
    LR              = 5e-5            # was 2e-4 (v5); slower to limit style drift
    WARMUP_RATIO    = 0.1             # was 0.03 (v5); more steps to find good local minima
    MAX_TRACE_CHARS = 14000  # drop traces longer than this
elif MODE == "dpo":
    PAIRS_PATH      = "data/preference_pairs.jsonl"
    OUTPUT_DIR      = "adapters/qwen3-lora-v4-dpo"
    LR              = 5e-6   # DPO is much more sensitive than SFT
    WARMUP_RATIO    = 0.1
    BETA            = 0.1    # KL strength against reference policy
else:
    raise SystemExit(f"Unknown MODE={MODE!r}; expected 'sft' or 'dpo'")

LOG_PATH = f"{OUTPUT_DIR}/training.log"

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

# SFT prompts (used only for MODE="sft" — DPO pairs come pre-formatted from build_pref_pairs.py)
SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)
SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)


def format_target(item: dict) -> str:
    """SFT assistant message: prefer the teacher trace, fall back to a bare \\boxed{}."""
    if item.get("trace"):
        return item["trace"]
    answer = item["answer"]
    body = ", ".join(str(a) for a in answer) if isinstance(answer, list) else str(answer)
    return f"\\boxed{{{body}}}"


def build_sft_pair(item: dict, tokenizer) -> dict:
    if item.get("options"):
        labels = [chr(65 + i) for i in range(len(item["options"]))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, item["options"]))
        user = f"{item['question']}\n\nOptions:\n{opts_text}"
        system = SYSTEM_PROMPT_MCQ
    else:
        user = item["question"]
        system = SYSTEM_PROMPT_MATH
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    completion = format_target(item) + tokenizer.eos_token
    return {"prompt": prompt, "completion": completion}


def log(msg: str, fp=None) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fp is not None:
        fp.write(line + "\n")
        fp.flush()


def load_sft_dataset(tokenizer, log_fp) -> Dataset:
    raw = []
    for src_path, label in [
        (RSFT_TRACES_PATH,     "RSFT base-model traces"),
        (MATH_TRACES_PATH,     "Hendrycks MATH Levels 4-5"),
        (DEEPMATH_TRACES_PATH, "DeepMath-103K R1 traces"),
    ]:
        if not src_path:
            continue
        if not Path(src_path).exists():
            log(f"  WARNING: {src_path} not found — skipping {label}", log_fp)
            continue
        items = [json.loads(line) for line in open(src_path)]
        before = len(items)
        items = [it for it in items if it.get("trace") and len(it["trace"]) <= MAX_TRACE_CHARS]
        log(f"  {src_path}: {len(items)}/{before} items kept (have trace, ≤{MAX_TRACE_CHARS} chars) [{label}]", log_fp)
        raw.extend(items)
    n_mcq  = sum(1 for d in raw if d.get("options"))
    n_free = len(raw) - n_mcq
    log(f"Combined SFT dataset: {len(raw)} items ({n_mcq} MCQ, {n_free} free-form)", log_fp)
    return Dataset.from_list([build_sft_pair(item, tokenizer) for item in raw])


def load_dpo_dataset(log_fp) -> Dataset:
    pairs = [json.loads(l) for l in open(PAIRS_PATH)]
    log(f"Loaded {len(pairs)} preference pairs from {PAIRS_PATH}", log_fp)
    n_mcq = sum(1 for p in pairs if p.get("is_mcq"))
    log(f"  {n_mcq} MCQ pairs, {len(pairs) - n_mcq} free-form pairs", log_fp)
    return Dataset.from_list(
        [{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]} for p in pairs]
    )


def main() -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    log_fp = open(LOG_PATH, "w", buffering=1)
    log(f"MODE = {MODE}", log_fp)

    log(f"Loading tokenizer + 4-bit base model ({MODEL_ID})", log_fp)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if MODE == "sft":
        dataset = load_sft_dataset(tokenizer, log_fp)
    else:
        dataset = load_dpo_dataset(log_fp)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,   # match vLLM inference default
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    log("Attaching LoRA adapter", log_fp)
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    log(f"  trainable params: {trainable / 1e6:.2f}M / {total / 1e6:.0f}M ({trainable / total * 100:.3f}%)", log_fp)

    if MODE == "sft":
        from trl import SFTTrainer, SFTConfig
        sft_config = SFTConfig(
            output_dir=OUTPUT_DIR,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LR,
            warmup_ratio=WARMUP_RATIO,
            lr_scheduler_type="cosine",
            bf16=True,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            logging_steps=5,
            save_strategy="epoch",
            save_total_limit=2,
            max_length=MAX_SEQ_LEN,
            report_to="none",
            completion_only_loss=True,
        )
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=sft_config,
        )
    else:
        from trl import DPOTrainer, DPOConfig
        dpo_config = DPOConfig(
            output_dir=OUTPUT_DIR,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LR,
            warmup_ratio=WARMUP_RATIO,
            lr_scheduler_type="cosine",
            beta=BETA,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="paged_adamw_8bit",
            logging_steps=5,
            save_strategy="epoch",
            save_total_limit=2,
            max_length=MAX_SEQ_LEN,
            report_to="none",
        )
        # ref_model=None + PEFT model → TRL builds the reference by disabling the LoRA adapter,
        # avoiding a second full copy of the 4B model in VRAM.
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

    log(f"Starting {MODE.upper()} training...", log_fp)
    t0 = time.time()
    trainer.train()
    log(f"Training finished in {(time.time() - t0) / 60:.1f} min", log_fp)

    log(f"Saving adapter to {OUTPUT_DIR}", log_fp)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    log("Done.", log_fp)
    log_fp.close()


if __name__ == "__main__":
    main()
