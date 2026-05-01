"""QLoRA fine-tune Qwen3-4B-Thinking on data/train.jsonl.

This is a pragmatic first sketch:
- Loads the base model in 4-bit (NF4) via bitsandbytes (QLoRA)
- Adds a rank-16 LoRA adapter on attention + MLP projections
- Trains on (system + question) -> "\\boxed{<gold answer>}" pairs
- Loss is masked on the prompt, only assistant response tokens contribute

Caveat: we do NOT have ground-truth reasoning traces in train.jsonl, so the
assistant message is just the boxed answer. This trains the model to output
the correct final answer in the right format. To preserve / improve thinking
quality, the better recipe is rejection-sampling fine-tuning: generate
reasoning traces with the base model, keep only ones that reach the gold
answer, then train on (prompt, generated_reasoning + boxed_answer). That's a
follow-up — get this minimal version working first.

Outputs:
  adapters/qwen3-lora-v1/         # LoRA weights to load at inference time
  adapters/qwen3-lora-v1/training.log
"""
import json
import os
import time
from pathlib import Path
from typing import Optional

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = "7"
TRAIN_PATH  = "data/train.jsonl"
TRACES_PATH = "data/train_with_traces.jsonl"     # if exists, use these as targets (RSFT)
OUTPUT_DIR  = "adapters/qwen3-lora-v2"
LOG_PATH    = f"{OUTPUT_DIR}/training.log"

# LoRA hyperparams
LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Training hyperparams
EPOCHS         = 2
BATCH_SIZE     = 1         # per-device (lowered from 4 to fit memory)
GRAD_ACCUM     = 16        # effective batch = 16
LR             = 2e-4
WARMUP_RATIO   = 0.03
MAX_SEQ_LEN     = 6144     # prompt + truncated reasoning trace; long-tail traces filtered out
MAX_TRACE_CHARS = 14000    # drop traces longer than this to avoid OOM (~4500 tokens)

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig


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
    """Assistant message.

    If the item has a "trace" field (from gen_rsft_data.py) use that — it includes
    the model's reasoning + final boxed answer, which preserves thinking behavior.

    Otherwise fall back to just the boxed gold answer (rough, breaks reasoning).
    """
    if item.get("trace"):
        return item["trace"]
    answer = item["answer"]
    if isinstance(answer, list):
        body = ", ".join(str(a) for a in answer)
    else:
        body = str(answer)
    return f"\\boxed{{{body}}}"


def build_pair(item: dict, tokenizer) -> dict:
    """Return {prompt, completion}. Loss will be computed only on the completion."""
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


def main() -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    log_fp = open(LOG_PATH, "w", buffering=1)

    if Path(TRACES_PATH).exists():
        log(f"Loading data from {TRACES_PATH} (RSFT mode — targets include reasoning)", log_fp)
        raw = [json.loads(line) for line in open(TRACES_PATH)]
        before = len(raw)
        raw = [item for item in raw if len(item.get("trace", "")) <= MAX_TRACE_CHARS]
        log(f"  Filtered traces > {MAX_TRACE_CHARS} chars: kept {len(raw)}/{before}", log_fp)
    else:
        log(f"Loading data from {TRAIN_PATH} (no traces file — targets are bare boxed answers; reasoning will degrade)", log_fp)
        raw = [json.loads(line) for line in open(TRAIN_PATH)]
    n_mcq  = sum(1 for d in raw if d.get("options"))
    n_free = len(raw) - n_mcq
    log(f"  {len(raw)} train items ({n_mcq} MCQ, {n_free} free-form)", log_fp)

    log(f"Loading tokenizer + 4-bit base model ({MODEL_ID})", log_fp)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_list([build_pair(item, tokenizer) for item in raw])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
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
        optim="paged_adamw_8bit",        # 8-bit Adam → ~4x less optimizer state memory
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        max_length=MAX_SEQ_LEN,
        report_to="none",                # disable wandb/tensorboard
        completion_only_loss=True,       # mask loss on prompt, train only on completion tokens
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    log("Starting training...", log_fp)
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
