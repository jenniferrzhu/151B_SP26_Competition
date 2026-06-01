"""Rejection-sampling fine-tuning data generation.

Run the BASE model on data/train.jsonl with K samples per prompt, score each
sample against the gold answer, and keep one correct trace per item. The
resulting traces include the model's reasoning, which preserves thinking
behavior when used as fine-tuning targets.

Outputs:
  data/train_with_traces.jsonl    one row per item that produced >=1 correct trace
                                  fields: {id, is_mcq, gold, prompt, trace, ...meta}
  results/rsft_gen.log            generation progress + stats
"""
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = "7"
TRAIN_PATH  = "data/train.jsonl"
OUT_PATH    = "data/train_with_traces.jsonl"
LOG_PATH    = "results/rsft_gen.log"
N_SAMPLES   = 4                   # candidates per prompt
MAX_TOKENS  = 8192                # most reasoning fits in 8k; saves time + KV cache
CHUNK_SIZE  = 32

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

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


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def log(msg: str, fp=None) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fp is not None:
        fp.write(line + "\n")
        fp.flush()


def main() -> None:
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(LOG_PATH, "w", buffering=1)

    data = [json.loads(line) for line in open(TRAIN_PATH)]
    log(f"Loaded {len(data)} train items from {TRAIN_PATH}", log_fp)

    log(f"Loading tokenizer + vLLM engine (BASE model, no LoRA)...", log_fp)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=MODEL_ID,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=True,            # share prefix across n=K samples
        gpu_memory_utilization=0.85,
        max_model_len=MAX_TOKENS + 2048,       # response + room for prompt
        trust_remote_code=True,
        max_num_seqs=64,
        max_num_batched_tokens=16384,
        kv_cache_memory_bytes=14 * 1024**3,
    )

    sampling_params = SamplingParams(
        n=N_SAMPLES,                # <-- generate K samples per prompt in one call
        max_tokens=MAX_TOKENS,
        temperature=0.7,            # slightly higher for diversity across K samples
        top_p=0.95,
        top_k=20,
        min_p=0.0,
    )

    prompts = []
    for item in data:
        system, user = build_prompt(item["question"], item.get("options"))
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        ))

    n = len(prompts)
    log(f"Engine ready. Generating {N_SAMPLES} samples for each of {n} prompts in chunks of {CHUNK_SIZE}.", log_fp)
    log(f"  (Total candidate generations: {n * N_SAMPLES})", log_fp)

    sys.path.insert(0, ".")
    from judger import Judger
    judger = Judger(strict_extract=False)

    out_fp = open(OUT_PATH, "w")
    n_with_correct = 0
    n_total_correct = 0
    t0 = time.time()

    for i in range(0, n, CHUNK_SIZE):
        chunk_prompts = prompts[i : i + CHUNK_SIZE]
        chunk_items = data[i : i + CHUNK_SIZE]
        outputs = llm.generate(chunk_prompts, sampling_params=sampling_params, use_tqdm=False)

        for item, out in zip(chunk_items, outputs):
            is_mcq = bool(item.get("options"))
            gold = item["answer"]
            candidates = [c.text.strip() for c in out.outputs]

            # score each candidate
            correct_traces = []
            for cand in candidates:
                if is_mcq:
                    is_correct = extract_letter(cand) == str(gold).strip().upper()
                else:
                    gold_list = gold if isinstance(gold, list) else [gold]
                    try:
                        is_correct = judger.auto_judge(
                            pred=cand, gold=gold_list, options=[[]] * len(gold_list),
                        )
                    except Exception:
                        is_correct = False
                if is_correct:
                    correct_traces.append(cand)

            n_total_correct += len(correct_traces)
            if correct_traces:
                n_with_correct += 1
                # keep the SHORTEST correct trace (less noise, faster training)
                trace = min(correct_traces, key=len)
                out_fp.write(json.dumps({
                    "id": item.get("id"),
                    "is_mcq": is_mcq,
                    "gold": gold,
                    "question": item["question"],
                    "options": item.get("options"),
                    "trace": trace,
                    "n_correct_of_k": f"{len(correct_traces)}/{N_SAMPLES}",
                }) + "\n")
                out_fp.flush()

        done = i + len(chunk_prompts)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n - done) / rate if rate > 0 else float("inf")
        log(
            f"chunk {i // CHUNK_SIZE + 1}: {done}/{n} prompts processed"
            f" | {n_with_correct} kept ({n_with_correct / done * 100:.1f}%)"
            f" | {n_total_correct}/{done * N_SAMPLES} correct candidates ({n_total_correct / (done * N_SAMPLES) * 100:.1f}%)"
            f" | rate {rate * 60:.1f} prompts/min, ETA {eta / 60:.1f} min",
            log_fp,
        )

    out_fp.close()
    gen_secs = time.time() - t0

    log(f"\nDone in {gen_secs / 60:.1f} min.", log_fp)
    log(f"  {n_with_correct}/{n} items have >=1 correct trace ({n_with_correct / n * 100:.1f}%)", log_fp)
    log(f"  {n_total_correct}/{n * N_SAMPLES} candidates correct ({n_total_correct / (n * N_SAMPLES) * 100:.1f}%)", log_fp)
    log(f"  Wrote {OUT_PATH}", log_fp)
    log_fp.close()


if __name__ == "__main__":
    main()
