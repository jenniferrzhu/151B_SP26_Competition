"""Generate submission responses for the private test set using the v2 LoRA adapter.

Same prompt construction and sampling params as run_eval.py (Altered2 setup).
Loads the LoRA adapter from adapters/qwen3-lora-v2 at inference time.

Output format follows the starter notebook submission convention:
  {id, is_mcq, response}     — no gold/correct (private set has no answers)

Outputs:
  results/submission_lora_v2.jsonl   — final submission
  results/submission_lora_v2.log     — incremental progress
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID      = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID        = "2"
PRIVATE_PATH  = "data/private.jsonl"
ADAPTER_PATH  = "adapters/qwen3-lora-v2"
OUT_PATH      = "results/submission_lora_v2.jsonl"
PROGRESS_PATH = "results/submission_lora_v2.log"
MAX_TOKENS    = 32768
CHUNK_SIZE    = 32
LORA_RANK     = 16

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Same prompts as run_eval.py (Altered2 setup — 3 examples, no symbolic line)
SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}.\n\n"
    "Example 1 (single numeric answer):\n"
    "Problem: What angle (in degrees) corresponds to 17.4 rotations around the unit circle? "
    "17.4 rotations is an angle of [ANS] degrees.\n"
    "Final answer: \\boxed{6264}\n\n"
    "Example 2 (multiple sub-answers, one boxed):\n"
    "Problem: For each of the following, find an angle phi satisfying the given equation "
    "(round to the nearest 0.001 degrees, with 0 <= phi <= 90). "
    "(a) sin(phi) = 0.561, phi = [ANS] degrees. "
    "(b) cos(phi) = 0.612, phi = [ANS] degrees. "
    "(c) tan(phi) = 721.863, phi = [ANS] degrees.\n"
    "Final answer: \\boxed{34.125, 52.266, 89.921}\n\n"
    "Example 3 (symbolic expression answer):\n"
    "Problem: Find the half-life of an element which decays by 3.416% each day. "
    "The half-life is [ANS] days.\n"
    "Final answer: \\boxed{[ln(0.5)]/[ln(0.96584)]}"
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


def log(msg: str, fp=None) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fp is not None:
        fp.write(line + "\n")
        fp.flush()


def main() -> None:
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    progress_fp = open(PROGRESS_PATH, "w", buffering=1)

    if not Path(ADAPTER_PATH).exists():
        log(f"ERROR: adapter not found at {ADAPTER_PATH}", progress_fp)
        sys.exit(1)

    data = [json.loads(line) for line in open(PRIVATE_PATH)]
    n_mcq = sum(1 for d in data if d.get("options"))
    n_free = len(data) - n_mcq
    log(f"Loaded {len(data)} private items ({n_mcq} MCQ, {n_free} free-form)", progress_fp)

    log(f"Loading tokenizer + vLLM engine with LoRA ({ADAPTER_PATH})...", progress_fp)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=MODEL_ID,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=False,
        gpu_memory_utilization=0.85,
        max_model_len=16384,
        trust_remote_code=True,
        max_num_seqs=64,
        max_num_batched_tokens=16384,
        kv_cache_memory_bytes=14 * 1024**3,
        enable_lora=True,
        max_lora_rank=LORA_RANK,
        max_loras=1,
    )

    lora_request = LoRARequest(lora_name="trained_v2", lora_int_id=1, lora_path=ADAPTER_PATH)

    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.6,
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
    log(f"Engine ready. Generating responses for {n} prompts in chunks of {CHUNK_SIZE}.", progress_fp)
    t0 = time.time()
    responses = []

    for i in range(0, n, CHUNK_SIZE):
        chunk = prompts[i : i + CHUNK_SIZE]
        outputs = llm.generate(
            chunk,
            sampling_params=sampling_params,
            lora_request=lora_request,
            use_tqdm=False,
        )
        responses.extend([out.outputs[0].text.strip() for out in outputs])

        done = i + len(chunk)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n - done) / rate if rate > 0 else float("inf")
        log(
            f"chunk {i // CHUNK_SIZE + 1}: {done}/{n} prompts done"
            f" | {rate * 60:.1f} prompts/min"
            f" | elapsed {elapsed / 60:.1f} min, ETA {eta / 60:.1f} min",
            progress_fp,
        )

    gen_secs = time.time() - t0
    log(f"Generation finished. {n} prompts in {gen_secs / 60:.1f} min.", progress_fp)

    log(f"Writing submission file to {OUT_PATH}", progress_fp)
    with open(OUT_PATH, "w") as f:
        for item, response in zip(data, responses):
            record = {
                "id": item.get("id"),
                "is_mcq": bool(item.get("options")),
                "response": response,
            }
            f.write(json.dumps(record) + "\n")
    log(f"Saved {n} records to {OUT_PATH}.", progress_fp)
    progress_fp.close()


if __name__ == "__main__":
    main()
