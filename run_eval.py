"""Run Qwen3-4B-Thinking on the held-out test split with incremental progress logging.

Outputs:
  results/{test_name}/predictions.jsonl   per-item {id, is_mcq, gold, response, correct}
  results/{test_name}/accuracy.txt        accuracy summary
  results/{test_name}/progress.log        live progress (one line per chunk)
"""
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID      = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID        = "6"
TEST_PATH     = "data/test.jsonl"
test_name     = "Altered2 FRQ Prompt"
PRED_PATH     = f"results/{test_name}/predictions.jsonl"
ACC_PATH      = f"results/{test_name}/accuracy.txt"
PROGRESS_PATH = f"results/{test_name}/progress.log"
MAX_TOKENS    = 32768
CHUNK_SIZE    = 32                       # log progress every N prompts

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

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


def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def log(msg: str, fp=None) -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    if fp is not None:
        fp.write(line + "\n")
        fp.flush()


def main() -> None:
    Path(PRED_PATH).parent.mkdir(parents=True, exist_ok=True)
    progress_fp = open(PROGRESS_PATH, "w", buffering=1)

    data = [json.loads(line) for line in open(TEST_PATH)]
    log(f"Loaded {len(data)} test items from {TEST_PATH}", progress_fp)

    log("Loading tokenizer + vLLM engine (CUDA graph capture takes ~1-2 min)...", progress_fp)
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
        kv_cache_memory_bytes=14 * 1024**3,   # 14 GiB → high concurrency on idle 24 GB GPU
    )

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
        outputs = llm.generate(chunk, sampling_params=sampling_params, use_tqdm=False)
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

    sys.path.insert(0, ".")
    from judger import Judger
    judger = Judger(strict_extract=False)

    log("Scoring responses...", progress_fp)
    results = []
    for item, response in zip(data, responses):
        is_mcq = bool(item.get("options"))
        gold = item["answer"]
        if is_mcq:
            correct = extract_letter(response) == str(gold).strip().upper()
        else:
            gold_list = gold if isinstance(gold, list) else [gold]
            try:
                correct = judger.auto_judge(
                    pred=response, gold=gold_list, options=[[]] * len(gold_list),
                )
            except Exception:
                correct = False
        results.append({
            "id": item.get("id"),
            "is_mcq": is_mcq,
            "gold": gold,
            "response": response,
            "correct": correct,
        })

    with open(PRED_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    mcq_res = [r for r in results if r["is_mcq"]]
    free_res = [r for r in results if not r["is_mcq"]]

    def acc(subset):
        return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0

    summary = (
        f"Baseline evaluation - {MODEL_ID} (no training)\n"
        f"GPU: {GPU_ID}\n"
        f"Test set: {TEST_PATH} ({len(results)} items)\n"
        f"Generation time: {gen_secs / 60:.1f} min\n"
        f"\n"
        f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)\n"
        f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)\n"
        f"  Overall    : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)\n"
    )
    log("\n" + summary, progress_fp)
    with open(ACC_PATH, "w") as f:
        f.write(summary)

    log(f"Saved predictions to {PRED_PATH}", progress_fp)
    log(f"Saved accuracy summary to {ACC_PATH}", progress_fp)
    progress_fp.close()


if __name__ == "__main__":
    main()
