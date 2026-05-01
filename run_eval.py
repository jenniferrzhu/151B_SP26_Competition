"""Run Qwen3-4B-Thinking on the held-out test split with incremental progress logging.

Outputs:
  results/{test_name}/predictions.jsonl   per-item {id, is_mcq, gold, response, candidates, correct}
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
test_name     = "Altered FRQ Prompt"
PRED_PATH     = f"results/{test_name}/predictions.jsonl"
ACC_PATH      = f"results/{test_name}/accuracy.txt"
PROGRESS_PATH = f"results/{test_name}/progress.log"
MAX_TOKENS    = 32768
CHUNK_SIZE    = 32                       # log progress every N prompts
NUM_CANDIDATES = 5
SELECT_CHUNK_SIZE = 16
SELECT_MAX_TOKENS = 4096
CANDIDATE_SNIPPET_CHARS = 1800
MAX_NUM_SEQS = 64

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}. Symbolic expressions are acceptable; you do not need to evaluate to a "
    "decimal unless the question explicitly asks for one.\n\n"
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
SYSTEM_PROMPT_SELECT = (
    "You are an expert math judge. You will see one math problem and several candidate "
    "solutions from another model. Decide which candidate final answer is most likely "
    "correct. Check the math yourself; do not trust an answer only because it is stated "
    "confidently. Consensus between candidates is useful evidence, but the reasoning and "
    "the problem requirements matter more. Return only the selected final answer inside "
    "\\boxed{}. For multiple sub-answers, put comma-separated values inside one \\boxed{}. "
    "For multiple-choice questions, return only the option letter inside \\boxed{}."
)


def format_problem(question: str, options: Optional[list]) -> str:
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return f"{question}\n\nOptions:\n{opts_text}"
    return question


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    if options:
        return SYSTEM_PROMPT_MCQ, format_problem(question, options)
    return SYSTEM_PROMPT_MATH, question


def build_chat_prompt(tokenizer, system: str, user: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_boxed_values(text: str) -> list[str]:
    values = []
    start = 0
    while True:
        idx = text.find("\\boxed{", start)
        if idx < 0:
            break
        brace_start = idx + len("\\boxed{")
        depth = 1
        i = brace_start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            values.append(text[brace_start:i - 1].strip())
        start = i
    return values


def answer_visible_text(text: str) -> str:
    think_end = text.rfind("</think>")
    return text[think_end + len("</think>"):] if think_end >= 0 else text


def visible_answer_text(text: str) -> str:
    answer_text = answer_visible_text(text)
    answer_text = answer_text.strip()
    if len(answer_text) <= CANDIDATE_SNIPPET_CHARS:
        return answer_text
    return "... " + answer_text[-CANDIDATE_SNIPPET_CHARS:]


def build_selection_prompt(item: dict, candidates: list[str]) -> tuple[str, str]:
    problem = format_problem(item["question"], item.get("options"))
    candidate_blocks = []
    for idx, candidate in enumerate(candidates, start=1):
        boxed_values = extract_boxed_values(candidate)
        boxed_text = ", ".join(boxed_values[-3:]) if boxed_values else "(no boxed answer found)"
        candidate_blocks.append(
            f"Candidate {idx}\n"
            f"Extracted boxed answer(s): {boxed_text}\n"
            f"Visible response excerpt:\n{visible_answer_text(candidate)}"
        )

    user = (
        f"Problem:\n{problem}\n\n"
        "Candidate solutions:\n\n"
        + "\n\n".join(candidate_blocks)
        + "\n\nChoose the best final answer from these candidates. "
        "Output only that answer in the required \\boxed{} format."
    )
    return SYSTEM_PROMPT_SELECT, user


def selected_or_fallback(selector_response: str, candidates: list[str]) -> str:
    if "\\boxed{" in selector_response:
        return selector_response.strip()

    m = re.search(r"\b(?:candidate|option|choice)\s*#?\s*([1-5])\b", selector_response, re.IGNORECASE)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]

    return selector_response.strip()


def extract_letter(text: str) -> str:
    search_text = answer_visible_text(text)
    m = re.search(r"\\boxed\{\s*([A-Za-z])\s*\}", search_text)
    if not m:
        m = re.search(r"\\boxed\{\s*([A-Za-z])\s*\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", search_text.upper())
    if not matches:
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
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=16384,
        kv_cache_memory_bytes=14 * 1024**3,   # 14 GiB → high concurrency on idle 24 GB GPU
    )

    sampling_params = SamplingParams(
        n=NUM_CANDIDATES,
        max_tokens=MAX_TOKENS,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
    )
    select_sampling_params = SamplingParams(
        max_tokens=SELECT_MAX_TOKENS,
        temperature=0.0,
    )

    prompts = []
    for item in data:
        system, user = build_prompt(item["question"], item.get("options"))
        prompts.append(build_chat_prompt(tokenizer, system, user))

    n = len(prompts)
    candidate_chunk_size = min(CHUNK_SIZE, max(1, MAX_NUM_SEQS // NUM_CANDIDATES))
    log(
        f"Engine ready. Generating {NUM_CANDIDATES} candidates for {n} prompts "
        f"in chunks of {candidate_chunk_size}.",
        progress_fp,
    )
    t0 = time.time()
    candidate_sets = []

    for i in range(0, n, candidate_chunk_size):
        chunk = prompts[i : i + candidate_chunk_size]
        outputs = llm.generate(chunk, sampling_params=sampling_params, use_tqdm=False)
        candidate_sets.extend([
            [candidate.text.strip() for candidate in out.outputs]
            for out in outputs
        ])

        done = i + len(chunk)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n - done) / rate if rate > 0 else float("inf")
        log(
            f"chunk {i // candidate_chunk_size + 1}: {done}/{n} prompts done"
            f" | {rate * 60:.1f} prompts/min"
            f" | elapsed {elapsed / 60:.1f} min, ETA {eta / 60:.1f} min",
            progress_fp,
        )

    candidate_secs = time.time() - t0
    log(
        f"Candidate generation finished. {n * NUM_CANDIDATES} completions for "
        f"{n} prompts in {candidate_secs / 60:.1f} min.",
        progress_fp,
    )

    selection_prompts = []
    for item, candidates in zip(data, candidate_sets):
        system, user = build_selection_prompt(item, candidates)
        selection_prompts.append(build_chat_prompt(tokenizer, system, user))

    log(
        f"Selecting final answers for {n} prompts in chunks of {SELECT_CHUNK_SIZE}.",
        progress_fp,
    )
    select_t0 = time.time()
    selector_responses = []

    for i in range(0, n, SELECT_CHUNK_SIZE):
        chunk = selection_prompts[i : i + SELECT_CHUNK_SIZE]
        outputs = llm.generate(chunk, sampling_params=select_sampling_params, use_tqdm=False)
        selector_responses.extend([out.outputs[0].text.strip() for out in outputs])

        done = i + len(chunk)
        elapsed = time.time() - select_t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n - done) / rate if rate > 0 else float("inf")
        log(
            f"select chunk {i // SELECT_CHUNK_SIZE + 1}: {done}/{n} prompts done"
            f" | {rate * 60:.1f} prompts/min"
            f" | elapsed {elapsed / 60:.1f} min, ETA {eta / 60:.1f} min",
            progress_fp,
        )

    selection_secs = time.time() - select_t0
    gen_secs = time.time() - t0
    responses = [
        selected_or_fallback(selector_response, candidates)
        for selector_response, candidates in zip(selector_responses, candidate_sets)
    ]
    log(
        f"Selection finished. Total generation time {gen_secs / 60:.1f} min "
        f"({candidate_secs / 60:.1f} candidate + {selection_secs / 60:.1f} selector).",
        progress_fp,
    )

    sys.path.insert(0, ".")
    from judger import Judger
    judger = Judger(strict_extract=False)

    log("Scoring responses...", progress_fp)
    results = []
    for item, response, candidates, selector_response in zip(
        data, responses, candidate_sets, selector_responses
    ):
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
            "candidates": candidates,
            "selector_response": selector_response,
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
        f"Candidates per prompt: {NUM_CANDIDATES}\n"
        f"Generation time: {gen_secs / 60:.1f} min\n"
        f"  Candidate pass: {candidate_secs / 60:.1f} min\n"
        f"  Selector pass : {selection_secs / 60:.1f} min\n"
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
