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
from collections import Counter
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
SELECT_CHUNK_SIZE = 16
SELECT_MAX_TOKENS = 4096
CANDIDATE_SNIPPET_CHARS = 1800
MAX_NUM_SEQS = 64
CANDIDATE_VARIANTS = [
    ("baseline_deterministic", ""),
    (
        "answer_order_audit",
        "First identify every answer the problem asks for, especially each real [ANS] "
        "blank. Solve them in order and put all final sub-answers in one boxed list.",
    ),
    (
        "formula_first_exact",
        "Before arithmetic, write down the relevant formula or theorem. Keep exact "
        "values until the final step and round only when the problem explicitly asks.",
    ),
    (
        "independent_then_options",
        "Solve independently before looking at answer choices. For multiple choice, "
        "compare your result to every option and watch for common distractors.",
    ),
    (
        "sanity_check",
        "After solving, check units, signs, ranges, rounding, and whether the answer "
        "is reasonable. Correct the final answer before boxing it if the check fails.",
    ),
]
NUM_CANDIDATES = len(CANDIDATE_VARIANTS)

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
    "solutions from the same model. Re-solve the problem independently, then compare "
    "your result against the candidates. Check the math yourself; do not trust an answer "
    "only because it is stated confidently. Consensus between candidates is useful "
    "evidence, but the reasoning, answer count, order, units, rounding, and problem "
    "requirements matter more. If every candidate is flawed but you can solve the "
    "problem, return your corrected final answer. Return only the selected or corrected "
    "final answer inside \\boxed{}. For multiple sub-answers, put comma-separated values "
    "inside one \\boxed{}. For multiple-choice questions, return only the option letter "
    "inside \\boxed{}."
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


def build_candidate_prompt(tokenizer, item: dict, variant: tuple[str, str]) -> str:
    variant_name, variant_instruction = variant
    system, user = build_prompt(item["question"], item.get("options"))
    if variant_instruction:
        user = (
            f"{user}\n\n"
            f"Attempt style: {variant_name}\n"
            f"{variant_instruction}\n"
            "Follow the original problem exactly. Put the final answer inside \\boxed{}."
        )
    return build_chat_prompt(tokenizer, system, user)


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


def answer_format_hint(item: dict) -> str:
    if item.get("options"):
        return "This is a multiple-choice problem. The final answer should be one option letter."
    blank_count = item["question"].count("[ANS]")
    if blank_count == 1:
        return "This appears to request one free-form answer."
    if blank_count > 1:
        return (
            f"The prompt contains {blank_count} [ANS] placeholders. Answer the actual "
            "requested blanks in order; ignore [ANS] tokens that are only part of copied "
            "choice labels or formatting noise."
        )
    return "This is a free-form problem. Follow the requested final-answer format."


def build_selection_prompt(
    item: dict,
    candidates: list[str],
    candidate_variant_names: Optional[list[str]] = None,
) -> tuple[str, str]:
    problem = format_problem(item["question"], item.get("options"))
    candidate_blocks = []
    for idx, candidate in enumerate(candidates, start=1):
        variant_name = ""
        if candidate_variant_names and idx <= len(candidate_variant_names):
            variant_name = f" ({candidate_variant_names[idx - 1]})"
        boxed_values = extract_boxed_values(candidate)
        boxed_text = ", ".join(boxed_values[-3:]) if boxed_values else "(no boxed answer found)"
        candidate_blocks.append(
            f"Candidate {idx}{variant_name}\n"
            f"Extracted boxed answer(s): {boxed_text}\n"
            f"Visible response excerpt:\n{visible_answer_text(candidate)}"
        )

    user = (
        f"Problem:\n{problem}\n\n"
        f"Answer format hint: {answer_format_hint(item)}\n\n"
        "Candidate solutions:\n\n"
        + "\n\n".join(candidate_blocks)
        + "\n\nChoose the best final answer from these candidates, or correct them if needed. "
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


def generate_single_outputs(
    llm,
    prompts: list[str],
    sampling_params,
    chunk_size: int,
    label: str,
    progress_fp,
) -> tuple[list[str], float]:
    if not prompts:
        return [], 0.0

    outputs_text = []
    t0 = time.time()
    n = len(prompts)
    log(f"{label}: generating {n} completions in chunks of {chunk_size}.", progress_fp)

    for i in range(0, n, chunk_size):
        chunk = prompts[i : i + chunk_size]
        outputs = llm.generate(chunk, sampling_params=sampling_params, use_tqdm=False)
        outputs_text.extend([out.outputs[0].text.strip() for out in outputs])

        done = i + len(chunk)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n - done) / rate if rate > 0 else float("inf")
        log(
            f"{label} chunk {i // chunk_size + 1}: {done}/{n} completions done"
            f" | {rate * 60:.1f} completions/min"
            f" | elapsed {elapsed / 60:.1f} min, ETA {eta / 60:.1f} min",
            progress_fp,
        )

    elapsed = time.time() - t0
    log(f"{label}: finished in {elapsed / 60:.1f} min.", progress_fp)
    return outputs_text, elapsed


def score_model_response(item: dict, response: str, judger) -> Optional[bool]:
    if "answer" not in item:
        return None

    gold = item["answer"]
    if item.get("options"):
        return extract_letter(response) == str(gold).strip().upper()

    gold_list = gold if isinstance(gold, list) else [gold]
    try:
        return judger.auto_judge(
            pred=response, gold=gold_list, options=[[]] * len(gold_list),
        )
    except Exception:
        return False


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

    deterministic_sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.0,
    )
    sampled_sampling_params = SamplingParams(
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

    n = len(data)
    candidate_chunk_size = min(CHUNK_SIZE, MAX_NUM_SEQS)
    log(
        f"Engine ready. Generating {NUM_CANDIDATES} diverse candidates for {n} prompts.",
        progress_fp,
    )
    t0 = time.time()

    candidate_sets = [[] for _ in data]
    candidate_variant_sets = [[] for _ in data]

    baseline_variant = CANDIDATE_VARIANTS[0]
    baseline_prompts = [
        build_candidate_prompt(tokenizer, item, baseline_variant)
        for item in data
    ]
    baseline_outputs, baseline_secs = generate_single_outputs(
        llm=llm,
        prompts=baseline_prompts,
        sampling_params=deterministic_sampling_params,
        chunk_size=candidate_chunk_size,
        label="baseline deterministic pass",
        progress_fp=progress_fp,
    )
    for item_idx, output in enumerate(baseline_outputs):
        candidate_sets[item_idx].append(output)
        candidate_variant_sets[item_idx].append(baseline_variant[0])

    sampled_jobs = []
    sampled_prompts = []
    for item_idx, item in enumerate(data):
        for variant in CANDIDATE_VARIANTS[1:]:
            sampled_jobs.append((item_idx, variant[0]))
            sampled_prompts.append(build_candidate_prompt(tokenizer, item, variant))

    sampled_outputs, sampled_secs = generate_single_outputs(
        llm=llm,
        prompts=sampled_prompts,
        sampling_params=sampled_sampling_params,
        chunk_size=candidate_chunk_size,
        label="diverse sampled pass",
        progress_fp=progress_fp,
    )
    for (item_idx, variant_name), output in zip(sampled_jobs, sampled_outputs):
        candidate_sets[item_idx].append(output)
        candidate_variant_sets[item_idx].append(variant_name)

    candidate_secs = time.time() - t0
    log(
        f"Candidate generation finished. {n * NUM_CANDIDATES} completions for "
        f"{n} prompts in {candidate_secs / 60:.1f} min "
        f"({baseline_secs / 60:.1f} deterministic + {sampled_secs / 60:.1f} sampled).",
        progress_fp,
    )

    selection_prompts = []
    for item, candidates, candidate_variant_names in zip(
        data, candidate_sets, candidate_variant_sets
    ):
        system, user = build_selection_prompt(item, candidates, candidate_variant_names)
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
    for item, response, candidates, candidate_variant_names, selector_response in zip(
        data, responses, candidate_sets, candidate_variant_sets, selector_responses
    ):
        is_mcq = bool(item.get("options"))
        gold = item.get("answer")
        correct = score_model_response(item, response, judger)
        candidate_correct = [
            score_model_response(item, candidate, judger)
            for candidate in candidates
        ]
        oracle_correct = (
            any(value is True for value in candidate_correct)
            if "answer" in item else None
        )
        results.append({
            "id": item.get("id"),
            "is_mcq": is_mcq,
            "gold": gold,
            "candidate_variants": candidate_variant_names,
            "candidates": candidates,
            "candidate_correct": candidate_correct,
            "oracle_correct": oracle_correct,
            "selector_response": selector_response,
            "response": response,
            "correct": correct,
        })

    with open(PRED_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    scored_results = [r for r in results if r["correct"] is not None]
    mcq_res = [r for r in scored_results if r["is_mcq"]]
    free_res = [r for r in scored_results if not r["is_mcq"]]

    def acc(subset):
        return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0

    if scored_results:
        oracle_count = sum(1 for r in scored_results if r["oracle_correct"])
        oracle_acc = oracle_count / len(scored_results) * 100
        selector_missed_oracle = sum(
            r["oracle_correct"] and not r["correct"]
            for r in scored_results
        )
        variant_total = Counter()
        variant_correct = Counter()
        for r in scored_results:
            for variant_name, is_correct in zip(
                r["candidate_variants"], r["candidate_correct"]
            ):
                variant_total[variant_name] += 1
                if is_correct:
                    variant_correct[variant_name] += 1
        variant_lines = "\n".join(
            f"    {name:24s}: {variant_correct[name]:4d} / {variant_total[name]:4d}  "
            f"({variant_correct[name] / variant_total[name] * 100:.2f}%)"
            for name, _ in CANDIDATE_VARIANTS
            if variant_total[name]
        )
        accuracy_block = (
            f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)\n"
            f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)\n"
            f"  Overall    : {sum(r['correct'] for r in scored_results):4d} / {len(scored_results):4d}  ({acc(scored_results):.2f}%)\n"
            f"  Oracle@{NUM_CANDIDATES}: {oracle_count:4d} / {len(scored_results):4d}  ({oracle_acc:.2f}%)\n"
            f"  Selector missed available correct candidate: {selector_missed_oracle:4d}\n"
            f"\n"
            f"  Candidate accuracy by variant:\n{variant_lines}\n"
        )
    else:
        accuracy_block = "  No ground-truth answers found; skipped scoring and oracle diagnostics.\n"

    summary = (
        f"Diverse-candidate evaluation - {MODEL_ID} (no training)\n"
        f"GPU: {GPU_ID}\n"
        f"Test set: {TEST_PATH} ({len(results)} items, {len(scored_results)} scored)\n"
        f"Candidates per prompt: {NUM_CANDIDATES}\n"
        f"Candidate variants: {', '.join(name for name, _ in CANDIDATE_VARIANTS)}\n"
        f"Generation time: {gen_secs / 60:.1f} min\n"
        f"  Candidate pass: {candidate_secs / 60:.1f} min\n"
        f"  Selector pass : {selection_secs / 60:.1f} min\n"
        f"\n"
        f"{accuracy_block}"
    )
    log("\n" + summary, progress_fp)
    with open(ACC_PATH, "w") as f:
        f.write(summary)

    log(f"Saved predictions to {PRED_PATH}", progress_fp)
    log(f"Saved accuracy summary to {ACC_PATH}", progress_fp)
    progress_fp.close()


if __name__ == "__main__":
    main()
