"""GEPA-style prompt optimization for the CSE 151B math evaluator.

This script is intentionally self-contained: it implements the GEPA loop
locally instead of requiring a new `gepa` package install. It optimizes prompt
text on a labeled split such as data/public.jsonl, then writes a prompt config
that run_eval.py can consume with:

    $env:GEPA_PROMPT_CONFIG="results/gepa_prompt_optimization/best_prompt_config.json"
    python run_eval.py

The implemented loop follows the main GEPA ingredients:
  * evaluate prompt candidates on real examples and keep execution traces,
  * select parents from a Pareto frontier over per-example scores,
  * reflect over failures/successes in natural language,
  * mutate prompt text and retain candidates that improve locally.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID = "6"
TRAIN_PATH = "data/public.jsonl"
OUTPUT_DIR = "results/gepa_prompt_optimization"

DEFAULT_SYSTEM_PROMPT_MATH = (
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

DEFAULT_SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

DEFAULT_SYSTEM_PROMPT_SELECT = (
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

SEED_VARIANTS = [
    (
        "baseline_deterministic",
        "",
    ),
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

REFLECTION_SYSTEM_PROMPT = (
    "You are implementing GEPA, a reflective prompt optimizer. Read execution "
    "traces, identify generalizable failure causes, and propose a better prompt. "
    "Your update must help on future math problems, not memorize the shown answers. "
    "Return only a JSON object."
)


@dataclass
class PromptCandidate:
    candidate_id: str
    name: str
    math_system_prompt: str
    mcq_system_prompt: str
    variant_instruction: str
    parent_id: Optional[str] = None
    notes: str = ""
    created_at_step: int = 0
    train_scores: dict[int, float] = field(default_factory=dict)
    val_scores: dict[int, float] = field(default_factory=dict)


@dataclass
class EvalTrace:
    candidate_id: str
    item_id: int
    is_mcq: bool
    question: str
    options: Optional[list[str]]
    gold: Any
    response: str
    prediction: str
    score: Optional[bool]
    feedback: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local GEPA-style prompt optimizer for this math task."
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--gpu-id", default=GPU_ID)
    parser.add_argument("--train-path", default=TRAIN_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=151)
    parser.add_argument("--train-size", type=int, default=80)
    parser.add_argument("--val-size", type=int, default=40)
    parser.add_argument("--initial-eval-size", type=int, default=16)
    parser.add_argument("--minibatch-size", type=int, default=8)
    parser.add_argument("--max-metric-calls", type=int, default=150)
    parser.add_argument("--finalists", type=int, default=6)
    parser.add_argument("--export-variants", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--reflection-max-tokens", type=int, default=4096)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--reflection-temperature", type=float, default=0.7)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--kv-cache-gb", type=int, default=14)
    return parser.parse_args()


def log(msg: str, fp=None) -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    if fp is not None:
        fp.write(line + "\n")
        fp.flush()


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def split_items(
    items: list[dict],
    train_size: int,
    val_size: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)

    train_items = shuffled[: min(train_size, len(shuffled))]
    val_start = len(train_items)
    val_items = shuffled[val_start : val_start + val_size]
    if not val_items:
        val_items = train_items[: min(val_size, len(train_items))]
    return train_items, val_items


def format_problem(question: str, options: Optional[list]) -> str:
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{label}. {opt.strip()}" for label, opt in zip(labels, options))
        return f"{question}\n\nOptions:\n{opts_text}"
    return question


def build_chat_prompt(tokenizer, system: str, user: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_candidate_prompt(tokenizer, candidate: PromptCandidate, item: dict) -> str:
    is_mcq = bool(item.get("options"))
    system = candidate.mcq_system_prompt if is_mcq else candidate.math_system_prompt
    user = format_problem(item["question"], item.get("options"))
    if candidate.variant_instruction.strip():
        user = (
            f"{user}\n\n"
            f"Additional solving instructions ({candidate.name}):\n"
            f"{candidate.variant_instruction.strip()}\n\n"
            "Follow the original problem exactly. Put the final answer inside \\boxed{}."
        )
    return build_chat_prompt(tokenizer, system, user)


def answer_visible_text(text: str) -> str:
    think_end = text.rfind("</think>")
    return text[think_end + len("</think>") :] if think_end >= 0 else text


def response_excerpt(text: str, limit: int = 1200) -> str:
    visible = answer_visible_text(text).strip()
    if len(visible) <= limit:
        return visible
    return "... " + visible[-limit:]


def extract_letter(text: str) -> str:
    search_text = answer_visible_text(text)
    match = re.search(r"\\boxed\{\s*([A-Za-z])\s*\}", search_text)
    if not match:
        match = re.search(r"\\boxed\{\s*([A-Za-z])\s*\}", text)
    if match:
        return match.group(1).upper()

    matches = re.findall(r"\b([A-Z])\b", search_text.upper())
    if not matches:
        matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_model_response(item: dict, response: str, judger) -> Optional[bool]:
    if "answer" not in item:
        return None

    gold = item["answer"]
    if item.get("options"):
        return extract_letter(response) == str(gold).strip().upper()

    gold_list = gold if isinstance(gold, list) else [gold]
    try:
        return judger.auto_judge(
            pred=response,
            gold=gold_list,
            options=[[]] * len(gold_list),
        )
    except Exception:
        return False


def extract_prediction(item: dict, response: str, judger) -> str:
    if item.get("options"):
        return extract_letter(response)
    try:
        return judger.extract_ans(response)
    except Exception:
        return ""


def make_feedback(item: dict, response: str, score: Optional[bool], judger) -> tuple[str, str]:
    prediction = extract_prediction(item, response, judger)
    gold = item.get("answer")
    if score is True:
        return prediction, f"Correct. Predicted {prediction!r}."

    if item.get("options"):
        return (
            prediction,
            f"Incorrect multiple-choice answer. Predicted option {prediction!r}; "
            f"gold option is {gold!r}. Diagnose whether the issue was calculation, "
            "answer-choice comparison, or output formatting.",
        )

    return (
        prediction,
        f"Incorrect free-form answer. Extracted prediction {prediction!r}; "
        f"gold answer is {gold!r}. Diagnose answer count, order, algebra, rounding, "
        "units, and boxed formatting.",
    )


def generate_outputs(
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
    start = time.time()
    for i in range(0, len(prompts), chunk_size):
        chunk = prompts[i : i + chunk_size]
        outputs = llm.generate(chunk, sampling_params=sampling_params, use_tqdm=False)
        outputs_text.extend([out.outputs[0].text.strip() for out in outputs])
        done = i + len(chunk)
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0.0
        log(
            f"{label}: {done}/{len(prompts)} completions "
            f"({rate * 60:.1f}/min)",
            progress_fp,
        )
    return outputs_text, time.time() - start


def evaluate_candidate(
    llm,
    tokenizer,
    candidate: PromptCandidate,
    items: list[dict],
    sampling_params,
    args: argparse.Namespace,
    judger,
    stage: str,
    cache: dict[tuple[str, int], EvalTrace],
    progress_fp,
) -> tuple[list[EvalTrace], int, float]:
    missing_items = []
    missing_prompts = []
    for item in items:
        key = (candidate.candidate_id, int(item["id"]))
        if key not in cache:
            missing_items.append(item)
            missing_prompts.append(build_candidate_prompt(tokenizer, candidate, item))

    outputs, elapsed = generate_outputs(
        llm=llm,
        prompts=missing_prompts,
        sampling_params=sampling_params,
        chunk_size=args.chunk_size,
        label=f"{stage} {candidate.candidate_id}",
        progress_fp=progress_fp,
    )

    for item, response in zip(missing_items, outputs):
        score = score_model_response(item, response, judger)
        prediction, feedback = make_feedback(item, response, score, judger)
        trace = EvalTrace(
            candidate_id=candidate.candidate_id,
            item_id=int(item["id"]),
            is_mcq=bool(item.get("options")),
            question=item["question"],
            options=item.get("options"),
            gold=item.get("answer"),
            response=response,
            prediction=prediction,
            score=score,
            feedback=feedback,
        )
        cache[(candidate.candidate_id, int(item["id"]))] = trace

    traces = [cache[(candidate.candidate_id, int(item["id"]))] for item in items]
    score_dict = candidate.val_scores if stage == "val" else candidate.train_scores
    for trace in traces:
        score_dict[trace.item_id] = 1.0 if trace.score is True else 0.0

    return traces, len(missing_items), elapsed


def mean_trace_score(traces: list[EvalTrace]) -> float:
    if not traces:
        return 0.0
    return sum(1.0 for trace in traces if trace.score is True) / len(traces)


def candidate_score(candidate: PromptCandidate, item_ids: list[int], stage: str) -> float:
    scores = candidate.val_scores if stage == "val" else candidate.train_scores
    known = [scores[item_id] for item_id in item_ids if item_id in scores]
    if not known:
        return 0.0
    return sum(known) / len(known)


def pareto_frontier(
    candidates: list[PromptCandidate],
    item_ids: list[int],
    stage: str = "train",
) -> list[tuple[PromptCandidate, int]]:
    coverage = Counter()
    for item_id in item_ids:
        scored = []
        for candidate in candidates:
            scores = candidate.val_scores if stage == "val" else candidate.train_scores
            if item_id in scores:
                scored.append((candidate, scores[item_id]))
        if not scored:
            continue

        best = max(score for _, score in scored)
        for candidate, score in scored:
            if score == best:
                coverage[candidate.candidate_id] += 1

    frontier = [
        (candidate, coverage[candidate.candidate_id])
        for candidate in candidates
        if coverage[candidate.candidate_id] > 0
    ]
    if frontier:
        return frontier

    return [(candidate, 1) for candidate in candidates]


def choose_parent(
    candidates: list[PromptCandidate],
    train_item_ids: list[int],
    rng: random.Random,
) -> PromptCandidate:
    frontier = pareto_frontier(candidates, train_item_ids, stage="train")
    total = sum(max(1, coverage) for _, coverage in frontier)
    pick = rng.uniform(0, total)
    upto = 0.0
    for candidate, coverage in frontier:
        upto += max(1, coverage)
        if upto >= pick:
            return candidate
    return frontier[-1][0]


def sample_minibatch(
    train_items: list[dict],
    parent: PromptCandidate,
    batch_size: int,
    rng: random.Random,
) -> list[dict]:
    failed = [
        item
        for item in train_items
        if parent.train_scores.get(int(item["id"])) == 0.0
    ]
    unknown = [
        item
        for item in train_items
        if int(item["id"]) not in parent.train_scores
    ]
    correct = [
        item
        for item in train_items
        if parent.train_scores.get(int(item["id"])) == 1.0
    ]
    rng.shuffle(failed)
    rng.shuffle(unknown)
    rng.shuffle(correct)

    batch = []
    for pool in (failed, unknown, correct):
        for item in pool:
            if len(batch) >= batch_size:
                break
            if item not in batch:
                batch.append(item)
        if len(batch) >= batch_size:
            break
    return batch


def trace_block(trace: EvalTrace, index: int) -> str:
    options = ""
    if trace.options:
        options = "\nOptions:\n" + "\n".join(
            f"{chr(65 + i)}. {option}" for i, option in enumerate(trace.options)
        )
    status = "CORRECT" if trace.score is True else "INCORRECT"
    return (
        f"Example {index} [{status}]\n"
        f"Problem:\n{trace.question}{options}\n"
        f"Gold answer: {trace.gold!r}\n"
        f"Extracted prediction: {trace.prediction!r}\n"
        f"Feedback: {trace.feedback}\n"
        f"Model response excerpt:\n{response_excerpt(trace.response)}"
    )


def build_reflection_prompt(
    tokenizer,
    parent: PromptCandidate,
    traces: list[EvalTrace],
) -> str:
    trace_text = "\n\n".join(
        trace_block(trace, idx)
        for idx, trace in enumerate(traces, start=1)
    )
    user = (
        "We are optimizing prompts for a math problem solver.\n\n"
        "Current candidate:\n"
        f"Name: {parent.name}\n"
        f"Math system prompt:\n{parent.math_system_prompt}\n\n"
        f"Multiple-choice system prompt:\n{parent.mcq_system_prompt}\n\n"
        f"Additional instruction:\n{parent.variant_instruction or '(none)'}\n\n"
        f"Candidate notes:\n{parent.notes or '(none)'}\n\n"
        "Execution traces and evaluator feedback:\n\n"
        f"{trace_text}\n\n"
        "Create one mutated prompt candidate. Preserve these invariants:\n"
        "- final answers must be inside \\boxed{};\n"
        "- multiple-choice output must be exactly one option letter inside \\boxed{};\n"
        "- free-form multi-answer output must put comma-separated answers in one \\boxed{};\n"
        "- the prompt must be general and must not mention the specific gold answers above.\n\n"
        "Return only JSON with exactly these keys:\n"
        "{\n"
        '  "name": "short_snake_case_name",\n'
        '  "math_system_prompt": "complete replacement math system prompt",\n'
        '  "mcq_system_prompt": "complete replacement multiple-choice system prompt",\n'
        '  "variant_instruction": "additional instruction appended to the problem",\n'
        '  "notes": "brief explanation of the diagnosis and mutation"\n'
        "}"
    )
    return build_chat_prompt(tokenizer, REFLECTION_SYSTEM_PROMPT, user)


def parse_json_object(text: str) -> Optional[dict]:
    clean = text.strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean)
    clean = re.sub(r"\s*```$", "", clean)
    try:
        value = json.loads(clean)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    start = clean.find("{")
    if start < 0:
        return None
    for end in range(len(clean), start, -1):
        snippet = clean[start:end]
        try:
            value = json.loads(snippet)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def safe_name(text: str, fallback: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return (name or fallback)[:48]


def mutate_candidate(
    llm,
    tokenizer,
    parent: PromptCandidate,
    traces: list[EvalTrace],
    reflection_params,
    step: int,
) -> tuple[PromptCandidate, str]:
    prompt = build_reflection_prompt(tokenizer, parent, traces)
    output = llm.generate([prompt], sampling_params=reflection_params, use_tqdm=False)[0]
    reflection = output.outputs[0].text.strip()
    parsed = parse_json_object(reflection) or {}

    fallback_name = f"gepa_step_{step:03d}"
    name = safe_name(str(parsed.get("name", "")), fallback_name)
    math_prompt = str(parsed.get("math_system_prompt") or parent.math_system_prompt).strip()
    mcq_prompt = str(parsed.get("mcq_system_prompt") or parent.mcq_system_prompt).strip()
    instruction = str(parsed.get("variant_instruction") or parent.variant_instruction).strip()
    notes = str(parsed.get("notes") or response_excerpt(reflection, limit=800)).strip()

    if not instruction and not parsed:
        instruction = (
            parent.variant_instruction.strip()
            + "\n\nGEPA reflection lesson: "
            + response_excerpt(reflection, limit=600)
        ).strip()

    child = PromptCandidate(
        candidate_id=f"cand_{step:03d}",
        name=name,
        math_system_prompt=math_prompt,
        mcq_system_prompt=mcq_prompt,
        variant_instruction=instruction,
        parent_id=parent.candidate_id,
        notes=notes,
        created_at_step=step,
    )
    return child, reflection


def candidate_record(candidate: PromptCandidate) -> dict:
    return {
        "candidate_id": candidate.candidate_id,
        "name": candidate.name,
        "parent_id": candidate.parent_id,
        "created_at_step": candidate.created_at_step,
        "notes": candidate.notes,
        "math_system_prompt": candidate.math_system_prompt,
        "mcq_system_prompt": candidate.mcq_system_prompt,
        "variant_instruction": candidate.variant_instruction,
        "train_correct": int(sum(candidate.train_scores.values())),
        "train_evaluated": len(candidate.train_scores),
        "val_correct": int(sum(candidate.val_scores.values())),
        "val_evaluated": len(candidate.val_scores),
    }


def trace_record(trace: EvalTrace) -> dict:
    return {
        "candidate_id": trace.candidate_id,
        "item_id": trace.item_id,
        "is_mcq": trace.is_mcq,
        "gold": trace.gold,
        "prediction": trace.prediction,
        "score": trace.score,
        "feedback": trace.feedback,
        "response": trace.response,
    }


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def make_seed_candidates() -> list[PromptCandidate]:
    candidates = []
    for idx, (name, instruction) in enumerate(SEED_VARIANTS):
        candidates.append(
            PromptCandidate(
                candidate_id=f"seed_{idx:02d}",
                name=name,
                math_system_prompt=DEFAULT_SYSTEM_PROMPT_MATH,
                mcq_system_prompt=DEFAULT_SYSTEM_PROMPT_MCQ,
                variant_instruction=instruction,
                created_at_step=0,
            )
        )
    return candidates


def unique_candidates(candidates: list[PromptCandidate]) -> list[PromptCandidate]:
    seen = set()
    unique = []
    for candidate in candidates:
        if candidate.candidate_id in seen:
            continue
        seen.add(candidate.candidate_id)
        unique.append(candidate)
    return unique


def candidate_variant(candidate: PromptCandidate, name: str) -> dict:
    return {
        "name": name,
        "instruction": candidate.variant_instruction,
        "math_system_prompt": candidate.math_system_prompt,
        "mcq_system_prompt": candidate.mcq_system_prompt,
    }


def export_prompt_config(
    best: PromptCandidate,
    finalists: list[PromptCandidate],
    args: argparse.Namespace,
    train_items: list[dict],
    val_items: list[dict],
    output_dir: Path,
) -> Path:
    ordered = [best] + [candidate for candidate in finalists if candidate.candidate_id != best.candidate_id]
    ordered = ordered[: max(1, args.export_variants)]

    variants = []
    for idx, candidate in enumerate(ordered):
        prefix = "gepa_best" if idx == 0 else f"gepa_pareto_{idx}"
        variants.append(candidate_variant(candidate, safe_name(f"{prefix}_{candidate.name}", prefix)))

    val_ids = [int(item["id"]) for item in val_items]
    train_ids = [int(item["id"]) for item in train_items]
    config = {
        "source": "gepa_optimize.py",
        "model_id": args.model_id,
        "best_candidate_id": best.candidate_id,
        "best_candidate_name": best.name,
        "math_system_prompt": best.math_system_prompt,
        "mcq_system_prompt": best.mcq_system_prompt,
        "selector_system_prompt": DEFAULT_SYSTEM_PROMPT_SELECT,
        "candidate_variants": variants,
        "optimizer": {
            "algorithm": "local_gepa_reflective_pareto",
            "max_metric_calls": args.max_metric_calls,
            "train_path": args.train_path,
            "train_size": len(train_items),
            "val_size": len(val_items),
            "best_train_score": candidate_score(best, train_ids, "train"),
            "best_val_score": candidate_score(best, val_ids, "val"),
        },
    }

    config_path = output_dir / "best_prompt_config.json"
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)
    return config_path


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_fp = (output_dir / "progress.log").open("w", buffering=1)
    steps_path = output_dir / "steps.jsonl"
    traces_path = output_dir / "traces.jsonl"
    candidates_path = output_dir / "candidates.json"
    summary_path = output_dir / "summary.json"

    rng = random.Random(args.seed)

    log("Loading labeled optimization data.", progress_fp)
    all_items = load_jsonl(args.train_path)
    train_items, val_items = split_items(
        all_items,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
    )
    train_item_ids = [int(item["id"]) for item in train_items]
    val_item_ids = [int(item["id"]) for item in val_items]
    log(
        f"Using {len(train_items)} train items and {len(val_items)} validation items "
        f"from {args.train_path}.",
        progress_fp,
    )

    sys.path.insert(0, ".")
    from judger import Judger
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    judger = Judger(strict_extract=False)

    log("Loading tokenizer + vLLM engine.", progress_fp)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs = {
        "model": args.model_id,
        "quantization": "bitsandbytes",
        "load_format": "bitsandbytes",
        "enable_prefix_caching": False,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "trust_remote_code": True,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_model_len,
    }
    if args.kv_cache_gb > 0:
        llm_kwargs["kv_cache_memory_bytes"] = args.kv_cache_gb * 1024**3
    llm = LLM(**llm_kwargs)

    eval_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.eval_temperature,
    )
    reflection_params = SamplingParams(
        max_tokens=args.reflection_max_tokens,
        temperature=args.reflection_temperature,
        top_p=0.95,
    )

    candidates = make_seed_candidates()
    cache: dict[tuple[str, int], EvalTrace] = {}
    metric_calls = 0
    reflection_calls = 0

    initial_items = train_items[: min(args.initial_eval_size, len(train_items))]
    log(
        f"Initial GEPA seed evaluation: {len(candidates)} candidates on "
        f"{len(initial_items)} examples.",
        progress_fp,
    )
    for candidate in candidates:
        remaining = args.max_metric_calls - metric_calls
        if remaining <= 0:
            break
        eval_items = initial_items[:remaining]
        traces, calls, _ = evaluate_candidate(
            llm=llm,
            tokenizer=tokenizer,
            candidate=candidate,
            items=eval_items,
            sampling_params=eval_params,
            args=args,
            judger=judger,
            stage="train",
            cache=cache,
            progress_fp=progress_fp,
        )
        metric_calls += calls
        for trace in traces:
            append_jsonl(traces_path, trace_record(trace))
        log(
            f"{candidate.candidate_id} {candidate.name}: "
            f"{mean_trace_score(traces) * 100:.1f}% on initial examples "
            f"({metric_calls}/{args.max_metric_calls} metric calls).",
            progress_fp,
        )

    step = 1
    while metric_calls < args.max_metric_calls:
        remaining = args.max_metric_calls - metric_calls
        batch_size = max(1, min(args.minibatch_size, remaining // 2 or 1))
        parent = choose_parent(candidates, train_item_ids, rng)
        batch = sample_minibatch(train_items, parent, batch_size, rng)
        if not batch:
            break

        log(
            f"GEPA step {step}: selected {parent.candidate_id} ({parent.name}) "
            f"for a {len(batch)}-example reflective mutation.",
            progress_fp,
        )

        parent_traces, parent_calls, _ = evaluate_candidate(
            llm=llm,
            tokenizer=tokenizer,
            candidate=parent,
            items=batch,
            sampling_params=eval_params,
            args=args,
            judger=judger,
            stage="train",
            cache=cache,
            progress_fp=progress_fp,
        )
        metric_calls += parent_calls
        for trace in parent_traces:
            append_jsonl(traces_path, trace_record(trace))

        if metric_calls >= args.max_metric_calls:
            break

        child, reflection = mutate_candidate(
            llm=llm,
            tokenizer=tokenizer,
            parent=parent,
            traces=parent_traces,
            reflection_params=reflection_params,
            step=step,
        )
        reflection_calls += 1

        remaining = args.max_metric_calls - metric_calls
        child_items = batch[:remaining]
        child_traces, child_calls, _ = evaluate_candidate(
            llm=llm,
            tokenizer=tokenizer,
            candidate=child,
            items=child_items,
            sampling_params=eval_params,
            args=args,
            judger=judger,
            stage="train",
            cache=cache,
            progress_fp=progress_fp,
        )
        metric_calls += child_calls
        for trace in child_traces:
            append_jsonl(traces_path, trace_record(trace))

        parent_score = mean_trace_score(parent_traces[: len(child_traces)])
        child_score = mean_trace_score(child_traces)
        improved_any = any(
            child_trace.score is True and parent_trace.score is not True
            for parent_trace, child_trace in zip(parent_traces, child_traces)
        )
        accepted = child_score >= parent_score or improved_any
        if accepted:
            candidates.append(child)

        append_jsonl(
            steps_path,
            {
                "step": step,
                "parent_id": parent.candidate_id,
                "child_id": child.candidate_id,
                "accepted": accepted,
                "parent_score": parent_score,
                "child_score": child_score,
                "metric_calls": metric_calls,
                "reflection": reflection,
                "child": candidate_record(child),
            },
        )
        log(
            f"GEPA step {step}: child {child.candidate_id} "
            f"{'accepted' if accepted else 'rejected'} "
            f"({child_score * 100:.1f}% vs parent {parent_score * 100:.1f}%). "
            f"Metric calls: {metric_calls}/{args.max_metric_calls}.",
            progress_fp,
        )
        step += 1

    frontier = [candidate for candidate, _ in pareto_frontier(candidates, train_item_ids)]
    top_by_train = sorted(
        candidates,
        key=lambda candidate: candidate_score(candidate, train_item_ids, "train"),
        reverse=True,
    )
    finalists = unique_candidates(frontier + top_by_train)[: max(1, args.finalists)]

    log(
        f"Final validation: evaluating {len(finalists)} finalist candidates on "
        f"{len(val_items)} held-out public examples.",
        progress_fp,
    )
    validation_calls = 0
    for candidate in finalists:
        traces, calls, _ = evaluate_candidate(
            llm=llm,
            tokenizer=tokenizer,
            candidate=candidate,
            items=val_items,
            sampling_params=eval_params,
            args=args,
            judger=judger,
            stage="val",
            cache=cache,
            progress_fp=progress_fp,
        )
        validation_calls += calls
        for trace in traces:
            append_jsonl(traces_path, trace_record(trace))
        log(
            f"val {candidate.candidate_id} {candidate.name}: "
            f"{mean_trace_score(traces) * 100:.1f}%",
            progress_fp,
        )

    best = max(
        finalists,
        key=lambda candidate: (
            candidate_score(candidate, val_item_ids, "val"),
            candidate_score(candidate, train_item_ids, "train"),
        ),
    )
    config_path = export_prompt_config(
        best=best,
        finalists=finalists,
        args=args,
        train_items=train_items,
        val_items=val_items,
        output_dir=output_dir,
    )

    summary = {
        "best_candidate_id": best.candidate_id,
        "best_candidate_name": best.name,
        "best_train_score": candidate_score(best, train_item_ids, "train"),
        "best_val_score": candidate_score(best, val_item_ids, "val"),
        "training_metric_calls": metric_calls,
        "validation_metric_calls": validation_calls,
        "reflection_calls": reflection_calls,
        "num_candidates": len(candidates),
        "num_finalists": len(finalists),
        "prompt_config": str(config_path),
    }
    with candidates_path.open("w") as f:
        json.dump([candidate_record(candidate) for candidate in candidates], f, indent=2)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    log(
        f"Best candidate: {best.candidate_id} {best.name} "
        f"with validation score {summary['best_val_score'] * 100:.1f}%.",
        progress_fp,
    )
    log(f"Saved prompt config to {config_path}", progress_fp)
    log(f"Use with: $env:GEPA_PROMPT_CONFIG='{config_path}'; python run_eval.py", progress_fp)
    progress_fp.close()


if __name__ == "__main__":
    main()
