"""
Consolidated Submission Script for CSE 151B Math Reasoning Competition.
Matches the logic and prompting of run_eval_hybrid_mcq_lora_frq.py.

Performs Hybrid Inference:
- MCQ: LoRA v6 Adapter, sampled (temperature 0.6).
- FRQ: Base Model, Multi-Prompt Ensemble (5 variants) + Selector.
- Output: submission.jsonl and submission.csv (id, response).
"""

import json
import os
import re
import sys
import time
import csv
import argparse
from collections import Counter
from pathlib import Path
from typing import Optional, List, Tuple

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID      = "Qwen/Qwen3-4B-Thinking-2507"
MCQ_ADAPTER_PATH = "adapters/qwen3-lora-v6-mixed-fmtfix"
LORA_RANK     = 16

MAX_TOKENS    = 32768
CHUNK_SIZE    = 32
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

# ── Prompts ──────────────────────────────────────────────────────────────────
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
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}. "
    "If you must reason, keep it brief and ensure the final letter is at the very end in its own box."
)

SYSTEM_PROMPT_SELECT = (
    "You are an expert math judge. You will see one math problem and 5 candidate "
    "solutions from the same model. Analyze each candidate for correctness and formatting. "
    "If one or more candidates are correct, state which one is best (e.g., 'The best choice is Candidate #1.') "
    "and provide the final answer clearly in a single \\boxed{} block. "
    "If multiple sub-answers are requested, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{ans1, ans2}. If all candidates are flawed but you can solve the "
    "problem, return your corrected final answer. Return only the selected or corrected "
    "final answer inside \\boxed{} at the end of your response."
)

# ── Normalization Utilities ──────────────────────────────────────────────────

def answer_visible_text(text: str) -> str:
    think_end = text.rfind("</think>")
    return text[think_end + len("</think>"):] if think_end >= 0 else text

def visible_answer_text(text: str) -> str:
    answer_text = answer_visible_text(text)
    answer_text = answer_text.strip()
    if len(answer_text) <= CANDIDATE_SNIPPET_CHARS:
        return answer_text
    return "... " + answer_text[-CANDIDATE_SNIPPET_CHARS:]

def extract_boxed_values(text: str) -> list[str]:
    values = []
    start = 0
    while True:
        idx = text.find("\\boxed{", start)
        if idx < 0: break
        brace_start = idx + len("\\boxed{")
        depth, i = 1, brace_start
        while i < len(text) and depth > 0:
            if text[i] == "{": depth += 1
            elif text[i] == "}": depth -= 1
            i += 1
        if depth == 0:
            values.append(text[brace_start:i - 1].strip())
        start = i
    return values

def extract_boxed_group(text: str) -> list[str]:
    entries = []
    start = 0
    while True:
        idx = text.find("\\boxed{", start)
        if idx < 0: break
        brace_start = idx + len("\\boxed{")
        depth, i = 1, brace_start
        while i < len(text) and depth > 0:
            if text[i] == "{": depth += 1
            elif text[i] == "}": depth -= 1
            i += 1
        if depth == 0:
            content = text[brace_start:i - 1].strip()
            if content: entries.append((idx, i, content))
        start = i
    if not entries: return []
    last_group = [entries[-1]]
    for j in range(len(entries) - 2, -1, -1):
        gap = text[entries[j][1]:entries[j + 1][0]]
        if re.match(r"^[\s,\$\.\;\:\-\&\\]*$", gap):
            last_group.insert(0, entries[j])
        else: break
    return [item[2] for item in last_group]

def normalize_tokens(text: str) -> str:
    cleaned = text.replace("\\,", "").replace("\\left", "").replace("\\right", "").replace("$", "").replace("\\", "").strip()
    t = cleaned.lower()
    if t in {"yes", "y", "true"}: return "true"
    if t in {"no", "n", "false"}: return "false"
    allowed = "".join(ch if (ch.isalnum() or ch in ".,/()[]{}+-*^=") else " " for ch in cleaned)
    return " ".join(allowed.split())

def extract_letter(text: str) -> str:
    search_text = answer_visible_text(text)
    matches = re.findall(r"\\boxed\{\s*([A-Za-z])\s*\}", search_text)
    if not matches: matches = re.findall(r"\\boxed\{\s*([A-Za-z])\s*\}", text)
    if matches: return matches[-1].upper()
    matches = re.findall(r"\b([A-Z])\b", search_text.upper())
    if not matches: matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""

def canonicalize_answer(answer: str, is_mcq: bool) -> str:
    if is_mcq: return extract_letter(answer)
    boxed_group = extract_boxed_group(answer_visible_text(answer))
    if boxed_group: return ", ".join(normalize_tokens(part) for part in boxed_group)
    lines = [l.strip() for l in answer_visible_text(answer).splitlines() if l.strip()]
    return normalize_tokens(lines[-1]) if lines else ""

def format_candidate_answer(item: dict, candidate: str) -> str:
    is_mcq = bool(item.get("options"))
    if is_mcq:
        letter = extract_letter(candidate)
        return f"\\boxed{{{letter}}}" if letter else candidate
    boxed_group = extract_boxed_group(answer_visible_text(candidate))
    if boxed_group:
        return f"\\boxed{{{', '.join(boxed_group)}}}"
    lines = [l.strip() for l in answer_visible_text(candidate).splitlines() if l.strip()]
    return f"\\boxed{{{lines[-1]}}}" if lines else candidate

def majority_vote_answer(candidates: list[str], is_mcq: bool) -> str:
    counts = Counter()
    canonical_to_raw = {}
    for cand in candidates:
        key = canonicalize_answer(cand, is_mcq)
        if not key: continue
        counts[key] += 1
        if key not in canonical_to_raw: canonical_to_raw[key] = cand
    if not counts: return candidates[0] if candidates else ""
    best_key = counts.most_common(1)[0][0]
    raw_winner = canonical_to_raw[best_key]
    if is_mcq: return f"\\boxed{{{best_key.upper()}}}"
    return format_candidate_answer({"options": None, "question": ""}, raw_winner)

# ── Prompt Building ──────────────────────────────────────────────────────────

def build_chat_prompt(tokenizer, system: str, user: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )

def format_problem(question: str, options: Optional[list]) -> str:
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return f"{question}\n\nOptions:\n{opts_text}"
    return question

def build_prompt(question: str, options: Optional[list], variant: Optional[dict] = None) -> tuple[str, str]:
    if options:
        system = variant.get("mcq_system_prompt") if variant and variant.get("mcq_system_prompt") else SYSTEM_PROMPT_MCQ
        return system, format_problem(question, options)
    system = variant.get("math_system_prompt") if variant and variant.get("math_system_prompt") else SYSTEM_PROMPT_MATH
    return system, question

def build_candidate_prompt(tokenizer, item: dict, variant: tuple) -> str:
    name, instruction = variant
    system, user = build_prompt(item["question"], item.get("options"))
    if instruction:
        user = (
            f"{user}\n\nAttempt style: {name}\n{instruction}\n"
            "Follow the original problem exactly. Put the final answer inside \\boxed{}."
        )
    return build_chat_prompt(tokenizer, system, user)

def answer_format_hint(item: dict) -> str:
    if item.get("options"): return "This is a multiple-choice problem. The final answer should be one option letter."
    blank_count = item["question"].count("[ANS]")
    if blank_count == 1: return "This appears to request one free-form answer."
    if blank_count > 1:
        return f"The prompt contains {blank_count} [ANS] placeholders. Answer the actual requested blanks in order."
    return "This is a free-form problem. Follow the requested final-answer format."

def build_selection_prompt(item: dict, candidates: list[str], variant_names: list[str]) -> str:
    problem = format_problem(item["question"], item.get("options"))
    candidate_blocks = []
    for idx, (candidate, v_name) in enumerate(zip(candidates, variant_names), start=1):
        boxed_values = extract_boxed_values(candidate)
        boxed_text = ", ".join(boxed_values[-3:]) if boxed_values else "(no boxed answer found)"
        candidate_blocks.append(
            f"Candidate {idx} ({v_name})\n"
            f"Extracted boxed answer(s): {boxed_text}\n"
            f"Visible response excerpt:\n{visible_answer_text(candidate)}"
        )
    user = (
        f"Problem:\n{problem}\n\nAnswer format hint: {answer_format_hint(item)}\n\n"
        "Candidate solutions:\n\n" + "\n\n".join(candidate_blocks) +
        "\n\nChoose the best final answer from these candidates, or correct them if needed. "
        "Output only that answer in the required \\boxed{} format."
    )
    return build_chat_prompt(None, SYSTEM_PROMPT_SELECT, user) # build_chat_prompt(tokenizer, ...) will be used in main

def selected_or_fallback(selector_response: str, candidates: list[str], is_mcq: bool) -> str:
    m = re.search(r"\b(?:candidate|option|choice)\s*#?\s*([1-5])\b", selector_response, re.IGNORECASE)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(candidates): return candidates[idx]
    boxed_vals = extract_boxed_group(answer_visible_text(selector_response))
    if boxed_vals: return f"\\boxed{{ {', '.join(boxed_vals)} }}"
    return majority_vote_answer(candidates, is_mcq)

# ── Utils ────────────────────────────────────────────────────────────────────

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def generate_single_outputs(llm, prompts, sampling_params, chunk_size, label, lora_request=None):
    if not prompts: return []
    outputs_text = []
    n = len(prompts)
    log(f"{label}: generating {n} completions...")
    for i in range(0, n, chunk_size):
        chunk = prompts[i : i + chunk_size]
        outputs = llm.generate(chunk, sampling_params=sampling_params, lora_request=lora_request, use_tqdm=False)
        outputs_text.extend([out.outputs[0].text.strip() for out in outputs])
        if (i // chunk_size) % 5 == 0 or i + chunk_size >= n:
            log(f"  {min(i + chunk_size, n)}/{n} done")
    return outputs_text

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/private.jsonl")
    parser.add_argument("--output_jsonl", type=str, default="results/submission.jsonl")
    parser.add_argument("--output_csv", type=str, default="results/submission.csv")
    parser.add_argument("--gpu_memory", type=float, default=0.85)
    args = parser.parse_args()

    log(f"Loading data from {args.input}...")
    data = [json.loads(line) for line in open(args.input, encoding='utf-8')]
    mcq_indices = [i for i, d in enumerate(data) if d.get("options")]
    frq_indices = [i for i, d in enumerate(data) if not d.get("options")]
    log(f"Loaded {len(data)} items ({len(mcq_indices)} MCQ, {len(frq_indices)} FRQ)")

    log("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    llm = LLM(
        model=MODEL_ID, quantization="bitsandbytes", load_format="bitsandbytes",
        enable_lora=True, max_lora_rank=LORA_RANK, max_loras=1,
        max_model_len=16384, gpu_memory_utilization=args.gpu_memory,
        trust_remote_code=True, max_num_seqs=MAX_NUM_SEQS,
    )
    lora_request = LoRARequest("mcq_v6", 1, MCQ_ADAPTER_PATH)

    deterministic_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)
    sampled_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.6, top_p=0.95, top_k=20)
    select_params = SamplingParams(max_tokens=SELECT_MAX_TOKENS, temperature=0.0)

    results = [""] * len(data)

    # 1. MCQ Inference (LoRA, sampled)
    if mcq_indices:
        mcq_prompts = []
        for idx in mcq_indices:
            item = data[idx]
            system, user = build_prompt(item["question"], item.get("options"))
            mcq_prompts.append(build_chat_prompt(tokenizer, system, user))
        
        outputs = generate_single_outputs(llm, mcq_prompts, sampled_params, CHUNK_SIZE, "MCQ LoRA pass", lora_request)
        for idx, out in zip(mcq_indices, outputs):
            results[idx] = format_candidate_answer(data[idx], out)

    # 2. FRQ Inference (Ensemble)
    if frq_indices:
        frq_items = [data[i] for i in frq_indices]
        candidate_sets = [[] for _ in frq_items]
        
        # Baseline (deterministic)
        log("FRQ Baseline pass...")
        baseline_prompts = [build_candidate_prompt(tokenizer, item, CANDIDATE_VARIANTS[0]) for item in frq_items]
        baseline_outputs = generate_single_outputs(llm, baseline_prompts, deterministic_params, CHUNK_SIZE, "FRQ Baseline")
        for i, out in enumerate(baseline_outputs): candidate_sets[i].append(out)
        
        # Sampled variants
        for variant in CANDIDATE_VARIANTS[1:]:
            log(f"FRQ Variant: {variant[0]}...")
            v_prompts = [build_candidate_prompt(tokenizer, item, variant) for item in frq_items]
            v_outputs = generate_single_outputs(llm, v_prompts, sampled_params, CHUNK_SIZE, f"FRQ {variant[0]}")
            for i, out in enumerate(v_outputs): candidate_sets[i].append(out)
        
        # Selection
        log("FRQ Selection pass...")
        variant_names = [v[0] for v in CANDIDATE_VARIANTS]
        selection_prompts = []
        for item, cands in zip(frq_items, candidate_sets):
            # Reuse build_selection_prompt logic but with tokenizer
            problem = format_problem(item["question"], item.get("options"))
            candidate_blocks = []
            for idx, (candidate, v_name) in enumerate(zip(cands, variant_names), start=1):
                boxed_values = extract_boxed_values(candidate)
                boxed_text = ", ".join(boxed_values[-3:]) if boxed_values else "(no boxed answer found)"
                candidate_blocks.append(
                    f"Candidate {idx} ({v_name})\n"
                    f"Extracted boxed answer(s): {boxed_text}\n"
                    f"Visible response excerpt:\n{visible_answer_text(candidate)}"
                )
            user = (
                f"Problem:\n{problem}\n\nAnswer format hint: {answer_format_hint(item)}\n\n"
                "Candidate solutions:\n\n" + "\n\n".join(candidate_blocks) +
                "\n\nChoose the best final answer from these candidates, or correct them if needed. "
                "Output only that answer in the required \\boxed{} format."
            )
            selection_prompts.append(build_chat_prompt(tokenizer, SYSTEM_PROMPT_SELECT, user))
        
        selector_outputs = generate_single_outputs(llm, selection_prompts, select_params, SELECT_CHUNK_SIZE, "FRQ Selection")
        for idx, sel_out, cands in zip(frq_indices, selector_outputs, candidate_sets):
            results[idx] = selected_or_fallback(sel_out, cands, is_mcq=False)

    # 3. Export
    log(f"Saving JSONL results to {args.output_jsonl}...")
    with open(args.output_jsonl, "w", encoding='utf-8') as f:
        for i, item in enumerate(data):
            f.write(json.dumps({"id": item["id"], "is_mcq": bool(item.get("options")), "response": results[i]}) + "\n")

    log(f"Saving CSV submission to {args.output_csv}...")
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "response"])
        for i, item in enumerate(data):
            writer.writerow([str(item["id"]), results[i]])

    log("Done!")

if __name__ == "__main__":
    main()
