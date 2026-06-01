"""
Consolidated Submission Script for CSE 151B Math Reasoning Competition.
Matches the logic and prompting of run_eval_v8_hybrid.py.

Performs Hybrid Inference:
- MCQ: LoRA v6 Adapter, sampled (temperature 0.6), exact training prompt.
- FRQ: Base Model, Multi-Prompt Ensemble (6 variants) + Weighted Majority Voting.
- Prompting: Updated symbolic-first math prompt.
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

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID      = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID        = "0"
MCQ_ADAPTER_PATH = "adapters/qwen3-lora-v6-mixed-fmtfix"
LORA_RANK     = 16

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

MAX_TOKENS    = 32768
CHUNK_SIZE    = 32
MAX_NUM_SEQS  = 64

# Variants matching v8 strategy
CANDIDATE_VARIANTS = [
    ("baseline_deterministic", "", 2), # weight = 2
    ("answer_order_audit", "First identify every answer the problem asks for, especially each real [ANS] blank. Solve them in order and put all final sub-answers in one boxed list.", 1),
    ("formula_first_exact", "Before arithmetic, write down the relevant formula or theorem. Keep exact values until the final step and round only when the problem explicitly asks.", 1),
    ("independent_then_options", "Solve independently before looking at answer choices. For multiple choice, compare your result to every option and watch for common distractors.", 1),
    ("sanity_check", "After solving, check units, signs, ranges, rounding, and whether the answer is reasonable. Correct the final answer before boxing it if the check fails.", 1),
    ("concise_reasoning", "Solve the problem concisely. Keep the reasoning short and direct. Focus on the final result and skip unnecessary intermediate steps.", 1),
]

# ── Prompts ──────────────────────────────────────────────────────────────────
# EXACTLY matches LoRA v6 fine-tuning prompt for MCQ
SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

# Symbolic-emphasis prompt for FRQ (base model)
SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step inside <think> tags. "
    "Always provide the EXACT fractional or symbolic form first (e.g., \\frac{13}{9} or \\pi). "
    "If the problem requests a decimal, you may provide it, but prioritize the exact form in the final \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}.\n\n"
    "Example 1:\nProblem: Simplify 38B^0.\nFinal answer: \\boxed{38}\n\n"
    "Example 2:\nProblem: Find the slope between (-2,-5) and (7,8).\nFinal answer: \\boxed{\\frac{13}{9}}"
)

# ── Normalization Utilities ──────────────────────────────────────────────────

def answer_visible_text(text: str) -> str:
    think_end = text.rfind("</think>")
    return text[think_end + len("</think>"):] if think_end >= 0 else text

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

# ── Weighted Consistency Logic ───────────────────────────────────────────────

def weighted_majority_vote(candidates: list[str], variants: list[tuple], is_mcq: bool) -> str:
    weights = [v[2] for v in variants]
    counts = {}
    canonical_to_raw = {}
    
    for cand, weight in zip(candidates, weights):
        key = canonicalize_answer(cand, is_mcq)
        if not key: continue
        counts[key] = counts.get(key, 0) + weight
        if key not in canonical_to_raw:
            canonical_to_raw[key] = cand
        if weight > 1: # Prefer baseline if it wins
            canonical_to_raw[key] = cand

    if not counts:
        return candidates[0] if candidates else ""
    best_key = max(counts, key=counts.get)
    return canonical_to_raw[best_key]

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

def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    if options: return SYSTEM_PROMPT_MCQ, format_problem(question, options)
    return SYSTEM_PROMPT_MATH, question

def build_candidate_prompt(tokenizer, item: dict, variant: tuple) -> str:
    name, instruction, weight = variant
    system, user = build_prompt(item["question"], item.get("options"))
    if instruction:
        user = (
            f"{user}\n\nAttempt style: {name}\n{instruction}\n"
            "Follow the original problem exactly. Put the final answer inside \\boxed{}."
        )
    return build_chat_prompt(tokenizer, system, user)

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
    parser.add_argument("--gpu_id", type=str, default=GPU_ID)
    parser.add_argument("--input", type=str, default="data/test.jsonl")
    parser.add_argument("--output_jsonl", type=str, default="submission.jsonl")
    parser.add_argument("--output_csv", type=str, default="submission.csv")
    parser.add_argument("--gpu_memory", type=float, default=0.85)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    log(f"Using GPU: {args.gpu_id}")
    log(f"Loading data from {args.input}...")

    data = [json.loads(line) for line in open(args.input, encoding='utf-8')]
    mcq_indices = [i for i, d in enumerate(data) if d.get("options")]
    frq_indices = [i for i, d in enumerate(data) if not d.get("options")]
    log(f"Loaded {len(data)} items ({len(mcq_indices)} MCQ, {len(frq_indices)} FRQ)")

    log("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    llm = LLM(
        model=MODEL_ID,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=16384,
        trust_remote_code=True,
        max_num_seqs=64,
        max_num_batched_tokens=16384,
        kv_cache_memory_bytes=14 * 1024**3,
        enable_lora=True,
        max_lora_rank=LORA_RANK,
        max_loras=1,
    )
    lora_request = LoRARequest("mcq_v6", 1, MCQ_ADAPTER_PATH)

    deterministic_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)
    sampled_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
    )

    results = [""] * len(data)

    # 1. MCQ Inference (Single pass, LoRA, Temp 0.6)
    if mcq_indices:
        mcq_prompts = []
        for idx in mcq_indices:
            item = data[idx]
            system, user = build_prompt(item["question"], item.get("options"))
            mcq_prompts.append(build_chat_prompt(tokenizer, system, user))
        
        outputs = generate_single_outputs(llm, mcq_prompts, sampled_params, CHUNK_SIZE, "MCQ LoRA pass", lora_request)
        for idx, out in zip(mcq_indices, outputs):
            results[idx] = out

    # 2. FRQ Inference (Weighted Ensemble, Base Model)
    if frq_indices:
        frq_items = [data[i] for i in frq_indices]
        frq_candidate_sets = [[] for _ in range(len(frq_indices))]
        
        # Baseline (Weight 2, Deterministic)
        v = CANDIDATE_VARIANTS[0]
        prompts = [build_candidate_prompt(tokenizer, item, v) for item in frq_items]
        outs = generate_single_outputs(llm, prompts, deterministic_params, CHUNK_SIZE, f"FRQ {v[0]}")
        for i, out in enumerate(outs): frq_candidate_sets[i].append(out)
        
        # Variants (Weight 1, Sampled 0.6)
        for v in CANDIDATE_VARIANTS[1:]:
            prompts = [build_candidate_prompt(tokenizer, item, v) for item in frq_items]
            outs = generate_single_outputs(llm, prompts, sampled_params, CHUNK_SIZE, f"FRQ {v[0]}")
            for i, out in enumerate(outs): frq_candidate_sets[i].append(out)
        
        # Weighted Voting
        for i, idx in enumerate(frq_indices):
            results[idx] = weighted_majority_vote(frq_candidate_sets[i], CANDIDATE_VARIANTS, is_mcq=False)

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
