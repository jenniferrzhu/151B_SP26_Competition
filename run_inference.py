"""
End-to-End Inference Script for CSE 151B Math Reasoning Competition.
Exposes run_inference() as the single entry point.

Strategy:
- MCQ: LoRA v6 Adapter, n=8 samples, Majority Voting.
- FRQ: Base Model, 5-Variant Ensemble (deterministic baseline + 4 sampled
       prompt-style variants), then an LLM-judge selector that picks the
       best candidate (with majority-vote fallback when the selector is
       ambiguous). Mirrors testing/run_eval_hybrid_mcq_lora_frq.py.
"""

import json
import os
import re
import sys
import time
import csv
import argparse
from pathlib import Path
from collections import Counter

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID      = "Qwen/Qwen3-4B-Thinking-2507" # DESIGNATED BASE MODEL
MCQ_ADAPTER_PATH = "lucashlaing/qwen3-lora-v6" # Path to fine-tuned LoRA on Hugging Face Hub
# Note: In production/Gradescope, this might be a HuggingFace Hub path like "username/lora-v6"

MAX_TOKENS              = 32768
CHUNK_SIZE              = 32
LORA_RANK               = 16
SELECT_CHUNK_SIZE       = 16
SELECT_MAX_TOKENS       = 4096
CANDIDATE_SNIPPET_CHARS = 1800

# 5-variant FRQ ensemble (matches hybrid eval). No weights — selector decides.
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

# ── Prompts ──────────────────────────────────────────────────────────────────
# EXACTLY matches LoRA v6 fine-tuning prompt for MCQ
SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

# FRQ system prompt (long few-shot), mirrors run_eval_hybrid_mcq_lora_frq.py
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

# Selector prompt for the FRQ judge pass
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
    """Selector-friendly view of a candidate: visible portion only, trimmed."""
    answer_text = answer_visible_text(text).strip()
    if len(answer_text) <= CANDIDATE_SNIPPET_CHARS:
        return answer_text
    return "... " + answer_text[-CANDIDATE_SNIPPET_CHARS:]


def extract_boxed_values(text: str) -> list[str]:
    """All \\boxed{} contents in order."""
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


def extract_boxed_group(text: str) -> list[str]:
    """The LAST contiguous run of \\boxed{} entries (e.g., \\boxed{a} \\boxed{b})."""
    entries = []
    start = 0
    while True:
        idx = text.find("\\boxed{", start)
        if idx < 0:
            break
        brace_start = idx + len("\\boxed{")
        depth, i = 1, brace_start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            content = text[brace_start:i - 1].strip()
            if content:
                entries.append((idx, i, content))
        start = i

    if not entries:
        return []

    last_group = [entries[-1]]
    for j in range(len(entries) - 2, -1, -1):
        gap = text[entries[j][1]:entries[j + 1][0]]
        if re.match(r"^[\s,\$\.\;\:\-\&\\]*$", gap):
            last_group.insert(0, entries[j])
        else:
            break
    return [item[2] for item in last_group]


def normalize_tokens(text: str) -> str:
    cleaned = text.replace("\\,", "").replace("\\left", "").replace("\\right", "")
    cleaned = cleaned.replace("$", "").replace("\\", "").strip()
    t = cleaned.lower()
    if t in {"yes", "y", "true"}: return "true"
    if t in {"no", "n", "false"}: return "false"
    allowed = []
    for ch in cleaned:
        if ch.isalnum() or ch in ".,/()[]{}+-*^=":
            allowed.append(ch)
        elif ch.isspace():
            allowed.append(" ")
    return " ".join("".join(allowed).split())


def extract_letter(text: str) -> str:
    search_text = answer_visible_text(text)
    matches = re.findall(r"\\boxed\{\s*([A-Za-z])\s*\}", search_text)
    if not matches:
        matches = re.findall(r"\\boxed\{\s*([A-Za-z])\s*\}", text)
    if matches:
        return matches[-1].upper()
    matches = re.findall(r"\b([A-Z])\b", search_text.upper())
    if not matches:
        matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def canonicalize_answer(answer: str, is_mcq: bool) -> str:
    if is_mcq:
        return extract_letter(answer) or ""
    boxed_group = extract_boxed_group(answer_visible_text(answer))
    if boxed_group:
        return ", ".join(normalize_tokens(part) for part in boxed_group)
    lines = [l.strip() for l in answer_visible_text(answer).splitlines() if l.strip()]
    return normalize_tokens(lines[-1]) if lines else ""


def format_candidate_answer(item: dict, candidate: str) -> str:
    """Coerce a raw candidate into a clean \\boxed{} answer string."""
    is_mcq = bool(item.get("options"))
    if is_mcq:
        letter = extract_letter(candidate)
        return f"\\boxed{{{letter}}}" if letter else candidate
    boxed_group = extract_boxed_group(answer_visible_text(candidate))
    if boxed_group:
        return f"\\boxed{{{', '.join(boxed_group)}}}"
    lines = [l.strip() for l in answer_visible_text(candidate).splitlines() if l.strip()]
    if lines:
        return f"\\boxed{{{lines[-1]}}}"
    return candidate


def weighted_majority_vote(candidates: list[str], is_mcq: bool) -> str:
    """Uniform-weight equivalence-class majority vote. Used for MCQ aggregation."""
    counts: dict[str, int] = {}
    canonical_to_raw: dict[str, str] = {}
    for cand in candidates:
        key = canonicalize_answer(cand, is_mcq)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
        if key not in canonical_to_raw:
            canonical_to_raw[key] = cand
    if not counts:
        return candidates[0] if candidates else ""
    best_key = max(counts, key=counts.get)
    return canonical_to_raw[best_key]



def format_problem(question: str, options) -> str:
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return f"{question}\n\nOptions:\n{opts_text}"
    return question


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


def build_selection_user_msg(item: dict, candidates: list[str], variant_names: list[str]) -> str:
    problem = format_problem(item["question"], item.get("options"))
    candidate_blocks = []
    for idx, (candidate, vname) in enumerate(zip(candidates, variant_names), start=1):
        boxed_values = extract_boxed_values(candidate)
        boxed_text = ", ".join(boxed_values[-3:]) if boxed_values else "(no boxed answer found)"
        candidate_blocks.append(
            f"Candidate {idx} ({vname})\n"
            f"Extracted boxed answer(s): {boxed_text}\n"
            f"Visible response excerpt:\n{visible_answer_text(candidate)}"
        )
    return (
        f"Problem:\n{problem}\n\n"
        f"Answer format hint: {answer_format_hint(item)}\n\n"
        "Candidate solutions:\n\n"
        + "\n\n".join(candidate_blocks)
        + "\n\nChoose the best final answer from these candidates, or correct them if needed. "
        "Output only that answer in the required \\boxed{} format."
    )


def selected_or_fallback(selector_response: str, candidates: list[str], is_mcq: bool) -> str:
    """Apply selector's pick, fall back to majority vote when ambiguous."""
    # 1. Explicit selection: "Candidate #3"
    m = re.search(r"\b(?:candidate|option|choice)\s*#?\s*([1-5])\b", selector_response, re.IGNORECASE)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]
    # 2. Selector wrote its own boxed answer in the visible portion → use selector verbatim
    if "\\boxed{" in answer_visible_text(selector_response):
        return selector_response.strip()
    # 3. Fallback: equivalence-class majority vote over candidates
    return weighted_majority_vote(candidates, is_mcq=is_mcq)

# ── Inference Engine ─────────────────────────────────────────────────────────

def run_inference(input_path="data/private.jsonl", output_csv="submission.csv"):
    """
    Performs the full pipeline end-to-end.
    """
    print(f"[{time.strftime('%H:%M:%S')}] Starting end-to-end inference...")
    
    # 1. Load Data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    mcq_indices = [i for i, d in enumerate(data) if d.get("options")]
    frq_indices = [i for i, d in enumerate(data) if not d.get("options")]
    print(f"Loaded {len(data)} items: {len(mcq_indices)} MCQ, {len(frq_indices)} FRQ.")

    # 2. Initialize vLLM
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    print(f"Loading model {MODEL_ID} with LoRA support...")
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
    
    # LoRA Request for MCQ
    lora_request = LoRARequest(lora_name="trained_v1", lora_int_id=1, lora_path=MCQ_ADAPTER_PATH)

    results = [""] * len(data)

    # 3. MCQ Phase (n=8 Majority Vote)
    if mcq_indices:
        print(f"Starting MCQ Phase ({len(mcq_indices)} items)...")
        mcq_prompts = []
        for idx in mcq_indices:
            item = data[idx]
            labels = [chr(65 + i) for i in range(len(item["options"]))]
            opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, item["options"]))
            user = f"{item['question']}\n\nOptions:\n{opts_text}"
            
            mcq_prompts.append(tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT_MCQ}, {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True
            ))
        
        sampling_params_mcq = SamplingParams(
            n=8, max_tokens=MAX_TOKENS, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0
        )
        
        mcq_responses = []
        for i in range(0, len(mcq_prompts), CHUNK_SIZE):
            chunk = mcq_prompts[i : i + CHUNK_SIZE]
            outputs = llm.generate(chunk, sampling_params=sampling_params_mcq, lora_request=lora_request, use_tqdm=False)
            for out in outputs:
                mcq_responses.append([o.text.strip() for o in out.outputs])
            print(f"  {min(i + CHUNK_SIZE, len(mcq_prompts))}/{len(mcq_prompts)} done")
            
        for i, idx in enumerate(mcq_indices):
            results[idx] = weighted_majority_vote(mcq_responses[i], is_mcq=True)

    # 4. FRQ Phase (5-variant ensemble + LLM-judge selector with majority fallback)
    if frq_indices:
        print(f"Starting FRQ Phase ({len(frq_indices)} items, {NUM_CANDIDATES} variants)...")
        frq_candidate_sets   = [[] for _ in range(len(frq_indices))]
        frq_variant_names    = [[] for _ in range(len(frq_indices))]

        det_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)
        sampled_params = SamplingParams(
            max_tokens=MAX_TOKENS, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0
        )
        select_params = SamplingParams(max_tokens=SELECT_MAX_TOKENS, temperature=0.0)

        # ── Generate all 5 variants ─────────────────────────────────────────
        for v_idx, (name, instruction) in enumerate(CANDIDATE_VARIANTS):
            print(f"  Generating variant {v_idx + 1}/{NUM_CANDIDATES}: {name}...")
            v_prompts = []
            for idx in frq_indices:
                item = data[idx]
                user = item["question"]
                if instruction:
                    user = (
                        f"{user}\n\n"
                        f"Attempt style: {name}\n"
                        f"{instruction}\n"
                        "Follow the original problem exactly. Put the final answer inside \\boxed{}."
                    )
                v_prompts.append(tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT_MATH}, {"role": "user", "content": user}],
                    tokenize=False, add_generation_prompt=True,
                ))

            params = det_params if v_idx == 0 else sampled_params
            v_outputs = []
            for i in range(0, len(v_prompts), CHUNK_SIZE):
                chunk = v_prompts[i : i + CHUNK_SIZE]
                outputs = llm.generate(chunk, sampling_params=params, use_tqdm=False)
                v_outputs.extend([out.outputs[0].text.strip() for out in outputs])

            for i, resp in enumerate(v_outputs):
                frq_candidate_sets[i].append(resp)
                frq_variant_names[i].append(name)

        # ── LLM-judge selector pass ────────────────────────────────────────
        print(f"  Running selector over {len(frq_indices)} FRQ items...")
        selection_prompts = []
        for i, idx in enumerate(frq_indices):
            user = build_selection_user_msg(
                data[idx], frq_candidate_sets[i], frq_variant_names[i]
            )
            selection_prompts.append(tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT_SELECT}, {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True,
            ))

        selector_responses = []
        for i in range(0, len(selection_prompts), SELECT_CHUNK_SIZE):
            chunk = selection_prompts[i : i + SELECT_CHUNK_SIZE]
            outputs = llm.generate(chunk, sampling_params=select_params, use_tqdm=False)
            selector_responses.extend([out.outputs[0].text.strip() for out in outputs])
            print(f"  selector {min(i + SELECT_CHUNK_SIZE, len(selection_prompts))}/{len(selection_prompts)} done")

        # ── Apply selector pick with majority-vote fallback ────────────────
        for i, idx in enumerate(frq_indices):
            results[idx] = selected_or_fallback(
                selector_responses[i],
                frq_candidate_sets[i],
                is_mcq=False,
            )

    # 5. Output CSV
    print(f"Saving final submission to {output_csv}...")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "response"])
        for i, item in enumerate(data):
            writer.writerow([str(item["id"]), results[i]])
            
    print(f"[{time.strftime('%H:%M:%S')}] Inference Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/private.jsonl")
    parser.add_argument("--output", type=str, default="submission.csv")
    args = parser.parse_args()
    
    run_inference(input_path=args.input, output_csv=args.output)
