"""
End-to-End Inference Script for CSE 151B Math Reasoning Competition.
Exposes run_inference() as the single entry point.

Strategy:
- MCQ: LoRA v6 Adapter, n=8 samples, Majority Voting.
- FRQ: Base Model, 6-Variant Ensemble, Weighted Majority Voting.
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

MAX_TOKENS    = 32768
CHUNK_SIZE    = 32
LORA_RANK     = 16

# Variants for FRQ Ensemble
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
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}."
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
            if content: entries.append(content)
        start = i
    return entries

def normalize_tokens(text: str) -> str:
    cleaned = text.replace("\\,", "").replace("\\left", "").replace("\\right", "").replace("$", "").replace("\\", "").strip()
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

def weighted_majority_vote(candidates: list[str], is_mcq: bool) -> str:
    if is_mcq:
        # Uniform weight for MCQ samples
        weights = [1] * len(candidates)
    else:
        # v8 Weighted logic for FRQ variants
        base_weights = [v[2] for v in CANDIDATE_VARIANTS]
        weights = base_weights + [1] * (len(candidates) - len(base_weights))
        weights = weights[:len(candidates)]
        
    counts = {}
    canonical_to_raw = {}
    for cand, weight in zip(candidates, weights):
        key = canonicalize_answer(cand, is_mcq)
        if not key: continue
        counts[key] = counts.get(key, 0) + weight
        if key not in canonical_to_raw or weight > 1:
            canonical_to_raw[key] = cand
            
    if not counts: 
        return candidates[0] if candidates else ""
    
    best_key = max(counts, key=counts.get)
    return canonical_to_raw[best_key]

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

    # 4. FRQ Phase (6-Variant Ensemble)
    if frq_indices:
        print(f"Starting FRQ Phase ({len(frq_indices)} items)...")
        frq_candidate_sets = [[] for _ in range(len(frq_indices))]
        
        det_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)
        sampled_params = SamplingParams(
            max_tokens=MAX_TOKENS, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0
        )
        
        for v_idx, (name, instruction, weight) in enumerate(CANDIDATE_VARIANTS):
            print(f"  Generating Variant {v_idx+1}/6: {name}...")
            v_prompts = []
            for idx in frq_indices:
                item = data[idx]
                user = item["question"]
                if instruction:
                    user = f"{user}\n\nAttempt style: {name}\n{instruction}\nFollow the original problem exactly. Put final answer in \\boxed{{}}."
                
                v_prompts.append(tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT_MATH}, {"role": "user", "content": user}],
                    tokenize=False, add_generation_prompt=True
                ))
            
            params = det_params if v_idx == 0 else sampled_params
            v_outputs = []
            for i in range(0, len(v_prompts), CHUNK_SIZE):
                chunk = v_prompts[i : i + CHUNK_SIZE]
                outputs = llm.generate(chunk, sampling_params=params, use_tqdm=False)
                v_outputs.extend([out.outputs[0].text.strip() for out in outputs])
                
            for i, resp in enumerate(v_outputs):
                frq_candidate_sets[i].append(resp)
        
        for i, idx in enumerate(frq_indices):
            results[idx] = weighted_majority_vote(frq_candidate_sets[i], is_mcq=False)

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
