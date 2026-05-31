"""
Evaluation Script v8: Hybrid LoRA MCQ + Weighted Majority FRQ.

Key Features:
- MCQ: LoRA v6 adapter, Temperature 0.6, EXACT fine-tuning prompt (fixes v2 drop).
- FRQ: Base model ensemble (6 variants), Weighted Majority Voting (ditch selector).
- Prompt: Updated symbolic-first math prompt from v7.
- Oracle Tracking: Monitor potential for each item.
"""

import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional, List, Tuple

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID      = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID        = "2"
TEST_PATH     = "data/test.jsonl"
MCQ_ADAPTER_PATH = "adapters/qwen3-lora-v6-mixed-fmtfix"
LORA_RANK     = 16

test_name     = "Hybrid v8 MCQ-Lora + FRQ-WeightedMajority"
PRED_PATH     = f"results/{test_name}/predictions.jsonl"
ACC_PATH      = f"results/{test_name}/accuracy.txt"
PROGRESS_PATH = f"results/{test_name}/progress.log"

MAX_TOKENS    = 32768
CHUNK_SIZE    = 32
MAX_NUM_SEQS  = 64

# Variants matching v7 strategy
CANDIDATE_VARIANTS = [
    ("baseline_deterministic", "", 2), # weight = 2
    ("answer_order_audit", "First identify every answer the problem asks for, especially each real [ANS] blank. Solve them in order and put all final sub-answers in one boxed list.", 1),
    ("formula_first_exact", "Before arithmetic, write down the relevant formula or theorem. Keep exact values until the final step and round only when the problem explicitly asks.", 1),
    ("independent_then_options", "Solve independently before looking at answer choices. For multiple choice, compare your result to every option and watch for common distractors.", 1),
    ("sanity_check", "After solving, check units, signs, ranges, rounding, and whether the answer is reasonable. Correct the final answer before boxing it if the check fails.", 1),
    ("concise_reasoning", "Solve the problem concisely. Keep the reasoning short and direct. Focus on the final result and skip unnecessary intermediate steps.", 1),
]

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

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

def log(msg: str, fp=None):
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    if fp: fp.write(line + "\n")

def generate_single_outputs(llm, prompts, sampling_params, chunk_size, label, progress_fp, lora_request=None):
    if not prompts: return [], 0.0
    outputs_text = []
    t0 = time.time()
    n = len(prompts)
    log(f"{label}: generating {n} completions...", progress_fp)
    for i in range(0, n, chunk_size):
        chunk = prompts[i : i + chunk_size]
        outputs = llm.generate(chunk, sampling_params=sampling_params, lora_request=lora_request, use_tqdm=False)
        outputs_text.extend([out.outputs[0].text.strip() for out in outputs])
        done = i + len(chunk)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n - done) / rate if rate > 0 else 0
        log(f"  {done}/{n} done | Rate: {rate*60:.1f}/min | ETA: {eta/60:.1f}min", progress_fp)
    return outputs_text, time.time() - t0

# ── Scoring ──────────────────────────────────────────────────────────────────

def score_model_response(item: dict, response: str, judger) -> Optional[bool]:
    if "answer" not in item: return None
    is_mcq = bool(item.get("options"))
    gold = item["answer"]
    if is_mcq:
        return extract_letter(response) == str(gold).strip().upper()
    gold_list = gold if isinstance(gold, list) else [gold]
    try:
        return judger.auto_judge(pred=response, gold=gold_list, options=[[]] * len(gold_list))
    except:
        return False

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    Path(PRED_PATH).parent.mkdir(parents=True, exist_ok=True)
    progress_fp = open(PROGRESS_PATH, "w", buffering=1)

    data = [json.loads(line) for line in open(TEST_PATH)]
    mcq_indices = [i for i, d in enumerate(data) if d.get("options")]
    frq_indices = [i for i, d in enumerate(data) if not d.get("options")]
    log(f"Loaded {len(data)} items ({len(mcq_indices)} MCQ, {len(frq_indices)} FRQ)", progress_fp)

    log("Loading model...", progress_fp)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    llm = LLM(
        model=MODEL_ID, quantization="bitsandbytes", load_format="bitsandbytes",
        enable_lora=True, max_lora_rank=LORA_RANK, max_loras=1,
        max_model_len=16384, gpu_memory_utilization=0.85,
        trust_remote_code=True, max_num_seqs=MAX_NUM_SEQS,
    )
    lora_request = LoRARequest("mcq_v6", 1, MCQ_ADAPTER_PATH)

    deterministic_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)
    sampled_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.6, top_p=0.95, top_k=20)

    n = len(data)
    responses = [""] * n
    candidate_sets = [[] for _ in range(n)]

    # 1. MCQ Inference (Single pass, LoRA, Temperature 0.6 to match standalone v6)
    if mcq_indices:
        mcq_prompts = []
        for idx in mcq_indices:
            item = data[idx]
            system, user = build_prompt(item["question"], item.get("options"))
            mcq_prompts.append(build_chat_prompt(tokenizer, system, user))
        
        mcq_outputs, _ = generate_single_outputs(llm, mcq_prompts, sampled_params, CHUNK_SIZE, "MCQ LoRA", progress_fp, lora_request)
        for idx, out in zip(mcq_indices, mcq_outputs):
            responses[idx] = out
            candidate_sets[idx] = [out]

    # 2. FRQ Inference (Weighted Ensemble, Base Model)
    if frq_indices:
        frq_items = [data[i] for i in frq_indices]
        frq_candidate_sets = [[] for _ in range(len(frq_indices))]
        
        # Baseline (Weight 2, Deterministic)
        v = CANDIDATE_VARIANTS[0]
        prompts = [build_candidate_prompt(tokenizer, item, v) for item in frq_items]
        outs, _ = generate_single_outputs(llm, prompts, deterministic_params, CHUNK_SIZE, f"FRQ {v[0]}", progress_fp)
        for i, out in enumerate(outs): frq_candidate_sets[i].append(out)
        
        # Variants (Weight 1, Sampled 0.6)
        for v in CANDIDATE_VARIANTS[1:]:
            prompts = [build_candidate_prompt(tokenizer, item, v) for item in frq_items]
            outs, _ = generate_single_outputs(llm, prompts, sampled_params, CHUNK_SIZE, f"FRQ {v[0]}", progress_fp)
            for i, out in enumerate(outs): frq_candidate_sets[i].append(out)
        
        # Weighted Voting
        for i, idx in enumerate(frq_indices):
            responses[idx] = weighted_majority_vote(frq_candidate_sets[i], CANDIDATE_VARIANTS, is_mcq=False)
            candidate_sets[idx] = frq_candidate_sets[i]

    # 3. Scoring
    sys.path.insert(0, ".")
    from judger import Judger
    judger = Judger(strict_extract=False)

    log("Scoring...", progress_fp)
    results = []
    for idx, item in enumerate(data):
        correct = score_model_response(item, responses[idx], judger)
        
        # Calculate Oracle
        candidate_correct = [score_model_response(item, cand, judger) for cand in candidate_sets[idx]]
        oracle_correct = any(candidate_correct) if candidate_correct else None

        results.append({
            "id": item.get("id"),
            "is_mcq": bool(item.get("options")),
            "gold": item.get("answer"),
            "candidates": candidate_sets[idx],
            "candidate_correct": candidate_correct,
            "oracle_correct": oracle_correct,
            "response": responses[idx],
            "correct": correct,
        })

    with open(PRED_PATH, "w") as f:
        for r in results: f.write(json.dumps(r) + "\n")

    scored = [r for r in results if r["correct"] is not None]
    if scored:
        mcq_sub = [r for r in scored if r["is_mcq"]]
        frq_sub = [r for r in scored if not r["is_mcq"]]
        
        mcq_acc = sum(r["correct"] for r in mcq_sub) / len(mcq_sub) * 100 if mcq_sub else 0
        frq_acc = sum(r["correct"] for r in frq_sub) / len(frq_sub) * 100 if frq_sub else 0
        overall = sum(r["correct"] for r in scored) / len(scored) * 100
        oracle_acc = sum(r["oracle_correct"] for r in scored if r["oracle_correct"] is not None) / len(scored) * 100
        
        summary = (
            f"Hybrid v8 Results:\n"
            f"MCQ Accuracy : {mcq_acc:.2f}% ({sum(r['correct'] for r in mcq_sub)}/{len(mcq_sub)})\n"
            f"FRQ Accuracy : {frq_acc:.2f}% ({sum(r['correct'] for r in frq_sub)}/{len(frq_sub)})\n"
            f"Overall      : {overall:.2f}%\n"
            f"Oracle@Ensemble: {oracle_acc:.2f}%\n"
            f"Weighted Voting missed correct candidate: {sum(1 for r in scored if r['oracle_correct'] and not r['correct'])}\n"
        )
        log("\n" + summary, progress_fp)
        with open(ACC_PATH, "w") as f: f.write(summary)

    progress_fp.close()

if __name__ == "__main__":
    main()
