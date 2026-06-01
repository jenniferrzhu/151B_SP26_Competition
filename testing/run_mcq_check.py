"""
Standalone MCQ Check with Majority Voting: Verifying LoRA v6 performance.
Uses an ensemble of samples per question to stabilize and maximize accuracy.
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional
from collections import Counter

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID      = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID        = "0"
TEST_PATH     = "data/test.jsonl"
MCQ_ADAPTER_PATH = "lucashlaing/qwen3-lora-v6"
LORA_RANK     = 16

NUM_SAMPLES   = 8  # Majority voting size
test_name     = f"MCQ Check LoRA v6 - Maj{NUM_SAMPLES}"
PRED_PATH     = f"results/{test_name}/predictions.jsonl"
ACC_PATH      = f"results/{test_name}/accuracy.txt"
PROGRESS_PATH = f"results/{test_name}/progress.log"

MAX_TOKENS    = 32768
CHUNK_SIZE    = 8  # Reduced chunk size because we are generating n=8 per prompt

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

# ── Utilities ────────────────────────────────────────────────────────────────

def answer_visible_text(text: str) -> str:
    think_end = text.rfind("</think>")
    return text[think_end + len("</think>"):] if think_end >= 0 else text

def extract_letter(text: str) -> str:
    search_text = answer_visible_text(text)
    matches = re.findall(r"\\boxed\{\s*([A-Za-z])\s*\}", search_text)
    if not matches: matches = re.findall(r"\\boxed\{\s*([A-Za-z])\s*\}", text)
    if matches: return matches[-1].upper()
    matches = re.findall(r"\b([A-Z])\b", search_text.upper())
    if not matches: matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""

def format_problem(question: str, options: list) -> str:
    labels = [chr(65 + i) for i in range(len(options))]
    opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
    return f"{question}\n\nOptions:\n{opts_text}"

def log(msg: str, fp=None):
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    if fp: fp.write(line + "\n")

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    Path(PRED_PATH).parent.mkdir(parents=True, exist_ok=True)
    progress_fp = open(PROGRESS_PATH, "w", buffering=1)

    data = [json.loads(line) for line in open(TEST_PATH)]
    mcq_items = [d for d in data if d.get("options")]
    log(f"Loaded {len(data)} items. Filtering for MCQ: {len(mcq_items)} items.", progress_fp)

    log("Loading model...", progress_fp)
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
    lora_request = LoRARequest(lora_name="trained_v1", lora_int_id=1, lora_path=MCQ_ADAPTER_PATH)

    sampling_params = SamplingParams(
        n=NUM_SAMPLES,
        max_tokens=MAX_TOKENS,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
    )

    prompts = []
    for item in mcq_items:
        user_content = format_problem(item["question"], item["options"])
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT_MCQ}, {"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True,
        ))

    log(f"Generating {NUM_SAMPLES} responses per prompt for {len(prompts)} MCQ items in chunks of {CHUNK_SIZE}...")
    t0 = time.time()
    all_responses = []
    for i in range(0, len(prompts), CHUNK_SIZE):
        chunk = prompts[i : i + CHUNK_SIZE]
        outputs = llm.generate(
            chunk, 
            sampling_params=sampling_params, 
            lora_request=lora_request, 
            use_tqdm=False
        )
        for out in outputs:
            all_responses.append([o.text.strip() for o in out.outputs])
        
        log(f"  {min(i + CHUNK_SIZE, len(prompts))}/{len(prompts)} done")
    
    gen_time = time.time() - t0
    log(f"Inference complete in {gen_time/60:.2f} min.", progress_fp)

    log("Scoring with Majority Voting...", progress_fp)
    results = []
    for item, responses in zip(mcq_items, all_responses):
        preds = [extract_letter(r) for r in responses]
        
        # Majority Vote
        counts = Counter([p for p in preds if p])
        if not counts:
            final_pred = preds[0] if preds else ""
        else:
            final_pred = counts.most_common(1)[0][0]
            
        gold = str(item["answer"]).strip().upper()
        correct = (final_pred == gold)
        
        results.append({
            "id": item.get("id"),
            "gold": gold,
            "pred": final_pred,
            "all_preds": preds,
            "vote_counts": dict(counts),
            "correct": correct,
            "responses": responses
        })

    with open(PRED_PATH, "w") as f:
        for r in results: f.write(json.dumps(r) + "\n")

    acc = sum(r["correct"] for r in results) / len(results) * 100
    summary = (
        f"Standalone MCQ Evaluation (Majority Voting n={NUM_SAMPLES}) - {MODEL_ID}\n"
        f"Adapter: {MCQ_ADAPTER_PATH}\n"
        f"Test set: {TEST_PATH} ({len(results)} MCQ items)\n"
        f"Accuracy: {acc:.2f}% ({sum(r['correct'] for r in results)}/{len(results)})\n"
    )
    log("\n" + summary, progress_fp)
    with open(ACC_PATH, "w") as f: f.write(summary)

    progress_fp.close()

if __name__ == "__main__":
    main()
