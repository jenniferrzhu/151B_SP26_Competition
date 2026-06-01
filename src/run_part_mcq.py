"""
Parallel MCQ Script (LoRA v6 + Majority Voting n=8).
Optimized for stable 77%+ MCQ accuracy.
Run once on any GPU: python run_part_mcq.py --gpu_id 0 --input data/private.jsonl
"""

import json
import os
import re
import sys
import time
import argparse
from pathlib import Path
from collections import Counter

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID      = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_INPUT_PATH  = "data/private.jsonl"
ADAPTER_PATH  = "lucashlaing/qwen3-lora-v6"
LORA_RANK     = 16
NUM_SAMPLES   = 8
MAX_TOKENS    = 32768
CHUNK_SIZE    = 8 # Reduced because we generate 8 per prompt

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_PATH)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    # 1. Load data
    log(f"Loading data from {args.input}...")
    data = [json.loads(line) for line in open(args.input, encoding='utf-8')]
    mcq_items = [d for d in data if d.get("options")]
    log(f"Processing {len(mcq_items)} MCQ items with n={NUM_SAMPLES} voting.")

    if not mcq_items:
        log("No MCQ items found. Exiting.")
        return

    # 2. Init Model with LoRA
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
    lora_request = LoRARequest(lora_name="trained_v1", lora_int_id=1, lora_path=ADAPTER_PATH)
    sampling_params = SamplingParams(
        n=NUM_SAMPLES,
        max_tokens=MAX_TOKENS,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
    )

    # 3. Generate
    prompts = []
    for item in mcq_items:
        labels = [chr(65 + i) for i in range(len(item["options"]))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, item["options"]))
        user = f"{item['question']}\n\nOptions:\n{opts_text}"
        
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT_MCQ}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True
        ))

    log(f"Generating {NUM_SAMPLES} samples per prompt in chunks of {CHUNK_SIZE}...")
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
    
    results = []
    for item, responses in zip(mcq_items, all_responses):
        preds = [extract_letter(r) for r in responses]
        counts = Counter([p for p in preds if p])
        final_pred = counts.most_common(1)[0][0] if counts else (preds[0] if preds else "")
        
        results.append({
            "id": item["id"],
            "is_mcq": True,
            "gold": item.get("answer"),
            "response": f"\\boxed{{{final_pred}}}",
            "candidates": [f"\\boxed{{{p}}}" for p in preds],
        })

    # 4. Save
    out_dir = Path("results/shards")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mcq_voted.jsonl"
    with open(out_path, "w", encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    log(f"Saved voted MCQ results to {out_path}")

if __name__ == "__main__":
    main()
