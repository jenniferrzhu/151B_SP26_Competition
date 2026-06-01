"""
Parallel MCQ Shard Script (LoRA v6 + Majority Voting n=8).
Matches the logic of run_mcq_check.py but supports sharding across multiple GPUs.
Run on multiple GPUs:
GPU 0: python run_shard_mcq.py --gpu_id 0 --shard 0 --num_shards 2 --input data/private.jsonl
GPU 1: python run_shard_mcq.py --gpu_id 1 --shard 1 --num_shards 2 --input data/private.jsonl
"""

import json
import os
import time
import argparse
import re
from pathlib import Path
from collections import Counter

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID      = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_INPUT_PATH  = "data/private.jsonl"
ADAPTER_PATH  = "adapters/qwen3-lora-v6-mixed-fmtfix"
LORA_RANK     = 16
NUM_SAMPLES   = 8
MAX_TOKENS    = 32768
CHUNK_SIZE    = 8 # Reduced because we generate 8 per prompt

# EXACTLY matches LoRA v6 fine-tuning prompt for MCQ
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

def format_mcq(question: str, options: list) -> str:
    labels = [chr(65 + i) for i in range(len(options))]
    opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
    return f"{question}\n\nOptions:\n{opts_text}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=str, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_PATH)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    # 1. Load and Filter
    log(f"Loading data from {args.input}...")
    data = [json.loads(line) for line in open(args.input, encoding='utf-8')]
    mcq_items = [d for d in data if d.get("options")]
    
    # Slice shard
    n_total = len(mcq_items)
    shard_size = (n_total + args.num_shards - 1) // args.num_shards
    start_idx = args.shard * shard_size
    end_idx = min(start_idx + shard_size, n_total)
    my_items = mcq_items[start_idx:end_idx]
    
    log(f"MCQ Shard {args.shard}/{args.num_shards}: Processing items {start_idx} to {end_idx} ({len(my_items)} items)")

    if not my_items:
        log("No items in this shard. Exiting.")
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

    # 3. Build Prompts
    prompts = []
    for item in my_items:
        user_content = format_mcq(item["question"], item["options"])
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT_MCQ}, {"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True
        ))

    # 4. Generate with n=8
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

    # 5. Process and Save
    results = []
    for item, responses in zip(my_items, all_responses):
        preds = [extract_letter(r) for r in responses]
        counts = Counter([p for p in preds if p])
        final_pred = counts.most_common(1)[0][0] if counts else (preds[0] if preds else "")
        
        results.append({
            "id": item["id"],
            "is_mcq": True,
            "gold": item.get("answer"),
            "response": f"\\boxed{{{final_pred}}}",
            "candidates": [f"\\boxed{{{p}}}" for p in preds], # For assembler compatibility
            "all_preds": preds,
            "vote_counts": dict(counts)
        })

    out_dir = Path("results/shards")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mcq2_shard_{args.shard}.jsonl"
    with open(out_path, "w", encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    log(f"Saved MCQ shard to {out_path}")

if __name__ == "__main__":
    main()
