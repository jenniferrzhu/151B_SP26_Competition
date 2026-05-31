"""
Parallel MCQ Shard Script (Base Model + 6 Variants).
Matches the logic of run_shard_frq.py but specifically for MCQ items.
Run on multiple GPUs:
GPU 0: python run_shard_mcq.py --gpu_id 0 --shard 0 --num_shards 1 --input data/private.jsonl
"""

import json
import os
import time
import argparse
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID   = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_INPUT_PATH  = "data/private.jsonl"
MAX_TOKENS = 32768
CHUNK_SIZE = 32

CANDIDATE_VARIANTS = [
    ("baseline_deterministic", "", 2),
    ("answer_order_audit", "First identify every answer the problem asks for, especially each real [ANS] blank. Solve them in order and put all final sub-answers in one boxed list.", 1),
    ("formula_first_exact", "Before arithmetic, write down the relevant formula or theorem. Keep exact values until the final step and round only when the problem explicitly asks.", 1),
    ("independent_then_options", "Solve independently before looking at answer choices. For multiple choice, compare your result to every option and watch for common distractors.", 1),
    ("sanity_check", "After solving, check units, signs, ranges, rounding, and whether the answer is reasonable. Correct the final answer before boxing it if the check fails.", 1),
    ("concise_reasoning", "Solve the problem concisely. Keep the reasoning short and direct. Focus on the final result and skip unnecessary intermediate steps.", 1),
]

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. Solve the problem step-by-step inside <think> tags, "
    "then select the best option from the list below. "
    "Put ONLY the final letter inside \\boxed{}, e.g. \\boxed{C}."
)

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

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

    # 2. Init Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    llm = LLM(
        model=MODEL_ID, quantization="bitsandbytes", load_format="bitsandbytes",
        enable_prefix_caching=False, gpu_memory_utilization=0.85, 
        max_model_len=16384, trust_remote_code=True, max_num_seqs=64,
        max_num_batched_tokens=16384, kv_cache_memory_bytes=14 * 1024**3,
    )

    deterministic_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)
    sampled_params = SamplingParams(
        max_tokens=MAX_TOKENS, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0
    )

    # 3. Pre-initialize results
    results = []
    for item in my_items:
        results.append({
            "id": item["id"],
            "is_mcq": True,
            "gold": item.get("answer"),
            "candidates": [],
            "candidate_variants": [v[0] for v in CANDIDATE_VARIANTS]
        })

    # 4. Generate Variants
    for v_idx, v in enumerate(CANDIDATE_VARIANTS):
        name, instruction, weight = v
        prompts = []
        for item in my_items:
            user = format_mcq(item["question"], item["options"])
            if instruction:
                user = f"{user}\n\nAttempt style: {name}\n{instruction}\nFollow the original problem exactly. Put final letter in \\boxed{{}}."
            
            prompts.append(tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT_MCQ}, {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True
            ))
        
        log(f"Generating MCQ Variant {v_idx+1}/6: {name} in chunks of {CHUNK_SIZE}...")
        params = deterministic_params if v_idx == 0 else sampled_params
        
        v_responses = []
        for i in range(0, len(prompts), CHUNK_SIZE):
            chunk = prompts[i : i + CHUNK_SIZE]
            outputs = llm.generate(chunk, sampling_params=params, use_tqdm=False)
            v_responses.extend([out.outputs[0].text.strip() for out in outputs])
            log(f"  {min(i + CHUNK_SIZE, len(prompts))}/{len(prompts)} done")
        
        for i, resp in enumerate(v_responses):
            results[i]["candidates"].append(resp)

    # 5. Save
    out_dir = Path("results/shards")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mcq_shard_{args.shard}.jsonl"
    with open(out_path, "w", encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    log(f"Saved MCQ shard to {out_path}")

if __name__ == "__main__":
    main()
