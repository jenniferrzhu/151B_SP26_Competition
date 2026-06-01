"""
Parallel FRQ Shard Script (v8 Logic).
Run on multiple GPUs:
GPU 0: python run_shard_frq.py --gpu_id 0 --shard 0 --num_shards 2 --input data/private.jsonl
GPU 1: python run_shard_frq.py --gpu_id 1 --shard 1 --num_shards 2 --input data/private.jsonl
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

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step inside <think> tags. "
    "Always provide the EXACT fractional or symbolic form first (e.g., \\frac{13}{9} or \\pi). "
    "If the problem requests a decimal, you may provide it, but prioritize the exact form in the final \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}."
)

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=str, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num_shards", type=int, default=2)
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_PATH)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # 1. Load and Filter
    log(f"Loading data from {args.input}...")
    data = [json.loads(line) for line in open(args.input, encoding='utf-8')]
    frq_items = [d for d in data if not d.get("options")]
    
    # Slice shard
    n_total = len(frq_items)
    shard_size = (n_total + args.num_shards - 1) // args.num_shards
    start_idx = args.shard * shard_size
    end_idx = min(start_idx + shard_size, n_total)
    my_items = frq_items[start_idx:end_idx]
    
    log(f"Shard {args.shard}/{args.num_shards}: Processing items {start_idx} to {end_idx} ({len(my_items)} items)")

    if not my_items:
        log("No items in this shard. Exiting.")
        return

    # 2. Init Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    llm = LLM(
        model=MODEL_ID, quantization="bitsandbytes", load_format="bitsandbytes",
        max_model_len=16384, gpu_memory_utilization=0.85, trust_remote_code=True,
        max_num_seqs=64,
    )

    deterministic_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)
    sampled_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.6, top_p=0.95, top_k=20)

    # 3. Pre-initialize results
    results = []
    for item in my_items:
        results.append({
            "id": item["id"],
            "is_mcq": False,
            "gold": item.get("answer"),
            "candidates": [],
            "candidate_variants": [v[0] for v in CANDIDATE_VARIANTS]
        })

    # 4. Generate Variants
    for v_idx, v in enumerate(CANDIDATE_VARIANTS):
        name, instruction, weight = v
        prompts = []
        for item in my_items:
            user = item["question"]
            if instruction:
                user = f"{user}\n\nAttempt style: {name}\n{instruction}\nFollow the original problem exactly. Put final answer in \\boxed{{}}."
            
            prompts.append(tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT_MATH}, {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True
            ))
        
        log(f"Generating Variant {v_idx+1}/6: {name}")
        params = deterministic_params if v_idx == 0 else sampled_params
        outputs = llm.generate(prompts, sampling_params=params, use_tqdm=True)
        
        for i, out in enumerate(outputs):
            results[i]["candidates"].append(out.outputs[0].text.strip())

    # 5. Save
    out_dir = Path("results/shards")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"frq_shard_{args.shard}.jsonl"
    with open(out_path, "w", encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    log(f"Saved shard to {out_path}")

if __name__ == "__main__":
    main()
