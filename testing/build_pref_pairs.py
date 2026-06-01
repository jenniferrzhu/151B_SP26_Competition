"""Convert raw per-item samples (data/pref_candidates.jsonl) into DPO pairs.

For each item:
- If it has at least 1 correct AND at least 1 incorrect sample,
  emit (chosen, rejected) pairs by crossing correct × incorrect.
- Cap pairs/item at MAX_PAIRS_PER_ITEM to avoid heavy duplication.

Output schema matches TRL's DPOTrainer convention:
  {"prompt": <chat-formatted prompt str>,
   "chosen":   <correct trace + EOS>,
   "rejected": <incorrect trace + EOS>,
   "id": ..., "is_mcq": ...}

Usage:
  python build_pref_pairs.py
  python build_pref_pairs.py --max-pairs-per-item 4
"""
import argparse
import json
import random
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)
SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)


def build_prompt(question: str, options: Optional[list], tokenizer) -> str:
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        user = f"{question}\n\nOptions:\n{opts_text}"
        system = SYSTEM_PROMPT_MCQ
    else:
        user = question
        system = SYSTEM_PROMPT_MATH
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/pref_candidates.jsonl")
    parser.add_argument("--output", default="data/preference_pairs.jsonl")
    parser.add_argument("--max-pairs-per-item", type=int, default=3,
                        help="cap on (correct × incorrect) pairs per item")
    parser.add_argument("--max-completion-chars", type=int, default=14000,
                        help="drop pairs where either completion exceeds this length")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos = tokenizer.eos_token

    rows = [json.loads(l) for l in open(args.input)]
    print(f"Loaded {len(rows)} raw items from {args.input}")

    n_pair_items = 0
    n_pairs_emitted = 0
    n_dropped_len  = 0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as out_fp:
        for row in rows:
            correct = [s["text"] for s in row["samples"] if s["correct"]]
            wrong   = [s["text"] for s in row["samples"] if not s["correct"]]
            if not correct or not wrong:
                continue
            n_pair_items += 1

            pairs = [(c, w) for c in correct for w in wrong]
            random.shuffle(pairs)
            pairs = pairs[: args.max_pairs_per_item]

            prompt = build_prompt(row["question"], row.get("options"), tokenizer)

            for chosen_text, rejected_text in pairs:
                chosen   = chosen_text + eos
                rejected = rejected_text + eos
                if len(chosen) > args.max_completion_chars or len(rejected) > args.max_completion_chars:
                    n_dropped_len += 1
                    continue
                out_fp.write(json.dumps({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "id": row.get("id"),
                    "is_mcq": row.get("is_mcq", False),
                }) + "\n")
                n_pairs_emitted += 1

    print(f"\nWrote {n_pairs_emitted} pairs from {n_pair_items} items → {args.output}")
    print(f"  Dropped due to length cap: {n_dropped_len}")
    print(f"  Avg pairs per item: {n_pairs_emitted / max(n_pair_items, 1):.2f}")


if __name__ == "__main__":
    main()
