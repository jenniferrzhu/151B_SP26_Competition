"""Build a filtered subset of DeepMath-103K for external-teacher DPO.

Pipeline per item:
- Filter to difficulty range matching our test set
- Pick the SHORTEST of the 3 R1 traces (keeps in-memory cost low for DPO)
- Drop items where even the shortest trace exceeds MAX_R1_CHARS (so the chosen
  side will fit in MAX_SEQ_LEN at training time without truncation)
- Contamination filter against data/{public,private,test}.jsonl

Output schema (one row per kept item):
  {"id": str, "question": str, "answer": str, "r1_trace": str,
   "difficulty": float, "topic": str}

This subset feeds:
  1. gen_pref_data_deepmath.py — samples our base model on these problems
  2. build_pref_pairs.py (or a deepmath-flavored variant) — pairs R1 traces as
     chosen with our model's wrong samples as rejected
"""
import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

DATASET = "zwhe99/DeepMath-103K"
OUT_PATH = "data/deepmath_subset.jsonl"
COMP_PATHS = ["data/public.jsonl", "data/private.jsonl", "data/test.jsonl"]


def normalize_loose(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def normalize_tight(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[.,;:!?'\"\-]+", "", s)
    return s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-difficulty", type=float, default=4.0)
    parser.add_argument("--max-difficulty", type=float, default=7.0)
    parser.add_argument("--max-r1-chars", type=int, default=10000,
                        help="drop items where even the shortest R1 trace exceeds this")
    parser.add_argument("--topics", default="Algebra,Calculus,Precalculus,Applied Mathematics",
                        help="comma-separated top-level topics to keep")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap final output size (None = no cap)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=OUT_PATH)
    args = parser.parse_args()
    keep_topics = {t.strip() for t in args.topics.split(",")}

    # Load comp questions for contamination check
    loose_pre80 = set()
    tight_pre120 = set()
    tight_full = set()
    for path in COMP_PATHS:
        if Path(path).exists():
            for line in open(path):
                d = json.loads(line)
                q = d["question"]
                loose_pre80.add(normalize_loose(q)[:80])
                t = normalize_tight(q)
                tight_pre120.add(t[:120])
                tight_full.add(t)
    print(f"Loaded {len(loose_pre80)} loose / {len(tight_pre120)} tight prefixes for contamination check")

    print(f"Loading {DATASET}...")
    ds = load_dataset(DATASET, split="train")
    print(f"  raw size: {len(ds)}")

    n_other_diff = 0
    n_other_topic = 0
    n_too_long = 0
    n_contam = 0
    n_kept = 0
    out_records = []

    for row in ds:
        diff = row["difficulty"]
        if diff < args.min_difficulty or diff > args.max_difficulty:
            n_other_diff += 1
            continue

        topic = row["topic"]
        # Parse "Mathematics -> X -> Y" → level-2 topic = X
        parts = [p.strip() for p in topic.split("->")]
        level2 = parts[1] if len(parts) >= 2 else topic
        if level2 not in keep_topics:
            n_other_topic += 1
            continue

        # Pick shortest of 3 R1 traces
        traces = [row[f"r1_solution_{i}"] for i in (1, 2, 3) if row.get(f"r1_solution_{i}")]
        traces = [t for t in traces if t]
        if not traces:
            continue
        r1 = min(traces, key=len)
        if len(r1) > args.max_r1_chars:
            n_too_long += 1
            continue

        # Contamination check
        q = row["question"]
        q_loose = normalize_loose(q)[:80]
        q_tight = normalize_tight(q)
        contaminated = (
            q_loose in loose_pre80
            or q_tight[:120] in tight_pre120
            or (len(q_tight) > 60 and any(q_tight in cf or cf in q_tight for cf in tight_full))
        )
        if contaminated:
            n_contam += 1
            continue

        out_records.append({
            "id": f"deepmath_{n_kept}",
            "question": q,
            "options": None,
            "answer": row["final_answer"],
            "trace": r1,            # field name matches train_lora.py SFT loader
            "difficulty": diff,
            "topic": topic,
            "level2": level2,
        })
        n_kept += 1

    # Optional cap (random subsample) for tractable downstream sampling
    if args.limit is not None and len(out_records) > args.limit:
        import random
        random.seed(args.seed)
        random.shuffle(out_records)
        out_records = out_records[: args.limit]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")

    print(f"\nFilter results:")
    print(f"  Skipped (difficulty out of [{args.min_difficulty},{args.max_difficulty}]): {n_other_diff}")
    print(f"  Skipped (off-topic):     {n_other_topic}")
    print(f"  Skipped (R1 too long):   {n_too_long}")
    print(f"  Skipped (contaminated):  {n_contam}")
    print(f"  Kept after filters:      {len(out_records)}")
    print(f"\nWrote {len(out_records)} records → {args.output}")

    # Summary stats
    if out_records:
        diffs = [r["difficulty"] for r in out_records]
        r1_lens = sorted(len(r["trace"]) for r in out_records)
        from collections import Counter
        topics = Counter(r["level2"] for r in out_records)
        print(f"\nDifficulty: min={min(diffs):.1f} max={max(diffs):.1f} avg={sum(diffs)/len(diffs):.1f}")
        print(f"R1 chars:   p50={r1_lens[len(r1_lens)//2]}  p90={r1_lens[int(len(r1_lens)*0.9)]}  max={r1_lens[-1]}")
        print(f"Topics:")
        for t, n in topics.most_common():
            print(f"  {t:30s} {n:>5}")


if __name__ == "__main__":
    main()
