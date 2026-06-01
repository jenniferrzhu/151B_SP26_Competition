"""Convert the Hendrycks MATH dataset (Levels 4-5) to our LoRA-training schema.

For each problem:
- Extract the final \\boxed{...} from `solution` as the gold answer
- Keep the full `solution` text as the teacher trace (already ends in \\boxed{})

Output schema matches data/train_with_traces.jsonl so train_lora.py can mix them:
  {id, is_mcq=False, gold=[answer], question, options=None, trace=solution}

Also runs a contamination check: flags any MATH question whose first 80 chars
appear in data/public.jsonl or data/private.jsonl.

Usage:
  python prep_math_dataset.py                  # Levels 4 and 5
  python prep_math_dataset.py --levels 3,4,5   # custom levels
"""
import argparse
import glob
import json
import re
from pathlib import Path

DEFAULT_LEVELS = ["Level 4", "Level 5"]
MATH_GLOB = "MATH/*/*/*.json"
OUT_PATH = "data/train_math.jsonl"
COMP_PATHS = ["data/public.jsonl", "data/private.jsonl", "data/test.jsonl"]


def extract_boxed(text: str) -> str | None:
    """Extract content of the LAST \\boxed{...} (with balanced braces)."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    i = idx + len(r"\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
            out.append(c)
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(c)
        else:
            out.append(c)
        i += 1
    return "".join(out).strip() if depth == 0 else None


def normalize_loose(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def normalize_tight(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[.,;:!?'\"\-]+", "", s)
    return s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", default="4,5",
                        help="comma-separated levels to keep (default: 4,5)")
    parser.add_argument("--output", default=OUT_PATH)
    args = parser.parse_args()
    keep_levels = {f"Level {x.strip()}" for x in args.levels.split(",")}

    # Load competition questions for contamination check (two normalizations + substring)
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

    files = glob.glob(MATH_GLOB)
    print(f"Scanning {len(files)} MATH files...")

    n_total = 0
    n_kept = 0
    n_no_box = 0
    n_contam = 0
    n_other_level = 0
    out_records = []

    for f in files:
        n_total += 1
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get("level") not in keep_levels:
            n_other_level += 1
            continue
        ans = extract_boxed(d["solution"])
        if ans is None:
            n_no_box += 1
            continue
        q_loose = normalize_loose(d["problem"])[:80]
        q_tight = normalize_tight(d["problem"])
        contaminated = (
            q_loose in loose_pre80
            or q_tight[:120] in tight_pre120
            or (len(q_tight) > 60 and any(q_tight in cf or cf in q_tight for cf in tight_full))
        )
        if contaminated:
            n_contam += 1
            continue
        # synthesize an id from the file path so it doesn't collide with our 0-1125 range
        split, subject, fname = f.split("/")[-3:]
        rec_id = f"math_{split}_{subject}_{fname.replace('.json', '')}"
        out_records.append({
            "id": rec_id,
            "is_mcq": False,
            "gold": [ans],
            "question": d["problem"],
            "options": None,
            "trace": d["solution"],
            "level": d["level"],
            "type": d.get("type"),
        })
        n_kept += 1

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for rec in out_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nScanned: {n_total}")
    print(f"  Skipped (other level): {n_other_level}")
    print(f"  Skipped (no \\boxed):  {n_no_box}")
    print(f"  Skipped (contaminated): {n_contam}")
    print(f"  Kept:                 {n_kept}")
    print(f"\nWrote {n_kept} records → {args.output}")

    # Trace length stats
    if out_records:
        lens = sorted(len(r["trace"]) for r in out_records)
        avg = sum(lens) / len(lens)
        p50, p90, p99 = lens[len(lens)//2], lens[int(len(lens)*0.9)], lens[int(len(lens)*0.99)]
        print(f"\nTrace char length: avg={avg:.0f}  p50={p50}  p90={p90}  p99={p99}  max={lens[-1]}")
        print(f"  > 14000 chars (would be filtered by train_lora MAX_TRACE_CHARS): "
              f"{sum(1 for l in lens if l > 14000)}")


if __name__ == "__main__":
    main()
