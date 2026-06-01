"""Rewrite R1 traces so only the FINAL \\boxed{} survives.

99.96% of DeepMath R1 traces have multiple \\boxed{} — every key intermediate
result is boxed for emphasis, plus the final answer. When used as SFT targets,
the model learns to box intermediate results too, which breaks our judger that
expects a single final \\boxed{} (and a single comma-separated boxed for
multi-sub-answer items).

This script processes data/deepmath_subset.jsonl and produces a corrected file
where every intermediate \\boxed{X} is replaced with just X, leaving only the
LAST \\boxed{...} intact as the final answer.

Usage:
  python fix_r1_format.py
  python fix_r1_format.py --input X --output Y
"""
import argparse
import json
import re


def extract_boxed_spans(text: str) -> list[tuple[int, int, str]]:
    """Return list of (start, end, inner) for every \\boxed{...} with balanced braces."""
    spans = []
    i = 0
    while True:
        idx = text.find(r"\boxed{", i)
        if idx < 0:
            break
        brace_start = idx + len(r"\boxed{")
        depth = 1
        j = brace_start
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth == 0:
            # span covers from \boxed{ through the matching closing }
            spans.append((idx, j + 1, text[brace_start:j]))
            i = j + 1
        else:
            # unbalanced — stop
            break
    return spans


def strip_intermediate_boxed(text: str) -> str:
    """Keep only the LAST \\boxed{...}; replace earlier ones with their inner content."""
    spans = extract_boxed_spans(text)
    if len(spans) <= 1:
        return text
    # Rewrite right-to-left so earlier indices stay valid as we mutate later positions
    out = text
    # Skip the last span (keep it); replace all earlier ones
    for start, end, inner in reversed(spans[:-1]):
        out = out[:start] + inner + out[end:]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/deepmath_subset.jsonl")
    parser.add_argument("--output", default="data/deepmath_subset_fmtfix.jsonl")
    args = parser.parse_args()

    rows = [json.loads(l) for l in open(args.input)]
    multi_before = 0
    after_one = 0
    total = len(rows)

    with open(args.output, "w") as f:
        for r in rows:
            spans = extract_boxed_spans(r["trace"])
            if len(spans) > 1:
                multi_before += 1
                r["trace"] = strip_intermediate_boxed(r["trace"])
            # sanity-recount
            if len(extract_boxed_spans(r["trace"])) == 1:
                after_one += 1
            f.write(json.dumps(r) + "\n")

    print(f"Processed {total} items")
    print(f"  Multi-boxed before: {multi_before}")
    print(f"  Single-boxed after fix: {after_one}/{total}")
    print(f"\nWrote → {args.output}")


if __name__ == "__main__":
    main()
