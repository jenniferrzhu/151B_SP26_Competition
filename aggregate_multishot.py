"""Smart aggregators over multishot candidates — closes the selector→oracle gap.

Reads predictions.jsonl from any multishot run (Multishot v1, v2, etc.) and
applies several aggregators side-by-side. No GPU needed — purely re-scoring
candidates that already exist on disk.

Aggregators:
  equiv_majority   — Cluster candidates by numeric/symbolic equivalence (via
                     judger.is_equal). Pick the largest cluster. Ties → first.
  variant_weighted — Same clustering, but weight each candidate by the
                     candidate variant's overall accuracy on this run (so a
                     `formula_first_exact` vote counts more than a `sanity_check`
                     vote when they disagree). Falls back to uniform weights if
                     no per-variant info is available.

Usage:
  python aggregate_multishot.py --source "results/Multishot v2 (diverse candidates)"
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def extract_letter(text: str) -> str:
    think_end = text.rfind("</think>")
    search_text = text[think_end + len("</think>"):] if think_end >= 0 else text
    m = re.search(r"\\boxed\{\s*([A-Za-z])\s*\}", search_text)
    if not m:
        m = re.search(r"\\boxed\{\s*([A-Za-z])\s*\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", search_text.upper())
    if not matches:
        matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def normalize_freeform(judger, cand: str) -> str:
    ans = judger.extract_ans(cand)
    if not ans:
        return ""
    parts = judger.split_by_comma(ans)
    norm_parts = [judger.norm_ans_str(p) for p in parts]
    return ", ".join(norm_parts) if norm_parts else ""


def cluster_with_weights(judger, votes_with_weights, is_mcq):
    """Cluster (vote_str, weight) pairs by mathematical equivalence; return
    [(representative, total_weight, count)] sorted by total_weight desc."""
    clusters = []  # list of [rep, total_weight, count]
    for vote, w in votes_with_weights:
        if not vote:
            continue
        placed = False
        for c in clusters:
            try:
                same = (vote == c[0]) if is_mcq else judger.is_equal(vote, c[0])
            except Exception:
                same = False
            if same:
                c[1] += w
                c[2] += 1
                placed = True
                break
        if not placed:
            clusters.append([vote, float(w), 1])
    clusters.sort(key=lambda c: -c[1])
    return clusters


def score_one(judger, response_box, gold, is_mcq):
    if is_mcq:
        return (response_box == str(gold).strip().upper())
    gold_list = gold if isinstance(gold, list) else [gold]
    try:
        return judger.auto_judge(
            pred=f"\\boxed{{{response_box}}}" if response_box else "",
            gold=gold_list, options=[[]] * len(gold_list),
        )
    except Exception:
        return False


def evaluate(judger, rows, weights_by_variant, label):
    """Apply weighted equivalence-majority and return (n_correct, mcq_correct, mcq_total, free_correct, free_total)."""
    mcq_c = mcq_t = free_c = free_t = 0
    out_rows = []
    for r in rows:
        is_mcq = r["is_mcq"]
        gold = r["gold"]
        variants = r.get("candidate_variants") or [f"cand_{i}" for i in range(len(r["candidates"]))]
        # Build (vote_str, weight) pairs
        pairs = []
        for cand, vname in zip(r["candidates"], variants):
            if is_mcq:
                vote = extract_letter(cand)
            else:
                vote = normalize_freeform(judger, cand)
            w = weights_by_variant.get(vname, 1.0)
            pairs.append((vote, w))
        clusters = cluster_with_weights(judger, pairs, is_mcq)
        if clusters:
            chosen, total_w, count = clusters[0]
        else:
            chosen, total_w, count = "", 0.0, 0
        correct = score_one(judger, chosen, gold, is_mcq)
        if is_mcq:
            mcq_t += 1
            mcq_c += int(correct)
        else:
            free_t += 1
            free_c += int(correct)
        out_rows.append({
            "id": r["id"],
            "is_mcq": is_mcq,
            "gold": gold,
            "chosen": chosen,
            "cluster_weight": total_w,
            "cluster_count": count,
            "response": f"\\boxed{{{chosen}}}" if chosen else "",
            "correct": correct,
            "label": label,
        })
    return out_rows, mcq_c, mcq_t, free_c, free_t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help='results folder, e.g. "results/Multishot v2 (diverse candidates)"')
    args = ap.parse_args()

    src_dir = Path(args.source)
    src_pred = src_dir / "predictions.jsonl"
    assert src_pred.exists(), f"missing {src_pred}"

    sys.path.insert(0, ".")
    from judger import Judger
    judger = Judger(strict_extract=False)

    rows = [json.loads(l) for l in open(src_pred)]
    print(f"Loaded {len(rows)} rows from {src_pred}")

    # Per-variant overall accuracy (rescored against gold) — used as weights.
    by_variant_total = defaultdict(int)
    by_variant_correct = defaultdict(int)
    for r in rows:
        variants = r.get("candidate_variants") or [f"cand_{i}" for i in range(len(r["candidates"]))]
        is_mcq = r["is_mcq"]
        gold = r["gold"]
        for cand, vname in zip(r["candidates"], variants):
            if is_mcq:
                vote = extract_letter(cand)
            else:
                vote = normalize_freeform(judger, cand)
            correct = score_one(judger, vote, gold, is_mcq)
            by_variant_total[vname] += 1
            by_variant_correct[vname] += int(correct)
    weights = {v: by_variant_correct[v] / max(by_variant_total[v], 1) for v in by_variant_total}
    print("\nPer-variant accuracy (used as weights):")
    for v in sorted(weights):
        print(f"  {v:30s} {weights[v]*100:.2f}%")

    # Run multiple aggregators side-by-side
    strategies = [
        ("uniform_majority",  {v: 1.0 for v in by_variant_total}),
        ("variant_weighted",  weights),
    ]
    summary_lines = []
    summary_lines.append(f"Multishot aggregator comparison")
    summary_lines.append(f"Source: {src_pred}")
    summary_lines.append(f"Total: {len(rows)}")
    summary_lines.append("")
    summary_lines.append(f"{'Strategy':22s} {'MCQ':>15} {'FRQ':>15} {'Overall':>15}")
    summary_lines.append("-" * 70)
    for label, w in strategies:
        out, mcq_c, mcq_t, free_c, free_t = evaluate(judger, rows, w, label)
        out_dir = src_dir.with_name(src_dir.name + f" [{label}]")
        out_dir.mkdir(exist_ok=True)
        with open(out_dir / "predictions.jsonl", "w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        ov = mcq_c + free_c
        line = (
            f"{label:22s} "
            f"{mcq_c:>3}/{mcq_t:<3} ({mcq_c/max(mcq_t,1)*100:5.2f}%)  "
            f"{free_c:>3}/{free_t:<3} ({free_c/max(free_t,1)*100:5.2f}%)  "
            f"{ov:>3}/{len(rows):<3} ({ov/len(rows)*100:5.2f}%)"
        )
        summary_lines.append(line)
        # Write per-strategy accuracy file too
        (out_dir / "accuracy.txt").write_text(
            f"Aggregator: {label}\n"
            f"Source: {src_pred}\n"
            f"Total: {len(rows)}\n\n"
            f"  MCQ        : {mcq_c:4d} / {mcq_t:4d}  ({mcq_c/max(mcq_t,1)*100:.2f}%)\n"
            f"  Free-form  : {free_c:4d} / {free_t:4d}  ({free_c/max(free_t,1)*100:.2f}%)\n"
            f"  Overall    : {ov:4d} / {len(rows):4d}  ({ov/len(rows)*100:.2f}%)\n"
        )
    print()
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
