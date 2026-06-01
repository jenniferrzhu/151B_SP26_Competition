"""
Assembler Script for Hybrid Parallel Shards.
Merges ALL shards (frq_shard_*.jsonl and mcq2_shard_*.jsonl) from results/shards/.
Applies Weighted Majority Voting and generates submission.csv.
"""

import json
import os
import re
import sys
import time
import csv
from pathlib import Path
from collections import Counter

# ── Weighted Consistency Config ─────────────────────────────────────────────
CANDIDATE_VARIANTS = [
    ("baseline_deterministic", 2),
    ("answer_order_audit", 1),
    ("formula_first_exact", 1),
    ("independent_then_options", 1),
    ("sanity_check", 1),
    ("concise_reasoning", 1),
]

# ── Normalization Utilities ──────────────────────────────────────────────────

def answer_visible_text(text: str) -> str:
    think_end = text.rfind("</think>")
    return text[think_end + len("</think>"):] if think_end >= 0 else text

def extract_boxed_group(text: str) -> list[str]:
    entries = []
    start = 0
    while True:
        idx = text.find("\\boxed{", start)
        if idx < 0: break
        brace_start = idx + len("\\boxed{")
        depth, i = 1, brace_start
        while i < len(text) and depth > 0:
            if text[i] == "{": depth += 1
            elif text[i] == "}": depth -= 1
            i += 1
        if depth == 0:
            content = text[brace_start:i - 1].strip()
            if content: entries.append((idx, i, content))
        start = i
    if not entries: return []
    last_group = [entries[-1]]
    for j in range(len(entries) - 2, -1, -1):
        gap = text[entries[j][1]:entries[j + 1][0]]
        if re.match(r"^[\s,\$\.\;\:\-\&\\]*$", gap):
            last_group.insert(0, entries[j])
        else: break
    return [item[2] for item in last_group]

def normalize_tokens(text: str) -> str:
    cleaned = text.replace("\\,", "").replace("\\left", "").replace("\\right", "").replace("$", "").replace("\\", "").strip()
    t = cleaned.lower()
    if t in {"yes", "y", "true"}: return "true"
    if t in {"no", "n", "false"}: return "false"
    allowed = "".join(ch if (ch.isalnum() or ch in ".,/()[]{}+-*^=") else " " for ch in cleaned)
    return " ".join(allowed.split())

def extract_letter(text: str) -> str:
    search_text = answer_visible_text(text)
    matches = re.findall(r"\\boxed\{\s*([A-Za-z])\s*\}", search_text)
    if not matches: matches = re.findall(r"\\boxed\{\s*([A-Za-z])\s*\}", text)
    if matches: return matches[-1].upper()
    matches = re.findall(r"\b([A-Z])\b", search_text.upper())
    if not matches: matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""

def canonicalize_answer(answer: str, is_mcq: bool) -> str:
    if is_mcq: return extract_letter(answer)
    boxed_group = extract_boxed_group(answer_visible_text(answer))
    if boxed_group: return ", ".join(normalize_tokens(part) for part in boxed_group)
    lines = [l.strip() for l in answer_visible_text(answer).splitlines() if l.strip()]
    return normalize_tokens(lines[-1]) if lines else ""

def weighted_majority_vote(candidates: list[str], is_mcq: bool) -> str:
    if is_mcq:
        # Matches run_mcq_check: uniform weight 1, choices are letters
        weights = [1] * len(candidates)
    else:
        # Matches v8: baseline weight 2, others weight 1
        base_weights = [v[1] for v in CANDIDATE_VARIANTS]
        weights = base_weights + [1] * (len(candidates) - len(base_weights))
        weights = weights[:len(candidates)]
        
    counts = {}
    canonical_to_raw = {}
    for cand, weight in zip(candidates, weights):
        key = canonicalize_answer(cand, is_mcq)
        if not key: continue
        counts[key] = counts.get(key, 0) + weight
        
        # In case of ties or multiple raw traces for same canonical key:
        # Prefer the raw trace of the HIGHER weighted candidate (like Baseline)
        if key not in canonical_to_raw or weight > 1:
            canonical_to_raw[key] = cand
            
    if not counts: 
        return candidates[0] if candidates else ""
    
    # Pick the key with highest total weight
    best_key = max(counts, key=counts.get)
    return canonical_to_raw[best_key]

# ── Main ────────────────────────────────────────────────────────────────────

def main():
    shard_dir = Path("results/shards")
    output_name = "Submission_Consolidated_v9"
    out_dir = Path(f"results/{output_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Voted MCQ or MCQ shards
    mcq_path = shard_dir / "mcq_voted.jsonl"
    mcq_results = []
    if mcq_path.exists():
        print(f"Loading Voted MCQ: {mcq_path.name}")
        mcq_results = [json.loads(l) for l in open(mcq_path, encoding='utf-8')]
    else:
        # Update prefix to mcq2_shard_ as requested
        mcq_shard_files = sorted(list(shard_dir.glob("mcq2_shard_*.jsonl")))
        if not mcq_shard_files:
            # Fallback to mcq_shard_ if mcq2 is not found
            mcq_shard_files = sorted(list(shard_dir.glob("mcq_shard_*.jsonl")))
            
        for sf in mcq_shard_files:
            print(f"Loading MCQ shard: {sf.name}")
            shard_data = [json.loads(l) for l in open(sf, encoding='utf-8')]
            for item in shard_data:
                # Apply voting if it has candidates
                if "candidates" in item and len(item["candidates"]) > 1:
                    winner_text = weighted_majority_vote(item["candidates"], is_mcq=True)
                    item["response"] = winner_text
                elif "response" not in item and "candidates" in item:
                    item["response"] = item["candidates"][0]
                mcq_results.append(item)

    # 2. Load FRQ Shards
    frq_results = []
    frq_shard_files = sorted(list(shard_dir.glob("frq_shard_*.jsonl")))
    for sf in frq_shard_files:
        print(f"Loading FRQ shard: {sf.name}")
        shard_data = [json.loads(l) for l in open(sf, encoding='utf-8')]
        for item in shard_data:
            if "candidates" in item and len(item["candidates"]) > 1:
                winner_text = weighted_majority_vote(item["candidates"], is_mcq=False)
                item["response"] = winner_text
            elif "response" not in item and "candidates" in item:
                item["response"] = item["candidates"][0]
            frq_results.append(item)

    # Union the results into combined
    combined = mcq_results + frq_results
    
    if not combined:
        print(f"Error: No results found in {shard_dir}")
        return

    # Sort to match original order
    combined.sort(key=lambda x: x["id"])
    
    # 3. Optional Scoring
    has_gold = any(r.get("gold") is not None for r in combined)
    if has_gold:
        sys.path.insert(0, ".")
        from judger import Judger
        judger = Judger(strict_extract=False)

        print("Scoring...")
        for r in combined:
            gold = r.get("gold")
            if gold is None:
                r["correct"] = None
                continue
            resp = r["response"]
            if r["is_mcq"]:
                r["correct"] = extract_letter(resp) == str(gold).strip().upper()
            else:
                gold_list = gold if isinstance(gold, list) else [gold]
                try:
                    r["correct"] = judger.auto_judge(pred=resp, gold=gold_list, options=[[]]*len(gold_list))
                except:
                    r["correct"] = False

    # 4. Save JSONL
    pred_path = out_dir / "predictions.jsonl"
    with open(pred_path, "w", encoding='utf-8') as f:
        for r in combined:
            f.write(json.dumps(r) + "\n")

    # 5. Save CSV Submission
    csv_path = out_dir / "submission.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "response"])
        for r in combined:
            writer.writerow([str(r["id"]), r["response"]])
    print(f"Saved submission to {csv_path}")

    # Summary
    if has_gold:
        scored = [r for r in combined if r.get("correct") is not None]
        if scored:
            mcq_sub = [r for r in scored if r["is_mcq"]]
            frq_sub = [r for r in scored if not r["is_mcq"]]
            
            mcq_acc = sum(r["correct"] for r in mcq_sub) / len(mcq_sub) * 100 if mcq_sub else 0
            frq_acc = sum(r["correct"] for r in frq_sub) / len(frq_sub) * 100 if frq_sub else 0
            overall = sum(r["correct"] for r in scored) / len(scored) * 100
            
            summary = (
                f"Consolidated Results:\n"
                f"MCQ Accuracy : {mcq_acc:.2f}% ({sum(r['correct'] for r in mcq_sub)}/{len(mcq_sub)})\n"
                f"FRQ Accuracy : {frq_acc:.2f}% ({sum(r['correct'] for r in frq_sub)}/{len(frq_sub)})\n"
                f"Overall      : {overall:.2f}%\n"
            )
            print("\n" + summary)
            with open(out_dir / "accuracy.txt", "w") as f:
                f.write(summary)
    else:
        print("No gold answers found. Skipping accuracy summary.")

if __name__ == "__main__":
    main()
