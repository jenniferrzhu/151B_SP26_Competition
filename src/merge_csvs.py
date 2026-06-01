"""
Submission CSV Merger.
Takes MCQ answers from one file and FRQ answers from another.
Uses private.jsonl to determine which IDs are MCQ vs FRQ.
"""

import json
import csv
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcq_csv", type=str, required=True, help="CSV file to take MCQ answers from")
    parser.add_argument("--frq_csv", type=str, required=True, help="CSV file to take FRQ answers from")
    parser.add_argument("--input_jsonl", type=str, default="data/private.jsonl", help="Source of truth for IDs")
    parser.add_argument("--output", type=str, default="merged_submission.csv")
    args = parser.parse_args()

    # 1. Map IDs to Type (MCQ vs FRQ)
    mcq_ids = set()
    frq_ids = set()
    print(f"Loading metadata from {args.input_jsonl}...")
    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if item.get("options"):
                mcq_ids.add(str(item["id"]))
            else:
                frq_ids.add(str(item["id"]))
    
    print(f"Metadata: {len(mcq_ids)} MCQ items, {len(frq_ids)} FRQ items.")

    # 2. Load MCQ answers
    mcq_responses = {}
    print(f"Loading MCQ answers from {args.mcq_csv}...")
    with open(args.mcq_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["id"] in mcq_ids:
                mcq_responses[row["id"]] = row["response"]

    # 3. Load FRQ answers
    frq_responses = {}
    print(f"Loading FRQ answers from {args.frq_csv}...")
    with open(args.frq_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["id"] in frq_ids:
                frq_responses[row["id"]] = row["response"]

    # 4. Combine and Sort
    all_ids = sorted(list(mcq_ids | frq_ids), key=lambda x: int(x))
    final_rows = []
    
    missing_mcq = 0
    missing_frq = 0

    for i in all_ids:
        if i in mcq_ids:
            if i in mcq_responses:
                final_rows.append({"id": i, "response": mcq_responses[i]})
            else:
                missing_mcq += 1
                final_rows.append({"id": i, "response": ""})
        else:
            if i in frq_responses:
                final_rows.append({"id": i, "response": frq_responses[i]})
            else:
                missing_frq += 1
                final_rows.append({"id": i, "response": ""})

    # 5. Write output
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"\nMerging Complete:")
    print(f"  Output saved to: {args.output}")
    print(f"  Total Rows: {len(final_rows)}")
    if missing_mcq: print(f"  WARNING: {missing_mcq} MCQ items were missing from {args.mcq_csv}")
    if missing_frq: print(f"  WARNING: {missing_frq} FRQ items were missing from {args.frq_csv}")

if __name__ == "__main__":
    main()
