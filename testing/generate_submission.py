import json
import csv
import argparse
import sys
from pathlib import Path

def convert_jsonl_to_submission_csv(input_path: str, output_path: str):
    """
    Converts a predictions.jsonl file into a Kaggle-ready submission.csv.
    Format: "id","response"
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        print(f"Error: Input file {input_path} not found.")
        sys.exit(1)

    print(f"Reading from {input_path}...")
    
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            
            # The competition expects the 'id' and the raw 'response' string.
            # We use the 'response' field which contains the model's finalized output.
            if "id" in data and "response" in data:
                records.append({
                    "id": str(data["id"]),
                    "response": data["response"]
                })
            else:
                print(f"Warning: Missing 'id' or 'response' in line: {line[:100]}...")

    # Sort by ID to ensure a consistent submission file
    try:
        records.sort(key=lambda x: int(x["id"]))
    except ValueError:
        records.sort(key=lambda x: x["id"])

    print(f"Writing {len(records)} records to {output_path}...")
    
    # We use csv.QUOTE_ALL to match the provided sample submission format
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(records)

    print("Success! File is ready for submission.")

def main():
    parser = argparse.ArgumentParser(description="Generate Kaggle submission CSV from JSONL predictions.")
    parser.add_argument("--input", type=str, default="results/Hybrid v6 MCQ + Majority FRQ/predictions.jsonl", 
                        help="Path to the predictions.jsonl file")
    parser.add_argument("--output", type=str, default="submission.csv", 
                        help="Path to save the submission.csv")
    
    args = parser.parse_args()
    convert_jsonl_to_submission_csv(args.input, args.output)

if __name__ == "__main__":
    main()
