import json

def get_mcq_results(path):
    results = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["is_mcq"]:
                results[r["id"]] = {
                    "correct": r["correct"],
                    "response": r["response"],
                    "gold": r["gold"]
                }
    return results

v6_path = "results/Trained LoRA v6 (DeepMath fmtfix + RSFT)/predictions.jsonl"
v8_path = "results/Hybrid v8 MCQ-Lora + FRQ-WeightedMajority/predictions.jsonl"

v6_mcq = get_mcq_results(v6_path)
v8_mcq = get_mcq_results(v8_path)

ids = sorted(v6_mcq.keys())
diff_count = 0
for id in ids:
    v6 = v6_mcq[id]
    v8 = v8_mcq[id]
    if v6["correct"] != v8["correct"]:
        diff_count += 1
        print(f"ID {id}: v6={v6['correct']} (Gold {v6['gold']}) | v8={v8['correct']} (Gold {v8['gold']})")
        # Print responses if they are short, otherwise snippets
        print(f"  v6 response: {v6['response']}")
        print(f"  v8 response: {v8['response']}")
        print("-" * 20)

print(f"Total MCQ differences: {diff_count}")
print(f"v6 Accuracy: {sum(1 for r in v6_mcq.values() if r['correct'])} / {len(v6_mcq)}")
print(f"v8 Accuracy: {sum(1 for r in v8_mcq.values() if r['correct'])} / {len(v8_mcq)}")
