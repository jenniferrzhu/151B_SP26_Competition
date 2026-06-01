import json

def get_mcq_results(path):
    results = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("is_mcq") or "pred" in r: # Handle different schemas
                results[r["id"]] = {
                    "correct": r["correct"],
                    "response": r["response"]
                }
    return results

v6_path = "results/Trained LoRA v6 (DeepMath fmtfix + RSFT)/predictions.jsonl"
check_path = "results/MCQ Check LoRA v6/predictions.jsonl"

v6 = get_mcq_results(v6_path)
check = get_mcq_results(check_path)

diffs = []
for id, res in v6.items():
    if id in check:
        if res["correct"] and not check[id]["correct"]:
            diffs.append(id)

print(f"IDs v6 got right but check got wrong: {diffs}")
print(f"Total: {len(diffs)}")

if diffs:
    id = diffs[0]
    print(f"\nExample ID {id}:")
    print(f"v6 response (first 200 chars): {v6[id]['response'][:200]}...")
    print(f"check response (first 200 chars): {check[id]['response'][:200]}...")
