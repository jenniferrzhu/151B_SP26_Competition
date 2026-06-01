import json

def get_mcq_results(path):
    results = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["is_mcq"]:
                results[r["id"]] = r
    return results

v6_path = "results/Trained LoRA v6 (DeepMath fmtfix + RSFT)/predictions.jsonl"
v8_path = "results/Hybrid v8 MCQ-Lora + FRQ-WeightedMajority/predictions.jsonl"

v6_mcq = get_mcq_results(v6_path)
v8_mcq = get_mcq_results(v8_path)

ids = sorted(v6_mcq.keys())
for id in ids[:10]: # Just look at the first 10
    v6 = v6_mcq[id]
    v8 = v8_mcq[id]
    print(f"ID {id}: v6_correct={v6['correct']}, v8_correct={v8['correct']}")
    print(f"  v6 response length: {len(v6['response'])}")
    print(f"  v8 response length: {len(v8['response'])}")
    if len(v6['response']) > 100:
        print(f"  v6 starts with: {v6['response'][:100]}...")
    else:
        print(f"  v6 response: {v6['response']}")
    if len(v8['response']) > 100:
        print(f"  v8 starts with: {v8['response'][:100]}...")
    else:
        print(f"  v8 response: {v8['response']}")
    print("-" * 20)
