import json

def get_results(path):
    res = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            # Standalone v6 has 'is_mcq' key, check script might have 'pred'
            if d.get("is_mcq") or "pred" in d:
                res[d["id"]] = d
    return res

v6_path = "results/Trained LoRA v6 (DeepMath fmtfix + RSFT)/predictions.jsonl"
check_path = "results/MCQ Check LoRA v6/predictions.jsonl"

v6_data = get_results(v6_path)
check_data = get_results(check_path)

# Questions that were CORRECT in v6 but WRONG in check
dropped_ids = []
for id, v6 in v6_data.items():
    if id in check_data:
        if v6["correct"] and not check_data[id]["correct"]:
            dropped_ids.append(id)

# Questions that were WRONG in v6 but CORRECT in check (random gain)
gained_ids = []
for id, v6 in v6_data.items():
    if id in check_data:
        if not v6["correct"] and check_data[id]["correct"]:
            gained_ids.append(id)

print(f"Dropped IDs (Correct in v6 -> Wrong in Check):")
print(dropped_ids)
print(f"Total Dropped: {len(dropped_ids)}")

print(f"\nGained IDs (Wrong in v6 -> Correct in Check):")
print(gained_ids)
print(f"Total Gained: {len(gained_ids)}")

# Analysis of a few dropped IDs
if dropped_ids:
    print("\nSample Analysis of Dropped Items:")
    for id in dropped_ids[:3]:
        v6 = v6_data[id]
        ck = check_data[id]
        print(f"ID {id}: Gold {v6['gold']}")
        # In standalone v6, responses were long strings. In check, we might have stored them differently.
        v6_resp = v6["response"]
        ck_resp = ck["response"]
        
        # Check for thinking truncation or logic flip
        v6_think_len = v6_resp.find("</think>")
        ck_think_len = ck_resp.find("</think>")
        
        print(f"  v6 thinking length: {v6_think_len}")
        print(f"  check thinking length: {ck_think_len}")
        
        # Extract letters
        import re
        def extract(t):
            m = re.findall(r"\\boxed\{\s*([A-Za-z])\s*\}", t)
            return m[-1] if m else "None"
        
        print(f"  v6 extracted: {extract(v6_resp)}")
        print(f"  check extracted: {extract(ck_resp)}")
        print("-" * 20)
