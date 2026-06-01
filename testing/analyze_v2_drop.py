import json

def get_results(path):
    res = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            res[d["id"]] = d
    return res

v6_path = "results/Trained LoRA v6 (DeepMath fmtfix + RSFT)/predictions.jsonl"
v2_path = "results/MCQv2 Check LoRA v6/predictions.jsonl"

v6_data = get_results(v6_path)
v2_data = get_results(v2_path)

# Filter for MCQ
v6_mcq = {id: d for id, d in v6_data.items() if d.get("is_mcq")}
v2_mcq = {id: d for id, d in v2_data.items()} # Check script only has MCQ

dropped = []
gained = []

for id, v6 in v6_mcq.items():
    if id in v2_mcq:
        if v6["correct"] and not v2_mcq[id]["correct"]:
            dropped.append(id)
        elif not v6["correct"] and v2_mcq[id]["correct"]:
            gained.append(id)

print(f"v6 MCQ Accuracy: {sum(1 for d in v6_mcq.values() if d['correct'])} / {len(v6_mcq)} ({sum(1 for d in v6_mcq.values() if d['correct'])/len(v6_mcq)*100:.2f}%)")
print(f"v2 MCQ Accuracy: {sum(1 for d in v2_mcq.values() if d['correct'])} / {len(v2_mcq)} ({sum(1 for d in v2_mcq.values() if d['correct'])/len(v2_mcq)*100:.2f}%)")
print(f"Dropped items: {len(dropped)}")
print(f"Gained items: {len(gained)}")
print(f"Dropped IDs: {dropped}")
print(f"Gained IDs: {gained}")
