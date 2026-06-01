import json
import re

def extract_letter_v6(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""

def extract_letter_v8(text: str) -> str:
    think_end = text.rfind("</think>")
    search_text = text[think_end + len("</think>"):] if think_end >= 0 else text
    matches = re.findall(r"\\boxed\{\s*([A-Za-z])\s*\}", search_text)
    if not matches: matches = re.findall(r"\\boxed\{\s*([A-Za-z])\s*\}", text)
    if matches: return matches[-1].upper()
    matches = re.findall(r"\b([A-Z])\b", search_text.upper())
    if not matches: matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""

v6_path = "results/Trained LoRA v6 (DeepMath fmtfix + RSFT)/predictions.jsonl"
v8_path = "results/Hybrid v8 MCQ-Lora + FRQ-WeightedMajority/predictions.jsonl"

v6_data = [json.loads(line) for line in open(v6_path) if json.loads(line)["is_mcq"]]
v8_data = [json.loads(line) for line in open(v8_path) if json.loads(line)["is_mcq"]]

# Cross-check extraction logic on same data
print("Checking extraction logic consistency on v6 data:")
v6_extractions_mismatch = 0
for r in v6_data:
    ev6 = extract_letter_v6(r["response"])
    ev8 = extract_letter_v8(r["response"])
    if ev6 != ev8:
        v6_extractions_mismatch += 1
        print(f"ID {r['id']}: v6_extract={ev6}, v8_extract={ev8} | Gold={r['gold']}")
        # print(f"  Response: {r['response'][-200:]}")

print(f"Total extraction mismatches on v6 data: {v6_extractions_mismatch}")
