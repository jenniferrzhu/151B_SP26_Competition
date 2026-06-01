import json
import re
from collections import Counter
from typing import Optional, List, Tuple

def extract_boxed_group(text: str) -> list[str]:
    entries = []
    start = 0
    while True:
        idx = text.find("\\boxed{", start)
        if idx < 0: break
        brace_start = idx + len("\\boxed{")
        depth = 1
        i = brace_start
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

def answer_visible_text(text: str) -> str:
    think_end = text.rfind("</think>")
    return text[think_end + len("</think>"):] if think_end >= 0 else text

def normalize_tokens(text: str) -> str:
    cleaned = text.replace("\\,", "").replace("\\left", "").replace("\\right", "")
    cleaned = cleaned.replace("$", "").replace("\\", "").strip()
    t = cleaned.lower()
    if t in {"yes", "y", "true"}: return "true"
    if t in {"no", "n", "false"}: return "false"
    allowed = []
    for ch in cleaned:
        if ch.isalnum() or ch in ".,/()[]{}+-*^=": allowed.append(ch)
        elif ch.isspace(): allowed.append(" ")
    return " ".join("".join(allowed).split())

def extract_letter(text: str) -> str:
    search_text = answer_visible_text(text)
    matches = re.findall(r"\\boxed\{\s*([A-Za-z])\s*\}", search_text)
    if not matches: matches = re.findall(r"\\boxed\{\s*([A-Za-z])\s*\}", text)
    if matches: return matches[-1].upper()
    matches = re.findall(r"\b([A-Z])\b", search_text.upper())
    return matches[-1] if matches else ""

def canonicalize_answer(answer: str, is_mcq: bool) -> str:
    if is_mcq: return extract_letter(answer)
    boxed_group = extract_boxed_group(answer_visible_text(answer))
    if boxed_group: return ", ".join(normalize_tokens(part) for part in boxed_group)
    lines = [l.strip() for l in answer_visible_text(answer).splitlines() if l.strip()]
    return normalize_tokens(lines[-1]) if lines else ""

def majority_vote_answer(candidates: list[str], is_mcq: bool) -> str:
    counts = Counter()
    canonical_to_raw = {}
    for cand in candidates:
        key = canonicalize_answer(cand, is_mcq)
        if not key: continue
        counts[key] += 1
        if key not in canonical_to_raw: canonical_to_raw[key] = cand
    if not counts: return candidates[0]
    best_key = counts.most_common(1)[0][0]
    return canonical_to_raw[best_key]

def main():
    import signal
    if not hasattr(signal, "SIGALRM"):
        signal.SIGALRM = signal.SIGINT 
    if not hasattr(signal, "alarm"):
        signal.alarm = lambda x: None
    
    PRED_PATH = "results/GEPA Optimized/predictions.jsonl"
    from judger import Judger
    judger = Judger(strict_extract=False)
    
    total_recovered = 0
    total_missed_before = 0
    
    print(f"{'ID':<6} | {'Type':<5} | {'Maj Correct?':<12} | {'Recovery Status'}")
    print("-" * 60)

    with open(PRED_PATH, "r") as f:
        for line in f:
            ev = json.loads(line)
            is_mcq = ev["is_mcq"]
            oracle = ev.get("oracle_correct")
            old_correct = ev.get("correct")
            
            if oracle and not old_correct:
                total_missed_before += 1
                maj_resp = majority_vote_answer(ev["candidates"], is_mcq)
                gold = ev["gold"]
                gold_list = gold if isinstance(gold, list) else [gold]
                
                if is_mcq:
                    clean_pred = extract_letter(maj_resp)
                else:
                    boxed_vals = extract_boxed_group(answer_visible_text(maj_resp))
                    clean_pred = ", ".join(boxed_vals) if boxed_vals else maj_resp
                
                maj_correct = judger.auto_judge(clean_pred, gold_list, [[]]*len(gold_list))
                if maj_correct:
                    total_recovered += 1
                    status = "RECOVERED \u2705"
                else:
                    status = "Still Missed \u274c"
                
                type_str = "MCQ" if is_mcq else "FRQ"
                print(f"{ev['id']:<6} | {type_str:<5} | {str(maj_correct):<12} | {status}")
                if not maj_correct:
                    print(f"    Gold: {gold}")
                    print(f"    Maj Pred: {clean_pred}")

    print("-" * 60)
    print(f"Total Missed with Old Selector: {total_missed_before}")
    print(f"Total Recovered by Majority Vote: {total_recovered}")
    if total_missed_before > 0:
        print(f"Recovery Rate: {total_recovered/total_missed_before:.2%}")

if __name__ == "__main__":
    main()
