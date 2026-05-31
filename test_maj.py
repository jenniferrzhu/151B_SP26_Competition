import json
from judger import Judger
from collections import Counter

judger = Judger()

with open('results/GEPA Optimized/predictions.jsonl') as f:
    evals = [json.loads(line) for line in f]

maj_correct = 0
for ev in evals:
    # 1. extract answers for each candidate
    extracted = []
    for cand in ev['candidates']:
        ext = judger.extract_ans(cand)
        # ext can be empty string or list if split_by_comma? Actually extract_ans returns a string.
        # Oh, parse it exactly like judger does for matching.
        # But wait, we just want to group them by exact string equivalence initially 
        # (or using judger is_equiv).
        ext_str = str(ext).strip()
        if ext_str:
            extracted.append(ext_str)
            
    if not extracted:
        # no candidates produced An answer, fail
        continue
        
    counts = Counter(extracted)
    most_common_ans = counts.most_common(1)[0][0]
    
    # 2. To check if most_common_ans is mathematically equivalent to gold:
    # We can just see if the 'is_correct' flag for a candidate that returned this exact string was True!
    # This avoids calling the full judger.get_match... logic that needs problem types etc.
    
    # Let's map cand extracted string to its correct bool
    cand_is_correct = dict()
    for cand, corr in zip(ev['candidates'], ev['candidate_correct']):
        e = str(judger.extract_ans(cand)).strip()
        if e:
            # If any candidate that output this string is correct, mark string as correct
            cand_is_correct[e] = cand_is_correct.get(e, False) or corr
            
    if cand_is_correct.get(most_common_ans, False):
        maj_correct += 1

print(f"Majority Voting Correct: {maj_correct} / {len(evals)} ({(maj_correct/len(evals))*100:.2f}%)")
