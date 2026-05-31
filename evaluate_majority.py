import json
import collections
from utils import strip_answer_string

def get_majority_answer(candidates):
    answers = []
    for c in candidates:
        ans = strip_answer_string(c)
        if ans:
            answers.append(ans)
    
    if not answers:
        return ""
        
    counts = collections.Counter(answers)
    # Return the most common answer
    return counts.most_common(1)[0][0]

with open('results/GEPA Optimized/predictions.jsonl') as f:
    evals = [json.loads(line) for line in f]

# To properly grade, we can check if get_majority_answer matches gold using judger
# But an easier approximation for now is checking if our majority answer matches one of the correct candidate's extracted answers.
majority_correct = 0
for ev in evals:
    maj = get_majority_answer(ev['candidates'])
    
    # Let's say if it exactly matches the text of ANY candidate that is marked correct, then the majority choice would be graded correct.
    # Note: Because strip_answer_string may normalize it, we should compare normalized string
    
    cand_strings = [strip_answer_string(c) for c in ev['candidates']]
    
    # Is the majority string in the set of "correct" strings for this instance?
    correct_strings = [s for s, c in zip(cand_strings, ev['candidate_correct']) if c]
    
    if maj in correct_strings:
        majority_correct += 1

print(f"Majority Voting Correct: {majority_correct} / {len(evals)} ({(majority_correct/len(evals))*100:.2f}%)")
print(f"Current Selector Correct: {sum(1 for ev in evals if ev['correct'])} / {len(evals)} ({sum(1 for ev in evals if ev['correct'])/len(evals)*100:.2f}%)")

