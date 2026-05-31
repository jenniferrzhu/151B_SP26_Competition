import json
import re
from typing import List, Optional, Tuple
from judger import Judger

PRED_PATH = "results/GEPA Optimized/predictions.jsonl"
SHOW_LOOSE_EXAMPLES = True


MCQ_LETTER_RE = re.compile(r"\b([A-J])\b", re.IGNORECASE)
NUM_RE = re.compile(r"^-?\d*\.?\d+(?:[eE][-+]?\d+)?$")
NUM_TOL = 1e-4


def extract_boxed(text: str) -> Optional[str]:
    entries = []
    start = 0
    while True:
        idx = text.find("\\boxed{", start)
        if idx < 0:
            break
        brace_start = idx + len("\\boxed{")
        depth = 1
        i = brace_start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            content = text[brace_start:i - 1]
            if content:
                entries.append((idx, i, content.strip()))
        start = i

    if not entries:
        return None

    last_group = [entries[-1]]
    for j in range(len(entries) - 2, -1, -1):
        gap = text[entries[j][1]:entries[j + 1][0]]
        if re.match(r"^[\s,\$\.\;\:\-\&\\]*$", gap):
            last_group.insert(0, entries[j])
        else:
            break

    if len(last_group) > 1:
        return ", ".join(item[2] for item in last_group)
    return last_group[0][2]


def extract_boxed_group(text: str) -> List[str]:
    entries = []
    start = 0
    while True:
        idx = text.find("\\boxed{", start)
        if idx < 0:
            break
        brace_start = idx + len("\\boxed{")
        depth = 1
        i = brace_start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            content = text[brace_start:i - 1]
            if content:
                entries.append((idx, i, content.strip()))
        start = i

    if not entries:
        return []

    last_group = [entries[-1]]
    for j in range(len(entries) - 2, -1, -1):
        gap = text[entries[j][1]:entries[j + 1][0]]
        if re.match(r"^[\s,\$\.\;\:\-\&\\]*$", gap):
            last_group.insert(0, entries[j])
        else:
            break

    return [item[2] for item in last_group]


def last_non_empty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ""


def basic_cleanup(text: str) -> str:
    cleaned = text
    cleaned = cleaned.replace("\\,", "")
    cleaned = cleaned.replace("\\left", "").replace("\\right", "")
    cleaned = cleaned.replace("$", "").replace("\\", "")
    cleaned = cleaned.strip()
    return cleaned


def normalize_tokens(text: str) -> str:
    text = basic_cleanup(text)
    # Keep common math tokens, letters, digits, punctuation.
    allowed = []
    for ch in text:
        if ch.isalnum() or ch in ".,/()[]{}+-*^=":
            allowed.append(ch)
        elif ch.isspace():
            allowed.append(" ")
    normalized = "".join(allowed)
    normalized = " ".join(normalized.split())
    return normalized


def try_parse_number(text: str) -> Optional[float]:
    cleaned = text.replace(",", "").strip()
    if NUM_RE.match(cleaned):
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def extract_mcq_letter(text: str) -> Optional[str]:
    boxed = extract_boxed(text)
    search_text = boxed if boxed is not None else text
    matches = MCQ_LETTER_RE.findall(search_text)
    if not matches:
        return None
    return matches[-1].upper()


def extract_candidate_answer(text: str, is_mcq: bool) -> str:
    if is_mcq:
        letter = extract_mcq_letter(text)
        return letter or ""

    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed

    return last_non_empty_line(text)


def answer_visible_text(text: str) -> str:
    think_end = text.rfind("</think>")
    return text[think_end + len("</think>"):] if think_end >= 0 else text


def normalize_yes_no(text: str) -> str:
    t = text.strip().lower()
    if t in {"yes", "y"}:
        return "yes"
    if t in {"no", "n"}:
        return "no"
    return text


def split_multi_answer(text: str) -> List[str]:
    # Split on commas that are likely separators.
    parts = [p.strip() for p in text.split(",")]
    return [p for p in parts if p]


def normalize_answer(text: str, gold_len: int, is_mcq: bool) -> List[str]:
    if is_mcq:
        letter = text.strip().upper()
        return [letter] if letter else []

    cleaned = normalize_tokens(text)
    cleaned = normalize_yes_no(cleaned)

    if gold_len > 1:
        parts = split_multi_answer(cleaned)
        return [normalize_yes_no(p) for p in parts]

    return [cleaned] if cleaned else []


def normalize_gold(gold, is_mcq: bool) -> List[str]:
    if is_mcq:
        return [str(gold).strip().upper()]
    if isinstance(gold, list):
        return [normalize_yes_no(normalize_tokens(g)) for g in gold]
    return [normalize_yes_no(normalize_tokens(str(gold)))]


def candidate_matches_gold(candidate_text: str, gold, is_mcq: bool) -> bool:
    gold_list = normalize_gold(gold, is_mcq)
    cand = extract_candidate_answer(candidate_text, is_mcq)
    cand_list = normalize_answer(cand, len(gold_list), is_mcq)
    if not cand_list:
        return False
    if len(gold_list) == 1 and len(cand_list) == 1:
        return compare_items(gold_list[0], cand_list[0])
    if len(gold_list) == len(cand_list):
        return all(compare_items(g, c) for g, c in zip(gold_list, cand_list))
    return False


def compare_items(gold_item: str, cand_item: str) -> bool:
    g = gold_item.strip()
    c = cand_item.strip()

    g_num = try_parse_number(g)
    c_num = try_parse_number(c)
    if g_num is not None and c_num is not None:
        if g_num == 0:
            return abs(c_num) <= NUM_TOL
        return abs((c_num - g_num) / g_num) <= NUM_TOL

    return g.lower() == c.lower()



def candidate_format_score(candidate_text: str, is_mcq: bool) -> int:
    if is_mcq:
        return 2 if extract_mcq_letter(candidate_text) else 0
    if extract_boxed(candidate_text):
        return 2
    if last_non_empty_line(candidate_text):
        return 1
    return 0


def select_best_candidate(candidates: List[str], is_mcq: bool) -> Tuple[int, str]:
    scored = []
    for idx, cand in enumerate(candidates):
        score = candidate_format_score(cand, is_mcq)
        score -= len(normalize_tokens(extract_candidate_answer(cand, is_mcq))) // 200
        scored.append((score, idx, cand))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][1], scored[0][2]


def selected_or_fallback(selector_response: str, candidates: List[str]) -> Tuple[str, Optional[int]]:
    if "\\boxed{" in selector_response:
        return selector_response.strip(), None

    m = re.search(r"\b(?:candidate|option|choice)\s*#?\s*([1-5])\b", selector_response, re.IGNORECASE)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx], idx

    return selector_response.strip(), None


def format_candidate_answer(candidate: str, is_mcq: bool) -> str:
    if is_mcq:
        letter = extract_mcq_letter(candidate)
        if letter:
            return f"\\boxed{{{letter}}}"
        return candidate

    boxed_group = extract_boxed_group(answer_visible_text(candidate))
    if boxed_group:
        return f"\\boxed{{{', '.join(boxed_group)}}}"
    return candidate


def extract_selector_answer(selector_response: str, is_mcq: bool) -> Optional[str]:
    if "\\boxed{" in selector_response:
        return selector_response.strip()

    if is_mcq:
        letter = extract_mcq_letter(selector_response)
        if letter:
            return f"\\boxed{{{letter}}}"
        return None

    boxed_group = extract_boxed_group(answer_visible_text(selector_response))
    if boxed_group:
        return f"\\boxed{{{', '.join(boxed_group)}}}"

    tail = last_non_empty_line(answer_visible_text(selector_response))
    if tail:
        return f"\\boxed{{{tail}}}"
    return None


def choose_best_formatted_candidate(candidates: List[str], is_mcq: bool) -> str:
    scored = []
    for idx, cand in enumerate(candidates):
        score = candidate_format_score(cand, is_mcq)
        score -= len(normalize_tokens(extract_candidate_answer(cand, is_mcq))) // 200
        scored.append((score, idx, cand))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][2] if scored else ""


def canonicalize_answer(answer: str, is_mcq: bool) -> str:
    if is_mcq:
        letter = extract_mcq_letter(answer)
        return letter or ""

    boxed_group = extract_boxed_group(answer_visible_text(answer))
    if boxed_group:
        return ", ".join(normalize_tokens(part) for part in boxed_group)

    extracted = extract_candidate_answer(answer, False)
    return normalize_tokens(extracted)


def majority_vote_answer(candidates: List[str], is_mcq: bool) -> Optional[str]:
    counts = {}
    order = []
    for cand in candidates:
        key = canonicalize_answer(cand, is_mcq)
        if not key:
            continue
        if key not in counts:
            counts[key] = 0
            order.append(key)
        counts[key] += 1

    if not counts:
        return None

    best_key = max(order, key=lambda k: (counts[k], -order.index(k)))
    if is_mcq:
        return f"\\boxed{{{best_key}}}" if best_key else None
    return f"\\boxed{{{best_key}}}"


def normalize_gold_list(gold) -> List[str]:
    if isinstance(gold, list):
        return gold
    return [gold]


def normalize_options_list(options, gold_len: int) -> List:
    if options is None:
        return [[] for _ in range(gold_len)]
    if isinstance(options, list) and gold_len == 1:
        return options
    if isinstance(options, list) and len(options) == gold_len:
        return options
    return [[] for _ in range(gold_len)]


def judge_candidate(judger: Judger, candidate_text: str, ev: dict) -> bool:
    gold_list = normalize_gold_list(ev.get("gold"))
    options = normalize_options_list(ev.get("options"), len(gold_list))
    type_sequence = ev.get("type_sequence")
    precision = ev.get("precision", 1e-8)
    return judger.auto_judge(candidate_text, gold_list, options, type_sequence, precision)


def main() -> None:
    judger = Judger(strict_extract=False)
    with open(PRED_PATH) as f:
        evals = [json.loads(line) for line in f]

    total = len(evals)
    correct = 0
    oracle_correct = 0
    selected_changed = 0

    for idx, ev in enumerate(evals, start=1):
        candidates = ev.get("candidates", [])
        gold = ev.get("gold")
        is_mcq = bool(ev.get("is_mcq"))

        selector_response = ev.get("selector_response", "")
        selected_text, selected_idx = selected_or_fallback(selector_response, candidates)
        if "\\boxed{" not in selected_text:
            selector_extracted = extract_selector_answer(selector_response, is_mcq)
            if selector_extracted:
                selected_text = selector_extracted
            else:
                voted = majority_vote_answer(candidates, is_mcq)
                if voted:
                    selected_text = voted
                elif selected_idx is not None:
                    selected_text = format_candidate_answer(candidates[selected_idx], is_mcq)
                else:
                    selected_text = format_candidate_answer(
                        choose_best_formatted_candidate(candidates, is_mcq),
                        is_mcq,
                    )

        best_ok = judge_candidate(judger, selected_text, ev)
        correct += 1 if best_ok else 0

        any_ok = any(judge_candidate(judger, c, ev) for c in candidates)
        oracle_correct += 1 if any_ok else 0


        prev_resp = ev.get("selector_response")
        if prev_resp and prev_resp != selected_text:
            selected_changed += 1

        if idx % 25 == 0:
            print(f"Progress: {idx}/{total}", flush=True)

    print(f"Total: {total}")
    print(f"New selector correct: {correct} ({correct / total:.2%})")
    print(f"Oracle@5: {oracle_correct} ({oracle_correct / total:.2%})")
    print(f"Selector changes vs prior: {selected_changed}")



if __name__ == "__main__":
    main()
