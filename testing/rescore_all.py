"""Re-score every results/*/predictions.jsonl with the (now updated) judger.

For each folder, writes accuracy_rescored.txt next to the original accuracy.txt
so we can diff the deltas without destroying history.

For multishot folders, also re-scores each individual candidate (so oracle
counts get the updated judger applied) and re-derives oracle correctness.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")
from judger import Judger
judger = Judger(strict_extract=False)


def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def judge_one(response: str, gold, is_mcq: bool) -> bool:
    if is_mcq:
        return extract_letter(response) == str(gold).strip().upper()
    gold_list = gold if isinstance(gold, list) else [gold]
    try:
        return judger.auto_judge(
            pred=response, gold=gold_list, options=[[]] * len(gold_list),
        )
    except Exception:
        return False


def rescore_folder(folder: Path) -> dict:
    preds_path = folder / "predictions.jsonl"
    if not preds_path.exists():
        return {}
    rows = [json.loads(l) for l in open(preds_path)]
    n_old_correct = sum(1 for r in rows if r.get("correct"))
    n_new_correct = 0
    mcq_correct = mcq_total = 0
    free_correct = free_total = 0
    # multishot fields (optional)
    has_oracle = "oracle_correct" in rows[0] if rows else False
    has_candidates = "candidates" in rows[0] if rows else False
    new_oracle_correct = 0
    candidate_acc_by_variant: dict[str, list[int]] = {}

    for r in rows:
        is_mcq = bool(r["is_mcq"])
        gold = r["gold"]
        new_c = judge_one(r["response"], gold, is_mcq)
        if is_mcq:
            mcq_total += 1
            mcq_correct += int(new_c)
        else:
            free_total += 1
            free_correct += int(new_c)
        n_new_correct += int(new_c)
        if has_candidates:
            variants = r.get("candidate_variants") or [f"cand_{i}" for i in range(len(r["candidates"]))]
            new_cand_correct = []
            for cand, vname in zip(r["candidates"], variants):
                cc = judge_one(cand, gold, is_mcq)
                new_cand_correct.append(cc)
                candidate_acc_by_variant.setdefault(vname, []).append(int(cc))
            new_oracle_correct += int(any(new_cand_correct))

    out = {
        "n_total": len(rows),
        "n_old_correct": n_old_correct,
        "n_new_correct": n_new_correct,
        "mcq_correct": mcq_correct,
        "mcq_total": mcq_total,
        "free_correct": free_correct,
        "free_total": free_total,
        "has_oracle": has_oracle,
        "has_candidates": has_candidates,
        "new_oracle_correct": new_oracle_correct,
        "candidate_acc_by_variant": candidate_acc_by_variant,
    }

    # Write accuracy_rescored.txt
    lines = []
    lines.append(f"Rescored with updated judger.py")
    lines.append(f"Folder: {folder.name}")
    lines.append(f"Total items: {len(rows)}")
    lines.append("")

    def fmt(c, t):
        return f"{c:4d} / {t:4d}  ({c/t*100:.2f}%)" if t else "n/a"

    lines.append(f"  MCQ        : {fmt(mcq_correct, mcq_total)}")
    lines.append(f"  Free-form  : {fmt(free_correct, free_total)}")
    lines.append(f"  Overall    : {fmt(n_new_correct, len(rows))}")
    lines.append("")
    delta = n_new_correct - n_old_correct
    sign = "+" if delta >= 0 else ""
    lines.append(f"  Old (saved) overall : {n_old_correct} / {len(rows)} ({n_old_correct/len(rows)*100:.2f}%)")
    lines.append(f"  Delta vs old        : {sign}{delta} items ({sign}{delta/len(rows)*100:.2f} pts)")
    if has_oracle:
        old_oracle = sum(1 for r in rows if r.get("oracle_correct"))
        lines.append("")
        lines.append(f"  Oracle@k (old)      : {old_oracle} / {len(rows)} ({old_oracle/len(rows)*100:.2f}%)")
        lines.append(f"  Oracle@k (rescored) : {new_oracle_correct} / {len(rows)} ({new_oracle_correct/len(rows)*100:.2f}%)")
        lines.append(f"  Selector miss (rescored) : {new_oracle_correct - n_new_correct}")
    if candidate_acc_by_variant:
        lines.append("")
        lines.append("  Per-variant candidate accuracy (rescored):")
        for v, hits in sorted(candidate_acc_by_variant.items()):
            lines.append(f"    {v:30s} {sum(hits):4d} / {len(hits):4d}  ({sum(hits)/len(hits)*100:.2f}%)")

    (folder / "accuracy_rescored.txt").write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    summary_rows = []
    for folder in sorted(Path("results").iterdir()):
        if not folder.is_dir():
            continue
        if not (folder / "predictions.jsonl").exists():
            continue
        out = rescore_folder(folder)
        if not out:
            continue
        summary_rows.append((folder.name, out))
        print(
            f"{folder.name:50s}  "
            f"old={out['n_old_correct']:3d}/{out['n_total']:3d}  "
            f"new={out['n_new_correct']:3d}/{out['n_total']:3d}  "
            f"Δ={out['n_new_correct']-out['n_old_correct']:+d}"
        )

    # Top-level summary file
    with open("results/SUMMARY_rescored.txt", "w") as f:
        f.write("Rescored leaderboard (using updated judger.py)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Run':50s}  {'MCQ':>15}  {'FRQ':>15}  {'Overall':>15}  {'Δ':>5}\n")
        f.write("-" * 110 + "\n")
        for name, o in summary_rows:
            mcq = f"{o['mcq_correct']}/{o['mcq_total']} ({o['mcq_correct']/max(o['mcq_total'],1)*100:.2f}%)" if o['mcq_total'] else "n/a"
            frq = f"{o['free_correct']}/{o['free_total']} ({o['free_correct']/max(o['free_total'],1)*100:.2f}%)" if o['free_total'] else "n/a"
            ov  = f"{o['n_new_correct']}/{o['n_total']} ({o['n_new_correct']/o['n_total']*100:.2f}%)"
            d   = f"{o['n_new_correct']-o['n_old_correct']:+d}"
            f.write(f"{name:50s}  {mcq:>15}  {frq:>15}  {ov:>15}  {d:>5}\n")
    print(f"\nWrote results/SUMMARY_rescored.txt")


if __name__ == "__main__":
    main()
