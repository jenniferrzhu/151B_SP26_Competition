# Multishot Evaluation Changes

This document summarizes the multishot prompt-engineering changes made in
`run_eval.py`.

## Goal

The model is nondeterministic when sampling, so a single run may produce a
wrong answer even when another sampled run would solve the problem correctly.
The new evaluation flow generates 5 candidate answers per question, then asks
the LLM to choose the answer it thinks is most likely correct.

## Previous Flow

Before this change, each question followed this process:

1. Build one prompt for the question.
2. Generate one model response.
3. Score that response against the ground truth.

## New Flow

Each question now follows this process:

1. Build the normal math or multiple-choice prompt.
2. Generate `NUM_CANDIDATES = 5` sampled responses for the same question.
3. Build a second selector prompt containing:
   - the original problem,
   - the answer choices, if it is multiple choice,
   - each candidate response,
   - each candidate's extracted `\boxed{}` answer when available.
4. Ask the model to act as an expert math judge and pick the most likely
   correct final answer.
5. Use the selector's `\boxed{}` answer as the final response that gets scored.

## Important Constants

The new constants are near the top of `run_eval.py`:

```python
NUM_CANDIDATES = 5
SELECT_CHUNK_SIZE = 16
SELECT_MAX_TOKENS = 4096
CANDIDATE_SNIPPET_CHARS = 1800
MAX_NUM_SEQS = 64
```

What they control:

- `NUM_CANDIDATES`: how many independent sampled answers are generated per
  question.
- `SELECT_CHUNK_SIZE`: how many selector prompts are processed per batch.
- `SELECT_MAX_TOKENS`: maximum output length for the selector model.
- `CANDIDATE_SNIPPET_CHARS`: how much of each candidate response is shown to
  the selector.
- `MAX_NUM_SEQS`: vLLM concurrency limit used to keep the 5-candidate batches
  within engine capacity.

## Prompt Additions

A new system prompt, `SYSTEM_PROMPT_SELECT`, was added. It tells the model to:

- act as an expert math judge,
- check the math itself,
- use consensus only as supporting evidence,
- return only the selected final answer inside `\boxed{}`,
- return only the option letter for multiple-choice questions.

## Output Changes

`results/{test_name}/predictions.jsonl` now includes extra debugging fields:

```json
{
  "id": 60,
  "is_mcq": false,
  "gold": ["38"],
  "candidates": ["...", "...", "...", "...", "..."],
  "selector_response": "\\boxed{38}",
  "response": "\\boxed{38}",
  "correct": true
}
```

Field meanings:

- `candidates`: the 5 raw sampled responses.
- `selector_response`: the model's judge-pass output.
- `response`: the final answer used by the scorer.
- `correct`: whether the final selected answer matched the ground truth.

## Scoring Behavior

The scoring logic still uses the existing `Judger` for free-form math and
letter extraction for multiple-choice questions. The only difference is that
the scored `response` is now the selected answer from the judge pass instead of
the first sampled model response.

## Safety Details

Because `NUM_CANDIDATES = 5` multiplies the number of generated sequences, the
script now computes a safe candidate chunk size:

```python
candidate_chunk_size = min(CHUNK_SIZE, max(1, MAX_NUM_SEQS // NUM_CANDIDATES))
```

This keeps a chunk from requesting more simultaneous sequences than vLLM is
configured to handle.

## Validation

The script was syntax-checked with:

```bash
python -m py_compile run_eval.py
```

The full model evaluation was not run during this change because it requires
loading the GPU model.

## Additional Changes: Diverse Candidates and Oracle Diagnostics

After the first multishot experiment did not improve accuracy, `run_eval.py`
was expanded again to make the five candidates more meaningfully different and
to diagnose whether the selector or the candidate generation is the bottleneck.

### Candidate Variants

Instead of generating five sampled responses from the exact same prompt, the
script now defines named prompt variants:

```python
CANDIDATE_VARIANTS = [
    ("baseline_deterministic", ""),
    ("answer_order_audit", "..."),
    ("formula_first_exact", "..."),
    ("independent_then_options", "..."),
    ("sanity_check", "..."),
]
NUM_CANDIDATES = len(CANDIDATE_VARIANTS)
```

The first candidate is a deterministic baseline generated with
`temperature=0.0`. The other four candidates are sampled attempts with
different instructions:

- `answer_order_audit`: identify every requested answer and preserve the order
  of the real `[ANS]` blanks.
- `formula_first_exact`: write the relevant formula first, keep exact values
  until the final step, and round only when requested.
- `independent_then_options`: solve before looking at choices, then compare
  against every option for multiple-choice problems.
- `sanity_check`: check units, signs, ranges, rounding, and reasonableness
  before boxing the final answer.

### Candidate Generation Flow

The old code used one vLLM call with `n=NUM_CANDIDATES`. The new code generates
single-output prompts through a helper:

```python
generate_single_outputs(...)
```

This allows each candidate slot to use a different prompt. The baseline pass
uses deterministic sampling parameters, while the four variant prompts use the
existing sampled settings.

### Selector Prompt Expansion

The selector prompt now asks the model to:

- re-solve the problem independently,
- compare its answer against the candidate answers,
- check answer count, order, units, signs, rounding, and requirements,
- use consensus only as supporting evidence,
- return a corrected final answer if all candidates are flawed but the problem
  can be solved.

The selector prompt also includes an answer-format hint. For multiple-choice
problems, it reminds the model to return one option letter. For free-form
problems, it reports the number of `[ANS]` placeholders and warns that some
placeholders may be formatting noise or copied choice labels.

### Oracle Diagnostics

When ground-truth answers are available, the script now scores each individual
candidate in addition to the final selected response. Each result row includes:

```json
{
  "candidate_variants": ["baseline_deterministic", "answer_order_audit"],
  "candidate_correct": [false, true],
  "oracle_correct": true
}
```

Field meanings:

- `candidate_variants`: the prompt strategy used for each candidate.
- `candidate_correct`: whether each candidate was locally correct.
- `oracle_correct`: whether at least one candidate was correct.

The accuracy summary now includes:

- final MCQ, free-form, and overall accuracy,
- `Oracle@5`, which shows how often at least one candidate was correct,
- the number of cases where the selector missed an available correct candidate,
- candidate accuracy broken down by prompt variant.

### Diagnostic Interpretation

Use the new summary this way:

- High `Oracle@5` but low final accuracy means the selector is failing to pick
  the right candidate or is over-correcting.
- Low `Oracle@5` means the prompts are not producing enough correct candidate
  answers, so the next step should be better variants, retrieval-style few-shot
  prompting, or supervised fine-tuning.
- If one variant clearly beats the others, promote that prompt style into the
  main prompt and build new variants around it.

### Current Validation

After these changes, the script was syntax-checked again with:

```bash
python -m py_compile run_eval.py
```

The full model evaluation was not run during this change because it requires
loading the GPU model.
