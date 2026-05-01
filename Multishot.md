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
