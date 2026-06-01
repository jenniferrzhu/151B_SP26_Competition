# Math Reasoning Competition Submission - Team G069 CSE 151B

This repository contains the inference pipeline and fine-tuned models for the Math Reasoning Competition.

## Strategy Overview
Our final strategy utilizes a hybrid approach:
- **Multiple-Choice Questions (MCQ)**: Powered by a fine-tuned **LoRA v6** adapter on top of Qwen3-4B-Thinking. We use **Majority Voting (n=8)** at Temperature 0.6 to stabilize reasoning.
- **Free-Response Questions (FRQ)**: Powered by the **Base Model** (Qwen3-4B-Thinking-2507) with a **5-variant prompt ensemble** — 1 deterministic baseline (temp 0.0) + 4 sampled variants at temp 0.6 (`answer_order_audit`, `formula_first_exact`, `independent_then_options`, `sanity_check`) under a long few-shot system prompt. An **LLM-judge selector** (temp 0, 4096 tokens) then picks the best candidate: it names "Candidate #N" → return that candidate verbatim, writes its own `\boxed{}` → return the selector's response, otherwise → equivalence-class majority vote over the 5 boxed answers. The submitted `response` always carries the full chain-of-thought trace.

## Hardware & Performance
- **GPU Used**: RTX 4090 
- **Total Inference Time**: 
  - ~2-3 hours for MCQ (300 items) when sharded across 2-4 GPUs.
  - ~2-3 hours for FRQ (643 items) when sharded across 2-4 GPUs.
  - ~12-14 hours total on a single GPU for the full private dataset.

## Model Weights

- **Base Model**: `Qwen/Qwen3-4B-Thinking-2507` (Automatically downloaded from HuggingFace).
- **LoRA Adapter**: Our fine-tuned adapter is hosted on Hugging Face Hub: [lucashlaing/qwen3-lora-v6](https://huggingface.co/lucashlaing/qwen3-lora-v6).
  - The `run_inference.py` script and sharding scripts are pre-configured to load this adapter automatically. No manual download is required.


## How to Reproduce Results

Run `run_inference()` in `run_inference.py`.

### End-to-End Inference
To perform the full pipeline on the private dataset:
```bash
python run_inference.py --input data/private.jsonl --output submission.csv
```