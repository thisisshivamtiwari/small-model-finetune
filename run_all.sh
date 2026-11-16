#!/usr/bin/env bash
# Full run on Apple silicon: baseline, three LoRA fine-tunes, every LLM arm, then scoring. Each step resumes or skips.
set -u
cd "$(dirname "$0")"
step() { echo "== $(date +%H:%M:%S) $*"; "$@" || echo "!! FAILED: $*"; }
step uv run python -m smft baseline
for k in 8 32 all; do
  name=$([ "$k" = all ] && echo all || echo "k$k")
  [ -f "adapters/$name/adapters.safetensors" ] || step uv run python -m smft train --k "$k"
done
for arm in lora-qwen0.5b-k8 lora-qwen0.5b-k32 lora-qwen0.5b-all zeroshot-qwen0.5b zeroshot-qwen1.5b zeroshot-qwen3b fewshot77-qwen3b; do
  if [ -f "predictions/$arm.jsonl" ] && [ "$(wc -l < "predictions/$arm.jsonl")" -ge 3080 ]; then continue; fi
  step uv run python -m smft infer --arm "$arm"
done
step uv run python -m smft score
echo "== $(date +%H:%M:%S) done"
