#!/usr/bin/env bash
# Eval the raw Llama-3.1-8B base on GSM8K / MATH / MMLU / HellaSwag.
# Run on 1× A100 80GB (RunPod). Assumes venv is activated.

set -euo pipefail

python -m src.eval \
    --mode baseline \
    --output-dir ./eval_results/baseline

echo "Baseline eval done. Verify: GSM8K ~52%, MMLU ~67% (per Llama-3.1 paper)."
echo "If far off, DEBUG eval pipeline before any training."
