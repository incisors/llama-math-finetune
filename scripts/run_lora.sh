#!/usr/bin/env bash
# LoRA bf16 training + eval. Hardware: 1× A100 80GB PCIe (RunPod).
# Assumes venv is activated and setup.sh has been run on the pod.

set -euo pipefail

# 1. Prepare dataset (idempotent)
if [ ! -d ./data/metamathqa_50k_seed42 ]; then
    python -m src.prepare_data
fi

# 2. Train
python -m src.train --config configs/lora_bf16.yaml

# 3. Eval adapter
python -m src.eval \
    --mode peft \
    --model-path ./checkpoints/lora-r16 \
    --output-dir ./eval_results/lora-r16
