#!/usr/bin/env bash
# QLoRA training + eval. Hardware: 1× A100 80GB SXM (RunPod).
# Assumes venv is activated.

set -euo pipefail

# 1. Prepare dataset (idempotent — skips if already cached)
if [ ! -d ./data/metamathqa_50k_seed42 ]; then
    python -m src.prepare_data
fi

# 2. Train
python -m src.train --config configs/qlora.yaml

# 3. Eval the resulting adapter
python -m src.eval \
    --mode peft \
    --model-path ./checkpoints/qlora-r16 \
    --output-dir ./eval_results/qlora-r16
