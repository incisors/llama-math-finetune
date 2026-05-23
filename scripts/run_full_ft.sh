#!/usr/bin/env bash
# Full FT training + eval. Hardware: 2× A100 80GB SXM (RunPod, NVLink required).
# Assumes venv is activated and setup.sh has been run.

set -euo pipefail

# 1. Prepare dataset (idempotent)
if [ ! -d ./data/metamathqa_50k_seed42 ]; then
    python -m src.prepare_data
fi

# 2. Train via accelerate (FSDP shards across 4 GPUs)
# Use --module so relative imports in src/train.py work
# (accelerate launch src/train.py treats it as a plain script, breaking `from .utils import`).
accelerate launch \
    --config_file configs/accelerate_fsdp.yaml \
    --module src.train \
    --config configs/full_ft_fsdp.yaml

# 3. Eval the merged checkpoint
python -m src.eval \
    --mode full_ft \
    --model-path ./checkpoints/full-ft \
    --output-dir ./eval_results/full-ft
