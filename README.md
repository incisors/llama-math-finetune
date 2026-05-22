# Llama-3.1-8B Math Fine-tuning: QLoRA vs LoRA vs Full FT

Empirical comparison of three fine-tuning methods on **Llama-3.1-8B** for math reasoning, under identical hardware, data, and token budget. The interesting axis is the trade-off between **math gain** (GSM8K, MATH) and **catastrophic forgetting** of general knowledge (MMLU, HellaSwag).

**Status**: baseline evaluated; QLoRA / LoRA / Full FT runs upcoming.

## Results

All scores are accuracy (%). Baseline is raw Llama-3.1-8B without fine-tuning. Δ columns will fill in as each phase completes.

| Method      | GSM8K | MATH | MMLU | HellaSwag | Δ GSM8K | Δ MMLU | Peak VRAM | Train time | Cost |
|-------------|------:|-----:|-----:|----------:|--------:|-------:|----------:|-----------:|-----:|
| Baseline    |  48.9 | 13.4 | 66.8 |      73.2 |       — |      — |         — |          — |    — |
| QLoRA r=16  |     — |    — |    — |         — |       — |      — |         — |          — |    — |
| LoRA r=16   |     — |    — |    — |         — |       — |      — |         — |          — |    — |
| Full FT     |     — |    — |    — |         — |       — |      — |         — |          — |    — |

Raw lm-eval-harness output for each phase lives in [`eval_results/`](eval_results/). Headline metrics used in the table:

- **GSM8K**: `exact_match,flexible-extract`, full 1319-item set, 8-shot
- **MATH** (`hendrycks_math`): `exact_match`, 500-item subset, 4-shot
- **MMLU**: `acc`, 500-item subset, 5-shot
- **HellaSwag**: `acc_norm`, 500-item subset, 10-shot

## What's being compared

| Method        | Trainable params | Base in memory       | Notes                                                              |
|---------------|------------------|----------------------|--------------------------------------------------------------------|
| **QLoRA**     | ~42M (LoRA only) | 4-bit NF4 (~5 GB)    | Memory-efficient; lets a 7-8B model fit on a 16 GB consumer GPU.   |
| **LoRA bf16** | ~42M (LoRA only) | bf16 frozen (~16 GB) | Standard LoRA, no quantization overhead.                           |
| **Full FT**   | 8.07B            | bf16 trainable       | All parameters trainable; needs FSDP to shard across 2 GPUs.       |

All three share the same training recipe: **50K MetaMathQA subset (seed=42), 1 epoch, effective batch 16, cosine LR with 3% warmup, max sequence 1024**. The only things that vary across methods are the learning rate, per-device batch size, and the quantization / parallelism setup. See [`configs/`](configs/) for exact hyperparameters.

## Setup

Tested on RunPod with A100 80GB pods and a 100 GB Network Volume mounted at `/workspace`.

```bash
git clone https://github.com/incisors/llama-math-finetune.git
cd llama-math-finetune
bash setup.sh
source ~/.bashrc                   # picks up HF_HOME on the network volume
huggingface-cli login              # needs Llama-3.1 access on your HF account
wandb login
```

## Reproduce a phase

```bash
# Phase 1: baseline eval (done — see eval_results/baseline/)
bash scripts/run_baseline.sh

# Phase 2: QLoRA training + eval  (~3h, 1× A100 80GB PCIe)
bash scripts/run_qlora.sh

# Phase 3: LoRA bf16 training + eval  (~4-5h, 1× A100 80GB PCIe)
bash scripts/run_lora.sh

# Phase 4: Full FT training + eval  (~10-12h, 2× A100 80GB SXM, FSDP)
bash scripts/run_full_ft.sh
```

## Repo layout

```
configs/                YAML hyperparameter configs, one per method
scripts/                Bash entrypoints per phase
src/
  prepare_data.py       MetaMathQA → 50K seed=42 → Problem:/Solution: format
  train.py              YAML-driven training (qlora / lora / full_ft branches)
  eval.py               lm-eval-harness wrapper for the 4 benchmarks
  utils.py              seed, config loader, W&B init
eval_results/           Per-phase results.json from lm-eval (committed)
checkpoints/            Saved adapters / models (gitignored, multi-GB)
```

## Stack

PyTorch 2.4 (+CUDA 12.1), Transformers 4.45, PEFT 0.13, TRL 0.11, bitsandbytes 0.44, lm-evaluation-harness 0.4.5, W&B for live monitoring. Full pinned versions in [`requirements.txt`](requirements.txt).
