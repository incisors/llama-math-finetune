# Llama-3.1-8B Math Fine-tuning: QLoRA vs LoRA vs Full FT

Empirical comparison of three fine-tuning methods on **Llama-3.1-8B** for math reasoning, under identical hardware, data, and token budget. The interesting axis is the trade-off between **math gain** (GSM8K, MATH) and **catastrophic forgetting** of general knowledge (MMLU, HellaSwag).

**Status**: baseline + QLoRA + LoRA bf16 done; Full FT upcoming.

## Results

Baseline is raw Llama-3.1-8B without fine-tuning. Δ columns are vs baseline. Full FT row fills in once Phase 4 completes.

### Quality benchmarks (accuracy %)

| Method      | GSM8K | MATH | MMLU | HellaSwag | Δ GSM8K | Δ MMLU |
|-------------|------:|-----:|-----:|----------:|--------:|-------:|
| Baseline    |  48.9 | 13.4 | 66.8 |      73.2 |       — |      — |
| QLoRA r=16  |  68.2 | 14.3 | 65.9 |      69.6 |   +19.3 |   -0.9 |
| LoRA r=16   |  68.7 | 14.1 | 66.0 |      69.8 |   +19.8 |   -0.8 |
| Full FT     |     — |    — |    — |         — |       — |      — |

Numbers are pulled directly from [`eval_results/`](eval_results/) (lm-eval-harness JSON output). Metrics used:

- **GSM8K**: `exact_match,flexible-extract`, full 1319-item set, 8-shot
- **MATH** (`hendrycks_math`): `exact_match`, 500-item subset, 4-shot
- **MMLU**: `acc`, 500-item subset, 5-shot
- **HellaSwag**: `acc_norm`, 500-item subset, 10-shot

### System efficiency

Measured via W&B run summary + system-metric time series. Extraction script: [`src/extract_wandb_metrics.py`](src/extract_wandb_metrics.py). Raw output: [`results/system_metrics.json`](results/system_metrics.json).

| Method      | Peak GPU mem | Train time | Throughput     | Avg GPU util | Avg power | Final loss | total FLOPs | Cost  |
|-------------|-------------:|-----------:|---------------:|-------------:|----------:|-----------:|------------:|------:|
| QLoRA r=16  |     35.5 GB  |    3.09 h  | 4.50 samples/s |       97.4 % |   380 W   |     0.376  |    7.77e17  | $4.63 |
| LoRA r=16   |     64.8 GB  |    1.43 h  | 9.72 samples/s |       94.3 % |   380 W   |     0.374  |    7.77e17  | $2.14 |
| Full FT     |          —   |        —   |             —  |          —   |       —   |         —  |         —   |    —  |

All methods share `effective_batch_size = 16` on the identical 50K seed=42 subset → identical `total_flos` confirms apples-to-apples comparison at the compute level.

**Key system trade-offs (QLoRA vs LoRA bf16)**:

- **Quality**: indistinguishable. Final train loss differs by 0.7%; all 4 downstream eval scores within 1 σ stderr (e.g. GSM8K diff 0.46 pts vs stderr 1.28 pts).
- **Memory**: QLoRA uses **1.83× less peak GPU memory** (35.5 GB vs 64.8 GB). Note both are far above the static weight footprint — dequantized activations + LoRA-bf16 activations dominate.
- **Time**: QLoRA is **2.16× slower** wall-clock (3.09 h vs 1.43 h) for the same total FLOPs. Cost: also 2.16×. The slowdown is purely the per-forward NF4 dequant tax — both methods saturate the GPU (>94 % util, 380 W).
- **Energy**: identical avg power × 2.16× longer runtime → QLoRA consumes ~2.16× more energy per run despite fitting on smaller hardware.

**When to pick which** (single-GPU regime): if you have ≥80 GB VRAM, LoRA bf16 is strictly better — faster, cheaper, identical quality. QLoRA wins only when memory is the hard constraint (e.g. on a 24 GB consumer GPU, where bf16 base + gradients + activations would OOM but the 4-bit weights fit).

Caveat: LoRA bf16 here uses `gradient_checkpointing=false` (PEFT-frozen base + checkpointing breaks the grad chain unless `enable_input_require_grads()` is called); QLoRA uses `gradient_checkpointing=true` (handled automatically by `prepare_model_for_kbit_training`). A fully apples-to-apples memory comparison would require fixing the LoRA bf16 code path to also support checkpointing.

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
