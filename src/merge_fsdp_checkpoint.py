"""Merge an FSDP SHARDED_STATE_DICT checkpoint into a single HF-format checkpoint.

After Full FT training with FSDP + SHARDED_STATE_DICT, HF Trainer saves the model
into `checkpoint-NNNN/pytorch_model_fsdp_0/` as PyTorch Distributed Checkpoint
(DCP) shards. This isn't loadable by `AutoModelForCausalLM.from_pretrained()`.

This script:
  1. Converts the DCP shards → a single torch state_dict file
  2. Initializes the base Llama-3.1-8B architecture on CPU
  3. Loads the merged state_dict into the model
  4. Saves model + tokenizer in standard HF format (model.safetensors + config.json)

Usage:
    python -m src.merge_fsdp_checkpoint \\
        --sharded-dir ./checkpoints/full-ft/checkpoint-3125/pytorch_model_fsdp_0 \\
        --tokenizer-dir ./checkpoints/full-ft \\
        --output-dir ./checkpoints/full-ft-merged

Memory: needs ~16 GB CPU RAM (loads full bf16 model). Runs on a single process
(no distributed init required). Takes 3-5 min for an 8B model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="meta-llama/Llama-3.1-8B",
                   help="HF base model ID — used to get architecture/config")
    p.add_argument("--sharded-dir", required=True,
                   help="DCP shard directory (e.g. checkpoint-3125/pytorch_model_fsdp_0)")
    p.add_argument("--tokenizer-dir", required=True,
                   help="Directory containing tokenizer files (usually the checkpoint root)")
    p.add_argument("--output-dir", required=True,
                   help="Destination for merged HF-format model")
    p.add_argument("--tmp-file", default="/workspace/merged_weights.pt",
                   help="Where to write the intermediate single-file checkpoint")
    args = p.parse_args()

    tmp = Path(args.tmp_file)
    tmp.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Converting DCP shards → single .pt file ...")
    print(f"      src: {args.sharded_dir}")
    print(f"      dst: {tmp}")
    dcp_to_torch_save(args.sharded_dir, str(tmp))
    print(f"      done — {tmp.stat().st_size / 1e9:.1f} GB")

    print(f"\n[2/5] Loading merged tensors from {tmp} ...")
    blob = torch.load(tmp, weights_only=True, map_location="cpu")
    # Accelerate's FSDP save typically wraps under "model" key; handle both shapes.
    if isinstance(blob, dict) and "model" in blob and isinstance(blob["model"], dict):
        state_dict = blob["model"]
        print(f"      unwrapped 'model' key — {len(state_dict)} tensors")
    elif isinstance(blob, dict):
        state_dict = blob
        print(f"      flat dict — {len(state_dict)} tensors")
    else:
        raise ValueError(f"Unexpected DCP content type: {type(blob)}")
    sample_keys = list(state_dict.keys())[:3]
    print(f"      sample keys: {sample_keys}")

    print(f"\n[3/5] Initializing {args.base_model} architecture on CPU ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    hf_sample_keys = list(model.state_dict().keys())[:3]
    print(f"      HF sample keys: {hf_sample_keys}")

    print(f"\n[4/5] Loading merged state_dict into model ...")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"      WARNING: {len(missing)} missing keys, first few: {missing[:3]}")
    if unexpected:
        print(f"      WARNING: {len(unexpected)} unexpected keys, first few: {unexpected[:3]}")
    if not missing and not unexpected:
        print(f"      perfect match ✓")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n[5/5] Saving HF-format model + tokenizer to {out} ...")
    model.save_pretrained(out, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    tokenizer.save_pretrained(out)

    print(f"\nCleaning up {tmp} ...")
    tmp.unlink(missing_ok=True)

    out_size_gb = sum(f.stat().st_size for f in out.glob("*")) / 1e9
    print(f"\nDone. Merged checkpoint at {out} ({out_size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
