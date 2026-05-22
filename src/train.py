"""Train Llama-3.1-8B on MetaMathQA via QLoRA, LoRA bf16, or Full FT.

YAML-driven; the `method` field in the config selects the branch.

Usage:
    # Single-GPU (QLoRA, LoRA bf16)
    python -m src.train --config configs/qlora.yaml
    python -m src.train --config configs/lora_bf16.yaml

    # Multi-GPU Full FT via FSDP
    accelerate launch --config_file configs/accelerate_fsdp.yaml \\
        src/train.py --config configs/full_ft_fsdp.yaml

    # Smoke test (small subset + few steps)
    python -m src.prepare_data --subset-size 500 \\
        --output-dir ./data/metamathqa_500_smoke
    python -m src.train --config configs/qlora.yaml \\
        --data-dir ./data/metamathqa_500_smoke --max-steps 50
"""
from __future__ import annotations

import argparse
import os

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .utils import init_wandb, load_config, set_seed, validate_effective_batch


def build_model_and_tokenizer(cfg: dict):
    """Build model + tokenizer based on cfg['method']."""
    method = cfg["method"]
    model_name = cfg["model"]["name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Llama base has no pad token — required for SFTTrainer batching.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if method == "qlora":
        q = cfg["quantization"]
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=q["load_in_4bit"],
            bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
            bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    elif method == "lora":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    elif method == "full_ft":
        # Under FSDP, accelerate launch shards via device dispatch — do NOT
        # pass device_map (it conflicts with FSDP wrapping).
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"Unknown method: {method!r}")

    return model, tokenizer


def wrap_with_peft(model, cfg: dict):
    """Wrap a model with a LoRA adapter (and kbit prep for QLoRA)."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if cfg["method"] == "qlora":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        )

    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--data-dir", default="./data/metamathqa_50k_seed42",
                        help="Dataset directory (output of prepare_data.py)")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Override step count (for smoke test). -1 = use epochs.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    num_gpus = max(torch.cuda.device_count(), 1)
    validate_effective_batch(cfg, num_gpus=num_gpus)

    # Under FSDP, only rank 0 talks to W&B
    is_main = int(os.environ.get("RANK", "0")) == 0
    if is_main:
        init_wandb(cfg, extra_config={"num_gpus": num_gpus})

    print(f"Loading dataset from {args.data_dir} ...")
    dataset = load_from_disk(args.data_dir)
    print(f"  {len(dataset):,} examples")

    print(f"Building model + tokenizer (method={cfg['method']}) ...")
    model, tokenizer = build_model_and_tokenizer(cfg)
    if cfg["method"] in ("qlora", "lora"):
        model = wrap_with_peft(model, cfg)

    from trl import SFTConfig, SFTTrainer

    t = cfg["training"]
    out = cfg["output"]
    sft_config = SFTConfig(
        output_dir=out["checkpoint_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        gradient_checkpointing=t["gradient_checkpointing"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        optim=t["optim"],
        bf16=t["bf16"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        seed=t["seed"],
        report_to="wandb" if is_main else "none",
        run_name=out["wandb_run_name"],
        max_steps=args.max_steps,
        dataset_text_field="text",
        max_seq_length=cfg["data"]["max_seq_length"],
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print("Starting training ...")
    trainer.train()

    print(f"Saving final {'adapter' if cfg['method'] != 'full_ft' else 'model'} "
          f"to {out['checkpoint_dir']} ...")
    trainer.save_model(out["checkpoint_dir"])
    tokenizer.save_pretrained(out["checkpoint_dir"])

    if is_main:
        import wandb
        wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
