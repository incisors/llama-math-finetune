"""Evaluate a model on GSM8K + MATH + MMLU + HellaSwag via lm-eval-harness.

Three modes:
    baseline : eval the raw Llama-3.1-8B base
    peft     : eval a LoRA/QLoRA adapter on top of the base
    full_ft  : eval a merged full-FT checkpoint directly

Usage:
    python -m src.eval --mode baseline --output-dir ./eval_results/baseline
    python -m src.eval --mode peft     --model-path ./checkpoints/qlora-r16 \\
                       --output-dir ./eval_results/qlora-r16
    python -m src.eval --mode full_ft  --model-path ./checkpoints/full-ft \\
                       --output-dir ./eval_results/full-ft
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lm_eval
from lm_eval.models.huggingface import HFLM


# Per-task settings — these are the canonical CLAUDE.md table values.
# Keep gsm8k full (1319) so the number is directly comparable to Llama paper.
TASK_SPEC = {
    "gsm8k":          {"limit": None, "num_fewshot": 8},
    "hendrycks_math": {"limit": 500,  "num_fewshot": 4},
    "mmlu":           {"limit": 500,  "num_fewshot": 5},
    "hellaswag":      {"limit": 500,  "num_fewshot": 10},
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["baseline", "peft", "full_ft"], required=True)
    p.add_argument("--base-model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--model-path",
                   help="Adapter dir (peft mode) or merged model dir (full_ft mode)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tasks", nargs="+", default=list(TASK_SPEC.keys()))
    p.add_argument("--batch-size", default="auto",
                   help="Batch size; 'auto' probes the largest fit per task — "
                        "recommended because loglikelihood tasks (MMLU/HellaSwag) "
                        "need much smaller batches than generate_until (GSM8K/MATH).")
    args = p.parse_args()

    # Build the HFLM wrapper once and reuse across all tasks.
    if args.mode == "baseline":
        lm = HFLM(pretrained=args.base_model, dtype="bfloat16",
                  batch_size=args.batch_size)
    elif args.mode == "peft":
        if not args.model_path:
            raise SystemExit("--model-path required for peft mode")
        lm = HFLM(pretrained=args.base_model, peft=args.model_path,
                  dtype="bfloat16", batch_size=args.batch_size)
    elif args.mode == "full_ft":
        if not args.model_path:
            raise SystemExit("--model-path required for full_ft mode")
        lm = HFLM(pretrained=args.model_path, dtype="bfloat16",
                  batch_size=args.batch_size)
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"

    merged: dict[str, dict] = {"results": {}, "configs": {}, "task_spec": {}}

    for task in args.tasks:
        spec = TASK_SPEC.get(task, {"limit": None, "num_fewshot": 0})
        merged["task_spec"][task] = spec
        print(f"\n=== {task}  (limit={spec['limit']}, "
              f"num_fewshot={spec['num_fewshot']}) ===")
        r = lm_eval.simple_evaluate(
            model=lm,
            tasks=[task],
            num_fewshot=spec["num_fewshot"],
            limit=spec["limit"],
        )
        merged["results"].update(r["results"])
        if "configs" in r:
            merged["configs"].update(r["configs"])

        # Save after each task so a later crash doesn't wipe earlier results.
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, default=str)
        print(f"  Partial results saved → {out_path}")

    print(f"\nAll done. Aggregated results at {out_path}")

    # Print quick summary
    print("\n=== Summary ===")
    for task, res in merged["results"].items():
        primary = next(iter(res.items()))
        print(f"  {task}: {primary[0]} = {primary[1]}")


if __name__ == "__main__":
    main()
