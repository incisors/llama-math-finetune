"""Plot W&B training-time curves for the 3 fine-tuning runs.

Pulls per-step `train/loss`, `train/learning_rate`, `train/grad_norm` from each
run and saves a 3-panel comparison figure to `results/training_curves.png`.

Usage (from project root, venv activated):
    python -m src.plot_wandb_curves
    python -m src.plot_wandb_curves --runs qlora-r16 lora-r16 full-ft \\
        --output results/training_curves.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import wandb


ENTITY = "haj003-university-of-california-san-diego"
PROJECT = "llama-math-finetune"

RUN_STYLE = {
    "qlora-r16": {"color": "#1f77b4", "label": "QLoRA r=16"},
    "lora-r16":  {"color": "#ff7f0e", "label": "LoRA r=16"},
    "full-ft":   {"color": "#2ca02c", "label": "Full FT"},
}

# (metric_key, axis label, optional y-scale)
PANELS = [
    ("train/loss",          "Training loss",       "linear"),
    ("train/learning_rate", "Learning rate",       "linear"),
    ("train/grad_norm",     "Gradient norm",       "linear"),
]


def get_history(api, run_name: str):
    runs = list(api.runs(f"{ENTITY}/{PROJECT}", filters={"display_name": run_name}))
    if not runs:
        raise ValueError(f"No run named {run_name!r}")
    finished = sorted(
        (r for r in runs if r.state == "finished"),
        key=lambda r: r.created_at,
        reverse=True,
    )
    run = finished[0] if finished else runs[0]
    print(f"  picked id={run.id} (state={run.state})")

    keys = ["train/global_step", "train/loss", "train/learning_rate", "train/grad_norm"]
    hist = run.history(keys=keys, pandas=True)
    return hist


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+",
                   default=["qlora-r16", "lora-r16", "full-ft"])
    p.add_argument("--output", default="results/training_curves.png")
    args = p.parse_args()

    api = wandb.Api()
    histories: dict[str, "object"] = {}
    for name in args.runs:
        print(f"[{name}] fetching history ...")
        histories[name] = get_history(api, name)
        print(f"  {len(histories[name])} logged points")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.suptitle(
        "Training dynamics across methods (Llama-3.1-8B, 50K MetaMathQA, 1 epoch, eff. batch 16)",
        fontsize=12, y=1.02,
    )

    for ax, (metric, label, yscale) in zip(axes, PANELS):
        for name, hist in histories.items():
            if metric not in hist.columns or "train/global_step" not in hist.columns:
                print(f"  skip {name}/{metric} — not in history")
                continue
            style = RUN_STYLE.get(name, {"color": "gray", "label": name})
            ax.plot(
                hist["train/global_step"],
                hist[metric],
                color=style["color"],
                label=style["label"],
                linewidth=1.6,
                alpha=0.9,
            )
        ax.set_xlabel("Training step")
        ax.set_ylabel(label)
        ax.set_title(label)
        if yscale == "log":
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", frameon=True)

    plt.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
