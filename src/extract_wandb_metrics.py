"""Pull W&B run metrics (summary + system time-series) for cross-method comparison.

Run summary alone is not enough — peak GPU memory, utilization, power are
only in the system stream. This script fetches both for the configured run
names and writes a single JSON snapshot.

Usage (from project root, with the venv activated):
    python -m src.extract_wandb_metrics
    python -m src.extract_wandb_metrics --runs qlora-r16 lora-r16 full-ft
    python -m src.extract_wandb_metrics --output results/system_metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb


ENTITY = "haj003-university-of-california-san-diego"
PROJECT = "llama-math-finetune"

# W&B's system-metric key names have shifted across versions. Try in priority order.
PEAK_MEM_KEYS = [
    "system.gpu.0.memoryAllocatedBytes",
    "system.gpu.process.0.memoryAllocatedBytes",
    "system.gpu.0.memory",
]
GPU_UTIL_KEYS = [
    "system.gpu.0.gpu",
    "system.gpu.process.0.gpu",
]
POWER_KEYS = [
    "system.gpu.0.powerWatts",
    "system.gpu.0.powerPercent",
]
TEMP_KEYS = ["system.gpu.0.temp"]


def first_match(history, candidate_keys):
    """Return the first column from `candidate_keys` that exists in history."""
    for k in candidate_keys:
        if k in history.columns:
            return k
    return None


def extract_run(api, run_name: str) -> dict | None:
    print(f"\n[{run_name}] fetching ...")
    runs = list(api.runs(f"{ENTITY}/{PROJECT}", filters={"display_name": run_name}))
    if not runs:
        print(f"  WARNING: no run named {run_name!r}")
        return None
    # A name can map to multiple runs (after retries). Prefer the newest finished one.
    finished = sorted(
        (r for r in runs if r.state == "finished"),
        key=lambda r: r.created_at,
        reverse=True,
    )
    if finished:
        run = finished[0]
        if len(runs) > 1:
            print(f"  {len(runs)} runs share this name; picked newest finished: id={run.id}")
    else:
        runs.sort(key=lambda r: r.created_at, reverse=True)
        run = runs[0]
        print(f"  WARNING: no finished runs for {run_name!r}; using newest (state={run.state})")
    print(f"  id={run.id}, state={run.state}, created={run.created_at}")

    summary = dict(run.summary)

    # --- system metrics (separate stream) ---
    peak_mem_gb = avg_util = avg_power = max_temp = None
    sys_cols_found: list[str] = []
    try:
        hist = run.history(stream="events", pandas=True)
        if not hist.empty:
            sys_cols_found = [c for c in hist.columns if c.startswith("system.")]

            mem_key = first_match(hist, PEAK_MEM_KEYS)
            if mem_key:
                raw = hist[mem_key].dropna()
                # heuristic: bytes if values > 1e9, MB if 1e2-1e6, % if <=100
                m = raw.max()
                if m > 1e9:
                    peak_mem_gb = m / 1e9
                elif m > 1e6:
                    peak_mem_gb = m / 1e3  # MB → GB
                elif m <= 100:
                    peak_mem_gb = None  # it's a %
                else:
                    peak_mem_gb = m / 1024  # MiB → GB approx
                print(f"  peak GPU mem ({mem_key}, max={m}): {peak_mem_gb} GB")

            util_key = first_match(hist, GPU_UTIL_KEYS)
            if util_key:
                avg_util = float(hist[util_key].dropna().mean())
                print(f"  avg GPU util ({util_key}): {avg_util:.1f} %")

            power_key = first_match(hist, POWER_KEYS)
            if power_key:
                avg_power = float(hist[power_key].dropna().mean())
                print(f"  avg power ({power_key}): {avg_power:.1f}")

            temp_key = first_match(hist, TEMP_KEYS)
            if temp_key:
                max_temp = float(hist[temp_key].dropna().max())
                print(f"  max temp ({temp_key}): {max_temp:.1f} C")
    except Exception as e:
        print(f"  history(stream=events) failed: {e}")

    runtime_s = float(summary.get("train_runtime") or 0)
    # train.py logs num_gpus into wandb.config (via init_wandb's extra_config).
    # config != summary, so look there. Fall back to 1 if not present.
    num_gpus = int(dict(run.config).get("num_gpus") or 1)

    # RunPod A100 80GB SXM ≈ $1.80/h/GPU (rough, varies by region/cloud type).
    sxm_rate_per_gpu_h = 1.80

    return {
        "run_id": run.id,
        "name": run.name,
        "state": run.state,
        "created_at": str(run.created_at),
        "num_gpus": num_gpus,
        "runtime_s": runtime_s,
        "runtime_h": runtime_s / 3600 if runtime_s else None,
        "train_samples_per_sec": summary.get("train_samples_per_second"),
        "train_steps_per_sec": summary.get("train_steps_per_second"),
        "total_flos": summary.get("total_flos"),
        "final_loss": summary.get("train/loss"),
        "avg_loss": summary.get("train_loss"),
        "final_grad_norm": summary.get("train/grad_norm"),
        "peak_gpu_mem_gb": peak_mem_gb,
        "avg_gpu_util_pct": avg_util,
        "avg_power_w": avg_power,
        "max_temp_c": max_temp,
        "cost_usd_sxm": (runtime_s / 3600) * num_gpus * sxm_rate_per_gpu_h if runtime_s else None,
        "system_columns_available": sys_cols_found,
    }


def print_comparison(metrics: dict, runs: list[str]) -> None:
    rows = [
        ("num_gpus",               "# GPUs"),
        ("runtime_h",              "Runtime (h)"),
        ("peak_gpu_mem_gb",        "Peak GPU mem (GB)"),
        ("train_samples_per_sec",  "Throughput (samples/s)"),
        ("train_steps_per_sec",    "Throughput (steps/s)"),
        ("avg_gpu_util_pct",       "Avg GPU util (%)"),
        ("avg_power_w",            "Avg power (W)"),
        ("max_temp_c",             "Max temp (C)"),
        ("final_loss",             "Final train loss"),
        ("avg_loss",               "Avg train loss"),
        ("total_flos",             "Total FLOPs"),
        ("cost_usd_sxm",           "Cost (SXM, $1.80/GPU/h)"),
    ]
    width_label = 25
    width_col = 18
    print("\n=== System comparison ===")
    print(f"{'Metric':<{width_label}}", end="")
    for r in runs:
        print(f"{r:>{width_col}}", end="")
    print()
    print("-" * (width_label + width_col * len(runs)))
    for key, label in rows:
        print(f"{label:<{width_label}}", end="")
        for r in runs:
            v = metrics.get(r, {}).get(key)
            if v is None:
                cell = "N/A"
            elif isinstance(v, (int, float)):
                if abs(v) >= 1e6:
                    cell = f"{float(v):.3e}"
                else:
                    cell = f"{v:.4g}"
            else:
                cell = str(v)
            print(f"{cell:>{width_col}}", end="")
        print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", default=["qlora-r16", "lora-r16", "full-ft"])
    p.add_argument("--output", default="results/system_metrics.json")
    args = p.parse_args()

    api = wandb.Api()
    metrics: dict[str, dict] = {}
    for name in args.runs:
        result = extract_run(api, name)
        if result is not None:
            metrics[name] = result

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nSaved → {out_path}")

    print_comparison(metrics, args.runs)


if __name__ == "__main__":
    main()
