"""Download MetaMathQA, take the seed=42 50K subset, format as
'Problem: ...\\n\\nSolution: ...', and cache to disk for train.py.

Usage:
    python -m src.prepare_data                       # default: 50K subset, seed 42
    python -m src.prepare_data --subset-size 500     # for smoke tests
    python -m src.prepare_data --output-dir ./data/foo

The output directory will contain a `datasets.Dataset.save_to_disk` snapshot
with one "text" column (concatenated prompt + solution).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


# CRITICAL — train and eval MUST use this exact format. Mismatch silently
# tanks fine-tuned scores. See CLAUDE.md "Prompt format consistency".
def format_example(example: dict) -> dict:
    return {"text": f"Problem: {example['query']}\n\nSolution: {example['response']}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="meta-math/MetaMathQA")
    parser.add_argument("--subset-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./data/metamathqa_50k_seed42")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    print(f"[1/4] Loading {args.dataset_name} (split={args.split}) ...")
    ds = load_dataset(args.dataset_name, split=args.split)
    print(f"      Loaded {len(ds):,} examples. Columns: {ds.column_names}")

    print(f"[2/4] Shuffling with seed={args.seed} and selecting {args.subset_size:,} ...")
    ds = ds.shuffle(seed=args.seed).select(range(args.subset_size))

    print("[3/4] Applying prompt format ...")
    ds = ds.map(format_example, remove_columns=ds.column_names, num_proc=1)

    out = Path(args.output_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[4/4] Saving to {out} ...")
    ds.save_to_disk(str(out))

    # Sanity report
    lengths = [len(x.split()) for x in ds["text"][:1000]]
    print("\n=== Sanity Report ===")
    print(f"Examples saved : {len(ds):,}")
    print(f"Sample [0]     : {ds['text'][0][:200]}...")
    print(f"Word count (n=1000): min={min(lengths)}, "
          f"median={sorted(lengths)[len(lengths) // 2]}, max={max(lengths)}")


if __name__ == "__main__":
    main()
