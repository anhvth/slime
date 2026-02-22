#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk


def to_dataset(obj) -> Dataset:
    if isinstance(obj, Dataset):
        return obj
    if isinstance(obj, DatasetDict):
        return concatenate_datasets([obj[k] for k in obj.keys()])
    raise TypeError(f"Unsupported dataset type: {type(obj)}")


def build_prompt(messages):
    if not isinstance(messages, list) or not messages:
        return None

    # For distillation prompts, drop the final assistant turn when present.
    if messages[-1].get("role") == "assistant":
        prompt = messages[:-1]
    else:
        prompt = messages

    if not prompt:
        return None

    return [
        {
            "role": m.get("role"),
            "content": m.get("content"),
        }
        for m in prompt
        if isinstance(m, dict) and m.get("role") and m.get("content") is not None
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Export random prompt-only samples from a HF disk dataset for distillation."
    )
    parser.add_argument(
        "--input-dataset",
        type=str,
        required=True,
        help="Path to dataset directory readable by datasets.load_from_disk",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/200k_prompt_for_distillation.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--num-samples", type=int, default=200_000, help="Number of random samples")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    args = parser.parse_args()

    raw = load_from_disk(args.input_dataset)
    ds = to_dataset(raw)

    if len(ds) == 0:
        raise ValueError("Input dataset is empty")

    n = min(args.num_samples, len(ds))
    sampled = ds.shuffle(seed=args.seed).select(range(n))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in sampled:
            prompt = build_prompt(row.get("messages"))
            if not prompt:
                continue
            f.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
            written += 1

    print(f"Loaded: {len(ds)} rows")
    print(f"Requested: {args.num_samples}, sampled: {n}")
    print(f"Written: {written} rows -> {out_path}")


if __name__ == "__main__":
    main()
