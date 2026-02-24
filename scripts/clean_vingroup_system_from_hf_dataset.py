#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk


TMP_MATCH_COL = "__vingroup_system_match__"
TMP_UPDATED_COL = "__vingroup_system_updated__"
TMP_REMOVED_COL = "__vingroup_system_removed__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For rows where any system message contains a keyword (case-insensitive), "
            "remove all system messages from that row."
        )
    )
    parser.add_argument(
        "--input-dataset",
        required=True,
        help="Path to dataset directory readable by datasets.load_from_disk",
    )
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="Output dataset directory. Defaults to <input>_no_vingroup_system",
    )
    parser.add_argument(
        "--messages-key",
        default="messages",
        help="Column containing chat message list (default: messages)",
    )
    parser.add_argument(
        "--keyword",
        default="vingroup",
        help="Case-insensitive keyword used to trigger system-message removal",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=8,
        help="Number of processes used by datasets.map (default: 8)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size used by datasets.map and stats scan (default: 1024)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick debug run",
    )
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.name}_no_vingroup_system")


def _is_system_message(msg: Any) -> bool:
    if not isinstance(msg, dict):
        return False
    role = msg.get("role")
    return isinstance(role, str) and role.lower() == "system"


def _content_text(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return str(content)


def _transform_batch(batch: dict[str, list[Any]], *, messages_key: str, keyword: str) -> dict[str, list[Any]]:
    if messages_key not in batch:
        n = len(next(iter(batch.values()))) if batch else 0
        batch[TMP_MATCH_COL] = [False] * n
        batch[TMP_UPDATED_COL] = [False] * n
        batch[TMP_REMOVED_COL] = [0] * n
        return batch

    needle = keyword.lower()
    msgs_batch = batch[messages_key]
    out_msgs: list[Any] = []
    match_flags: list[bool] = []
    updated_flags: list[bool] = []
    removed_counts: list[int] = []

    for msgs in msgs_batch:
        if not isinstance(msgs, list):
            out_msgs.append(msgs)
            match_flags.append(False)
            updated_flags.append(False)
            removed_counts.append(0)
            continue

        row_match = any(_is_system_message(msg) and needle in _content_text(msg).lower() for msg in msgs)
        if not row_match:
            out_msgs.append(msgs)
            match_flags.append(False)
            updated_flags.append(False)
            removed_counts.append(0)
            continue

        filtered = [msg for msg in msgs if not _is_system_message(msg)]
        removed = len(msgs) - len(filtered)

        if removed > 0:
            out_msgs.append(filtered)
            updated_flags.append(True)
            removed_counts.append(removed)
        else:
            out_msgs.append(msgs)
            updated_flags.append(False)
            removed_counts.append(0)

        match_flags.append(True)

    batch[messages_key] = out_msgs
    batch[TMP_MATCH_COL] = match_flags
    batch[TMP_UPDATED_COL] = updated_flags
    batch[TMP_REMOVED_COL] = removed_counts
    return batch


def _collect_stats(ds: Dataset, batch_size: int) -> dict[str, int]:
    total = len(ds)
    keyword_match = 0
    rows_updated = 0
    system_msgs_removed = 0

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        chunk = ds[start:end]
        keyword_match += sum(1 for v in chunk[TMP_MATCH_COL] if v)
        rows_updated += sum(1 for v in chunk[TMP_UPDATED_COL] if v)
        system_msgs_removed += sum(int(v) for v in chunk[TMP_REMOVED_COL])

    return {
        "rows_total": total,
        "rows_keyword_match": keyword_match,
        "rows_updated": rows_updated,
        "system_msgs_removed": system_msgs_removed,
    }


def _process_dataset(
    ds: Dataset,
    *,
    messages_key: str,
    keyword: str,
    num_proc: int,
    batch_size: int,
    limit: int | None,
    desc: str,
) -> tuple[Dataset, dict[str, int]]:
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    mapped = ds.map(
        _transform_batch,
        fn_kwargs={"messages_key": messages_key, "keyword": keyword},
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc=desc,
    )

    stats = _collect_stats(mapped, batch_size=batch_size)
    cleaned = mapped.remove_columns([TMP_MATCH_COL, TMP_UPDATED_COL, TMP_REMOVED_COL])
    return cleaned, stats


def _merge_stats(stats_list: list[dict[str, int]]) -> dict[str, int]:
    out = {
        "rows_total": 0,
        "rows_keyword_match": 0,
        "rows_updated": 0,
        "system_msgs_removed": 0,
    }
    for stats in stats_list:
        for key in out:
            out[key] += stats.get(key, 0)
    return out


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_dataset).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    output_path = (
        Path(args.output_dataset).expanduser().resolve()
        if args.output_dataset
        else default_output_path(input_path)
    )
    if output_path.exists():
        raise FileExistsError(f"Output path already exists: {output_path}")

    raw = load_from_disk(str(input_path))

    if isinstance(raw, Dataset):
        cleaned, stats = _process_dataset(
            raw,
            messages_key=args.messages_key,
            keyword=args.keyword,
            num_proc=args.num_proc,
            batch_size=args.batch_size,
            limit=args.limit,
            desc="Cleaning system prompts",
        )
        cleaned.save_to_disk(str(output_path))
        split_stats = {"dataset": stats}
    elif isinstance(raw, DatasetDict):
        cleaned_splits = {}
        split_stats: dict[str, dict[str, int]] = {}
        for split_name, split_ds in raw.items():
            cleaned_split, stats = _process_dataset(
                split_ds,
                messages_key=args.messages_key,
                keyword=args.keyword,
                num_proc=args.num_proc,
                batch_size=args.batch_size,
                limit=args.limit,
                desc=f"Cleaning system prompts ({split_name})",
            )
            cleaned_splits[split_name] = cleaned_split
            split_stats[split_name] = stats
        DatasetDict(cleaned_splits).save_to_disk(str(output_path))
        stats = _merge_stats(list(split_stats.values()))
    else:
        raise TypeError(f"Unsupported dataset type: {type(raw)}")

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(
        "Total stats: "
        f"rows_total={stats['rows_total']}, "
        f"rows_keyword_match={stats['rows_keyword_match']}, "
        f"rows_updated={stats['rows_updated']}, "
        f"system_msgs_removed={stats['system_msgs_removed']}"
    )
    for split_name, split_stat in split_stats.items():
        print(
            f"Split {split_name}: "
            f"rows_total={split_stat['rows_total']}, "
            f"rows_keyword_match={split_stat['rows_keyword_match']}, "
            f"rows_updated={split_stat['rows_updated']}, "
            f"system_msgs_removed={split_stat['system_msgs_removed']}"
        )


if __name__ == "__main__":
    main()
