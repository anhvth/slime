#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove system prompts for samples where any system prompt contains a given keyword "
            "(case-insensitive)."
        )
    )
    parser.add_argument("--input", required=True, help="Input JSONL dataset path.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path. Defaults to <input>_no_vingroup_system.jsonl.",
    )
    parser.add_argument(
        "--keyword",
        default="vingroup",
        help="Case-insensitive keyword to search inside system prompt content.",
    )
    parser.add_argument(
        "--prompt-key",
        default="prompt",
        help="Key containing prompt messages (default: prompt).",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Write back to the input file atomically (ignores --output).",
    )
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    if input_path.suffix == ".jsonl":
        return input_path.with_name(f"{input_path.stem}_no_vingroup_system{input_path.suffix}")
    return input_path.with_name(f"{input_path.name}_no_vingroup_system.jsonl")


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


def should_strip_system(prompt: Any, keyword: str) -> bool:
    if not isinstance(prompt, list):
        return False
    needle = keyword.lower()
    for msg in prompt:
        if not _is_system_message(msg):
            continue
        if needle in _content_text(msg).lower():
            return True
    return False


def strip_system_messages(prompt: list[Any]) -> tuple[list[Any], int]:
    filtered = [msg for msg in prompt if not _is_system_message(msg)]
    return filtered, len(prompt) - len(filtered)


def process_file(
    *,
    input_path: Path,
    output_path: Path,
    prompt_key: str,
    keyword: str,
) -> dict[str, int]:
    stats = {
        "rows_total": 0,
        "rows_updated": 0,
        "rows_keyword_match": 0,
        "system_msgs_removed": 0,
    }

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            raw = line.strip()
            if not raw:
                continue
            stats["rows_total"] += 1

            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {input_path}: {exc}") from exc

            prompt = row.get(prompt_key)
            if should_strip_system(prompt, keyword):
                stats["rows_keyword_match"] += 1
                if isinstance(prompt, list):
                    filtered_prompt, removed = strip_system_messages(prompt)
                    if removed > 0:
                        row[prompt_key] = filtered_prompt
                        stats["rows_updated"] += 1
                        stats["system_msgs_removed"] += removed

            fout.write(json.dumps(row, ensure_ascii=False))
            fout.write("\n")

    return stats


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.inplace:
        output_path = input_path.with_name(f"{input_path.name}.tmp.{os.getpid()}")
    else:
        output_path = Path(args.output).expanduser().resolve() if args.output else default_output_path(input_path)

    stats = process_file(
        input_path=input_path,
        output_path=output_path,
        prompt_key=args.prompt_key,
        keyword=args.keyword,
    )

    if args.inplace:
        output_path.replace(input_path)
        final_path = input_path
    else:
        final_path = output_path

    print(f"Saved: {final_path}")
    print(
        "Stats: "
        f"rows_total={stats['rows_total']}, "
        f"rows_keyword_match={stats['rows_keyword_match']}, "
        f"rows_updated={stats['rows_updated']}, "
        f"system_msgs_removed={stats['system_msgs_removed']}"
    )


if __name__ == "__main__":
    main()
