#!/usr/bin/env python3
"""
Quick utility to remove /think or /no_think from JSONL files.
Uses fast text replacement instead of JSON parsing.
"""

import argparse
import sys
from pathlib import Path


def clean_file(file_path: str, dry_run: bool = False) -> tuple[int, int]:
    """Remove /think and /no_think from a JSONL file.

    Args:
        file_path: Path to the JSONL file
        dry_run: If True, only count occurrences without modifying

    Returns:
        Tuple of (think_count, no_think_count) occurrences found
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    content = path.read_text()

    think_count = content.count("/think")
    no_think_count = content.count("/no_think")

    if dry_run:
        return think_count, no_think_count

    # Replace /no_think first (to avoid partial match with /think)
    new_content = content.replace("/no_think", "")
    new_content = new_content.replace("/think", "")

    path.write_text(new_content)
    return think_count, no_think_count


def main():
    parser = argparse.ArgumentParser(
        description="Remove /think or /no_think from JSONL files"
    )
    parser.add_argument(
        "path",
        nargs="+",
        help="Path(s) to JSONL file(s) or directory containing JSONL files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count occurrences without modifying files",
    )
    args = parser.parse_args()

    files = []
    for p in args.path:
        path = Path(p)
        if path.is_dir():
            files.extend(path.glob("*.jsonl"))
        elif path.is_file():
            files.append(path)
        else:
            print(f"Warning: {p} not found, skipping", file=sys.stderr)

    if not files:
        print("No JSONL files found", file=sys.stderr)
        sys.exit(1)

    total_think = 0
    total_no_think = 0

    for f in files:
        think_count, no_think_count = clean_file(str(f), args.dry_run)
        total_think += think_count
        total_no_think += no_think_count

        action = "Would remove" if args.dry_run else "Removed"
        if think_count > 0 or no_think_count > 0:
            print(f"{action} from {f}: /think={think_count}, /no_think={no_think_count}")
        else:
            print(f"No occurrences in {f}")

    print(f"\nTotal: /think={total_think}, /no_think={total_no_think}")
    if args.dry_run:
        print("(dry run - no files were modified)")


if __name__ == "__main__":
    main()
