"""
Convert a Qwen3-32B checkpoint → Qwen3.5-compatible vocab

Strategy:
  - Load the Qwen3-32B finetune weights
  - Remap embed_tokens.weight and lm_head.weight from Qwen3 vocab IDs → Qwen3.5 vocab IDs
    using the shared-token mapping (mapping_qwen_35.csv)
  - Patch config.vocab_size to match Qwen3.5 (248320)
  - Copy the Qwen3.5 tokenizer into the output directory
  - Save a standard HF checkpoint ready for Megatron conversion and SGLang rollout

Usage:
  python my_exps/opd-397-32b/convert_qwen3_to_qwen35_vocab.py \
      --src ~/home-trained-model/Stage3_SFT_Epoch3 \
      --out ~/ckpt/hf_models/Qwen/Qwen3-32B-as-Qwen35 \
      --teacher-tokenizer ~/ckpt/hf_models/Qwen/Qwen3.5-397B-A17B-FP8 \
      --csv my_exps/opd-397-32b/mapping_qwen_35.csv
"""

import argparse
import csv
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Remap Qwen3 checkpoint vocab to Qwen3.5 vocab")
    p.add_argument("--src", type=str, required=True, help="Path to the source Qwen3 HF checkpoint")
    p.add_argument("--out", type=str, required=True, help="Output directory for the converted model")
    p.add_argument(
        "--teacher-tokenizer",
        type=str,
        default="~/ckpt/hf_models/Qwen/Qwen3.5-397B-A17B-FP8",
        help="Path or HF ID for the Qwen3.5 tokenizer",
    )
    p.add_argument(
        "--csv",
        type=str,
        default="my_exps/opd-397-32b/mapping_qwen_35.csv",
        help="CSV mapping file (qwen3_5_idx, qwen3_idx, token_str)",
    )
    p.add_argument(
        "--target-vocab-size",
        type=int,
        default=248320,
        help="Target (padded) vocab size for Qwen3.5",
    )
    return p.parse_args()


def load_mapping(csv_path: str) -> list[tuple[int, int]]:
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((int(row["qwen3_5_idx"]), int(row["qwen3_idx"])))
    return rows


def remap_weight(
    old_w: torch.Tensor,
    new_vocab_size: int,
    mapping: list[tuple[int, int]],
    label: str,
) -> torch.Tensor:
    """Remap old_w [old_vocab, hidden] → new_w [new_vocab, hidden] via mapping."""
    new_w = torch.zeros(new_vocab_size, old_w.shape[1], dtype=old_w.dtype)
    mapped = 0
    for id35, id3 in mapping:
        if id35 < new_vocab_size and id3 < old_w.shape[0]:
            new_w[id35] = old_w[id3]
            mapped += 1
    print(f"  {label}: mapped {mapped}/{new_vocab_size} rows ({100 * mapped / new_vocab_size:.1f}%)")
    return new_w


def main():
    args = parse_args()
    src = Path(args.src).expanduser()
    out = Path(args.out).expanduser()
    teacher_tok_path = str(Path(args.teacher_tokenizer).expanduser())

    print(f"Source checkpoint : {src}")
    print(f"Output directory  : {out}")
    print(f"Teacher tokenizer : {teacher_tok_path}")
    print(f"Mapping CSV       : {args.csv}")
    print(f"Target vocab size : {args.target_vocab_size}")

    # ── load mapping ──────────────────────────────────────────────────
    mapping = load_mapping(args.csv)
    print(f"\nLoaded {len(mapping)} shared-token pairs from mapping CSV")

    # ── load source model (keep in fp16/bf16 to save memory) ─────────
    print(f"\nLoading source model from {src} ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(src),
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    old_vocab = model.config.vocab_size
    hidden_size = model.lm_head.weight.shape[1]
    print(f"  embed_tokens : {model.model.embed_tokens.weight.shape}")
    print(f"  lm_head      : {model.lm_head.weight.shape}")

    # ── remap embed_tokens ────────────────────────────────────────────
    print(f"\nRemapping embed_tokens [{old_vocab} → {args.target_vocab_size}] ...")
    new_embed = remap_weight(
        model.model.embed_tokens.weight.data,
        args.target_vocab_size,
        mapping,
        "embed_tokens",
    )
    with torch.no_grad():
        model.model.embed_tokens.weight = torch.nn.Parameter(new_embed)

    # ── remap lm_head ─────────────────────────────────────────────────
    print(f"\nRemapping lm_head [{old_vocab} → {args.target_vocab_size}] ...")
    new_lm = remap_weight(
        model.lm_head.weight.data,
        args.target_vocab_size,
        mapping,
        "lm_head",
    )
    with torch.no_grad():
        model.lm_head.weight = torch.nn.Parameter(new_lm)

    # ── patch config ──────────────────────────────────────────────────
    model.config.vocab_size = args.target_vocab_size
    model.config.tie_word_embeddings = False

    # ── save model ────────────────────────────────────────────────────
    out.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving converted model to {out} ...")
    model.save_pretrained(str(out))

    # ── copy Qwen3.5 tokenizer ────────────────────────────────────────
    print(f"\nCopying Qwen3.5 tokenizer from {teacher_tok_path} ...")
    tok35 = AutoTokenizer.from_pretrained(teacher_tok_path)
    tok35.save_pretrained(str(out))

    print(f"\nDone! Converted model saved at: {out}")
    print(f"  vocab_size      : {args.target_vocab_size}")
    print(f"  hidden_size     : {hidden_size}")
    print(f"  shared tokens   : {len(mapping)}")


if __name__ == "__main__":
    main()
