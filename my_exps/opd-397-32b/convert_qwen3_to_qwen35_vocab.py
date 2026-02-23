"""
Convert a Qwen3-32B checkpoint → Qwen3.5-compatible vocab

Strategy:
  - Load the Qwen3-32B finetune weights
  - Remap embed_tokens.weight and lm_head.weight from Qwen3 vocab IDs → Qwen3.5 vocab IDs
    using the shared-token mapping (mapping_qwen_35.csv)
  - Patch config.vocab_size to match Qwen3.5 (248320)
  - Save supported/unmapped token artifacts for runtime masking
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
import json
import shutil
from pathlib import Path
from typing import Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_unmapped_weight_value(value: str) -> Union[float, str]:
    if value.lower() == "random":
        return "random"
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--unmapped-weight-value must be a float value or 'random'"
        ) from exc


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
    p.add_argument(
        "--unmapped-weight-value",
        type=parse_unmapped_weight_value,
        default=-1000.0,
        help=(
            "Initialization value for unmapped rows in BOTH embed_tokens and lm_head after remap. "
            "Pass a float (default: -1000.0) or 'random'."
        ),
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
    fill_value: Union[float, str] = 0.0,
) -> torch.Tensor:
    """Remap old_w [old_vocab, hidden] → new_w [new_vocab, hidden] via mapping."""
    if fill_value == "random":
        stats_w = old_w.float()
        mean = stats_w.mean().item()
        std = stats_w.std(unbiased=False).item()
        if std == 0:
            std = 1e-6
        random_dtype = torch.float32 if old_w.dtype in (torch.float16, torch.bfloat16) else old_w.dtype
        new_w = torch.empty(
            (new_vocab_size, old_w.shape[1]),
            dtype=random_dtype,
            device=old_w.device,
        )
        new_w.normal_(mean=mean, std=std)
        if random_dtype != old_w.dtype:
            new_w = new_w.to(dtype=old_w.dtype)
        print(f"  {label}: initialized unmapped rows with random N(mean={mean:.6f}, std={std:.6f})")
    else:
        new_w = torch.full(
            (new_vocab_size, old_w.shape[1]),
            float(fill_value),
            dtype=old_w.dtype,
            device=old_w.device,
        )

    mapped = 0
    for id35, id3 in mapping:
        if id35 < new_vocab_size and id3 < old_w.shape[0]:
            new_w[id35] = old_w[id3]
            mapped += 1
    print(f"  {label}: mapped {mapped}/{new_vocab_size} rows ({100 * mapped / new_vocab_size:.1f}%)")
    return new_w


def get_supported_token_ids(
    mapping: list[tuple[int, int]],
    *,
    new_vocab_size: int,
    old_vocab_size: int,
) -> list[int]:
    """Return sorted Qwen3.5 token IDs that have a valid mapped source row."""
    supported = set()
    for id35, id3 in mapping:
        if id35 < new_vocab_size and id3 < old_vocab_size:
            supported.add(id35)
    return sorted(supported)


def save_mask_artifacts(
    out: Path,
    *,
    supported_token_ids: list[int],
    target_vocab_size: int,
    report: dict,
) -> None:
    """Persist lightweight artifacts for rollout/train-time token masking."""
    supported_set = set(supported_token_ids)
    unmapped_token_ids = [i for i in range(target_vocab_size) if i not in supported_set]

    supported_mask = torch.zeros(target_vocab_size, dtype=torch.bool)
    if supported_token_ids:
        supported_mask[torch.tensor(supported_token_ids, dtype=torch.long)] = True

    (out / "supported_token_ids.txt").write_text(
        "\n".join(str(x) for x in supported_token_ids) + "\n",
        encoding="utf-8",
    )
    (out / "unmapped_token_ids.txt").write_text(
        "\n".join(str(x) for x in unmapped_token_ids) + "\n",
        encoding="utf-8",
    )
    torch.save(supported_mask, out / "supported_token_mask.pt")
    (out / "vocab_remap_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nSaved mask artifacts:")
    print(f"  supported_token_mask.pt : {out / 'supported_token_mask.pt'}")
    print(f"  supported_token_ids.txt : {out / 'supported_token_ids.txt'}")
    print(f"  unmapped_token_ids.txt  : {out / 'unmapped_token_ids.txt'}")
    print(f"  vocab_remap_report.json : {out / 'vocab_remap_report.json'}")
    print(
        f"  supported={len(supported_token_ids)}, "
        f"unmapped={len(unmapped_token_ids)}"
    )


TOKENIZER_ASSET_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "chat_template.jinja",
)


def copy_teacher_tokenizer_assets(teacher_tokenizer: str, out: Path) -> None:
    """Copy tokenizer artifacts from teacher path when local; fallback to AutoTokenizer save."""
    teacher_path = Path(teacher_tokenizer).expanduser()
    if teacher_path.exists():
        copied = []
        for name in TOKENIZER_ASSET_FILES:
            src = teacher_path / name
            if src.exists():
                shutil.copy2(src, out / name)
                copied.append(name)
        if not copied:
            raise FileNotFoundError(
                f"No tokenizer assets found in teacher path: {teacher_path}. "
                f"Expected one of: {', '.join(TOKENIZER_ASSET_FILES)}"
            )
        print(f"  copied {len(copied)} tokenizer files from local teacher path")
        return

    # Non-local source (e.g. HF repo id): fallback to tokenizer serialization.
    tok35 = AutoTokenizer.from_pretrained(teacher_tokenizer, trust_remote_code=True)
    tok35.save_pretrained(str(out))
    print("  teacher tokenizer was non-local; saved tokenizer via AutoTokenizer")


def sanitize_tokenizer_class(out: Path) -> None:
    """Some environments serialize unsupported tokenizer_class values (e.g. TokenizersBackend)."""
    cfg_path = out / "tokenizer_config.json"
    if not cfg_path.exists():
        return

    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    tokenizer_class = data.get("tokenizer_class")
    if tokenizer_class == "TokenizersBackend":
        data["tokenizer_class"] = "Qwen2Tokenizer"
        cfg_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("  patched tokenizer_config.json: tokenizer_class TokenizersBackend -> Qwen2Tokenizer")


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
    print(f"Unmapped weight value : {args.unmapped_weight_value}")

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
    if getattr(model.lm_head, "bias", None) is None:
        print("  NOTE: lm_head has no bias parameter; unmapped suppression uses lm_head row fill.")

    # ── remap embed_tokens ────────────────────────────────────────────
    print(f"\nRemapping embed_tokens [{old_vocab} → {args.target_vocab_size}] ...")
    new_embed = remap_weight(
        model.model.embed_tokens.weight.data,
        args.target_vocab_size,
        mapping,
        "embed_tokens",
        fill_value=args.unmapped_weight_value,
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
        fill_value=args.unmapped_weight_value,
    )
    with torch.no_grad():
        model.lm_head.weight = torch.nn.Parameter(new_lm)

    supported_token_ids = get_supported_token_ids(
        mapping,
        new_vocab_size=args.target_vocab_size,
        old_vocab_size=old_vocab,
    )
    remap_report = {
        "source_checkpoint": str(src),
        "output_dir": str(out),
        "teacher_tokenizer": teacher_tok_path,
        "mapping_csv": args.csv,
        "old_vocab_size": int(old_vocab),
        "target_vocab_size": int(args.target_vocab_size),
        "shared_pairs_in_csv": int(len(mapping)),
        "supported_token_count": int(len(supported_token_ids)),
        "unmapped_token_count": int(args.target_vocab_size - len(supported_token_ids)),
        "supported_ratio": float(len(supported_token_ids) / args.target_vocab_size),
        "unmapped_weight_value": (
            args.unmapped_weight_value
            if isinstance(args.unmapped_weight_value, str)
            else float(args.unmapped_weight_value)
        ),
    }

    # ── patch config ──────────────────────────────────────────────────
    model.config.vocab_size = args.target_vocab_size
    model.config.tie_word_embeddings = False

    # ── save model ────────────────────────────────────────────────────
    out.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving converted model to {out} ...")
    model.save_pretrained(str(out))

    save_mask_artifacts(
        out,
        supported_token_ids=supported_token_ids,
        target_vocab_size=args.target_vocab_size,
        report=remap_report,
    )

    # ── copy Qwen3.5 tokenizer ────────────────────────────────────────
    print(f"\nCopying Qwen3.5 tokenizer from {teacher_tok_path} ...")
    copy_teacher_tokenizer_assets(teacher_tok_path, out)
    sanitize_tokenizer_class(out)

    print(f"\nDone! Converted model saved at: {out}")
    print(f"  vocab_size      : {args.target_vocab_size}")
    print(f"  hidden_size     : {hidden_size}")
    print(f"  shared pairs csv: {len(mapping)}")
    print(f"  supported tokens: {len(supported_token_ids)}")
    print(f"  unmapped tokens : {args.target_vocab_size - len(supported_token_ids)}")


if __name__ == "__main__":
    main()
