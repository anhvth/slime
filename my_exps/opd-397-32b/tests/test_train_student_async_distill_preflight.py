from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LIB_PATH = REPO_ROOT / "my_exps/opd-397-32b/train_student_async_distill_lib.sh"


def _run_bash(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-lc", command],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def test_cross_preflight_fails_on_vocab_mismatch(tmp_path: Path) -> None:
    hf_dir = tmp_path / "hf_ckpt"
    hf_dir.mkdir(parents=True)
    (hf_dir / "config.json").write_text('{"vocab_size": 248320}\n', encoding="utf-8")

    command = "\n".join(
        [
            f"source {shlex.quote(str(LIB_PATH))}",
            "MODEL_ARGS=(--tensor-model-parallel-size 8 --vocab-size 151936)",
            (
                "preflight_cross_tokenizer_vocab_match "
                f"\"1\" \"scripts/models/qwen3-32B.sh\" {shlex.quote(str(hf_dir))} MODEL_ARGS"
            ),
        ]
    )
    proc = _run_bash(command)

    assert proc.returncode != 0
    assert "Cross-tokenizer preflight failed" in proc.stderr
    assert "model_args --vocab-size: 151936" in proc.stderr
    assert "hf config vocab_size: 248320" in proc.stderr


def test_cross_preflight_passes_on_vocab_match(tmp_path: Path) -> None:
    hf_dir = tmp_path / "hf_ckpt"
    hf_dir.mkdir(parents=True)
    (hf_dir / "config.json").write_text('{"vocab_size": 151936}\n', encoding="utf-8")

    command = "\n".join(
        [
            f"source {shlex.quote(str(LIB_PATH))}",
            "MODEL_ARGS=(--tensor-model-parallel-size 8 --vocab-size 151936)",
            (
                "preflight_cross_tokenizer_vocab_match "
                f"\"1\" \"scripts/models/qwen3-32B.sh\" {shlex.quote(str(hf_dir))} MODEL_ARGS"
            ),
            "echo preflight-ok",
        ]
    )
    proc = _run_bash(command)

    assert proc.returncode == 0
    assert "preflight-ok" in proc.stdout
