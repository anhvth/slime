#!/usr/bin/env python3
import argparse
import json
from typing import Any
from urllib import request

from transformers import AutoTokenizer


def http_json(method: str, url: str, payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any] | None]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=30) as resp:  # noqa: S310
        code = int(resp.getcode())
        body = resp.read()
    if not body:
        return code, None
    return code, json.loads(body.decode("utf-8"))


def first_non_null(rows: Any) -> Any:
    if not isinstance(rows, list):
        return None
    for row in rows:
        if row:
            return row
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe SGLang teacher /generate top-k logprob contract.")
    parser.add_argument("--base-url", default="http://worker-30:13141")
    parser.add_argument(
        "--tokenizer-path",
        default="/home/anhvth8/home-trained-model/Stage3_SFT_Epoch3-As-Qwen35-Aligned",
    )
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    args = parser.parse_args()

    code, _ = http_json("GET", f"{args.base_url}/health")
    print(f"health_http={code}")

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    messages = [{"role": "user", "content": "Return exactly one short word: blue"}]
    prompt_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

    payload_gen = {
        "input_ids": prompt_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": args.max_new_tokens,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
        "top_logprobs_num": args.topk,
    }
    code_gen, body_gen = http_json("POST", f"{args.base_url}/generate", payload_gen)
    assert body_gen is not None
    output_ids = body_gen.get("output_ids", []) or []
    full_ids = list(prompt_ids) + list(output_ids)
    print(
        f"generate_http={code_gen} prompt_len={len(prompt_ids)} output_len={len(output_ids)} full_len={len(full_ids)}"
    )

    payload_score = {
        "input_ids": full_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": len(prompt_ids),
        "top_logprobs_num": args.topk,
    }
    code_score, body_score = http_json("POST", f"{args.base_url}/generate", payload_score)
    assert body_score is not None
    meta = body_score.get("meta_info", {})
    input_top = meta.get("input_top_logprobs")
    input_lp = meta.get("input_token_logprobs")
    first_top = first_non_null(input_top)
    first_lp = first_non_null(input_lp)

    print(f"score_http={code_score}")
    print(f"prompt_tokens={meta.get('prompt_tokens')} completion_tokens={meta.get('completion_tokens')}")
    print(f"input_token_logprobs_len={len(input_lp) if isinstance(input_lp, list) else None}")
    print(f"input_top_logprobs_len={len(input_top) if isinstance(input_top, list) else None}")
    print(f"first_non_null_input_token_logprobs={first_lp}")
    print(f"first_non_null_input_top_logprobs={first_top}")
    if isinstance(first_top, list) and first_top:
        print(f"first_top_entry_types={[type(v).__name__ for v in first_top[0]]}")


if __name__ == "__main__":
    main()
