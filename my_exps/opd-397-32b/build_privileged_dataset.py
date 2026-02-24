#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from openai import OpenAI


SYSTEM_PROMPT = """You write concise reasoning context for teacher-guided distillation.

Hard requirements:
- Output only one block wrapped by <distill_context> and </distill_context>.
- Keep it concise and high-signal.
- Include only reasoning that materially leads to the answer.
- End with a clear final answer sentence.
- Do not add preamble, markdown fences, or extra sections outside tags.
- Do not discuss instructions, request analysis, or formatting.
- Do not output "Thinking Process" or similar meta headings.
"""

_CONTEXT_RE = re.compile(r"<(?:distill_context|privileged_context)>\s*(.*?)\s*</(?:distill_context|privileged_context)>", re.DOTALL | re.IGNORECASE)
_FINAL_RE = re.compile(r"(?:final answer|the correct answer is)\s*[:：]\s*(.+)", re.IGNORECASE)

_THREAD_LOCAL = threading.local()


def to_dataset(obj: Dataset | DatasetDict) -> Dataset:
    if isinstance(obj, Dataset):
        return obj
    if isinstance(obj, DatasetDict):
        return concatenate_datasets([obj[k] for k in obj.keys()])
    raise TypeError(f"Unsupported dataset type: {type(obj)}")


def _clean_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        return []
    clean: list[dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(role, str):
            continue
        if content is None:
            continue
        if not isinstance(content, str):
            content = str(content)
        clean.append({"role": role, "content": content})
    return clean


def split_prompt_and_reference(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], str | None]:
    if not messages:
        return [], None
    if messages[-1].get("role") == "assistant":
        return messages[:-1], messages[-1].get("content")
    return messages, None


def format_messages_for_prompt(messages: list[dict[str, str]], max_chars: int) -> str:
    lines: list[str] = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown").upper()
        content = (msg.get("content") or "").strip()
        lines.append(f"[{i}] {role}:\n{content}\n")
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def build_teacher_messages(
    *,
    prompt_messages: list[dict[str, str]],
    reference_answer: str | None,
    max_prompt_chars: int,
    max_reference_chars: int,
) -> list[dict[str, str]]:
    student_visible = format_messages_for_prompt(prompt_messages, max_chars=max_prompt_chars)
    if reference_answer is None:
        reference_answer = "N/A"
    reference_answer = reference_answer.strip()
    if len(reference_answer) > max_reference_chars:
        reference_answer = reference_answer[:max_reference_chars]

    user_prompt = f"""Write a privileged context for this sample.

Student-visible conversation (what student sees):
{student_visible}

Teacher-only reference answer (for correctness alignment):
{reference_answer}

Output format:
<distill_context>
### Step-by-Step Solution
- Step 1: ...
- Step 2: ...
- Step 3: ...

### Final Answer
...
</distill_context>

Rules:
- Keep it concise and useful for distillation.
- If reference answer exists, align final answer with it.
- Do not output anything outside the distill_context tags.
- Do not mention the request, instructions, or formatting itself.
"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def extract_privileged_context(raw: str) -> str:
    matches = [m.strip() for m in _CONTEXT_RE.findall(raw) if m and m.strip()]
    if matches:
        longest = max(matches, key=len)
        if len(longest) >= 80:
            return longest
    text = raw.strip()
    for tag in (
        "<distill_context>",
        "</distill_context>",
        "<privileged_context>",
        "</privileged_context>",
    ):
        text = text.replace(tag, "")
    return text.strip()


def extract_final_answer(privileged_context: str) -> str | None:
    match = _FINAL_RE.search(privileged_context)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in privileged_context.splitlines() if line.strip()]
    if not lines:
        return None
    return lines[-1]


def looks_low_quality(privileged_context: str) -> bool:
    text = privileged_context.strip()
    if len(text) < 80:
        return True
    if text.lower() in {"` and `", "and"}:
        return True
    if "thinking process" in text.lower():
        return True
    return False


def fallback_privileged_context(reference_answer: str | None) -> str:
    ref = (reference_answer or "").strip()
    if len(ref) > 2000:
        ref = ref[:2000]
    if not ref:
        ref = "No reference answer was provided."
    return (
        "### Step-by-Step Solution\n"
        "- Use the teacher reference answer as the authoritative solution.\n"
        "- Keep only the essential reasoning needed for correctness.\n\n"
        "### Final Answer\n"
        f"{ref}"
    )


def _get_thread_client(base_url: str, api_key: str, timeout: float) -> OpenAI:
    clients = getattr(_THREAD_LOCAL, "clients", None)
    if clients is None:
        clients = {}
        _THREAD_LOCAL.clients = clients
    client = clients.get(base_url)
    if client is None:
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        clients[base_url] = client
    return client


def get_model_id(base_url: str, api_key: str, timeout: float, model: str | None) -> str:
    if model:
        return model
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    models = client.models.list()
    if not models.data:
        raise RuntimeError(f"No models returned by {base_url}/models")
    return models.data[0].id


def parse_base_urls(base_urls_arg: str | None, base_url_arg: str | None) -> list[str]:
    candidates: list[str] = []
    if base_urls_arg:
        candidates.extend(base_urls_arg.split(","))
    if base_url_arg:
        candidates.append(base_url_arg)
    urls: list[str] = []
    for url in candidates:
        cleaned = url.strip().rstrip("/")
        if not cleaned:
            continue
        cleaned = re.sub(r"/v1(?:/v1)+$", "/v1", cleaned)
        if not cleaned.endswith("/v1"):
            cleaned = f"{cleaned}/v1"
        if cleaned not in urls:
            urls.append(cleaned)
    if not urls:
        raise ValueError("No teacher base URLs configured. Set --base-urls or --base-url.")
    return urls


def get_model_id_from_urls(base_urls: list[str], api_key: str, timeout: float, model: str | None) -> tuple[str, str]:
    if model:
        return model, base_urls[0]
    last_error: Exception | None = None
    for base_url in base_urls:
        try:
            return get_model_id(base_url, api_key, timeout, model=None), base_url
        except Exception as exc:  # pragma: no cover - network failures depend on runtime env
            last_error = exc
    raise RuntimeError(f"Failed to fetch model id from all teacher endpoints: {base_urls}") from last_error


def build_output_row(
    *,
    sample_id: int,
    prompt_messages: list[dict[str, str]],
    tools: Any,
    subset_name: str | None,
    source: str | None,
    privileged_context: str,
    final_answer: str | None,
    model_id: str,
) -> dict[str, Any]:
    metadata = {
        "privileged_context": privileged_context,
        "subset_name": subset_name,
        "source": source,
        "privileged_model": model_id,
    }
    if isinstance(final_answer, str):
        final_answer = final_answer.strip()
        if len(final_answer) > 512:
            final_answer = final_answer[:512]
    row: dict[str, Any] = {
        "prompt": prompt_messages,
        "label": final_answer,
        "metadata": metadata,
        "sample_id": sample_id,
    }
    if isinstance(tools, list) and len(tools) > 0:
        row["tools"] = tools
    return row
from speedy_utils import memoize

@memoize
def process_one(
    *,
    record: dict[str, Any],
    sample_id: int,
    base_urls: list[str],
    api_key: str,
    timeout: float,
    model_id: str,
    temperature: float,
    top_p: float,
    top_k: int | None,
    max_new_tokens: int,
    retries: int,
    retry_backoff_sec: float,
    max_prompt_chars: int,
    max_reference_chars: int,
    disable_thinking: bool,
) -> dict[str, Any] | None:
    clean_messages = _clean_messages(record.get("messages"))
    if not clean_messages:
        return None

    prompt_messages, reference_answer = split_prompt_and_reference(clean_messages)
    if not prompt_messages:
        return None

    teacher_messages = build_teacher_messages(
        prompt_messages=prompt_messages,
        reference_answer=reference_answer,
        max_prompt_chars=max_prompt_chars,
        max_reference_chars=max_reference_chars,
    )

    num_urls = len(base_urls)
    for attempt in range(retries + 1):
        try:
            base_url = base_urls[(sample_id + attempt) % num_urls]
            client = _get_thread_client(base_url=base_url, api_key=api_key, timeout=timeout)
            extra_body: dict[str, Any] = {}
            if top_k is not None:
                extra_body["top_k"] = top_k
            if disable_thinking:
                extra_body["chat_template_kwargs"] = {"enable_thinking": False}
            if not extra_body:
                extra_body = None  # type: ignore[assignment]
            resp = client.chat.completions.create(
                model=model_id,
                messages=teacher_messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                extra_body=extra_body,
            )
            raw = resp.choices[0].message.content or ""
            privileged_context = extract_privileged_context(raw)
            if looks_low_quality(privileged_context):
                privileged_context = fallback_privileged_context(reference_answer)
            final_answer = extract_final_answer(privileged_context)
            return build_output_row(
                sample_id=sample_id,
                prompt_messages=prompt_messages,
                tools=record.get("tools"),
                subset_name=record.get("subset_name"),
                source=record.get("source"),
                privileged_context=privileged_context,
                final_answer=final_answer,
                model_id=model_id,
            )
        except Exception:
            if attempt >= retries:
                return None
            sleep_s = retry_backoff_sec * (2**attempt) * (1.0 + random.random() * 0.2)
            time.sleep(sleep_s)

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build prompt/metadata dataset with teacher-generated privileged_context from HF SFT dataset."
    )
    parser.add_argument(
        "--input-dataset",
        type=str,
        default="/home/anhvth8/projects/SFT/data/SFT_merged_2.9M",
        help="Path readable by datasets.load_from_disk",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/200k_prompt_for_distillation_privileged.jsonl",
        help="Output JSONL path in SLIME prompt-data format",
    )
    parser.add_argument("--num-samples", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--base-url", type=str, default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/teacher/v1"))
    parser.add_argument(
        "--base-urls",
        type=str,
        default=os.getenv("OPENAI_BASE_URLS"),
        help="Comma-separated teacher OpenAI-compatible endpoints, e.g. http://head:8000/teacher/v1,http://node2:13142/v1",
    )
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY", "abc"))
    parser.add_argument("--model", type=str, default=None, help="Model id. If omitted, uses first from /models")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-backoff-sec", type=float, default=1.5)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-prompt-chars", type=int, default=24000)
    parser.add_argument("--max-reference-chars", type=int, default=8000)
    parser.add_argument("--max-write", type=int, default=None, help="Debug limit after sampling.")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Do not force chat_template_kwargs.enable_thinking=False.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call model; write placeholder privileged_context for plumbing checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_urls = parse_base_urls(args.base_urls, args.base_url)

    ds_obj = load_from_disk(args.input_dataset)
    ds = to_dataset(ds_obj)
    if len(ds) == 0:
        raise ValueError("Input dataset is empty.")

    n = min(args.num_samples, len(ds))
    sampled = ds.shuffle(seed=args.seed).select(range(n))
    if args.max_write is not None:
        sampled = sampled.select(range(min(args.max_write, len(sampled))))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_id = "dry-run"
    if not args.dry_run:
        model_id, model_probe_url = get_model_id_from_urls(base_urls, args.api_key, args.timeout, args.model)
    else:
        model_probe_url = "dry-run"

    print(
        f"Loaded rows={len(ds)}, sampled={len(sampled)}, "
        f"seed={args.seed}, model={model_id}, workers={args.workers}, endpoints={len(base_urls)}",
        flush=True,
    )
    print(f"Teacher endpoints: {', '.join(base_urls)}", flush=True)
    print(f"Model probe endpoint: {model_probe_url}", flush=True)
    print(f"Writing to: {output_path}", flush=True)

    total = len(sampled)
    written = 0
    failed = 0
    started = time.time()

    with output_path.open("w", encoding="utf-8") as out_f:
        if args.dry_run:
            for i, record in enumerate(sampled):
                clean_messages = _clean_messages(record.get("messages"))
                prompt_messages, reference_answer = split_prompt_and_reference(clean_messages)
                if not prompt_messages:
                    failed += 1
                    continue
                privileged_context = (
                    "### Key Reasoning\n"
                    f"- Reference available: {bool(reference_answer)}\n\n"
                    "### Final Answer\n"
                    f"{(reference_answer or 'N/A')[:200]}"
                )
                final_answer = extract_final_answer(privileged_context)
                row = build_output_row(
                    sample_id=i,
                    prompt_messages=prompt_messages,
                    tools=record.get("tools"),
                    subset_name=record.get("subset_name"),
                    source=record.get("source"),
                    privileged_context=privileged_context,
                    final_answer=final_answer,
                    model_id=model_id,
                )
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            print(f"Dry-run complete: written={written}, failed={failed}", flush=True)
            return

        executor = ThreadPoolExecutor(max_workers=args.workers)
        futures: dict[Future, int] = {}

        def submit(i: int, rec: dict[str, Any]) -> None:
            fut = executor.submit(
                process_one,
                record=rec,
                sample_id=i,
                base_urls=base_urls,
                api_key=args.api_key,
                timeout=args.timeout,
                model_id=model_id,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                retries=args.retries,
                retry_backoff_sec=args.retry_backoff_sec,
                max_prompt_chars=args.max_prompt_chars,
                max_reference_chars=args.max_reference_chars,
                disable_thinking=(not args.enable_thinking),
            )
            futures[fut] = i

        next_idx = 0
        while next_idx < total or futures:
            while next_idx < total and len(futures) < args.chunk_size:
                submit(next_idx, sampled[next_idx])
                next_idx += 1

            done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                sample_id = futures.pop(fut)
                result = fut.result()
                if result is None:
                    failed += 1
                else:
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    written += 1

                if (written + failed) % 50 == 0 or (written + failed) == total:
                    elapsed = time.time() - started
                    rate = (written + failed) / elapsed if elapsed > 0 else 0.0
                    print(
                        f"progress={written + failed}/{total} written={written} failed={failed} rate={rate:.2f}/s",
                        flush=True,
                    )

        executor.shutdown(wait=True)

    elapsed = time.time() - started
    print(
        f"Done: total={total}, written={written}, failed={failed}, elapsed_sec={elapsed:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
