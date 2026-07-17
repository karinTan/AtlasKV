#!/usr/bin/env python3
"""Smoke test the AtlasKV OpenAI-compatible server with AndroidWorld payloads."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from atlaskv.android_world.prompt_strategy import rewrite_chat_completion_payload


DEFAULT_URL = "http://127.0.0.1:8000/v1/chat/completions"

def _post_json(url: str, payload: Dict[str, Any], timeout: float) -> Tuple[int, Dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = response.getcode()
        data = response.read().decode("utf-8")
    return status, json.loads(data)


def _load_cases(path: Path, category: str, names: Optional[Iterable[str]]) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        payloads = json.load(f)

    categories = ["pure_text", "text_image"] if category == "all" else [category]
    cases: List[Dict[str, Any]] = []
    wanted = set(names or [])
    for item_category in categories:
        for case in payloads.get(item_category, []):
            if wanted and case.get("name") not in wanted:
                continue
            cases.append(case)
    return cases


def _build_request_payload(case: Dict[str, Any], prompt_strategy: str) -> Dict[str, Any]:
    return rewrite_chat_completion_payload(case["request"], prompt_strategy)


def _extract_assistant_content(response: Dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _extract_action_json(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    match = re.search(r"Action\s*:\s*", text)
    if not match:
        return None, "missing Action:"

    start = text.find("{", match.end())
    if start == -1:
        return None, "missing JSON object after Action:"

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                raw_action = text[start : idx + 1]
                try:
                    return json.loads(raw_action), None
                except json.JSONDecodeError as exc:
                    return None, f"invalid Action JSON: {exc}"
    return None, "unterminated Action JSON object"


def _apply_overrides(
    request_payload: Dict[str, Any],
    model: Optional[str],
    max_tokens: Optional[int],
    temperature: Optional[float],
) -> Dict[str, Any]:
    payload = json.loads(json.dumps(request_payload))
    if model is not None:
        payload["model"] = model
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    return payload


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def _write_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

def _build_output_prefix(
    kb_size: Optional[str],
    kb_layer_frequency: Optional[str],
    kb_scale_factor: Optional[str],
    kv_injected: Optional[bool],
) -> str:
    parts = []
    if kb_size and kb_size != "0":
        parts.append(f"{kb_size}")
    if kb_layer_frequency and kb_layer_frequency != "0":
        parts.append(f"{kb_layer_frequency}")
    if kb_scale_factor and kb_scale_factor != "0" and kb_scale_factor != "None":
        parts.append(f"{kb_scale_factor}")
    if kv_injected is not None:
        parts.append("kv" if kv_injected else "no_kv")
    return "_".join(parts) + "_" if parts else ""

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test AtlasKV server with AndroidWorld request payloads")
    parser.add_argument("--url", default=DEFAULT_URL, help="Chat completions endpoint")
    parser.add_argument(
        "--payloads",
        type=Path,
        default=Path(__file__).with_name("android_world_test_payloads.json"),
        help="Path to android_world_test_payloads.json",
    )
    parser.add_argument("--category", choices=["all", "pure_text", "text_image"], default="all")
    parser.add_argument("--case", action="append", help="Run only the named case; can be passed multiple times")
    parser.add_argument(
        "--prompt-strategy",
        choices=["original", "qkv_action_v1"],
        default="qkv_action_v1",
        help=(
            "Send the original request, use a compact output-focused request, or send only "
            "AtlasKV's textual Q while the server injects pre-encoded K/V vectors"
        ),
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--model", help="Override request.model")
    parser.add_argument("--max-tokens", type=int, help="Override request.max_tokens")
    parser.add_argument("--temperature", type=float, help="Override request.temperature")
    parser.add_argument("--show-raw", action="store_true", help="Print full response JSON")
    parser.add_argument("--show-prompt", action="store_true", help="Print the request prompt sent to the server")
    parser.add_argument("--dry-run", action="store_true", help="List selected cases without sending requests")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed request")
    parser.add_argument("--output-jsonl", type=Path, help="Write per-case results as JSONL")
    parser.add_argument("--output-json", type=Path, help="Write per-case results as JSON")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cases = _load_cases(args.payloads, args.category, args.case)
    if not cases:
        print("No cases selected.")
        return 1

    if args.dry_run:
        for case in cases:
            print(
                f"{case.get('name')} [{case.get('modality')}] "
                f"task={case.get('task_name') or case.get('task')} strategy={args.prompt_strategy}"
            )
            if args.show_prompt:
                request_payload = _build_request_payload(case, args.prompt_strategy)
                request_payload = _apply_overrides(request_payload, args.model, args.max_tokens, args.temperature)
                print(json.dumps(request_payload.get("messages", []), ensure_ascii=False, indent=2))
        return 0

    results: List[Dict[str, Any]] = []
    failures = 0
    last_kb_size: Optional[str] = None
    last_kb_layer_frequency: Optional[str] = None
    last_kb_scale_factor: Optional[str] = None
    last_kv_injected: Optional[bool] = None
    for index, case in enumerate(cases, start=1):
        name = case.get("name", f"case_{index}")
        task = case.get("task_name") or case.get("task", "unknown")
        modality = case.get("modality", "unknown")
        request_payload = _build_request_payload(case, args.prompt_strategy)
        request_payload = _apply_overrides(request_payload, args.model, args.max_tokens, args.temperature)

        print(f"\n[{index}/{len(cases)}] {name} ({task}, {modality})")
        if args.show_prompt:
            print("request_messages:")
            print(json.dumps(request_payload.get("messages", []), ensure_ascii=False, indent=2))
        started = time.perf_counter()
        row: Dict[str, Any] = {
            "name": name,
            "task": task,
            "modality": modality,
            "prompt_strategy": args.prompt_strategy,
        }
        try:
            status, response = _post_json(args.url, request_payload, args.timeout)
            elapsed = time.perf_counter() - started
            content = _extract_assistant_content(response)
            action, action_error = _extract_action_json(content)
            usage = response.get("usage", {})
            kb_size = str(response.get("kb_size", 0))
            if kb_size and kb_size != "0":
                last_kb_size = kb_size
            kb_layer_frequency = str(response.get("kb_layer_frequency", 0))
            if kb_layer_frequency and kb_layer_frequency != "0":
                last_kb_layer_frequency = kb_layer_frequency
            kb_scale_factor = str(response.get("kb_scale_factor", 0))
            if kb_scale_factor and kb_scale_factor != "0" and kb_scale_factor != "None":
                last_kb_scale_factor = kb_scale_factor
            kv_injected = response.get("kv_injected")
            if isinstance(kv_injected, bool):
                last_kv_injected = kv_injected

            row.update(
                {
                    "ok": action_error is None,
                    "http_status": status,
                    "elapsed_sec": round(elapsed, 3),
                    "content": content,
                    "action": action,
                    "action_error": action_error,
                    "usage": usage,
                    "kb_size": kb_size,
                    "kb_layer_frequency": kb_layer_frequency,
                    "kb_scale_factor": kb_scale_factor,
                    "kv_injected": kv_injected,
                    "server_android_world_error": response.get("android_world_error"),
                    "server_output_valid": response.get("output_valid"),
                    "response": response if args.show_raw else None,
                }
            )

            print(f"HTTP {status} in {elapsed:.2f}s")
            if usage:
                print(f"usage={usage}")
            print("assistant:")
            print(content.strip() or "<empty>")
            if action_error:
                failures += 1
                print(f"Action parse: FAIL ({action_error})")
            else:
                print(f"Action parse: OK {action}")
            if args.show_raw:
                print("raw_response:")
                print(json.dumps(response, ensure_ascii=False, indent=2))

        except urllib.error.HTTPError as exc:
            elapsed = time.perf_counter() - started
            body = exc.read().decode("utf-8", errors="replace")
            failures += 1
            row.update(
                {
                    "ok": False,
                    "http_status": exc.code,
                    "elapsed_sec": round(elapsed, 3),
                    "error": body,
                }
            )
            print(f"HTTP {exc.code} in {elapsed:.2f}s")
            print(body)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            elapsed = time.perf_counter() - started
            failures += 1
            row.update({"ok": False, "elapsed_sec": round(elapsed, 3), "error": str(exc)})
            print(f"Request failed in {elapsed:.2f}s: {exc}")

        results.append(row)
        if failures and args.fail_fast:
            break

    # If we observed KB metadata in responses, prefix output filename with it.
    prefix = _build_output_prefix(
        last_kb_size,
        last_kb_layer_frequency,
        last_kb_scale_factor,
        last_kv_injected,
    )

    if args.output_jsonl:
        output_filename = args.output_jsonl
        output_path = output_filename.with_name(prefix + output_filename.name) if prefix else output_filename
        _write_jsonl(output_path, results)
        print(f"\nWrote results to {output_path}")

    if args.output_json:
        output_filename = args.output_json
        output_path = output_filename.with_name(prefix + output_filename.name) if prefix else output_filename
        _write_json(output_path, results)
        print(f"\nWrote results to {output_path}")

    passed = len(results) - failures
    print(f"\nSummary: {passed}/{len(results)} passed Action JSON parsing; {failures} failed.")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
