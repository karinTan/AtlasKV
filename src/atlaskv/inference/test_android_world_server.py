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


DEFAULT_URL = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_ACTION_MAX_TOKENS = 128
STRICT_ACTION_SYSTEM_PROMPT = (
    "You are an AndroidWorld action emitter. Return exactly one Action line and nothing else. "
    "The Action line must contain one valid JSON object using an action type allowed by the request. "
    "Do not output Reason. Do not output Markdown. Do not use ellipsis."
)

STRICT_ACTION_OUTPUT_INSTRUCTION = """Now choose one action from the action list above.
Return exactly one line.
The line must start with Action: and then contain a concrete JSON object.

Rules:
- The JSON after Action: must be valid JSON.
- Use only action types described in the request.
- Do not output Reason.
- Do not copy placeholders such as <target_index>, <name>, or <text_input>.
- Fill every required field with a concrete value.

Your Answer:
"""

QKV_ALLOWED_ACTIONS = """Valid action_type values:
status, answer, click, long_press, input_text, keyboard_enter, navigate_home, navigate_back, scroll, open_app, wait.
Use one of these JSON action shapes:
Action: {"action_type": "click", "index": 0}
Action: {"action_type": "input_text", "index": 0, "text": "text"}
Action: {"action_type": "open_app", "app_name": "Contacts"}
Action: {"action_type": "scroll", "direction": "down"}
Action: {"action_type": "status", "goal_status": "complete"}"""

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


def _split_request_parts(original_request: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    texts: List[str] = []
    images: List[Dict[str, Any]] = []
    for message in original_request.get("messages", []):
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "text" and part.get("text"):
                    texts.append(part["text"])
                elif part.get("type") == "image_url":
                    images.append(part)
    return "\n\n".join(texts).strip(), images


def _replace_final_output_instruction(text: str) -> str:
    match = re.search(r"\nNow output an action.*?\nYour Answer:\s*$", text, flags=re.DOTALL)
    if match:
        return text[: match.start()].rstrip() + "\n\n" + STRICT_ACTION_OUTPUT_INSTRUCTION
    return text.rstrip() + "\n\n" + STRICT_ACTION_OUTPUT_INSTRUCTION


def _extract_section(text: str, start_pattern: str, end_patterns: Iterable[str]) -> str:
    start = re.search(start_pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not start:
        return ""
    start_idx = start.end()
    end_idx = len(text)
    for end_pattern in end_patterns:
        end = re.search(end_pattern, text[start_idx:], flags=re.IGNORECASE | re.DOTALL)
        if end:
            end_idx = min(end_idx, start_idx + end.start())
    return text[start_idx:end_idx].strip()


def _extract_action_reference(text: str) -> str:
    start = text.find("- If you think the task has been completed")
    end_match = re.search(r"\nThe current user goal/request is:", text)
    if start != -1 and end_match:
        return text[start : end_match.start()].strip()
    return _extract_section(
        text,
        r"you must choose to perform one of the action.*?\n",
        [r"\nThe current user goal/request is:"],
    )


def _extract_goal(text: str) -> str:
    return _extract_section(
        text,
        r"The current user goal/request is:\s*",
        [r"\n\nHere is a history", r"\n\nThe current screenshot", r"\n\nHere is a list"],
    )


def _extract_history(text: str) -> str:
    return _extract_section(
        text,
        r"Here is a history.*?:\s*\n",
        [r"\n\nThe current screenshot", r"\n\nHere is a list"],
    )


def _extract_ui_elements(text: str) -> str:
    return _extract_section(
        text,
        r"Here is a list .*?:\s*\n",
        [r"\n\nHere are some useful guidelines", r"\n\nNow output an action"],
    )


def _build_enhanced_text_prompt(original_text: str, has_images: bool) -> str:
    modality_note = (
        "The request includes screenshots. Use the screenshot information and UI element list, but still return only one Action line."
        if has_images
        else "This is a text-only request. Use the UI element list and history."
    )
    goal = _extract_goal(original_text) or "Unknown goal."
    history = _extract_history(original_text) or "No previous action."
    ui_elements = _extract_ui_elements(original_text) or "No UI element details were found."
    action_reference = _extract_action_reference(original_text) or "Use an action type described by AndroidWorld."
    # Keep the enhanced prompt compact. Re-sending the complete native prompt plus a
    # second set of instructions made the model copy sentences from the request.
    return f"""Choose the next AndroidWorld action.

Allowed actions:
{action_reference}

Goal: {goal}
History: {history}
Observation: {modality_note}
UI elements:
{ui_elements}

{STRICT_ACTION_OUTPUT_INSTRUCTION}"""


def _build_qkv_text_prompt(original_text: str, has_images: bool) -> str:
    """Build only AtlasKV's textual query (Q) in the training QA style.

    K and V are *not* prompt fields. They are the ``kb_size`` pre-encoded
    vectors supplied to the model as ``kb_kvs`` by the server; the model builds
    the per-token knowledge query with its separate query projection head.
    """
    goal = _extract_goal(original_text) or "Unknown goal."
    history = _extract_history(original_text) or "No previous action."
    ui_elements = _extract_ui_elements(original_text) or "No UI element details were found."
    image_note = (
        "Screenshots are attached as image inputs. Use the labeled screenshot together with UI element indexes."
        if has_images
        else "No screenshot is available. Use the text UI element descriptions."
    )
    return f"""What is the next AndroidWorld action?

The current AndroidWorld user goal is: {goal}
History: {history}
Observation: {image_note}
{QKV_ALLOWED_ACTIONS}

The visible UI elements are:
{ui_elements}

Please answer in exactly this format:
Reason: <one brief reason grounded in the goal, history, or visible UI elements>
Action: {{"action_type": "..."}}
Use concrete JSON values. Do not output UI element metadata or a second action."""


def _build_request_payload(case: Dict[str, Any], prompt_strategy: str) -> Dict[str, Any]:
    original_request = json.loads(json.dumps(case["request"]))
    # An action is tiny; leaving the server's 1024-token default in place lets a
    # malformed answer continue for hundreds of tokens before it is returned.
    requested_max_tokens = original_request.get("max_tokens", DEFAULT_ACTION_MAX_TOKENS)
    original_request["max_tokens"] = min(int(requested_max_tokens), DEFAULT_ACTION_MAX_TOKENS)
    if prompt_strategy == "original":
        return original_request

    original_text, image_parts = _split_request_parts(original_request)
    has_images = bool(image_parts)
    if prompt_strategy == "request_enhanced_v1":
        system_prompt = STRICT_ACTION_SYSTEM_PROMPT
        text_prompt = _build_enhanced_text_prompt(original_text, has_images)
    elif prompt_strategy == "qkv_action_v1":
        # This strategy differs in the model-side KV injection, not through
        # literal Q:/K:/V: sections in the natural-language prompt.
        system_prompt = None
        text_prompt = _build_qkv_text_prompt(original_text, has_images)
    else:
        raise ValueError(f"Unsupported prompt strategy: {prompt_strategy}")

    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if has_images:
        content = [{"type": "text", "text": text_prompt}]
        content.extend(image_parts)
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": text_prompt})

    original_request["messages"] = messages
    return original_request


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
        choices=["original", "request_enhanced_v1", "qkv_action_v1"],
        default="request_enhanced_v1",
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
