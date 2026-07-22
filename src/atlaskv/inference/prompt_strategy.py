"""Prompt rewriting strategies used by the OpenAI-compatible inference API."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from atlaskv.android_world.protocol import PromptKind, classify_t3a_prompt
from atlaskv.android_world.q_format import DEFAULT_HISTORY, format_qkv_question

PromptStrategy = Literal["original", "qkv_action_v1"]

DEFAULT_ACTION_MAX_TOKENS = 128


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


def build_qkv_text_prompt(original_text: str, has_images: bool) -> str:
    """Build only AtlasKV's textual query (Q) in the training QA style."""

    del has_images
    goal = _extract_goal(original_text) or "Unknown goal."
    history = _extract_history(original_text) or DEFAULT_HISTORY
    raw_ui_elements = _extract_ui_elements(original_text)
    return format_qkv_question(
        goal=goal,
        history=history,
        ui_elements=raw_ui_elements,
    )


def rewrite_chat_completion_payload(
    request_payload: Dict[str, Any],
    prompt_strategy: PromptStrategy,
    action_max_tokens: Optional[int] = DEFAULT_ACTION_MAX_TOKENS,
) -> Dict[str, Any]:
    """Rewrite AndroidWorld action prompts while leaving other requests unchanged."""

    payload = json.loads(json.dumps(request_payload))
    if prompt_strategy == "original":
        return payload

    original_text, image_parts = _split_request_parts(payload)
    if classify_t3a_prompt(original_text) is not PromptKind.ACTION_SELECTION:
        return payload

    if action_max_tokens is not None:
        requested_max_tokens = payload.get("max_tokens", action_max_tokens)
        payload["max_tokens"] = min(int(requested_max_tokens), action_max_tokens)

    has_images = bool(image_parts)
    if prompt_strategy == "qkv_action_v1":
        text_prompt = build_qkv_text_prompt(original_text, has_images)
    else:
        raise ValueError(f"Unsupported prompt_strategy: {prompt_strategy}")

    if has_images:
        content = [{"type": "text", "text": text_prompt}]
        content.extend(image_parts)
        payload["messages"] = [{"role": "user", "content": content}]
    else:
        payload["messages"] = [{"role": "user", "content": text_prompt}]
    return payload
