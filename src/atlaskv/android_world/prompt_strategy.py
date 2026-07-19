"""Prompt rewriting strategies for AndroidWorld action-selection requests."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from atlaskv.android_world.protocol import PromptKind, classify_t3a_prompt

PromptStrategy = Literal["original", "qkv_action_v1"]

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

QKV_ALLOWED_ACTIONS = """Allowed action JSON shapes:
Action: {"action_type":"status","goal_status":"complete"}
Action: {"action_type":"status","goal_status":"infeasible"}
Action: {"action_type":"answer","text":"..."}
Action: {"action_type":"click","index":0}
Action: {"action_type":"long_press","index":0}
Action: {"action_type":"input_text","text":"...","index":0}
Action: {"action_type":"keyboard_enter"}
Action: {"action_type":"navigate_home"}
Action: {"action_type":"navigate_back"}
Action: {"action_type":"scroll","direction":"up|down|left|right"}
Action: {"action_type":"scroll","direction":"up|down|left|right","index":0}
Action: {"action_type":"open_app","app_name":"..."}
Action: {"action_type":"wait"}"""

_NOISY_UI_PACKAGES = frozenset({
    "com.android.systemui",
})
_GENERIC_UI_RESOURCE_SUFFIXES = frozenset({
    "container",
    "content",
    "decor_view",
    "navigationBarBackground",
    "status_bar_launch_animation_container",
    "wifi_signal",
})
_INFORMATIVE_TRUE_FIELDS = (
    "is_clickable",
    "is_editable",
    "is_scrollable",
    "is_selected",
    "is_checked",
    "is_focused",
)
_TEXT_FIELDS = ("text", "content_description", "hint_text", "tooltip")


def _quoted_field_value(line: str, field_name: str) -> str:
    python_match = re.search(
        rf"\b{re.escape(field_name)}=('(?:\\'|[^'])*'|None|True|False)",
        line,
    )
    if python_match:
        raw = python_match.group(1)
        if raw in {"None", "False"}:
            return ""
        if raw == "True":
            return "True"
        return raw[1:-1].replace("\\'", "'")

    json_match = re.search(
        rf'"{re.escape(field_name)}":\s*("(?:\\"|[^"])*"|null|true|false)',
        line,
    )
    if json_match:
        raw = json_match.group(1)
        if raw in {"null", "false"}:
            return ""
        if raw == "true":
            return "True"
        try:
            return str(json.loads(raw))
        except json.JSONDecodeError:
            return raw.strip('"')
    return ""


def _is_true_field(line: str, field_name: str) -> bool:
    return bool(
        re.search(rf"\b{re.escape(field_name)}=True\b", line)
        or re.search(rf'"{re.escape(field_name)}":\s*true\b', line)
    )


def _is_generic_resource(resource_name: str) -> bool:
    suffix = resource_name.split("/")[-1]
    return suffix in _GENERIC_UI_RESOURCE_SUFFIXES


def _is_informative_ui_line(line: str) -> bool:
    if any(_quoted_field_value(line, field_name) for field_name in _TEXT_FIELDS):
        return True
    if any(_is_true_field(line, field_name) for field_name in _INFORMATIVE_TRUE_FIELDS):
        return True
    resource_name = _quoted_field_value(line, "resource_name")
    return bool(resource_name and not _is_generic_resource(resource_name))


def filter_android_world_ui_elements(
    ui_elements: str,
    *,
    include_system_ui: bool = False,
) -> str:
    """Remove AndroidWorld UI lines that are unlikely to help action selection."""

    kept_lines: List[str] = []
    for line in ui_elements.splitlines():
        if not line.strip():
            continue
        package_name = _quoted_field_value(line, "package_name")
        if package_name in _NOISY_UI_PACKAGES and not include_system_ui:
            continue
        if not _is_informative_ui_line(line):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


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


def build_enhanced_text_prompt(original_text: str, has_images: bool) -> str:
    modality_note = (
        "The request includes screenshots. Use the screenshot information and UI element list, but still return only one Action line."
        if has_images
        else "This is a text-only request. Use the UI element list and history."
    )
    goal = _extract_goal(original_text) or "Unknown goal."
    history = _extract_history(original_text) or "No previous action."
    ui_elements = _extract_ui_elements(original_text) or "No UI element details were found."
    action_reference = _extract_action_reference(original_text) or "Use an action type described by AndroidWorld."
    return f"""Choose the next AndroidWorld action.

Allowed actions:
{action_reference}

Goal: {goal}
History: {history}
Observation: {modality_note}
UI elements:
{ui_elements}

{STRICT_ACTION_OUTPUT_INSTRUCTION}"""


def build_qkv_text_prompt(original_text: str, has_images: bool) -> str:
    """Build only AtlasKV's textual query (Q) in the training QA style."""

    goal = _extract_goal(original_text) or "Unknown goal."
    history = _extract_history(original_text) or "No previous action."
    raw_ui_elements = _extract_ui_elements(original_text)
    ui_elements = (
        filter_android_world_ui_elements(raw_ui_elements)
        or "No UI element details were found."
    )
    image_note = (
        "Screenshots are attached as image inputs. Use the labeled screenshot together with UI element indexes."
        if has_images
        else "No screenshot is available. Use the text UI element descriptions."
    )
    return f"""What is the next AndroidWorld action?

The current AndroidWorld user goal is: {goal}
History: {history}
The visible UI elements are:
{ui_elements}

{QKV_ALLOWED_ACTIONS}

Please answer in exactly this format:
Reason: <one brief reason grounded in the goal, history, or visible UI elements>
Action: {{"action_type": "..."}}
Use concrete JSON values. Do not output UI element metadata or a second action."""


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
        system_prompt = None
        text_prompt = build_qkv_text_prompt(original_text, has_images)
    else:
        raise ValueError(f"Unsupported prompt_strategy: {prompt_strategy}")

    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if has_images:
        content = [{"type": "text", "text": text_prompt}]
        content.extend(image_parts)
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": text_prompt})

    payload["messages"] = messages
    return payload
