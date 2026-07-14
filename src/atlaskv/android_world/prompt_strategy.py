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

QKV_ALLOWED_ACTIONS = """Valid action_type values:
status, answer, click, long_press, input_text, keyboard_enter, navigate_home, navigate_back, scroll, open_app, wait.
Use one of these JSON action shapes:
- If you think the task has been completed, finish the task by using the status action with complete as goal_status:
Action: {"action_type": "status", "goal_status": "complete"}
- If you think the task is not feasible (including cases where you don't have enough information or cannot perform some necessary actions), finish the task by using the status action with infeasible as goal_status:
Action: {"action_type": "status", "goal_status": "infeasible"}
- Answer the user's question:
Action: {"action_type": "answer", "text": <answer_text>}
- Click/tap on a UI element (specified by its index) on the screen:
Action: {"action_type": "click", "index": <target_index>}
- Long press on a UI element (specified by its index) on the screen:
Action: {"action_type": "long_press", "index": <target_index>}
- Type text into a text field. This action automatically clicks the text field, enters the text, and presses Enter, so you do not need to click the field beforehand. Use the numeric label to indicate the target text field:
Action: {"action_type": "input_text", "text": <text_input>, "index": <target_index>}
- Press the Enter key:
Action: {"action_type": "keyboard_enter"}
- Navigate to the home screen:
Action: {"action_type": "navigate_home"}
- Navigate back:
Action: {"action_type": "navigate_back"}
- Scroll the screen or a scrollable UI element in one of the four directions. Specify the numeric index if scrolling a specific UI element; otherwise omit the index to scroll the whole screen:
Action: {"action_type": "scroll", "direction": <up|down|left|right>, "index": <optional_target_index>}
- Open an app (nothing will happen if the app is not installed):
Action: {"action_type": "open_app", "app_name": <app_name>}
- Wait for the screen to update:
Action: {"action_type": "wait"}"""


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
    ui_elements = _extract_ui_elements(original_text) or "No UI element details were found."
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
