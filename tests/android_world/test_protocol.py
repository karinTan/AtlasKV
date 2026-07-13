"""Tests for AndroidWorld T3A protocol handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlaskv.android_world.protocol import (
    AndroidWorldOutputError,
    PromptKind,
    classify_t3a_prompt,
    normalize_and_validate_action_output,
    process_t3a_output,
    validate_action,
)


ACTION_PROMPT = """You are an agent who can operate an Android phone on behalf of a user.
Here is a list of descriptions for some UI elements on the current screen:
UI element 0: Phone
UI element 2: Contacts
Now output an action from the above list in the correct JSON format.
Your answer should look like:
Reason: ...
Action: {"action_type":...}
"""

SUMMARY_PROMPT = """Now I want you to summerize the latest step.
Here is the description for the before screenshot: before
Here is the description for the after screenshot: after
This is the action you picked: {"action_type":"click","index":0}
Based on the reason: open it
Summary of this step:
"""


def test_classifies_native_t3a_requests() -> None:
    assert classify_t3a_prompt(ACTION_PROMPT) is PromptKind.ACTION_SELECTION
    assert classify_t3a_prompt(SUMMARY_PROMPT) is PromptKind.SUMMARIZATION
    assert classify_t3a_prompt("Please summarize Android actions.") is PromptKind.OTHER


def test_all_checked_in_native_payloads_are_classified_as_actions() -> None:
    payload_path = (
        Path(__file__).parents[2] / "src" / "atlaskv" / "inference" / "android_world_test_payloads.json"
    )
    with payload_path.open(encoding="utf-8") as file:
        payloads = json.load(file)

    for category in ("pure_text", "text_image"):
        for case in payloads[category]:
            text_parts = []
            for message in case["request"]["messages"]:
                content = message["content"]
                if isinstance(content, str):
                    text_parts.append(content)
                else:
                    text_parts.extend(
                        part["text"] for part in content if part.get("type") == "text" and part.get("text")
                    )
            assert classify_t3a_prompt("\n".join(text_parts)) is PromptKind.ACTION_SELECTION


def test_safely_normalizes_action_output() -> None:
    processed = process_t3a_output(
        ACTION_PROMPT,
        "```python\nThe answer is Action: {'action_type': 'click', 'index': '2'}\n``` extra explanation",
    )

    assert processed.prompt_kind is PromptKind.ACTION_SELECTION
    assert processed.action == {"action_type": "click", "index": 2}
    assert processed.content == (
        'Reason: Model selected the next action.\nAction: {"action_type":"click","index":2}'
    )


def test_preserves_reason_and_discards_trailing_explanation() -> None:
    content, action = normalize_and_validate_action_output(
        'Reason: Contacts is visible.\nAction: {"action_type":"open_app","app_name":"Contacts"}\nDone.'
    )

    assert content == (
        'Reason: Contacts is visible.\nAction: {"action_type":"open_app","app_name":"Contacts"}'
    )
    assert action["app_name"] == "Contacts"


@pytest.mark.parametrize(
    "action",
    [
        {"action_type": "status", "goal_status": "complete"},
        {"action_type": "answer", "text": "10 AM"},
        {"action_type": "click", "index": "2"},
        {"action_type": "click", "x": 100, "y": 200},
        {"action_type": "long_press", "index": 2},
        {"action_type": "input_text", "index": 2, "text": "Alice"},
        {"action_type": "keyboard_enter"},
        {"action_type": "navigate_home"},
        {"action_type": "navigate_back"},
        {"action_type": "scroll", "direction": "down"},
        {"action_type": "scroll", "direction": "up", "index": 2},
        {"action_type": "open_app", "app_name": "Contacts"},
        {"action_type": "wait"},
    ],
)
def test_accepts_supported_actions(action: dict[str, object]) -> None:
    normalized = validate_action(action, frozenset({0, 2}))
    assert normalized["action_type"] == action["action_type"]


@pytest.mark.parametrize(
    ("action", "code"),
    [
        ({"action_type": "send", "index": 0}, "unsupported_action_type"),
        ({"action_type": "click"}, "missing_action_field"),
        ({"action_type": "click", "index": 1}, "index_out_of_range"),
        ({"action_type": "click", "index": 0, "x": 1, "y": 2}, "conflicting_action_fields"),
        ({"action_type": "input_text", "text": "Alice"}, "missing_action_field"),
        ({"action_type": "scroll", "direction": "diagonal"}, "invalid_action_enum"),
        ({"action_type": "status", "goal_status": "success"}, "invalid_action_enum"),
        ({"action_type": "wait", "index": 0}, "unsupported_action_field"),
    ],
)
def test_rejects_invalid_actions_without_guessing(action: dict[str, object], code: str) -> None:
    with pytest.raises(AndroidWorldOutputError) as exc_info:
        validate_action(action, frozenset({0, 2}))
    assert exc_info.value.code == code


def test_rejects_multiple_actions() -> None:
    with pytest.raises(AndroidWorldOutputError, match="more than one"):
        normalize_and_validate_action_output(
            'Reason: choose one\nAction: {"action_type":"wait"}\nAction: {"action_type":"navigate_back"}'
        )


def test_summary_is_normalized_to_plain_single_line() -> None:
    processed = process_t3a_output(SUMMARY_PROMPT, "Summary of this step: Opened Contacts.\nIt worked.")
    assert processed.prompt_kind is PromptKind.SUMMARIZATION
    assert processed.content == "Opened Contacts. It worked."


def test_summary_rejects_action_block() -> None:
    with pytest.raises(AndroidWorldOutputError) as exc_info:
        process_t3a_output(SUMMARY_PROMPT, 'Action: {"action_type":"wait"}')
    assert exc_info.value.code == "invalid_summary_output"


def test_other_prompt_is_unchanged() -> None:
    output = "A normal response\nwith its original formatting."
    assert process_t3a_output("Tell me a joke", output).content == output
