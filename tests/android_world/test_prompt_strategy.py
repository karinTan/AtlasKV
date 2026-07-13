"""Tests for AndroidWorld prompt rewriting strategies."""

from __future__ import annotations

from atlaskv.android_world.prompt_strategy import rewrite_chat_completion_payload


ACTION_TEXT = """You are an agent who can operate an Android phone on behalf of a user.
- If you think the task has been completed, return status complete.
The current user goal/request is: Add Alice to contacts.

Here is a history of what you have done so far:
No previous action.

Here is a list of descriptions for some UI elements on the current screen:
UI element 0: Phone
UI element 2: Contacts

Now output an action from the above list in the correct JSON format.
Your answer should look like:
Reason: ...
Action: {"action_type":...}
"""

SUMMARY_TEXT = """Now I want you to summerize the latest step.
Here is the description for the before screenshot: before
Here is the description for the after screenshot: after
This is the action you picked: {"action_type":"click","index":0}
Based on the reason: open it
Summary of this step:
"""


def test_enhanced_strategy_rewrites_action_request() -> None:
    payload = {
        "model": "atlaskv",
        "messages": [{"role": "user", "content": ACTION_TEXT}],
        "max_tokens": 1024,
    }

    rewritten = rewrite_chat_completion_payload(payload, "request_enhanced_v1")

    assert rewritten["max_tokens"] == 128
    assert rewritten["messages"][0]["role"] == "system"
    assert rewritten["messages"][1]["role"] == "user"
    assert "Choose the next AndroidWorld action." in rewritten["messages"][1]["content"]
    assert "Goal: Add Alice to contacts." in rewritten["messages"][1]["content"]


def test_qkv_strategy_preserves_image_parts() -> None:
    payload = {
        "model": "atlaskv",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ACTION_TEXT},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            }
        ],
    }

    rewritten = rewrite_chat_completion_payload(payload, "qkv_action_v1")

    content = rewritten["messages"][0]["content"]
    assert rewritten["messages"][0]["role"] == "user"
    assert content[0]["type"] == "text"
    assert "What is the next AndroidWorld action?" in content[0]["text"]
    assert content[1]["type"] == "image_url"


def test_non_action_request_is_not_rewritten() -> None:
    payload = {
        "model": "atlaskv",
        "messages": [{"role": "user", "content": SUMMARY_TEXT}],
        "max_tokens": 1024,
    }

    rewritten = rewrite_chat_completion_payload(payload, "qkv_action_v1")

    assert rewritten == payload
