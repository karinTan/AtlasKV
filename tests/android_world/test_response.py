"""Tests for OpenAI-compatible AndroidWorld response envelopes."""

from __future__ import annotations

from atlaskv.android_world.response import openai_chat_completion_body, openai_error_body


def test_chat_completion_body() -> None:
    body = openai_chat_completion_body(
        "atlaskv",
        'Reason: ready\nAction: {"action_type":"wait"}',
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        kv_injected=True,
    )

    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["kv_injected"] is True


def test_error_body() -> None:
    body = openai_error_body("bad action", code="unsupported_action_type", param="completion")
    assert body == {
        "error": {
            "message": "bad action",
            "type": "invalid_request_error",
            "param": "completion",
            "code": "unsupported_action_type",
        }
    }
