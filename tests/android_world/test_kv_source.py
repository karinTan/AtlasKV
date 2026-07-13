"""Consistency tests for the AndroidWorld value-embedding source data."""

from __future__ import annotations

import json
from pathlib import Path

from atlaskv.android_world.protocol import normalize_and_validate_action_output


SOURCE_PATH = (
    Path(__file__).parents[2] / "src" / "atlaskv" / "inference" / "android_world_action_qkv_seed.json"
)


def test_android_world_value_source_uses_reason_action_format() -> None:
    with SOURCE_PATH.open(encoding="utf-8") as file:
        rows = json.load(file)

    assert len(rows) == 14
    for row in rows:
        reason = row["reason"]
        description = row["description"]

        assert reason.strip()
        assert description.startswith(f"Reason: {reason}\nAction: ")
        assert row["A"] == description
        normalized, action = normalize_and_validate_action_output(description)
        assert normalized.startswith(f"Reason: {reason}\nAction: ")
        assert action["action_type"]
