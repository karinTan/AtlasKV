"""T3A prompt classification, output normalization, and action validation.

This module deliberately performs only syntax-preserving normalization. It
never changes an action type, guesses a missing UI index, or substitutes an
application name. Invalid action semantics are reported to the caller so the
AndroidWorld client can decide whether to retry generation.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, FrozenSet, Optional, Tuple


DEFAULT_REASON = "Model selected the next action."
ALLOWED_ACTION_TYPES = frozenset(
    {
        "status",
        "answer",
        "click",
        "long_press",
        "input_text",
        "keyboard_enter",
        "navigate_home",
        "navigate_back",
        "scroll",
        "open_app",
        "wait",
    }
)
SCROLL_DIRECTIONS = frozenset({"up", "down", "left", "right"})
GOAL_STATUSES = frozenset({"complete", "infeasible"})

_ACTION_MARKER_RE = re.compile(r"\bAction\s*:\s*", re.IGNORECASE)
_REASON_MARKER_RE = re.compile(r"\bReason\s*:\s*", re.IGNORECASE)
_UI_ELEMENT_RE = re.compile(r"\bUI\s+element\s+(\d+)\s*:", re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```(?:json|python|text)?\s*|```", re.IGNORECASE)


class PromptKind(str, Enum):
    """Kinds of T3A requests understood by the service."""

    ACTION_SELECTION = "action_selection"
    SUMMARIZATION = "summarization"
    OTHER = "other"


class AndroidWorldOutputError(ValueError):
    """A model output cannot safely be converted to the T3A contract."""

    def __init__(self, message: str, code: str = "invalid_android_world_output") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ProcessedOutput:
    """A model output after request-aware processing."""

    content: str
    prompt_kind: PromptKind
    action: Optional[Dict[str, Any]] = None


def classify_t3a_prompt(prompt: str) -> PromptKind:
    """Classify current AndroidWorld T3A action and summary prompt templates.

    Multiple template-specific signals are required so ordinary chat prompts
    mentioning an action or a summary are not accidentally rewritten.
    """

    lowered = prompt.lower()
    summary_signals = (
        "summary of this step:",
        "description for the before screenshot",
        "description for the after screenshot",
        "this is the action you picked:",
        "based on the reason:",
    )
    if sum(signal in lowered for signal in summary_signals) >= 3:
        return PromptKind.SUMMARIZATION

    action_signals = (
        "now output an action from the above list",
        "your answer should look like:",
        "reason: ...",
        'action: {"action_type":...}',
        "here is a list of descriptions for some ui elements",
    )
    if sum(signal in lowered for signal in action_signals) >= 2 and "action_type" in lowered:
        return PromptKind.ACTION_SELECTION

    # AtlasKV's compact AndroidWorld action prompts use a different ending but
    # still explicitly identify the task and require a single Action JSON.
    compact_action_signals = (
        "androidworld",
        "action_type",
        "next androidworld action",
        "return exactly one action line",
    )
    if sum(signal in lowered for signal in compact_action_signals) >= 3:
        return PromptKind.ACTION_SELECTION
    return PromptKind.OTHER


def process_t3a_output(prompt: str, output: str) -> ProcessedOutput:
    """Process a model output according to the detected T3A request kind."""

    kind = classify_t3a_prompt(prompt)
    if kind is PromptKind.ACTION_SELECTION:
        visible_indices = _extract_visible_indices(prompt)
        normalized, action = normalize_and_validate_action_output(output, visible_indices)
        return ProcessedOutput(normalized, kind, action)
    if kind is PromptKind.SUMMARIZATION:
        return ProcessedOutput(_normalize_summary(output), kind)
    return ProcessedOutput(output, kind)


def normalize_and_validate_action_output(
    output: str,
    visible_indices: Optional[FrozenSet[int]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Safely normalize one ``Reason + Action`` output and validate its action."""

    if not isinstance(output, str) or not output.strip():
        raise AndroidWorldOutputError("The model returned an empty action output.", "empty_action_output")

    text = _CODE_FENCE_RE.sub("", output).strip()
    action_markers = list(_ACTION_MARKER_RE.finditer(text))
    if not action_markers:
        raise AndroidWorldOutputError("The model output is missing 'Action:'.", "missing_action")
    if len(action_markers) != 1:
        raise AndroidWorldOutputError("The model output contains more than one Action.", "multiple_actions")

    action_marker = action_markers[0]
    object_start = text.find("{", action_marker.end())
    if object_start < 0:
        raise AndroidWorldOutputError("Action must be followed by a JSON object.", "missing_action_object")
    if text[action_marker.end() : object_start].strip():
        raise AndroidWorldOutputError(
            "Only a JSON object may follow the 'Action:' marker.", "invalid_action_json"
        )
    object_end = _find_balanced_object_end(text, object_start)
    raw_object = text[object_start : object_end + 1]
    action = _parse_action_object(raw_object)

    trailing = text[object_end + 1 :]
    if _contains_json_object(trailing):
        raise AndroidWorldOutputError("The model output contains more than one JSON action object.", "multiple_actions")

    reason = _extract_reason(text[: action_marker.start()])
    normalized_action = validate_action(action, visible_indices)
    serialized = json.dumps(normalized_action, ensure_ascii=False, separators=(",", ":"))
    return f"Reason: {reason}\nAction: {serialized}", normalized_action


def validate_action(
    action: Dict[str, Any],
    visible_indices: Optional[FrozenSet[int]] = None,
) -> Dict[str, Any]:
    """Validate one T3A action and return its type-normalized representation."""

    action_type = action.get("action_type")
    if not isinstance(action_type, str) or not action_type:
        raise AndroidWorldOutputError("Action requires a non-empty string action_type.", "missing_action_type")
    if action_type not in ALLOWED_ACTION_TYPES:
        raise AndroidWorldOutputError(
            f"Unsupported action_type {action_type!r}; the action meaning will not be guessed.",
            "unsupported_action_type",
        )

    normalized = dict(action)
    if action_type in {"click", "long_press"}:
        _reject_unknown_fields(normalized, {"action_type", "index", "x", "y"})
        has_index = "index" in normalized
        has_x = "x" in normalized
        has_y = "y" in normalized
        if has_index and (has_x or has_y):
            raise AndroidWorldOutputError("index cannot be combined with x/y coordinates.", "conflicting_action_fields")
        if has_index:
            normalized["index"] = _normalize_index(normalized["index"], visible_indices)
        elif has_x and has_y:
            normalized["x"] = _require_number(normalized["x"], "x")
            normalized["y"] = _require_number(normalized["y"], "y")
        else:
            raise AndroidWorldOutputError(
                f"{action_type} requires index or both x and y.", "missing_action_field"
            )
    elif action_type == "input_text":
        _reject_unknown_fields(normalized, {"action_type", "index", "text"})
        _require_fields(normalized, "index", "text")
        normalized["index"] = _normalize_index(normalized["index"], visible_indices)
        normalized["text"] = _require_string(normalized["text"], "text", allow_empty=True)
    elif action_type == "scroll":
        _reject_unknown_fields(normalized, {"action_type", "direction", "index"})
        _require_fields(normalized, "direction")
        if normalized["direction"] not in SCROLL_DIRECTIONS:
            raise AndroidWorldOutputError(
                "scroll.direction must be one of: up, down, left, right.", "invalid_action_enum"
            )
        if "index" in normalized:
            normalized["index"] = _normalize_index(normalized["index"], visible_indices)
    elif action_type == "open_app":
        _reject_unknown_fields(normalized, {"action_type", "app_name"})
        _require_fields(normalized, "app_name")
        normalized["app_name"] = _require_string(normalized["app_name"], "app_name")
    elif action_type == "answer":
        _reject_unknown_fields(normalized, {"action_type", "text"})
        _require_fields(normalized, "text")
        normalized["text"] = _require_string(normalized["text"], "text")
    elif action_type == "status":
        _reject_unknown_fields(normalized, {"action_type", "goal_status"})
        _require_fields(normalized, "goal_status")
        if normalized["goal_status"] not in GOAL_STATUSES:
            raise AndroidWorldOutputError(
                "status.goal_status must be complete or infeasible.", "invalid_action_enum"
            )
    else:
        _reject_unknown_fields(normalized, {"action_type"})
    return normalized


def _extract_visible_indices(prompt: str) -> Optional[FrozenSet[int]]:
    indices = frozenset(int(match.group(1)) for match in _UI_ELEMENT_RE.finditer(prompt))
    return indices or None


def _normalize_summary(output: str) -> str:
    if not isinstance(output, str) or not output.strip():
        raise AndroidWorldOutputError("The model returned an empty summary.", "empty_summary_output")
    text = _CODE_FENCE_RE.sub("", output).strip()
    text = re.sub(r"^Summary(?: of this step)?\s*:\s*", "", text, flags=re.IGNORECASE)
    if _ACTION_MARKER_RE.search(text):
        raise AndroidWorldOutputError(
            "A summarization response must not contain an Action block.", "invalid_summary_output"
        )
    summary = " ".join(text.split())
    if not summary:
        raise AndroidWorldOutputError("The model returned an empty summary.", "empty_summary_output")
    return summary


def _extract_reason(prefix: str) -> str:
    matches = list(_REASON_MARKER_RE.finditer(prefix))
    if len(matches) > 1:
        raise AndroidWorldOutputError("The model output contains more than one Reason.", "multiple_reasons")
    reason = prefix[matches[0].end() :] if matches else ""
    reason = " ".join(reason.strip().split())
    if not reason:
        return DEFAULT_REASON
    if _ACTION_MARKER_RE.search(reason) or _contains_json_object(reason):
        raise AndroidWorldOutputError("Reason must not contain another action or JSON object.", "invalid_reason")
    return reason


def _find_balanced_object_end(text: str, start: int) -> int:
    depth = 0
    quote: Optional[str] = None
    escaped = False
    for position in range(start, len(text)):
        char = text[position]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return position
    raise AndroidWorldOutputError("Action contains an unterminated JSON object.", "invalid_action_json")


def _parse_action_object(raw_object: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(raw_object)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(raw_object)
        except (SyntaxError, ValueError) as exc:
            raise AndroidWorldOutputError(
                "Action is not valid JSON or a safe dictionary literal.", "invalid_action_json"
            ) from exc
    if not isinstance(parsed, dict):
        raise AndroidWorldOutputError("Action must be a JSON object.", "invalid_action_json")
    if not all(isinstance(key, str) for key in parsed):
        raise AndroidWorldOutputError("All Action field names must be strings.", "invalid_action_json")
    return parsed


def _contains_json_object(text: str) -> bool:
    return "{" in text or "}" in text


def _require_fields(action: Dict[str, Any], *fields: str) -> None:
    missing = [field for field in fields if field not in action]
    if missing:
        raise AndroidWorldOutputError(
            f"Action is missing required field(s): {', '.join(missing)}.", "missing_action_field"
        )


def _reject_unknown_fields(action: Dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(action) - allowed)
    if unknown:
        raise AndroidWorldOutputError(
            f"Action contains unsupported field(s): {', '.join(unknown)}.", "unsupported_action_field"
        )


def _normalize_index(value: Any, visible_indices: Optional[FrozenSet[int]]) -> int:
    if isinstance(value, bool):
        raise AndroidWorldOutputError("index must be an integer.", "invalid_action_field")
    if isinstance(value, str):
        stripped = value.strip()
        if not re.fullmatch(r"\d+", stripped):
            raise AndroidWorldOutputError("index must be an integer.", "invalid_action_field")
        value = int(stripped)
    if not isinstance(value, int) or value < 0:
        raise AndroidWorldOutputError("index must be a non-negative integer.", "invalid_action_field")
    if visible_indices is not None and value not in visible_indices:
        raise AndroidWorldOutputError(
            f"index {value} is not present in the current UI element list.", "index_out_of_range"
        )
    return value


def _require_string(value: Any, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise AndroidWorldOutputError(f"{field} must be a string.", "invalid_action_field")
    if not allow_empty and not value.strip():
        raise AndroidWorldOutputError(f"{field} must be non-empty.", "invalid_action_field")
    return value


def _require_number(value: Any, field: str) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AndroidWorldOutputError(f"{field} must be a number.", "invalid_action_field")
    return value
