"""Q prompt formatting and normalization for AndroidWorld QKV data."""

from __future__ import annotations

import re
from typing import Dict, Optional

from atlaskv.android_world.ui_elements import compact_android_world_ui_elements

DEFAULT_HISTORY = "You just started, no action has been performed yet."
NO_UI_ELEMENTS = "No UI element details were found."
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

_GOAL_MARKER = "The current AndroidWorld user goal is:"
_HISTORY_MARKER = "\nHistory:"
_UI_MARKER = "\nThe visible UI elements are:"
_Q_UI_END_MARKERS = (
    "\n\nAllowed action JSON shapes:",
    "\n\nValid action_type values:",
    "\n\nPlease answer in exactly this format:",
)
_KEY_PREFIX = "For the AndroidWorld goal of "
_KEY_HISTORY_MARKER = ", after "
_KEY_STATE_MARKER = ", the current screen shows "
_KEY_SUFFIXES = (
    ", and the next action should be",
    ", and the next action is",
)
_HISTORY_STEP_RE = re.compile(r"^(?P<prefix>\s*Step\s+\d+\s*[:\-]\s*)(?P<body>.*)$")
_NO_PREVIOUS_ACTIONS = frozenset({
    "no previous action",
    "no previous actions",
    "nothing has been done",
    "you just started",
})


def _clean_text(value: object, default: str = "") -> str:
    if value is None:
        return default
    return " ".join(str(value).split()) or default


def _clean_history(value: object) -> str:
    if value is None:
        return DEFAULT_HISTORY
    return str(value).strip() or DEFAULT_HISTORY


def _find_first(text: str, markers: tuple[str, ...], start: int = 0) -> int:
    matches = [idx for marker in markers if (idx := text.find(marker, start)) != -1]
    return min(matches) if matches else -1


def extract_q_sections(q: str) -> Optional[Dict[str, str]]:
    """Extract goal, history, UI elements, and suffix from a Q prompt."""

    goal_start = q.find(_GOAL_MARKER)
    if goal_start == -1:
        return None
    goal_start += len(_GOAL_MARKER)

    history_marker_start = q.find(_HISTORY_MARKER, goal_start)
    if history_marker_start == -1:
        return None

    history_start = history_marker_start + len(_HISTORY_MARKER)
    if history_start < len(q) and q[history_start] == " ":
        history_start += 1

    ui_marker_start = q.find(_UI_MARKER, history_start)
    if ui_marker_start == -1:
        return None

    ui_start = ui_marker_start + len(_UI_MARKER)
    if ui_start < len(q) and q[ui_start] == "\n":
        ui_start += 1

    ui_end = _find_first(q, _Q_UI_END_MARKERS, ui_start)
    if ui_end == -1:
        ui_end = len(q)

    return {
        "goal": q[goal_start:history_marker_start].strip(),
        "history": q[history_start:ui_marker_start].strip(),
        "ui_elements": q[ui_start:ui_end].strip(),
        "suffix": q[ui_end:].strip(),
    }


def _strip_key_suffix(key_string: str) -> Optional[str]:
    key_string = _clean_text(key_string)
    lowered = key_string.lower()
    for suffix in _KEY_SUFFIXES:
        if lowered.endswith(suffix):
            return key_string[: -len(suffix)].strip()
    return None


def key_string_to_history_summary(key_string: str) -> Optional[str]:
    """Convert a distilled key_string into an AndroidWorld-style step summary."""

    body = _strip_key_suffix(key_string)
    if body is None or not body.startswith(_KEY_PREFIX):
        return None
    body = body[len(_KEY_PREFIX) :]
    if _KEY_HISTORY_MARKER not in body or _KEY_STATE_MARKER not in body:
        return None

    _, rest = body.split(_KEY_HISTORY_MARKER, maxsplit=1)
    history_phrase, current_state = rest.split(_KEY_STATE_MARKER, maxsplit=1)
    history_phrase = _clean_text(history_phrase).strip(" ,.")
    current_state = _clean_text(current_state).strip(" ,.")
    if not history_phrase or not current_state:
        return None
    if history_phrase.lower() in _NO_PREVIOUS_ACTIONS:
        return None
    return f"After {history_phrase}, the current screen shows {current_state}."


def _extract_balanced_json(text: str) -> tuple[Optional[str], str]:
    start = text.find("{")
    if start == -1:
        return None, text

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
                return text[start : idx + 1], text[idx + 1 :]
    return None, text


def _normalize_summary_text(summary: str) -> str:
    summary = _clean_text(summary).strip(" .")
    if not summary:
        return ""
    return summary + "."


def _parse_history_step(line: str) -> Optional[tuple[str, str, str]]:
    match = _HISTORY_STEP_RE.match(line)
    if not match:
        return None

    body = match.group("body").strip()
    label = "Action selected:"
    if not body.startswith(label):
        return None

    action, remainder = _extract_balanced_json(body[len(label) :].strip())
    if action is None:
        return None

    remainder = remainder.strip()
    if remainder.startswith("."):
        remainder = remainder[1:].strip()
    if remainder.startswith("Instruction:"):
        remainder = remainder[len("Instruction:") :].strip()
    return match.group("prefix"), action, _normalize_summary_text(remainder)


def format_action_selected_summary(action: str, summary: str = "") -> str:
    """Match AndroidWorld T3A's `Action selected: {action}. {summary}` shape."""

    summary = _clean_text(summary)
    if summary:
        return f"Action selected: {action}. {summary}"
    return f"Action selected: {action}."


def normalize_android_world_history(history: str, key_string: str | None = None) -> str:
    """Normalize offline history to AndroidWorld T3A history when a key is available."""

    history = _clean_history(history)
    if not key_string:
        return history

    key_summary = key_string_to_history_summary(key_string)
    if not key_summary:
        return history

    lines = [line.strip() for line in history.splitlines() if line.strip()]
    if not lines or history == DEFAULT_HISTORY:
        return DEFAULT_HISTORY

    parsed_steps = [_parse_history_step(line) for line in lines]
    last_parseable = max(
        (idx for idx, parsed in enumerate(parsed_steps) if parsed is not None),
        default=-1,
    )
    if last_parseable == -1:
        return history

    normalized_lines = []
    for idx, (line, parsed) in enumerate(zip(lines, parsed_steps)):
        if parsed is None:
            normalized_lines.append(line)
            continue
        prefix, action, fallback_summary = parsed
        summary = key_summary if idx == last_parseable else fallback_summary
        normalized_lines.append(prefix + format_action_selected_summary(action, summary))
    return "\n".join(normalized_lines)


def format_qkv_question(
    *,
    goal: str,
    history: str,
    ui_elements: str,
    allowed_actions: str = QKV_ALLOWED_ACTIONS,
    key_string: str | None = None,
) -> str:
    """Build AtlasKV's textual query after applying shared Q normalizers."""

    compact_ui = compact_android_world_ui_elements(ui_elements) or NO_UI_ELEMENTS
    normalized_history = (
        normalize_android_world_history(history, key_string)
        if key_string is not None
        else history
    )
    return f"""What is the next AndroidWorld action?

The current AndroidWorld user goal is: {_clean_text(goal, 'Unknown goal.')}
History: {normalized_history}
The visible UI elements are:
{compact_ui}

{allowed_actions}

Please answer in exactly this format:
Reason: <one brief reason grounded in the goal, history, or visible UI elements>
Action: {{"action_type": "..."}}
Use concrete JSON values. Do not output UI element metadata or a second action."""


def normalize_qkv_question(q: str, key_string: str | None = None) -> str:
    """Normalize an existing Q prompt: compact UI first, then normalize history."""

    sections = extract_q_sections(q)
    if sections is None:
        return q

    compact_ui = (
        compact_android_world_ui_elements(sections["ui_elements"]) or NO_UI_ELEMENTS
    )
    normalized_history = normalize_android_world_history(
        sections["history"],
        key_string,
    )
    suffix = sections["suffix"]
    if suffix:
        suffix = "\n\n" + suffix
    return f"""What is the next AndroidWorld action?

The current AndroidWorld user goal is: {sections['goal']}
History: {normalized_history}
The visible UI elements are:
{compact_ui}{suffix}"""
