"""Q prompt formatting and normalization for AndroidWorld QKV data."""

from __future__ import annotations

import json
import re
from typing import Dict, Iterable, Optional

from atlaskv.android_world.ui_elements import compact_android_world_ui_elements_with_keep_indices

DEFAULT_HISTORY = "You just started, no action has been performed yet."
NO_UI_ELEMENTS = "No UI element details were found."
QKV_ALLOWED_ACTIONS = (
    "Allowed action JSON shapes: "
    "status(goal_status=complete|infeasible); answer(text); click(index); "
    "long_press(index); input_text(index,text); keyboard_enter; navigate_home; "
    "navigate_back; scroll(direction=up|down|left|right[,index]); open_app(app_name); wait."
)
QKV_ANSWER_FORMAT = """Please answer in exactly this format:
Reason: <one brief reason grounded in the goal, history, or visible UI elements>
Action: {"action_type": "..."}
Use concrete JSON values. Do not output UI element metadata or a second action."""

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
_HISTORY_STEP_NUMBER_RE = re.compile(r"\bStep\s+(?P<number>\d+)\b", re.IGNORECASE)
MAX_RECENT_HISTORY_STEPS = 5
_MAX_OLD_HISTORY_PARTS = 5
_MAX_OLD_HISTORY_CHARS = 360
_MAX_HISTORY_PART_CHARS = 80
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


def _truncate_text(text: str, limit: int) -> str:
    text = _clean_text(text).strip(" .")
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip(" .,;") + "..."


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


def _history_step_number(line: str) -> Optional[int]:
    match = _HISTORY_STEP_NUMBER_RE.search(line)
    if match is None:
        return None
    return int(match.group("number"))


def _action_json_summary(action_json: str) -> str:
    try:
        action = json.loads(action_json)
    except json.JSONDecodeError:
        return "selected an action"
    if not isinstance(action, dict):
        return "selected an action"

    action_type = action.get("action_type")
    if action_type == "open_app":
        return f"opened {_clean_text(action.get('app_name'), 'an app')}"
    if action_type == "click":
        return f"clicked UI element {action.get('index')}"
    if action_type == "long_press":
        return f"long pressed UI element {action.get('index')}"
    if action_type == "input_text":
        return f"typed text into UI element {action.get('index')}"
    if action_type == "scroll":
        direction = _clean_text(action.get("direction"), "a direction")
        if "index" in action:
            return f"scrolled {direction} in UI element {action.get('index')}"
        return f"scrolled {direction}"
    if action_type == "status":
        return f"marked the goal {_clean_text(action.get('goal_status'), 'done')}"
    if action_type == "answer":
        return "answered the user"
    if isinstance(action_type, str) and action_type:
        return action_type.replace("_", " ")
    return "selected an action"


def _history_line_summary(line: str) -> str:
    parsed = _parse_history_step(line)
    if parsed is None:
        return _truncate_text(line, _MAX_HISTORY_PART_CHARS)
    _, action_json, summary = parsed
    if summary:
        return _truncate_text(summary, _MAX_HISTORY_PART_CHARS)
    return _truncate_text(_action_json_summary(action_json), _MAX_HISTORY_PART_CHARS)


def _summarize_older_history(lines: list[str]) -> str:
    parts = [_history_line_summary(line) for line in lines]
    parts = [part for part in parts if part]
    if not parts:
        return f"{len(lines)} earlier actions omitted"
    if len(parts) > _MAX_OLD_HISTORY_PARTS:
        parts = parts[:3] + ["..."] + parts[-2:]
    return _truncate_text("; ".join(parts), _MAX_OLD_HISTORY_CHARS)


def compact_android_world_history(
    history: str,
    key_string: str | None = None,
    *,
    max_recent_steps: int = MAX_RECENT_HISTORY_STEPS,
) -> str:
    """Keep recent AndroidWorld steps verbatim and summarize older history."""

    normalized_history = normalize_android_world_history(history, key_string)
    lines = [line.strip() for line in normalized_history.splitlines() if line.strip()]
    if (
        max_recent_steps <= 0
        or not lines
        or normalized_history == DEFAULT_HISTORY
        or len(lines) <= max_recent_steps
    ):
        return normalized_history

    older_lines = lines[:-max_recent_steps]
    recent_lines = lines[-max_recent_steps:]
    step_numbers = [_history_step_number(line) for line in older_lines]
    step_numbers = [number for number in step_numbers if number is not None]
    if step_numbers:
        span = f"{min(step_numbers)}-{max(step_numbers)}"
    else:
        span = f"1-{len(older_lines)}"
    earlier = f"Earlier steps {span}: {_summarize_older_history(older_lines)}."
    return "\n".join([earlier, *recent_lines])


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


def qkv_prompt_suffix(allowed_actions: str = QKV_ALLOWED_ACTIONS) -> str:
    return f"{allowed_actions}\n\n{QKV_ANSWER_FORMAT}"


def _normalize_q_suffix(suffix: str) -> str:
    if (
        not suffix
        or "Allowed action JSON shapes:" in suffix
        or "Valid action_type values:" in suffix
    ):
        return qkv_prompt_suffix()
    return suffix


def format_qkv_question(
    *,
    goal: str,
    history: str,
    ui_elements: str,
    allowed_actions: str = QKV_ALLOWED_ACTIONS,
    key_string: str | None = None,
    keep_ui_indices: Iterable[int] | None = None,
) -> str:
    """Build AtlasKV's textual query after applying shared Q normalizers."""

    compact_ui = (
        compact_android_world_ui_elements_with_keep_indices(
            ui_elements,
            keep_ui_indices,
        )
        or NO_UI_ELEMENTS
    )
    compact_history = (
        compact_android_world_history(history, key_string)
        if key_string is not None
        else compact_android_world_history(history)
    )
    return f"""What is the next AndroidWorld action?

The current AndroidWorld user goal is: {_clean_text(goal, 'Unknown goal.')}
History: {compact_history}
The visible UI elements are:
{compact_ui}

{qkv_prompt_suffix(allowed_actions)}"""


def normalize_qkv_question(
    q: str,
    key_string: str | None = None,
    keep_ui_indices: Iterable[int] | None = None,
) -> str:
    """Normalize an existing Q prompt: compact UI first, then normalize history."""

    sections = extract_q_sections(q)
    if sections is None:
        return q

    compact_ui = (
        compact_android_world_ui_elements_with_keep_indices(
            sections["ui_elements"],
            keep_ui_indices,
        )
        or NO_UI_ELEMENTS
    )
    compact_history = compact_android_world_history(
        sections["history"],
        key_string,
    )
    suffix = _normalize_q_suffix(sections["suffix"])
    if suffix:
        suffix = "\n\n" + suffix
    return f"""What is the next AndroidWorld action?

The current AndroidWorld user goal is: {sections['goal']}
History: {compact_history}
The visible UI elements are:
{compact_ui}{suffix}"""
