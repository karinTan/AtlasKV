"""AndroidWorld UI element formatting utilities."""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

_NOISY_UI_PACKAGES = frozenset({
    "com.android.systemui",
})
_KEYBOARD_UI_PACKAGES = frozenset({
    "com.google.android.inputmethod.latin",
})
_SYSTEM_UI_PACKAGES = frozenset({
    "android",
    "com.android.packageinstaller",
    "com.android.permissioncontroller",
    "com.android.systemui",
    "com.google.android.permissioncontroller",
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
    "is_checkable",
    "is_clickable",
    "is_editable",
    "is_focusable",
    "is_scrollable",
    "is_selected",
    "is_checked",
    "is_focused",
    "is_long_clickable",
)
_TEXT_FIELDS = ("text", "content_description", "hint_text", "tooltip")
_COMPACT_TEXT_FIELDS = ("text", "content_description", "hint_text")
_COMPACT_TRUE_BOOL_FIELDS = (
    "is_checked",
    "is_clickable",
    "is_editable",
    "is_scrollable",
    "is_selected",
)
_GENERIC_CLASS_NAMES = frozenset({
    "FrameLayout",
    "ImageView",
    "LinearLayout",
    "RelativeLayout",
    "RecyclerView",
    "View",
    "ViewGroup",
})
_MAX_COMPACT_TEXT_CHARS = 96
_MAX_STATIC_TEXT_CHARS = 64
_MACHINE_ID_RE = re.compile(
    r"^(?:[0-9a-f]{12,}|[0-9a-f]{8,}(?:[-_][0-9a-f]{4,})+)(?:[-_]\d{6,})?$",
    re.IGNORECASE,
)
_ICON_TOKEN_RE = re.compile(
    r"\b(?:custom\s+)?icon[-_][A-Za-z0-9_-]+\b|\bcurrency_[A-Za-z]{3}\b"
)
_BBOX_LIST_RE = re.compile(r"\bbbox(?:_pixels)?=\[(?P<body>[^\]]+)\]")
_GENERIC_RESOURCE_TOKENS = frozenset({
    "background",
    "btn",
    "button",
    "cell",
    "constraint",
    "container",
    "content",
    "decor",
    "frame",
    "fragment",
    "host",
    "icon",
    "id",
    "image",
    "item",
    "items",
    "layout",
    "linear",
    "list",
    "nav",
    "navigation",
    "panel",
    "recycler",
    "relative",
    "root",
    "row",
    "screen",
    "scroll",
    "text",
    "view",
    "views",
})
_UI_ELEMENT_LINE_RE = re.compile(
    r"^(?P<prefix>\s*UI\s+element\s+(?P<index>\d+)\s*:\s*)(?P<body>.*)$",
    re.IGNORECASE,
)


def _quoted_field_value(line: str, field_name: str) -> str:
    python_match = re.search(
        rf"\b{re.escape(field_name)}=("
        r"'(?:\\'|[^'])*'|\"(?:\\\"|[^\"])*\"|None|True|False"
        r")",
        line,
    )
    if python_match:
        raw = python_match.group(1)
        if raw in {"None", "False"}:
            return ""
        if raw == "True":
            return "True"
        try:
            return str(ast.literal_eval(raw))
        except (SyntaxError, ValueError):
            return raw[1:-1]

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
        or re.search(rf"\b{re.escape(field_name)}\b(?=\s*(?:,|\)))", line)
    )


def _is_generic_resource(resource_name: str) -> bool:
    suffix = resource_name.split("/")[-1]
    return suffix in _GENERIC_UI_RESOURCE_SUFFIXES


def _resource_name_looks_semantic(resource_name: Any) -> bool:
    if not isinstance(resource_name, str) or not resource_name.strip():
        return False
    suffix = resource_name.rsplit("/", maxsplit=1)[-1].strip()
    if not suffix or not re.search(r"[A-Za-z]", suffix):
        return False
    normalized = suffix.lower()
    if normalized in _GENERIC_UI_RESOURCE_SUFFIXES:
        return False
    if "resource_name_obfuscated" in normalized:
        return False
    if normalized.startswith(("input_method_nav", "key_pos_", "navigationbar")):
        return False

    tokens = [token for token in re.split(r"[^a-z0-9]+", normalized) if token]
    if not tokens:
        return False
    return any(
        token not in _GENERIC_RESOURCE_TOKENS and not token.isdigit()
        for token in tokens
    )


def _package_kind(package_name: Any) -> Optional[str]:
    if not isinstance(package_name, str) or not package_name:
        return None
    if package_name in _KEYBOARD_UI_PACKAGES or "inputmethod" in package_name:
        return "keyboard"
    if package_name in _SYSTEM_UI_PACKAGES:
        return "system"
    return None


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _ast_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _ast_value(node.operand)
        if isinstance(value, (int, float)):
            return -value
    if isinstance(node, ast.Call) and _call_name(node.func) == "BoundingBox":
        return {
            keyword.arg: _ast_value(keyword.value)
            for keyword in node.keywords
            if keyword.arg is not None
        }
    try:
        return ast.literal_eval(node)
    except (TypeError, ValueError):
        return None


def _parse_ui_element_line(line: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    match = _UI_ELEMENT_LINE_RE.match(line)
    if not match:
        return None
    body = match.group("body").strip()
    if not body.startswith("UIElement("):
        return None
    try:
        parsed = ast.parse(body, mode="eval").body
    except SyntaxError:
        return _parse_loose_ui_element_line(match.group("prefix"), line)
    if not isinstance(parsed, ast.Call) or _call_name(parsed.func) != "UIElement":
        return None
    values = {
        keyword.arg: _ast_value(keyword.value)
        for keyword in parsed.keywords
        if keyword.arg is not None
    }
    return match.group("prefix"), values


def _parse_loose_ui_element_line(
    prefix: str,
    line: str,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    values: Dict[str, Any] = {}
    for field_name in (*_TEXT_FIELDS, "class_name", "package_name", "resource_name"):
        value = _quoted_field_value(line, field_name)
        if value:
            values[field_name] = value

    bbox_match = _BBOX_LIST_RE.search(line)
    if bbox_match is not None:
        try:
            bbox = ast.literal_eval("[" + bbox_match.group("body") + "]")
        except (SyntaxError, ValueError):
            bbox = None
        if isinstance(bbox, list) and len(bbox) == 4:
            values["bbox"] = bbox

    for field_name in _INFORMATIVE_TRUE_FIELDS:
        if _is_true_field(line, field_name):
            values[field_name] = True
    if re.search(r"\bis_enabled=False\b", line) or re.search(
        r'"is_enabled":\s*false\b',
        line,
    ):
        values["is_enabled"] = False

    return (prefix, values) if values else None


def _has_compact_text(values: Dict[str, Any]) -> bool:
    return any(
        _compact_text_value(values.get(field_name)) is not None
        for field_name in _COMPACT_TEXT_FIELDS
    )


def _compact_ui_element_is_informative(values: Dict[str, Any]) -> bool:
    if _is_long_static_text(values):
        return False
    if _has_compact_text(values):
        return True
    if values.get("is_editable") is True or values.get("is_scrollable") is True:
        return True
    if values.get("is_checked") is True or values.get("is_selected") is True:
        return True
    if values.get("is_clickable") is True and not _is_generic_container(values):
        return True
    if values.get("is_enabled") is False:
        return True
    return _resource_name_looks_semantic(values.get("resource_name"))


def _strip_private_use_chars(value: str) -> str:
    return "".join(
        char
        for char in value
        if not (0xE000 <= ord(char) <= 0xF8FF or 0xF0000 <= ord(char) <= 0x10FFFF)
    )


def _looks_like_machine_id(value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return False
    if _MACHINE_ID_RE.fullmatch(normalized):
        return True
    return bool(
        len(normalized) >= 20
        and any(char.isdigit() for char in normalized)
        and "-" in normalized
        and not re.search(r"\s", normalized)
    )


def _compact_text_value(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.replace("\xa0", " ")
    text = _strip_private_use_chars(text)
    text = _ICON_TOKEN_RE.sub(" ", text)
    text = " ".join(text.split()).strip()
    if not text or _looks_like_machine_id(text):
        return None
    if len(text) > _MAX_COMPACT_TEXT_CHARS:
        text = text[: _MAX_COMPACT_TEXT_CHARS - 1].rstrip(" .,;") + "..."
    return text


def _short_class_name_value(values: Dict[str, Any]) -> Optional[str]:
    return _short_class_name(values.get("class_name"))


def _is_generic_container(values: Dict[str, Any]) -> bool:
    class_name = _short_class_name_value(values)
    if class_name not in _GENERIC_CLASS_NAMES:
        return False
    if _has_compact_text(values):
        return False
    return not _resource_name_looks_semantic(values.get("resource_name"))


def _is_long_static_text(values: Dict[str, Any]) -> bool:
    raw_text = values.get("text")
    if not isinstance(raw_text, str):
        return False
    text = raw_text.replace("\xa0", " ")
    text = _strip_private_use_chars(text)
    text = _ICON_TOKEN_RE.sub(" ", text)
    text = " ".join(text.split()).strip()
    if not text or _looks_like_machine_id(text):
        return False
    class_name = _short_class_name_value(values)
    if class_name != "TextView":
        return False
    if any(values.get(field_name) is True for field_name in _COMPACT_TRUE_BOOL_FIELDS):
        return False
    if (
        _compact_text_value(values.get("content_description"))
        or _compact_text_value(values.get("hint_text"))
        or _resource_name_looks_semantic(values.get("resource_name"))
    ):
        return False
    return len(text) > _MAX_STATIC_TEXT_CHARS or _looks_like_static_body_text(text)


def _looks_like_static_body_text(text: str) -> bool:
    words = re.findall(r"[A-Za-z0-9]+", text)
    if len(words) >= 10:
        return True
    return len(words) >= 7 and bool(re.search(r"[a-z]", text)) and text[-1:] in ".,;:"


def _format_compact_value(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    return repr(value)


def _bbox_tuple(bbox: Any) -> Optional[Tuple[float, float, float, float]]:
    if isinstance(bbox, dict):
        required_fields = ("x_min", "x_max", "y_min", "y_max")
        if any(field_name not in bbox for field_name in required_fields):
            return None
        try:
            return (
                float(bbox["x_min"]),
                float(bbox["y_min"]),
                float(bbox["x_max"]),
                float(bbox["y_max"]),
            )
        except (TypeError, ValueError):
            return None
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            x_min, y_min, x_max, y_max = bbox
            return float(x_min), float(y_min), float(x_max), float(y_max)
        except (TypeError, ValueError):
            return None
    return None


def _best_bbox(values: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    return _bbox_tuple(values.get("bbox_pixels")) or _bbox_tuple(values.get("bbox"))


def _screen_size(parsed_elements: Iterable[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    max_x = 0.0
    max_y = 0.0
    for values in parsed_elements:
        bbox = _best_bbox(values)
        if bbox is None:
            continue
        _, _, x_max, y_max = bbox
        max_x = max(max_x, x_max)
        max_y = max(max_y, y_max)
    if max_x <= 0 or max_y <= 0:
        return None
    return max_x, max_y


def _axis_region(center: float, span: float, extent: float, labels: Tuple[str, str, str]) -> str:
    if extent <= 0:
        return labels[1]
    if span / extent >= 0.75:
        return "full"
    fraction = center / extent
    if fraction < 1 / 3:
        return labels[0]
    if fraction > 2 / 3:
        return labels[2]
    return labels[1]


def _format_compact_location(
    bbox: Optional[Tuple[float, float, float, float]],
    screen_size: Optional[Tuple[float, float]],
) -> Optional[str]:
    if bbox is None or screen_size is None:
        return None
    x_min, y_min, x_max, y_max = bbox
    width, height = screen_size
    if x_max <= x_min or y_max <= y_min:
        return None
    horizontal = _axis_region(
        (x_min + x_max) / 2.0,
        x_max - x_min,
        width,
        ("left", "center", "right"),
    )
    vertical = _axis_region(
        (y_min + y_max) / 2.0,
        y_max - y_min,
        height,
        ("top", "middle", "bottom"),
    )
    if horizontal == "full" and vertical == "full":
        return "full-screen"
    if horizontal == "full":
        return f"{vertical}-full"
    if vertical == "full":
        return f"full-{horizontal}"
    return f"{vertical}-{horizontal}"


def _short_class_name(class_name: Any) -> Optional[str]:
    if not isinstance(class_name, str) or not class_name.strip():
        return None
    return class_name.rsplit(".", maxsplit=1)[-1]


def _semantic_package_counter(parsed_elements: Iterable[Dict[str, Any]]) -> Counter[str]:
    packages: Counter[str] = Counter()
    for values in parsed_elements:
        package_name = values.get("package_name")
        if not isinstance(package_name, str) or not package_name:
            continue
        if _package_kind(package_name) is not None:
            continue
        packages[package_name] += 1
    return packages


def _main_package(parsed_elements: Iterable[Dict[str, Any]]) -> Optional[str]:
    packages = _semantic_package_counter(parsed_elements)
    if not packages:
        return None
    return sorted(packages.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _compact_ui_element_args(
    values: Dict[str, Any],
    *,
    main_package: Optional[str],
    screen_size: Optional[Tuple[float, float]],
) -> List[str]:
    args: List[str] = []

    for field_name in _COMPACT_TEXT_FIELDS:
        value = _compact_text_value(values.get(field_name))
        if value is not None:
            args.append(f"{field_name}={_format_compact_value(value)}")

    class_name = _short_class_name(values.get("class_name"))
    if class_name:
        args.append(f"class_name={_format_compact_value(class_name)}")

    location = _format_compact_location(_best_bbox(values), screen_size)
    if location is not None:
        args.append(f"loc={_format_compact_value(location)}")

    for field_name in _COMPACT_TRUE_BOOL_FIELDS:
        if values.get(field_name) is True:
            args.append(field_name)

    if values.get("is_enabled") is False:
        args.append("disabled")

    package_name = values.get("package_name")
    package_kind = _package_kind(package_name)
    if package_kind:
        args.append(f"package_name={_format_compact_value(package_kind)}")
    elif isinstance(package_name, str) and package_name and package_name != main_package:
        args.append(f"package_name={_format_compact_value(package_name)}")

    return args


def _compact_line_signature(args: List[str]) -> Tuple[str, ...]:
    return tuple(args)


def _compact_plain_ui_line(line: str) -> Optional[str]:
    match = _UI_ELEMENT_LINE_RE.match(line)
    if not match:
        return None
    body = " ".join(match.group("body").split()).strip()
    if not body or body.startswith("UIElement("):
        return None
    if _looks_like_machine_id(body):
        return None
    if len(body) > _MAX_COMPACT_TEXT_CHARS:
        body = body[: _MAX_COMPACT_TEXT_CHARS - 1].rstrip(" .,;") + "..."
    return f"{match.group('prefix')}{body}"


def compact_android_world_ui_elements(ui_elements: str) -> str:
    """Normalize AndroidWorld UIElement lines into the compact training format."""
    return compact_android_world_ui_elements_with_keep_indices(ui_elements)


def compact_android_world_ui_elements_with_keep_indices(
    ui_elements: str,
    keep_indices: Iterable[int] | None = None,
) -> str:
    """Normalize UIElement lines while preserving required target indices."""

    required_indices = frozenset(keep_indices or ())
    parsed_rows: List[Tuple[str, Optional[Dict[str, Any]]]] = []
    parsed_values: List[Dict[str, Any]] = []
    for line in ui_elements.splitlines():
        if not line.strip():
            continue
        parsed = _parse_ui_element_line(line)
        if parsed is None:
            parsed_rows.append((line, None))
            continue
        prefix, values = parsed
        parsed_rows.append((prefix, values))
        parsed_values.append(values)

    main_package = _main_package(parsed_values)
    screen_size = _screen_size(parsed_values)
    kept_lines: List[str] = []
    seen_signatures: set[Tuple[str, ...]] = set()
    for prefix_or_line, values in parsed_rows:
        index_match = _UI_ELEMENT_LINE_RE.match(prefix_or_line)
        row_index = int(index_match.group("index")) if index_match else None
        must_keep = row_index in required_indices
        if values is None:
            plain_line = _compact_plain_ui_line(prefix_or_line)
            if plain_line is not None:
                if plain_line not in kept_lines:
                    kept_lines.append(plain_line)
            elif _is_informative_ui_line(prefix_or_line):
                kept_lines.append(prefix_or_line)
            elif _UI_ELEMENT_LINE_RE.match(prefix_or_line) and "UIElement(" in prefix_or_line:
                kept_lines.append(prefix_or_line)
            continue
        if not must_keep and not _compact_ui_element_is_informative(values):
            continue
        args = _compact_ui_element_args(
            values,
            main_package=main_package,
            screen_size=screen_size,
        )
        if not args:
            continue
        signature = _compact_line_signature(args)
        if not must_keep and signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        kept_lines.append(f"{prefix_or_line}UIElement({', '.join(args)})")
    return "\n".join(kept_lines)


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
