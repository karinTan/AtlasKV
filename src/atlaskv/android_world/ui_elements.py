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
    "is_checkable",
    "is_clickable",
    "is_editable",
    "is_focused",
    "is_focusable",
    "is_long_clickable",
    "is_scrollable",
    "is_selected",
)
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
        return None
    if not isinstance(parsed, ast.Call) or _call_name(parsed.func) != "UIElement":
        return None
    values = {
        keyword.arg: _ast_value(keyword.value)
        for keyword in parsed.keywords
        if keyword.arg is not None
    }
    return match.group("prefix"), values


def _has_compact_text(values: Dict[str, Any]) -> bool:
    return any(
        isinstance(values.get(field_name), str) and values[field_name].strip()
        for field_name in _COMPACT_TEXT_FIELDS
    )


def _compact_ui_element_is_informative(values: Dict[str, Any]) -> bool:
    if _has_compact_text(values):
        return True
    if any(values.get(field_name) is True for field_name in _COMPACT_TRUE_BOOL_FIELDS):
        return True
    if values.get("is_enabled") is False:
        return True
    return _resource_name_looks_semantic(values.get("resource_name"))


def _format_compact_value(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    return repr(value)


def _format_compact_bbox(bbox: Any) -> Optional[str]:
    if not isinstance(bbox, dict):
        return None
    required_fields = ("x_min", "y_min", "x_max", "y_max")
    if any(field_name not in bbox for field_name in required_fields):
        return None
    values = [bbox[field_name] for field_name in required_fields]
    return "[" + ", ".join(_format_compact_value(value) for value in values) + "]"


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
) -> List[str]:
    args: List[str] = []

    for field_name in _COMPACT_TEXT_FIELDS:
        value = values.get(field_name)
        if isinstance(value, str) and value.strip():
            args.append(f"{field_name}={_format_compact_value(value)}")

    class_name = _short_class_name(values.get("class_name"))
    if class_name:
        args.append(f"class_name={_format_compact_value(class_name)}")

    bbox = _format_compact_bbox(values.get("bbox_pixels"))
    if bbox is not None:
        args.append(f"bbox={bbox}")

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

    if not _has_compact_text(values) and _resource_name_looks_semantic(
        values.get("resource_name")
    ):
        args.append(f"resource_name={_format_compact_value(values['resource_name'])}")

    return args


def compact_android_world_ui_elements(ui_elements: str) -> str:
    """Normalize AndroidWorld UIElement lines into the compact training format."""

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
    kept_lines: List[str] = []
    for prefix_or_line, values in parsed_rows:
        if values is None:
            if _is_informative_ui_line(prefix_or_line):
                kept_lines.append(prefix_or_line)
            elif _UI_ELEMENT_LINE_RE.match(prefix_or_line) and "UIElement(" in prefix_or_line:
                kept_lines.append(prefix_or_line)
            continue
        if not _compact_ui_element_is_informative(values):
            continue
        args = _compact_ui_element_args(values, main_package=main_package)
        if not args:
            continue
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
