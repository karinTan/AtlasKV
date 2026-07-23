#!/usr/bin/env python3
"""Normalize existing AndroidWorld QKV questions after key distillation."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
  sys.path.insert(0, str(SRC_ROOT))

from atlaskv.android_world.q_format import (  # pylint: disable=wrong-import-position
    extract_q_sections,
    normalize_qkv_question,
)
from atlaskv.android_world.protocol import (  # pylint: disable=wrong-import-position
    AndroidWorldOutputError,
    normalize_and_validate_action_output,
)


def _read_json(path: Path) -> Any:
  with path.open(encoding="utf-8") as file:
    return json.load(file)


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as file:
    json.dump(rows, file, ensure_ascii=False, indent=2)
    file.write("\n")


def _default_output_path(input_path: Path) -> Path:
  return input_path.with_name(f"{input_path.stem}_normalized{input_path.suffix}")


def _rows_by_name(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
  return {
      str(row.get("name")): row
      for row in rows
      if isinstance(row, dict) and row.get("name") is not None
  }


def _target_keep_indices(row: dict[str, Any]) -> list[int]:
  output = str(row.get("A") or row.get("description") or "")
  try:
    _, action = normalize_and_validate_action_output(output)
  except AndroidWorldOutputError:
    return []
  index = action.get("index")
  return [index] if isinstance(index, int) else []


def normalize_qkv_rows(
    rows: list[dict[str, Any]],
    source_rows_by_name: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
  stats = {
      "total_rows": 0,
      "q_restored_from_source_rows": 0,
      "missing_source_q_rows": 0,
      "q_changed_rows": 0,
      "ui_changed_rows": 0,
      "history_changed_rows": 0,
      "unparseable_q_rows": 0,
  }
  normalized_rows: list[dict[str, Any]] = []

  for row in rows:
    stats["total_rows"] += 1
    updated = copy.deepcopy(row)
    old_q = str(row.get("Q") or "")
    source_row = (
        source_rows_by_name.get(str(row.get("name")))
        if source_rows_by_name is not None
        else None
    )
    if source_rows_by_name is not None:
      source_q = str(source_row.get("Q") or "") if source_row is not None else ""
      if source_q:
        old_q = source_q
        stats["q_restored_from_source_rows"] += 1
      else:
        stats["missing_source_q_rows"] += 1

    old_sections = extract_q_sections(old_q)
    if old_sections is None:
      stats["unparseable_q_rows"] += 1
      normalized_rows.append(updated)
      continue

    new_q = normalize_qkv_question(
        old_q,
        str(row.get("key_string") or ""),
        keep_ui_indices=_target_keep_indices(row),
    )
    new_sections = extract_q_sections(new_q)
    updated["Q"] = new_q
    if new_q != old_q:
      stats["q_changed_rows"] += 1
    if new_sections is not None:
      if new_sections["ui_elements"] != old_sections["ui_elements"]:
        stats["ui_changed_rows"] += 1
      if new_sections["history"] != old_sections["history"]:
        stats["history_changed_rows"] += 1
    normalized_rows.append(updated)

  return normalized_rows, stats


def main() -> None:
  parser = argparse.ArgumentParser(
      description=(
          "Normalize Q fields in an existing AndroidWorld QKV JSON file. "
          "The Q normalizer compacts UI elements first, then rebuilds history "
          "from key_string when possible."
      )
  )
  parser.add_argument(
      "--input-json",
      default=str(REPO_ROOT / "data" / "out" / "version2" / "qkv_6000_deepseek_key.json"),
      help="Input QKV JSON file.",
  )
  parser.add_argument(
      "--output-json",
      help="Output JSON file. Defaults to <input>_normalized.json.",
  )
  parser.add_argument(
      "--source-q-json",
      help=(
          "Optional original QKV JSON. When provided, Q is restored by row name "
          "from this file before normalization while keeping distilled A/reason/key_string."
      ),
  )
  args = parser.parse_args()

  input_path = Path(args.input_json)
  output_path = Path(args.output_json) if args.output_json else _default_output_path(input_path)
  rows = _read_json(input_path)
  if not isinstance(rows, list):
    raise ValueError("input JSON must be a list of QKV rows")
  source_rows_by_name = None
  if args.source_q_json:
    source_rows = _read_json(Path(args.source_q_json))
    if not isinstance(source_rows, list):
      raise ValueError("source-q-json must be a list of QKV rows")
    source_rows_by_name = _rows_by_name(source_rows)

  normalized_rows, stats = normalize_qkv_rows(rows, source_rows_by_name)
  _write_json(output_path, normalized_rows)
  print(json.dumps({"output_json": str(output_path), **stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
