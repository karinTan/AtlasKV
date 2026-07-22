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


def normalize_qkv_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
  stats = {
      "total_rows": 0,
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
    old_sections = extract_q_sections(old_q)
    if old_sections is None:
      stats["unparseable_q_rows"] += 1
      normalized_rows.append(updated)
      continue

    new_q = normalize_qkv_question(old_q, str(row.get("key_string") or ""))
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
  args = parser.parse_args()

  input_path = Path(args.input_json)
  output_path = Path(args.output_json) if args.output_json else _default_output_path(input_path)
  rows = _read_json(input_path)
  if not isinstance(rows, list):
    raise ValueError("input JSON must be a list of QKV rows")

  normalized_rows, stats = normalize_qkv_rows(rows)
  _write_json(output_path, normalized_rows)
  print(json.dumps({"output_json": str(output_path), **stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
