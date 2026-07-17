#!/usr/bin/env python3
"""Summarize generated AndroidWorld QKV data."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
  sys.path.insert(0, str(SRC_ROOT))

from atlaskv.android_world.protocol import (  # pylint: disable=wrong-import-position
    AndroidWorldOutputError,
    normalize_and_validate_action_output,
)


_QKV_NAME_RE = re.compile(r'^aw_(?P<episode>.*)_(?P<step>\d+)(?:_(?P<suffix>.*))?$')


def _read_json(path: Path) -> Any:
  with path.open(encoding='utf-8') as file:
    return json.load(file)


def _write_json(path: Path, data: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=2, sort_keys=True)
    file.write('\n')


def _parse_qkv_name(name: str) -> tuple[str, int | None, str]:
  match = _QKV_NAME_RE.match(name)
  if not match:
    return 'unknown', None, ''
  return (
      match.group('episode') or 'unknown',
      int(match.group('step')),
      match.group('suffix') or '',
  )


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
  return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _episode_length_stats(lengths: dict[str, int]) -> dict[str, Any]:
  if not lengths:
    return {
        'avg': 0.0,
        'min': 0,
        'max': 0,
        'min_episodes': [],
        'max_episodes': [],
    }
  values = list(lengths.values())
  min_value = min(values)
  max_value = max(values)
  return {
      'avg': round(sum(values) / len(values), 3),
      'min': min_value,
      'max': max_value,
      'min_episodes': sorted(
          episode for episode, value in lengths.items() if value == min_value
      )[:10],
      'max_episodes': sorted(
          episode for episode, value in lengths.items() if value == max_value
      )[:10],
  }


def summarize_qkv_rows(
    qkv_rows: list[dict[str, Any]],
    generation_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Returns aggregate stats for generated QKV rows."""
  description_type_counts: Counter[str] = Counter()
  action_counts: Counter[str] = Counter()
  status_goal_counts: Counter[str] = Counter()
  synthetic_counts: Counter[str] = Counter()
  qkv_rows_by_episode: Counter[str] = Counter()
  original_steps_by_episode: dict[str, set[int]] = defaultdict(set)
  malformed_rows: list[dict[str, str]] = []

  for index, row in enumerate(qkv_rows):
    name = str(row.get('name') or f'row_{index}')
    episode_id, base_step, suffix = _parse_qkv_name(name)
    qkv_rows_by_episode[episode_id] += 1
    if base_step is not None:
      original_steps_by_episode[episode_id].add(base_step)
    if 'keyboard_enter_after_input_text' in suffix:
      synthetic_counts['keyboard_enter_after_input_text'] += 1
    if 'terminal_complete' in suffix or suffix.endswith('status_complete'):
      synthetic_counts['terminal_complete'] += 1
    if suffix.endswith('_infeasible') or suffix.endswith('infeasible'):
      synthetic_counts['invalid_action_to_infeasible'] += 1

    description_type_counts[str(row.get('description_type') or 'missing')] += 1
    description = str(row.get('description') or row.get('A') or '')
    try:
      _, action = normalize_and_validate_action_output(description)
    except AndroidWorldOutputError as exc:
      action_counts['malformed'] += 1
      if len(malformed_rows) < 20:
        malformed_rows.append({'name': name, 'error': str(exc)})
      continue
    action_type = str(action.get('action_type') or 'missing')
    action_counts[action_type] += 1
    if action_type == 'status':
      status_goal_counts[str(action.get('goal_status') or 'missing')] += 1

  original_step_lengths = {
      episode_id: len(steps) for episode_id, steps in original_steps_by_episode.items()
  }
  qkv_row_lengths = dict(qkv_rows_by_episode)

  summary = {
      'total_qkv_rows': len(qkv_rows),
      'episode_count': len(qkv_rows_by_episode),
      'description_type_counts': _sorted_counter(description_type_counts),
      'action_type_counts': _sorted_counter(action_counts),
      'status_goal_counts': _sorted_counter(status_goal_counts),
      'synthetic_counts': _sorted_counter(synthetic_counts),
      'original_steps_per_episode': _episode_length_stats(original_step_lengths),
      'qkv_rows_per_episode': _episode_length_stats(qkv_row_lengths),
      'malformed_action_rows': malformed_rows,
  }
  if generation_stats:
    summary['generation_stats'] = generation_stats
  return summary


def _format_counter(title: str, values: dict[str, int]) -> list[str]:
  lines = [f'{title}:']
  if not values:
    lines.append('  none: 0')
    return lines
  for key, value in values.items():
    lines.append(f'  {key}: {value}')
  return lines


def _format_episode_stats(title: str, values: dict[str, Any]) -> str:
  return (
      f'{title}: avg={values["avg"]}, min={values["min"]} '
      f'({", ".join(values["min_episodes"]) or "n/a"}), '
      f'max={values["max"]} ({", ".join(values["max_episodes"]) or "n/a"})'
  )


def format_summary(summary: dict[str, Any]) -> str:
  """Formats a QKV summary for terminal output."""
  lines = [
      'QKV generation summary',
      f'total_qkv_rows: {summary["total_qkv_rows"]}',
      f'episode_count: {summary["episode_count"]}',
  ]

  generation_stats = summary.get('generation_stats')
  if generation_stats:
    lines.append('generation_stats:')
    for key in sorted(generation_stats):
      lines.append(f'  {key}: {generation_stats[key]}')

  lines.extend(_format_counter('description_type_counts', summary['description_type_counts']))
  lines.extend(_format_counter('action_type_counts', summary['action_type_counts']))
  lines.extend(_format_counter('status_goal_counts', summary['status_goal_counts']))
  lines.extend(_format_counter('synthetic_counts', summary['synthetic_counts']))
  lines.append(
      _format_episode_stats(
          'original_steps_per_episode',
          summary['original_steps_per_episode'],
      )
  )
  lines.append(
      _format_episode_stats(
          'qkv_rows_per_episode',
          summary['qkv_rows_per_episode'],
      )
  )
  malformed_rows = summary.get('malformed_action_rows') or []
  if malformed_rows:
    lines.append('malformed_action_rows:')
    for row in malformed_rows:
      lines.append(f'  {row["name"]}: {row["error"]}')
  return '\n'.join(lines)


def print_summary(summary: dict[str, Any]) -> None:
  print(format_summary(summary))


def write_summary(path: Path, summary: dict[str, Any]) -> None:
  _write_json(path, summary)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description='Summarize AndroidWorld qkv.json.')
  parser.add_argument(
      '--qkv-json',
      default=str(REPO_ROOT / 'data' / 'out' / 'qkv.json'),
      help='Path to generated qkv.json.',
  )
  parser.add_argument(
      '--summary-json',
      help='Optional path for writing the structured summary JSON.',
  )
  return parser


def main() -> None:
  args = build_parser().parse_args()
  rows = _read_json(Path(args.qkv_json))
  if not isinstance(rows, list):
    raise ValueError('qkv JSON must be an array')
  summary = summarize_qkv_rows(rows)
  print_summary(summary)
  if args.summary_json:
    _write_json(Path(args.summary_json), summary)


if __name__ == '__main__':
  main()
