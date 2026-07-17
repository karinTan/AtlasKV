#!/usr/bin/env python3
"""Profile raw and converted AndroidWorld actions before QKV generation."""

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
    validate_action,
)


_UI_ELEMENT_RE = re.compile(r'\bUI\s+element\s+(\d+)\s*:', re.IGNORECASE)


def _read_json(path: Path) -> Any:
  with path.open(encoding='utf-8') as file:
    return json.load(file)


def _write_json(path: Path, data: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=2, sort_keys=True)
    file.write('\n')


def _field_type(value: Any) -> str:
  if value is None:
    return 'null'
  if isinstance(value, bool):
    return 'bool'
  if isinstance(value, int):
    return 'int'
  if isinstance(value, float):
    return 'float'
  if isinstance(value, str):
    return 'str'
  if isinstance(value, list):
    return 'list'
  if isinstance(value, dict):
    return 'dict'
  return type(value).__name__


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
  return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _visible_indices(ui_elements_description: str) -> frozenset[int] | None:
  indices = frozenset(
      int(match.group(1)) for match in _UI_ELEMENT_RE.finditer(ui_elements_description)
  )
  return indices or None


def _validate_qkv_action(
    action: dict[str, Any],
    visible_indices: frozenset[int] | None,
) -> tuple[bool, str | None]:
  try:
    normalized = validate_action(action, visible_indices)
    if normalized.get('action_type') in {'click', 'long_press'} and 'index' not in normalized:
      raise AndroidWorldOutputError(
          'click/long_press actions must use an index for QKV data.',
          'missing_index_action',
      )
  except AndroidWorldOutputError as exc:
    return False, f'{exc.code}: {exc}'
  return True, None


def _update_action_summary(
    summary: dict[str, Any],
    action: dict[str, Any],
    example: dict[str, Any],
    max_examples: int,
) -> None:
  action_type = str(action.get('action_type') or 'missing')
  details = summary.setdefault(
      action_type,
      {
          'count': 0,
          'field_counts': Counter(),
          'field_type_counts': defaultdict(Counter),
          'examples': [],
      },
  )
  details['count'] += 1
  for key, value in action.items():
    details['field_counts'][str(key)] += 1
    details['field_type_counts'][str(key)][_field_type(value)] += 1
  if len(details['examples']) < max_examples:
    details['examples'].append(example)


def _finalize_action_summary(summary: dict[str, Any]) -> dict[str, Any]:
  finalized = {}
  for action_type, details in sorted(summary.items()):
    finalized[action_type] = {
        'count': details['count'],
        'field_counts': _sorted_counter(details['field_counts']),
        'field_type_counts': {
            field: _sorted_counter(counter)
            for field, counter in sorted(details['field_type_counts'].items())
        },
        'examples': details['examples'],
    }
  return finalized


def summarize_prompt_rows(
    prompt_rows: list[dict[str, Any]],
    max_examples: int = 3,
) -> dict[str, Any]:
  """Profiles original actions, converted actions, and converted validity."""
  original_summary: dict[str, Any] = {}
  converted_summary: dict[str, Any] = {}
  validity_counts: Counter[str] = Counter()
  conversion_error_counts: Counter[str] = Counter()
  conversion_error_examples: list[dict[str, Any]] = []
  invalid_examples: list[dict[str, Any]] = []

  for row in prompt_rows:
    original_action = row.get('original_action') or {}
    converted_action = row.get('target_action') or {}
    base_example = {
        'episode_id': row.get('episode_id'),
        'step_index': row.get('step_index'),
        'step_instruction': row.get('step_instruction'),
    }
    conversion_error = row.get('conversion_error')
    if conversion_error:
      conversion_error_counts[str(conversion_error)] += 1
      if len(conversion_error_examples) < max_examples:
        conversion_error_examples.append(
            {
                **base_example,
                'original_action': original_action,
                'converted_action': converted_action,
                'conversion_error': conversion_error,
            }
        )
    _update_action_summary(
        original_summary,
        original_action,
        {**base_example, 'action': original_action},
        max_examples,
    )
    _update_action_summary(
        converted_summary,
        converted_action,
        {**base_example, 'action': converted_action},
        max_examples,
    )
    visible_indices = _visible_indices(row.get('ui_elements_description') or '')
    ok, error = _validate_qkv_action(converted_action, visible_indices)
    if ok:
      validity_counts['valid'] += 1
    else:
      validity_counts['invalid'] += 1
      if len(invalid_examples) < max_examples:
        invalid_examples.append(
            {
                **base_example,
                'original_action': original_action,
                'converted_action': converted_action,
                'error': error,
            }
        )

  return {
      'total_actions': len(prompt_rows),
      'original_action_type_counts': _sorted_counter(
          Counter(str((row.get('original_action') or {}).get('action_type') or 'missing')
                  for row in prompt_rows)
      ),
      'converted_action_type_counts': _sorted_counter(
          Counter(str((row.get('target_action') or {}).get('action_type') or 'missing')
                  for row in prompt_rows)
      ),
      'converted_qkv_validity_counts': _sorted_counter(validity_counts),
      'conversion_error_counts': _sorted_counter(conversion_error_counts),
      'original_action_details': _finalize_action_summary(original_summary),
      'converted_action_details': _finalize_action_summary(converted_summary),
      'conversion_error_examples': conversion_error_examples,
      'invalid_converted_examples': invalid_examples,
  }


def _format_counts(title: str, values: dict[str, int]) -> list[str]:
  lines = [f'{title}:']
  for key, value in values.items():
    lines.append(f'  {key}: {value}')
  if len(lines) == 1:
    lines.append('  none: 0')
  return lines


def format_summary(profile: dict[str, Any]) -> str:
  lines = [
      'Action profile summary',
      f'total_actions: {profile["total_actions"]}',
  ]
  lines.extend(_format_counts('original_action_type_counts', profile[
      'original_action_type_counts'
  ]))
  lines.extend(_format_counts('converted_action_type_counts', profile[
      'converted_action_type_counts'
  ]))
  lines.extend(_format_counts('converted_qkv_validity_counts', profile[
      'converted_qkv_validity_counts'
  ]))
  lines.extend(_format_counts('conversion_error_counts', profile.get(
      'conversion_error_counts', {}
  )))
  conversion_examples = profile.get('conversion_error_examples') or []
  if conversion_examples:
    lines.append('conversion_error_examples:')
    for example in conversion_examples:
      lines.append(
          '  episode={episode_id} step={step_index} error={conversion_error} '
          'original={original_action} converted={converted_action}'.format(
              **example
          )
      )
  invalid_examples = profile.get('invalid_converted_examples') or []
  if invalid_examples:
    lines.append('invalid_converted_examples:')
    for example in invalid_examples:
      lines.append(
          '  episode={episode_id} step={step_index} error={error} '
          'original={original_action} converted={converted_action}'.format(
              **example
          )
      )
  return '\n'.join(lines)


def print_summary(profile: dict[str, Any]) -> None:
  print(format_summary(profile))


def write_summary(path: Path, profile: dict[str, Any]) -> None:
  _write_json(path, profile)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description='Profile actions in prompts.json.')
  parser.add_argument(
      '--prompts-json',
      default=str(REPO_ROOT / 'data' / 'out' / 'prompts.json'),
      help='Path to prompt rows containing original_action and target_action.',
  )
  parser.add_argument(
      '--profile-json',
      help='Optional path for writing the structured action profile JSON.',
  )
  parser.add_argument('--max-examples', type=int, default=3)
  return parser


def main() -> None:
  args = build_parser().parse_args()
  rows = _read_json(Path(args.prompts_json))
  if not isinstance(rows, list):
    raise ValueError('prompts JSON must be an array')
  profile = summarize_prompt_rows(rows, max_examples=args.max_examples)
  print_summary(profile)
  if args.profile_json:
    write_summary(Path(args.profile_json), profile)


if __name__ == '__main__':
  main()
