#!/usr/bin/env python3
"""Distill AndroidWorld AtlasKV key_string values with DeepSeek."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
  sys.path.insert(0, str(SRC_ROOT))

from atlaskv.android_world.ui_elements import (  # pylint: disable=wrong-import-position
    compact_android_world_ui_elements,
)
from atlaskv.android_world.protocol import (  # pylint: disable=wrong-import-position
    normalize_and_validate_action_output,
)

_UI_SECTION_END_PATTERNS = [
    r'\n\nAllowed action JSON shapes:',
    r'\n\nValid action_type values:',
    r'\n\nPlease answer in exactly this format:',
]


SYSTEM_PROMPT = """You write AtlasKV key_string and reason values for AndroidWorld action-selection examples.

The key_string is a compact query-side semantic key. It is used for matching the current prompt to a value embedding. It must not contain the correct answer.
The reason is a value-side rationale. It will be placed before the supplied target Action in the final V/description field.

Rules for key_string:
- Use only the goal, history, and visible UI elements provided by the user.
- Infer a concise current screen description from the visible UI elements.
- Ignore Android system status bar, Android system navigation bar, notifications, generic containers, machine ids, duplicated controls, and other UI noise unless task-relevant.
- Do not include or infer the target action, action_type, target index, app_name to open, or final answer.
- Write one natural English sentence.
- Use a short history phrase. If no previous action has been performed, write "no previous actions".
- The sentence must follow this shape:
  For the AndroidWorld goal of {goal}, after {history}, the current screen shows {current state}, and the next action should be
- End the sentence with the phrase "and the next action should be". Do not add the answer after "should be".

Rules for reason:
- Use the supplied Target Action only to explain why that action is correct.
- Ground the reason in the goal, history, visible UI elements, and Target Action.
- Write one brief natural English sentence.
- Mention the target UI index, text, app name, direction, answer, or status only when that field appears in the Target Action.
- Do not include a JSON object.

General output rules:
- Do not mention "Reason:" or "Action:" inside JSON field values.
- Do not use markdown, bullets, pipe separators, or extra text.
- Return valid JSON only: {"key_string":"...","reason":"..."}"""

ONE_SHOT_USER = """Row name: aw_0_0_open_app
Goal: Open the Zoho Meet app, view the scheduled meetings.
History: You just started, no action has been performed yet.
Visible UI elements excerpt:
UI element 12: UIElement(content_description='Create new event or other calendar entries', is_clickable)
UI element 13: UIElement(content_description='Show Calendar List and Settings drawer', is_clickable)
UI element 53: UIElement(text='July')
Target Action:
Action: {"action_type":"open_app","app_name":"Zoho Meet"}

Write the AtlasKV key_string and reason."""

ONE_SHOT_ASSISTANT = json.dumps(
    {
        'key_string': (
            'For the AndroidWorld goal of opening the Zoho Meet app and '
            'viewing the scheduled meetings, after no previous actions, '
            'the current screen shows Google Calendar with July calendar '
            'entries and visible calendar controls, and the next action '
            'should be'
        ),
        'reason': (
            'The task requires Zoho Meet but the current screen is Google '
            'Calendar, so the next step is to open Zoho Meet.'
        ),
    },
    ensure_ascii=False,
)


def _read_json(path: Path) -> Any:
  with path.open(encoding='utf-8') as file:
    return json.load(file)


def _write_json(path: Path, data: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=2)
    file.write('\n')


def _extract_section(text: str, start_pattern: str, end_patterns: list[str]) -> str:
  start = re.search(start_pattern, text, flags=re.IGNORECASE | re.DOTALL)
  if not start:
    return ''
  start_idx = start.end()
  end_idx = len(text)
  for end_pattern in end_patterns:
    end = re.search(end_pattern, text[start_idx:], flags=re.IGNORECASE | re.DOTALL)
    if end:
      end_idx = min(end_idx, start_idx + end.start())
  return text[start_idx:end_idx].strip()


def _shorten(text: str, limit: int) -> str:
  text = text.strip()
  if len(text) <= limit:
    return text
  return text[:limit].rstrip() + '\n...[truncated]'


def _target_action_json(row: dict[str, Any]) -> str:
  output = str(row.get('A') or row.get('description') or '')
  _, action = normalize_and_validate_action_output(output)
  return json.dumps(action, ensure_ascii=False, separators=(',', ':'))


def _prepare_ui_excerpt(
    ui_elements: str,
    max_chars: int,
    include_system_ui: bool = False,
) -> str:
  del include_system_ui
  raw_lines = [line for line in ui_elements.splitlines() if line.strip()]
  filtered = compact_android_world_ui_elements(ui_elements)
  filtered_lines = filtered.splitlines()
  if filtered_lines:
    excerpt = '\n'.join(filtered_lines)
    dropped = max(0, len(raw_lines) - len(filtered_lines))
  else:
    excerpt = '\n'.join(raw_lines)
    dropped = 0

  prefix = ''
  if dropped:
    prefix = f'[Filtered out {dropped} generic UI lines.]\n'
  return _shorten(prefix + excerpt, max_chars)


def _row_source(
    row: dict[str, Any],
    max_ui_chars: int,
    include_system_ui: bool = False,
) -> dict[str, str]:
  q = str(row.get('Q') or '')
  goal = _extract_section(
      q,
      r'The current AndroidWorld user goal is:\s*',
      [r'\nHistory:'],
  )
  history = _extract_section(
      q,
      r'\nHistory:\s*',
      [r'\nThe visible UI elements are:'],
  )
  ui_elements = _extract_section(
      q,
      r'\nThe visible UI elements are:\s*',
      _UI_SECTION_END_PATTERNS,
  )
  return {
      'name': str(row.get('name') or ''),
      'goal': goal or 'Unknown goal.',
      'history': history or 'No previous action.',
      'ui_elements_excerpt': _prepare_ui_excerpt(
          ui_elements,
          max_ui_chars,
          include_system_ui=include_system_ui,
      ),
      'target_action': _target_action_json(row),
  }


def _user_prompt(source: dict[str, str], retry_error: str | None = None) -> str:
  retry_note = ''
  if retry_error:
    retry_note = (
        '\nThe previous answer was invalid because: '
        f'{retry_error}\nReturn corrected JSON only.\n'
    )
  return f"""Row name: {source['name']}
Goal: {source['goal']}
History: {source['history']}
Visible UI elements excerpt:
{source['ui_elements_excerpt']}
Target Action:
Action: {source['target_action']}
{retry_note}
Write the AtlasKV key_string and reason."""


def _messages(source: dict[str, str], retry_error: str | None = None) -> list[dict[str, str]]:
  return [
      {'role': 'system', 'content': SYSTEM_PROMPT},
      {'role': 'user', 'content': ONE_SHOT_USER},
      {'role': 'assistant', 'content': ONE_SHOT_ASSISTANT},
      {'role': 'user', 'content': _user_prompt(source, retry_error)},
  ]


def _chat_completion(args: argparse.Namespace, messages: list[dict[str, str]]) -> str:
  api_key = os.environ.get(args.api_key_env)
  if not api_key:
    raise RuntimeError(f'Environment variable {args.api_key_env} is not set')

  payload: dict[str, Any] = {
      'model': args.model,
      'messages': messages,
      'temperature': args.temperature,
      'max_tokens': args.max_tokens,
      'stream': False,
  }
  if args.thinking != 'omit':
    payload['thinking'] = {'type': args.thinking}
  if args.reasoning_effort:
    payload['reasoning_effort'] = args.reasoning_effort

  url = args.base_url.rstrip('/') + '/chat/completions'
  request = urllib.request.Request(
      url,
      data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
      headers={
          'Content-Type': 'application/json',
          'Authorization': f'Bearer {api_key}',
      },
      method='POST',
  )
  try:
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
      data = json.loads(response.read().decode('utf-8'))
  except urllib.error.HTTPError as exc:
    body = exc.read().decode('utf-8', errors='replace')
    raise RuntimeError(f'DeepSeek HTTP {exc.code}: {body[:1000]}') from exc

  choices = data.get('choices') or []
  if not choices:
    raise RuntimeError(f'DeepSeek response had no choices: {data}')
  message = choices[0].get('message') or {}
  content = message.get('content')
  if not isinstance(content, str) or not content.strip():
    raise RuntimeError(f'DeepSeek response had empty content: {data}')
  return content.strip()


def _balanced_json_object(text: str) -> str | None:
  start = text.find('{')
  while start != -1:
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
      char = text[idx]
      if in_string:
        if escaped:
          escaped = False
        elif char == '\\':
          escaped = True
        elif char == '"':
          in_string = False
        continue
      if char == '"':
        in_string = True
      elif char == '{':
        depth += 1
      elif char == '}':
        depth -= 1
        if depth == 0:
          return text[start:idx + 1]
    start = text.find('{', start + 1)
  return None


def _parse_distillation(
    text: str,
    max_reason_chars: int,
) -> tuple[str | None, str | None, str | None]:
  json_text = _balanced_json_object(text)
  if not json_text:
    return None, None, 'no JSON object found'
  try:
    data = json.loads(json_text)
  except json.JSONDecodeError as exc:
    return None, None, f'invalid JSON: {exc}'
  if not isinstance(data, dict):
    return None, None, 'JSON root is not an object'
  key = data.get('key_string') or data.get('key')
  if not isinstance(key, str):
    return None, None, 'JSON did not contain string field key_string'
  key = ' '.join(key.split()).strip()
  if key.endswith('.'):
    key = key[:-1].rstrip()
  error = _validate_key_string(key)
  if error:
    return None, None, error

  reason = data.get('reason')
  if not isinstance(reason, str):
    return None, None, 'JSON did not contain string field reason'
  reason = ' '.join(reason.split()).strip()
  if reason and reason[-1] not in '.!?':
    reason += '.'
  error = _validate_reason(reason, max_reason_chars)
  if error:
    return None, None, error
  return key, reason, None


def _validate_key_string(key: str) -> str | None:
  if not key:
    return 'key_string is empty'
  if '|' in key:
    return 'key_string contains pipe separators'
  if '\n' in key or '\r' in key:
    return 'key_string is not one line'
  if 'Reason:' in key or 'Action:' in key or '{"action_type"' in key:
    return 'key_string appears to contain the target answer'
  if not key.startswith('For the AndroidWorld goal of '):
    return 'key_string does not start with the required template'
  if ', after ' not in key:
    return 'key_string does not contain ", after "'
  if ', the current screen shows ' not in key:
    return 'key_string does not contain ", the current screen shows "'
  if not key.endswith(', and the next action should be'):
    return 'key_string must end with ", and the next action should be"'
  return None


def _validate_reason(reason: str, max_chars: int) -> str | None:
  if not reason:
    return 'reason is empty'
  if '\n' in reason or '\r' in reason:
    return 'reason is not one line'
  if 'Reason:' in reason or 'Action:' in reason:
    return 'reason contains a Reason or Action marker'
  if '{' in reason or '}' in reason or '"action_type"' in reason:
    return 'reason appears to contain an action JSON object'
  if len(reason) > max_chars:
    return f'reason is too long ({len(reason)} > {max_chars} chars)'
  return None


def _distill_one(row: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any] | None, str | None]:
  source = _row_source(row, args.max_ui_chars, args.include_system_ui)
  retry_error = None
  last_error = None
  for attempt in range(1, args.retries + 1):
    try:
      content = _chat_completion(args, _messages(source, retry_error))
      key, reason, error = _parse_distillation(content, args.max_reason_chars)
      if key and reason:
        updated = dict(row)
        updated['key_string'] = key
        updated['reason'] = reason
        description = f'Reason: {reason}\nAction: {source["target_action"]}'
        updated['description'] = description
        updated['A'] = description
        return updated, None
      last_error = error or 'unknown parse error'
      retry_error = last_error
    except Exception as exc:  # pylint: disable=broad-except
      last_error = str(exc)
      retry_error = last_error
    if attempt < args.retries and args.retry_sleep > 0:
      time.sleep(args.retry_sleep)
  return None, last_error


def distill_file(args: argparse.Namespace) -> None:
  rows = _read_json(Path(args.input_json))
  if not isinstance(rows, list):
    raise ValueError('input JSON must be a list of QKV rows')

  output_path = Path(args.output_json) if args.output_json else _default_output_path(Path(args.input_json))
  distilled_rows: list[dict[str, Any]] = []
  failed_rows: list[dict[str, Any]] = []

  selected_rows = rows[args.start:]
  if args.limit is not None:
    selected_rows = selected_rows[:args.limit]

  for offset, row in enumerate(selected_rows, start=args.start):
    if not isinstance(row, dict):
      failed_rows.append({'index': offset, 'error': 'row is not an object'})
      continue

    if args.dry_run:
      source = _row_source(row, args.max_ui_chars, args.include_system_ui)
      print(_user_prompt(source))
      return

    updated, error = _distill_one(row, args)
    if updated is None:
      failed_rows.append({
          'index': offset,
          'name': row.get('name'),
          'error': error or 'unknown failure',
      })
      print(f'SKIP index={offset} name={row.get("name")} error={error}', file=sys.stderr)
      continue

    distilled_rows.append(updated)
    if args.checkpoint_every > 0 and len(distilled_rows) % args.checkpoint_every == 0:
      _write_json(output_path, distilled_rows)
      print(f'Checkpoint wrote {len(distilled_rows)} rows to {output_path}')

  _write_json(output_path, distilled_rows)
  if args.failed_json:
    _write_json(Path(args.failed_json), failed_rows)

  print(
      f'Wrote {len(distilled_rows)} distilled rows to {output_path}; '
      f'skipped {len(failed_rows)} rows.'
  )


def _default_output_path(input_path: Path) -> Path:
  return input_path.with_name(input_path.stem + '_deepseek_key.json')


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description='Use DeepSeek to distill AtlasKV key_string values for AndroidWorld QKV rows.'
  )
  parser.add_argument(
      '--input-json',
      default=str(REPO_ROOT / 'data' / 'out' / 'android_control_seed_qkv.json'),
      help='Input QKV JSON file.',
  )
  parser.add_argument(
      '--output-json',
      help='Output QKV JSON file. Defaults to <input>_deepseek_key.json.',
  )
  parser.add_argument(
      '--failed-json',
      help='Optional JSON file for skipped rows and errors.',
  )
  parser.add_argument('--api-key-env', default='DEEPSEEK_API_KEY')
  parser.add_argument('--base-url', default='https://api.deepseek.com')
  parser.add_argument('--model', default='deepseek-v4-pro')
  parser.add_argument('--thinking', choices=['disabled', 'enabled', 'omit'], default='disabled')
  parser.add_argument('--reasoning-effort', choices=['high', 'max'], default=None)
  parser.add_argument('--temperature', type=float, default=0.0)
  parser.add_argument('--max-tokens', type=int, default=500)
  parser.add_argument('--timeout', type=int, default=120)
  parser.add_argument('--retries', type=int, default=3)
  parser.add_argument('--retry-sleep', type=float, default=1.0)
  parser.add_argument('--checkpoint-every', type=int, default=20)
  parser.add_argument('--max-ui-chars', type=int, default=8000)
  parser.add_argument('--max-reason-chars', type=int, default=320)
  parser.add_argument(
      '--include-system-ui',
      action='store_true',
      help='Keep Android system UI lines in the DeepSeek UI excerpt.',
  )
  parser.add_argument('--start', type=int, default=0)
  parser.add_argument('--limit', type=int)
  parser.add_argument(
      '--dry-run',
      action='store_true',
      help='Print the first prompt that would be sent and exit without calling the API.',
  )
  return parser


def main() -> None:
  args = build_parser().parse_args()
  if args.retries < 1:
    raise ValueError('--retries must be at least 1')
  if args.max_reason_chars < 1:
    raise ValueError('--max-reason-chars must be at least 1')
  distill_file(args)


if __name__ == '__main__':
  main()
