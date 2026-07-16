#!/usr/bin/env python3
"""Entry point for converting Android Control TFRecords into AW prompts/QKV."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Iterable

PACKAGE_PARENT = Path(__file__).resolve().parent.parent
if str(PACKAGE_PARENT) not in sys.path:
  sys.path.insert(0, str(PACKAGE_PARENT))

from offline_android_world_prompt import build_qkv
from offline_android_world_prompt import prompt_utils
from offline_android_world_prompt import representation_utils


def _require_tensorflow():
  try:
    import tensorflow as tf  # pylint: disable=import-outside-toplevel
  except ModuleNotFoundError as exc:
    raise SystemExit(
        'TensorFlow is required to read TFRecord files. Run this in the same '
        'environment where /Users/kailing.tan/AtlasKV/data.py works.'
    ) from exc
  return tf


def _new_forest_proto():
  try:
    from android_env.proto.a11y import (  # pylint: disable=import-outside-toplevel
        android_accessibility_forest_pb2,
    )
  except ModuleNotFoundError as exc:
    raise SystemExit(
        'android_env is required to parse accessibility_trees bytes. Run this '
        'in the same environment where /Users/kailing.tan/AtlasKV/data.py works.'
    ) from exc
  return android_accessibility_forest_pb2.AndroidAccessibilityForest()


def _bytes_list(features: Any, name: str) -> list[bytes]:
  if name not in features:
    return []
  return list(features[name].bytes_list.value)


def _int_list(features: Any, name: str) -> list[int]:
  if name not in features:
    return []
  return list(features[name].int64_list.value)


def _decode_text(value: bytes) -> str:
  return value.decode('utf-8')


def _decode_json_list(values: list[bytes]) -> list[dict[str, Any]]:
  return [json.loads(_decode_text(value)) for value in values]


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('w', encoding='utf-8') as out:
    json.dump(rows, out, ensure_ascii=False, indent=2)
    out.write('\n')


def _load_teacher_histories(path: str | None) -> dict[str, list[str]]:
  if not path:
    return {}
  with open(path, encoding='utf-8') as f:
    data = json.load(f)
  if not isinstance(data, dict):
    raise ValueError('teacher history JSON must be an object keyed by episode_id')
  return {str(key): [str(item) for item in value] for key, value in data.items()}


def _iter_examples(files: list[str]) -> Iterable[tuple[str, int, Any]]:
  tf = _require_tensorflow()
  for file_path in files:
    dataset = tf.data.TFRecordDataset([file_path], compression_type='GZIP')
    for record_index, raw_record in enumerate(dataset):
      example = tf.train.Example.FromString(raw_record.numpy())
      yield file_path, record_index, example


def _parse_forest(tree_bytes: bytes) -> Any:
  forest = _new_forest_proto()
  forest.ParseFromString(tree_bytes)
  return forest


def _screen_size(
    widths: list[int],
    heights: list[int],
    step_index: int,
) -> tuple[int, int]:
  if step_index < len(widths) and step_index < len(heights):
    return widths[step_index], heights[step_index]
  if widths and heights:
    return widths[0], heights[0]
  return 1080, 2400


def _convert_action(
    action: dict[str, Any],
    ui_elements: list[representation_utils.UIElement],
    screen_size: tuple[int, int],
) -> dict[str, Any]:
  """Converts coordinate actions to Android World index actions when possible."""
  converted = dict(action)
  action_type = converted.get('action_type')
  if action_type in {'click', 'long_press', 'input_text'}:
    if 'index' not in converted and 'x' in converted and 'y' in converted:
      index = representation_utils.find_element_index_for_point(
          ui_elements, screen_size, converted['x'], converted['y']
      )
      if index is not None:
        converted['index'] = index
        converted.pop('x', None)
        converted.pop('y', None)
  return converted


def _history_lines(
    episode_id: str,
    step_index: int,
    agent: str,
    prior_actions: list[dict[str, Any]],
    step_instructions: list[str],
    teacher_histories: dict[str, list[str]],
) -> list[str]:
  prefix = 'Step {step}- ' if agent == 'm3a' else 'Step {step}: '
  teacher = teacher_histories.get(episode_id, [])
  lines = []

  for i in range(step_index):
    if i < len(teacher):
      summary = teacher[i]
    else:
      instruction = step_instructions[i] if i < len(step_instructions) else None
      action = prior_actions[i] if i < len(prior_actions) else {}
      summary = prompt_utils.action_to_history_summary(action, instruction)
    lines.append(prefix.format(step=i + 1) + summary)
  return lines


def _goal_for_step(
    episode_goal: str,
    step_instructions: list[str],
    step_index: int,
    goal_mode: str,
) -> str:
  if goal_mode == 'step_instruction' and step_index < len(step_instructions):
    return step_instructions[step_index]
  return episode_goal


def process_examples(args: argparse.Namespace) -> int:
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  prompts_path = output_dir / 'prompts.json'
  qkv_path = Path(args.qkv_output_json) if args.qkv_output_json else output_dir / 'qkv.json'
  teacher_histories = _load_teacher_histories(args.teacher_history_json)

  files = sorted(glob.glob(args.input_glob))
  if not files:
    raise FileNotFoundError(f'No files matched {args.input_glob!r}')

  rows_written = 0
  records_seen = 0
  rows: list[dict[str, Any]] = []
  for source_file, record_index, example in _iter_examples(files):
    if args.max_records is not None and records_seen >= args.max_records:
      break
    records_seen += 1

    features = example.features.feature
    episode_ids = _int_list(features, 'episode_id')
    episode_id = str(episode_ids[0]) if episode_ids else str(record_index)
    episode_goal_values = _bytes_list(features, 'goal')
    episode_goal = _decode_text(episode_goal_values[0]) if episode_goal_values else ''
    accessibility_trees = _bytes_list(features, 'accessibility_trees')
    widths = _int_list(features, 'screenshot_widths')
    heights = _int_list(features, 'screenshot_heights')
    actions = _decode_json_list(_bytes_list(features, 'actions'))
    step_instructions = [
      _decode_text(value) for value in _bytes_list(features, 'step_instructions')
    ]

    num_steps = min(len(accessibility_trees), len(actions))
    converted_actions: list[dict[str, Any]] = []

    for step_index in range(num_steps):
      forest = _parse_forest(accessibility_trees[step_index])
      screen_size = _screen_size(widths, heights, step_index)
      ui_elements = representation_utils.forest_to_ui_elements(
          forest,
          exclude_invisible_elements=True,
      )
      target_action = _convert_action(actions[step_index], ui_elements, screen_size)
      converted_actions.append(target_action)

      if args.agent == 'm3a':
        ui_text = prompt_utils.generate_m3a_ui_elements_description_list(
            ui_elements, screen_size
        )
      else:
        ui_text = prompt_utils.generate_t3a_ui_elements_description_list_full(
            ui_elements, screen_size
        )

      current_goal = _goal_for_step(
          episode_goal, step_instructions, step_index, args.goal_mode
      )
      history = _history_lines(
          episode_id,
          step_index,
          args.agent,
          converted_actions,
          step_instructions,
          teacher_histories,
      )
      prompt = ''
      if args.output_format in {'prompts', 'both'}:
        prompt = prompt_utils.action_selection_prompt(
            current_goal,
            history,
            ui_text,
            agent=args.agent,
        )

      row = {
        'source_file': source_file,
        'record_index': record_index,
        'episode_id': episode_id,
        'step_index': step_index,
        'episode_goal': episode_goal,
        'goal': current_goal,
        'step_instruction': (
          step_instructions[step_index]
          if step_index < len(step_instructions)
          else None
        ),
        'history': (
          '\n'.join(history)
          if history
          else 'You just started, no action has been performed yet.'
        ),
        'ui_elements_description': ui_text,
        'prompt': prompt,
        'original_action': actions[step_index],
        'target_action': target_action,
      }
      rows.append(row)
      rows_written += 1

  if args.output_format in {'prompts', 'both'}:
    _write_json(prompts_path, rows)
    print(f'Wrote {rows_written} prompt rows to {prompts_path}')

  if args.output_format in {'qkv', 'both'}:
    qkv_rows, stats = build_qkv.build_qkv_rows(
        rows,
        invalid_action_policy=args.invalid_action_policy,
        keyboard_enter_after_input_text=args.keyboard_enter_after_input_text,
        terminal_complete=args.terminal_complete,
    )
    _write_json(qkv_path, qkv_rows)
    invalid_note = (
        'invalid skipped'
        if args.invalid_action_policy == 'skip'
        else 'invalid -> status infeasible'
    )
    print(
        'Wrote {qkv_rows} QKV rows to {qkv_path} '
        'from {prompt_rows} prompt rows '
        '({invalid_actions} {invalid_note}, '
        '{keyboard_enter_rows} keyboard_enter, '
        '{terminal_rows} terminal complete).'.format(
            qkv_path=qkv_path,
            invalid_note=invalid_note,
            **stats,
        )
    )

  return rows_written


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input-glob',
      required=True,
      help='Glob for Android Control TFRecord files, e.g. "/path/android_control*"',
  )
  parser.add_argument('--output-dir', required=True)
  parser.add_argument(
      '--output-format',
      choices=['prompts', 'qkv', 'both'],
      default='prompts',
      help='Write prompt rows, final QKV rows, or both. qkv does not save prompts.json.',
  )
  parser.add_argument(
      '--qkv-output-json',
      help='QKV output path. Defaults to <output-dir>/qkv.json.',
  )
  parser.add_argument('--agent', choices=['t3a', 'm3a'], default='t3a')
  parser.add_argument(
      '--goal-mode',
      choices=['episode', 'step_instruction'],
      default='episode',
      help='Use episode goal like Android World, or current step instruction.',
  )
  parser.add_argument('--teacher-history-json')
  parser.add_argument(
      '--invalid-action-policy',
      choices=['status_infeasible', 'skip'],
      default='status_infeasible',
      help='How QKV output handles target actions that fail validation.',
  )
  parser.add_argument(
      '--no-keyboard-enter-after-input-text',
      action='store_false',
      dest='keyboard_enter_after_input_text',
      help='Do not add a synthetic keyboard_enter QKV row after input_text.',
  )
  parser.add_argument(
      '--no-terminal-complete',
      action='store_false',
      dest='terminal_complete',
      help='Do not add a synthetic status complete QKV row at episode end.',
  )
  parser.add_argument('--max-records', type=int)
  parser.set_defaults(keyboard_enter_after_input_text=True, terminal_complete=True)
  args = parser.parse_args()
  process_examples(args)


if __name__ == '__main__':
  main()
