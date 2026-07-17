#!/usr/bin/env python3
"""Convert offline AndroidWorld prompt rows into AtlasKV QKV data."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / 'src'
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
  sys.path.insert(0, str(SRC_ROOT))

from offline_android_world_prompt import qkv_stats  # pylint: disable=wrong-import-position

from atlaskv.android_world.protocol import (  # pylint: disable=wrong-import-position
    AndroidWorldOutputError,
    normalize_and_validate_action_output,
    validate_action,
)
from atlaskv.android_world.prompt_strategy import (  # pylint: disable=wrong-import-position
    QKV_ALLOWED_ACTIONS,
)


_UI_ELEMENT_RE = re.compile(r'\bUI\s+element\s+(\d+)\s*:', re.IGNORECASE)
_SAFE_NAME_RE = re.compile(r'[^0-9A-Za-z_]+')
_LEADING_WORD_RE = re.compile(r'^([A-Za-z]+)(\b.*)?$')
_LONG_PRESS_RE = re.compile(r'^(?:long[-\s]?press|long[-\s]?pressing|longing press)\b', re.IGNORECASE)
_HISTORY_INSTRUCTION_RE = re.compile(r'Instruction:\s*(.*?)(?:\.\s*(?:Step|$)|$)')
_VOWELS = frozenset('aeiou')

_IRREGULAR_GERUNDS = {
    'be': 'being',
    'begin': 'beginning',
    'cancel': 'canceling',
    'do': 'doing',
    'get': 'getting',
    'go': 'going',
    'have': 'having',
    'make': 'making',
    'open': 'opening',
    'run': 'running',
    'set': 'setting',
    'stop': 'stopping',
    'tap': 'tapping',
}

_IRREGULAR_PARTICIPLES = {
    'be': 'been',
    'begin': 'begun',
    'do': 'done',
    'get': 'gotten',
    'go': 'gone',
    'have': 'had',
    'input': 'input',
    'make': 'made',
    'open': 'opened',
    'run': 'run',
    'send': 'sent',
    'set': 'set',
    'take': 'taken',
    'visit': 'visited',
    'write': 'written',
}

_GERUND_PARTICIPLES = {
    'adding': 'added',
    'checking': 'checked',
    'choosing': 'chosen',
    'clearing': 'cleared',
    'clicking': 'clicked',
    'closing': 'closed',
    'creating': 'created',
    'decreasing': 'decreased',
    'deleting': 'deleted',
    'disabling': 'disabled',
    'enabling': 'enabled',
    'entering': 'entered',
    'getting': 'gotten',
    'going': 'gone',
    'increasing': 'increased',
    'making': 'made',
    'moving': 'moved',
    'navigating': 'navigated',
    'opening': 'opened',
    'pressing': 'pressed',
    'returning': 'returned',
    'saving': 'saved',
    'scrolling': 'scrolled',
    'searching': 'searched',
    'selecting': 'selected',
    'sending': 'sent',
    'setting': 'set',
    'sharing': 'shared',
    'starting': 'started',
    'stopping': 'stopped',
    'swiping': 'swiped',
    'tapping': 'tapped',
    'taking': 'taken',
    'typing': 'typed',
    'using': 'used',
    'viewing': 'viewed',
    'waiting': 'waited',
    'writing': 'written',
}


def _read_json(path: Path) -> Any:
  with path.open(encoding='utf-8') as file:
    return json.load(file)


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('w', encoding='utf-8') as file:
    json.dump(rows, file, ensure_ascii=False, indent=2)
    file.write('\n')


def _clean_text(value: Any, default: str = '') -> str:
  if value is None:
    return default
  return ' '.join(str(value).split()) or default


def _clean_instruction(value: Any, default: str = '') -> str:
  text = _clean_text(value, default)
  text = text.strip(' .')
  if text.lower().startswith('to '):
    text = text[3:].strip()
  return text


def _history_text(value: Any) -> str:
  if value is None:
    return 'You just started, no action has been performed yet.'
  text = str(value).strip()
  return text or 'You just started, no action has been performed yet.'


def _safe_name(value: Any, default: str = 'row') -> str:
  text = _SAFE_NAME_RE.sub('_', _clean_text(value, default)).strip('_')
  return text or default


def _to_gerund(verb: str) -> str:
  lower = verb.lower()
  if lower.endswith('ing'):
    return lower
  if lower in _IRREGULAR_GERUNDS:
    return _IRREGULAR_GERUNDS[lower]
  if lower.endswith('ie'):
    return lower[:-2] + 'ying'
  if lower.endswith('e') and not lower.endswith(('ee', 'oe', 'ye')):
    return lower[:-1] + 'ing'
  if (
      len(lower) >= 3
      and lower[-1] not in _VOWELS
      and lower[-2] in _VOWELS
      and lower[-3] not in _VOWELS
      and lower[-1] not in {'w', 'x', 'y'}
  ):
    return lower + lower[-1] + 'ing'
  return lower + 'ing'


def _to_past_participle(verb: str) -> str:
  lower = verb.lower()
  if lower.endswith('ing'):
    return _GERUND_PARTICIPLES.get(lower, _to_past_participle(_base_from_gerund(lower)))
  if lower in _IRREGULAR_PARTICIPLES:
    return _IRREGULAR_PARTICIPLES[lower]
  if lower.endswith('e'):
    return lower + 'd'
  if (
      len(lower) >= 3
      and lower[-1] not in _VOWELS
      and lower[-2] in _VOWELS
      and lower[-3] not in _VOWELS
      and lower[-1] not in {'w', 'x', 'y'}
  ):
    return lower + lower[-1] + 'ed'
  return lower + 'ed'


def _base_from_gerund(verb: str) -> str:
  lower = verb.lower()
  if not lower.endswith('ing') or len(lower) <= 4:
    return lower
  if lower.endswith('ying'):
    return lower[:-4] + 'ie'
  stem = lower[:-3]
  if (
      len(stem) >= 3
      and stem[-1] == stem[-2]
      and stem[-1] not in _VOWELS
      and stem[-3] in _VOWELS
  ):
    return stem[:-1]
  if stem.endswith(('ak', 'av', 'az', 'os', 'ov', 'yp', 'rit')):
    return stem + 'e'
  return stem


def _goal_phrase(value: Any) -> str:
  text = _clean_instruction(value, 'the current task')
  match = _LONG_PRESS_RE.match(text)
  if match:
    return 'long pressing' + text[match.end():]
  match = _LEADING_WORD_RE.match(text)
  if not match:
    return text
  verb, rest = match.group(1), match.group(2) or ''
  return f'{_to_gerund(verb)}{rest}'


def _completed_goal_phrase(value: Any | None) -> str:
  if not value:
    return 'starting from the initial state'
  text = _clean_instruction(value)
  if not text:
    return 'starting from the initial state'
  match = _LONG_PRESS_RE.match(text)
  if match:
    return 'having long pressed' + text[match.end():]
  match = _LEADING_WORD_RE.match(text)
  if not match:
    return f'having completed {text}'
  verb, rest = match.group(1), match.group(2) or ''
  return f'having {_to_past_participle(verb)}{rest}'


def _last_instruction_from_history(history: Any) -> str | None:
  if not history:
    return None
  matches = _HISTORY_INSTRUCTION_RE.findall(str(history))
  if not matches:
    return None
  return _clean_text(matches[-1])


def _visible_indices(ui_elements_description: str) -> frozenset[int] | None:
  indices = frozenset(
      int(match.group(1)) for match in _UI_ELEMENT_RE.finditer(ui_elements_description)
  )
  return indices or None


def _format_action(action: dict[str, Any]) -> str:
  return json.dumps(action, ensure_ascii=False, separators=(',', ':'))


def _history_action_summary(action: dict[str, Any], instruction: str | None) -> str:
  summary = f'Action selected: {_format_action(action)}.'
  if instruction:
    summary += f' Instruction: {_clean_text(instruction)}.'
  return summary


def _append_history(row: dict[str, Any], action: dict[str, Any]) -> str:
  history = _history_text(row.get('history'))
  if history == 'You just started, no action has been performed yet.':
    history_lines: list[str] = []
  else:
    history_lines = history.splitlines()
  step_number = int(row.get('step_index', len(history_lines))) + 1
  instruction = row.get('step_instruction')
  history_lines.append(
      f'Step {step_number}: {_history_action_summary(action, instruction)}'
  )
  return '\n'.join(history_lines)


def _build_q(row: dict[str, Any]) -> str:
  goal = _clean_text(row.get('goal') or row.get('episode_goal'), 'Unknown goal.')
  history = _history_text(row.get('history'))
  ui_elements = row.get('ui_elements_description') or 'No UI element details were found.'
  return f"""What is the next AndroidWorld action?

The current AndroidWorld user goal is: {goal}
History: {history}
The visible UI elements are:
{ui_elements}

{QKV_ALLOWED_ACTIONS}

Please answer in exactly this format:
Reason: <one brief reason grounded in the goal, history, or visible UI elements>
Action: {{"action_type": "..."}}
Use concrete JSON values. Do not output UI element metadata or a second action."""


def _action_phrase(action: dict[str, Any]) -> str:
  action_type = action.get('action_type')
  if action_type == 'open_app':
    return f"opening {_clean_text(action.get('app_name'), 'the requested app')}"
  if action_type == 'click':
    return f"tapping UI element {action.get('index')}"
  if action_type == 'long_press':
    return f"long pressing UI element {action.get('index')}"
  if action_type == 'input_text':
    text = _clean_text(action.get('text'), 'the requested text')
    return f"typing {text} into UI element {action.get('index')}"
  if action_type == 'keyboard_enter':
    return 'pressing Enter after text input'
  if action_type == 'scroll':
    direction = _clean_text(action.get('direction'), 'the needed direction')
    if 'index' in action:
      return f"scrolling {direction} on UI element {action.get('index')}"
    return f"scrolling {direction}"
  if action_type == 'navigate_home':
    return 'navigating to the home screen'
  if action_type == 'navigate_back':
    return 'navigating back'
  if action_type == 'wait':
    return 'waiting for the screen to update'
  if action_type == 'status':
    return f"marking the task {_clean_text(action.get('goal_status'), 'complete')}"
  if action_type == 'answer':
    return 'answering the user'
  return 'choosing the next action'


def _key_string(row: dict[str, Any], action: dict[str, Any]) -> str:
  current_goal = _goal_phrase(
      row.get('step_instruction') or row.get('goal') or row.get('episode_goal'),
  )
  previous_goal = (
      row.get('previous_step_instruction') or _last_instruction_from_history(row.get('history'))
  )
  return (
      'The next AndroidWorld action for '
      f'{current_goal} by {_completed_goal_phrase(previous_goal)}'
  )


def _reason_for_action(
    action: dict[str, Any],
    row: dict[str, Any],
    invalid_error: str | None = None,
) -> str:
  action_type = action.get('action_type')
  if invalid_error:
    return (
        'The recorded target action could not be converted into a valid '
        'index-based AndroidWorld action, so the task should be marked infeasible.'
    )
  if action_type == 'open_app':
    app_name = _clean_text(action.get('app_name'), 'the requested app')
    return f'The task requires {app_name}, so I should open that app.'
  if action_type == 'wait':
    return 'The previous action may need time to update the screen, so I should wait.'
  if action_type == 'click':
    return (
        f'The next step is to select the relevant visible UI element at index '
        f'{action.get("index")}.'
    )
  if action_type == 'long_press':
    return (
        f'The next step requires a long press on the relevant visible UI element '
        f'at index {action.get("index")}.'
    )
  if action_type == 'input_text':
    text = _clean_text(action.get('text'), 'the requested text')
    return (
        f'The target text field at index {action.get("index")} needs {text}, '
        'so I should enter it there.'
    )
  if action_type == 'keyboard_enter':
    return 'The text has been entered, so I should press Enter to submit it.'
  if action_type == 'navigate_home':
    return 'The next step requires returning to the Android home screen.'
  if action_type == 'navigate_back':
    return 'The current screen is not the needed destination, so I should go back.'
  if action_type == 'scroll':
    direction = _clean_text(action.get('direction'), 'down')
    if 'index' in action:
      return (
          f'The needed item is not visible yet, so I should scroll {direction} '
          f'on UI element {action.get("index")}.'
      )
    return f'The needed item is not visible yet, so I should scroll {direction}.'
  if action_type == 'status':
    if action.get('goal_status') == 'complete':
      return 'The recorded episode has reached its final state, so the task is complete.'
    conversion_error = _clean_text(row.get('conversion_error'))
    if conversion_error:
      return (
          'The recorded target action could not be converted into a valid '
          f'AndroidWorld action because {conversion_error}'
      )
    return 'The required action cannot be safely performed from this state.'
  if action_type == 'answer':
    return 'The requested information is available, so I should answer the user directly.'
  task = _clean_text(row.get('step_instruction') or row.get('goal'), 'the current task')
  return f'This action is the next recorded step for {task}.'


def _normalize_index_based_action(
    action: dict[str, Any],
    visible_indices: frozenset[int] | None,
) -> dict[str, Any]:
  normalized = validate_action(action, visible_indices)
  if normalized.get('action_type') in {'click', 'long_press'} and 'index' not in normalized:
    raise AndroidWorldOutputError(
        'click/long_press actions must use an index for QKV data.',
        'missing_index_action',
    )
  return normalized


def _description_and_action(
    reason: str,
    action: dict[str, Any],
    visible_indices: frozenset[int] | None,
) -> tuple[str, dict[str, Any]]:
  raw_description = f'Reason: {reason}\nAction: {_format_action(action)}'
  normalized_description, normalized_action = normalize_and_validate_action_output(
      raw_description, visible_indices
  )
  return normalized_description, normalized_action


def _build_qkv_row(
    row: dict[str, Any],
    action: dict[str, Any],
    name_suffix: str,
    invalid_error: str | None = None,
) -> dict[str, Any]:
  visible_indices = _visible_indices(row.get('ui_elements_description') or '')
  if invalid_error:
    action = {'action_type': 'status', 'goal_status': 'infeasible'}
  else:
    action = _normalize_index_based_action(action, visible_indices)

  reason = _reason_for_action(action, row, invalid_error)
  description, normalized_action = _description_and_action(reason, action, visible_indices)
  episode_id = _safe_name(row.get('episode_id'), 'episode')
  step_index = _safe_name(row.get('step_index'), 'step')
  name = f'aw_{episode_id}_{step_index}_{name_suffix}'

  return {
      'name': name,
      'description_type': 'next AndroidWorld action',
      'reason': reason,
      'description': description,
      'Q': _build_q(row),
      'A': description,
      'key_string': _key_string(row, normalized_action),
      'extended_Q': '',
      'extended_A': '',
  }


def _action_type_from_qkv_row(row: dict[str, Any]) -> str | None:
  _, action = normalize_and_validate_action_output(row['description'])
  action_type = action.get('action_type')
  return action_type if isinstance(action_type, str) else None


def _sort_episode_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
  return sorted(rows, key=lambda row: int(row.get('step_index', 0)))


def _make_keyboard_enter_context(
    row: dict[str, Any],
    next_row: dict[str, Any] | None,
    input_action: dict[str, Any],
) -> dict[str, Any]:
  if next_row is not None:
    context = copy.deepcopy(next_row)
  else:
    context = copy.deepcopy(row)
    context['history'] = _append_history(row, input_action)
  context['step_instruction'] = row.get('step_instruction')
  context['previous_step_instruction'] = row.get('step_instruction')
  context['step_index'] = f'{row.get("step_index", 0)}_keyboard_enter'
  return context


def _make_terminal_context(row: dict[str, Any], final_action: dict[str, Any]) -> dict[str, Any]:
  context = copy.deepcopy(row)
  context['history'] = _append_history(row, final_action)
  context['previous_step_instruction'] = row.get('step_instruction')
  context['step_index'] = f'{row.get("step_index", 0)}_terminal_complete'
  return context


def build_qkv_rows(
    prompt_rows: list[dict[str, Any]],
    invalid_action_policy: str = 'status_infeasible',
    keyboard_enter_after_input_text: bool = True,
    terminal_complete: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
  """Builds QKV rows from in-memory AndroidWorld prompt rows."""
  if not isinstance(prompt_rows, list):
    raise ValueError('prompt rows must be a list')

  episodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
  for row in prompt_rows:
    episodes[str(row.get('episode_id', ''))].append(row)

  qkv_rows: list[dict[str, Any]] = []
  invalid_actions = 0
  conversion_error_rows = 0
  keyboard_enter_rows = 0
  terminal_rows = 0

  for episode_id in sorted(episodes):
    episode_rows = _sort_episode_rows(episodes[episode_id])
    for index, row in enumerate(episode_rows):
      if row.get('conversion_error'):
        conversion_error_rows += 1
      target_action = row.get('target_action') or {}
      action_type = target_action.get('action_type', 'unknown')
      built_row: dict[str, Any] | None = None
      try:
        built_row = _build_qkv_row(row, target_action, _safe_name(action_type))
        qkv_rows.append(built_row)
      except AndroidWorldOutputError as exc:
        invalid_actions += 1
        if invalid_action_policy == 'skip':
          continue
        built_row = _build_qkv_row(
            row,
            target_action,
            f'{_safe_name(action_type)}_infeasible',
            invalid_error=str(exc),
        )
        qkv_rows.append(built_row)

      if (
          keyboard_enter_after_input_text
          and built_row is not None
          and _action_type_from_qkv_row(built_row) == 'input_text'
      ):
        next_row = episode_rows[index + 1] if index + 1 < len(episode_rows) else None
        keyboard_context = _make_keyboard_enter_context(row, next_row, target_action)
        qkv_rows.append(
            _build_qkv_row(
                keyboard_context,
                {'action_type': 'keyboard_enter'},
                'keyboard_enter_after_input_text',
            )
        )
        keyboard_enter_rows += 1

    if terminal_complete and episode_rows:
      final_row = episode_rows[-1]
      final_action = final_row.get('target_action') or {}
      terminal_context = _make_terminal_context(final_row, final_action)
      qkv_rows.append(
          _build_qkv_row(
              terminal_context,
              {'action_type': 'status', 'goal_status': 'complete'},
              'status_complete',
          )
      )
      terminal_rows += 1

  stats = {
      'prompt_rows': len(prompt_rows),
      'qkv_rows': len(qkv_rows),
      'invalid_actions': invalid_actions,
      'conversion_error_rows': conversion_error_rows,
      'keyboard_enter_rows': keyboard_enter_rows,
      'terminal_rows': terminal_rows,
  }
  return qkv_rows, stats


def convert_prompts_to_qkv(args: argparse.Namespace) -> dict[str, int]:
  prompt_rows = _read_json(Path(args.prompts_json))
  qkv_rows, stats = build_qkv_rows(
      prompt_rows,
      invalid_action_policy=args.invalid_action_policy,
      keyboard_enter_after_input_text=args.keyboard_enter_after_input_text,
      terminal_complete=args.terminal_complete,
  )
  _write_json(Path(args.output_json), qkv_rows)
  summary = qkv_stats.summarize_qkv_rows(qkv_rows, generation_stats=stats)
  qkv_stats.print_summary(summary)
  if args.qkv_stats_json:
    qkv_stats.write_summary(Path(args.qkv_stats_json), summary)
  return stats


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description='Convert offline AndroidWorld prompts.json into AtlasKV qkv.json.'
  )
  parser.add_argument(
      '--prompts-json',
      default=str(REPO_ROOT / 'data' / 'out' / 'prompts.json'),
      help='Path to prompts.json emitted by process_tfrecord.py.',
  )
  parser.add_argument(
      '--output-json',
      default=str(REPO_ROOT / 'data' / 'out' / 'qkv.json'),
      help='Path for generated qkv.json.',
  )
  parser.add_argument(
      '--qkv-stats-json',
      help='Optional path for writing QKV generation summary stats as JSON.',
  )
  parser.add_argument(
      '--invalid-action-policy',
      choices=['status_infeasible', 'skip'],
      default='status_infeasible',
      help='How to handle rows whose target_action is not valid index-based AndroidWorld.',
  )
  parser.add_argument(
      '--no-keyboard-enter-after-input-text',
      action='store_false',
      dest='keyboard_enter_after_input_text',
      help='Do not add a synthetic keyboard_enter row after input_text actions.',
  )
  parser.add_argument(
      '--no-terminal-complete',
      action='store_false',
      dest='terminal_complete',
      help='Do not add one synthetic status complete row at the end of each episode.',
  )
  parser.set_defaults(keyboard_enter_after_input_text=True, terminal_complete=True)
  return parser


def main() -> None:
  args = build_parser().parse_args()
  stats = convert_prompts_to_qkv(args)
  invalid_note = (
      'invalid skipped'
      if args.invalid_action_policy == 'skip'
      else 'invalid -> status infeasible'
  )
  print(
      'Wrote {qkv_rows} QKV rows to {output_json} '
      'from {prompt_rows} prompt rows '
      '({invalid_actions} {invalid_note}, '
      '{keyboard_enter_rows} keyboard_enter, {terminal_rows} terminal complete).'.format(
          output_json=args.output_json,
          invalid_note=invalid_note,
          **stats,
      )
  )


if __name__ == '__main__':
  main()
