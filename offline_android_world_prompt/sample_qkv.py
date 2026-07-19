#!/usr/bin/env python3
"""Sample a smaller AndroidWorld QKV set before key distillation."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / 'src'
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
  sys.path.insert(0, str(SRC_ROOT))

from atlaskv.android_world.protocol import (  # pylint: disable=wrong-import-position
    AndroidWorldOutputError,
    normalize_and_validate_action_output,
)
from offline_android_world_prompt import qkv_stats  # pylint: disable=wrong-import-position


_QKV_NAME_RE = re.compile(r'^aw_(?P<episode>.*)_(?P<step>\d+)(?:_(?P<suffix>.*))?$')


def _read_json(path: Path) -> Any:
  with path.open(encoding='utf-8') as file:
    return json.load(file)


def _write_json(path: Path, data: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=2)
    file.write('\n')


def _episode_id(row: dict[str, Any], fallback: str) -> str:
  name = str(row.get('name') or '')
  match = _QKV_NAME_RE.match(name)
  if match:
    return match.group('episode') or fallback
  return fallback


def _action_bucket(row: dict[str, Any]) -> str:
  description = str(row.get('description') or row.get('A') or '')
  try:
    _, action = normalize_and_validate_action_output(description)
  except AndroidWorldOutputError:
    return 'malformed'
  action_type = str(action.get('action_type') or 'missing')
  if action_type == 'status':
    goal_status = str(action.get('goal_status') or 'missing')
    return f'status_{goal_status}'
  return action_type


def _largest_remainder_counts(
    capacities: dict[str, int],
    target_total: int,
) -> dict[str, int]:
  if target_total <= 0 or not capacities:
    return {key: 0 for key in capacities}
  capacity_total = sum(capacities.values())
  if target_total >= capacity_total:
    return dict(capacities)

  counts: dict[str, int] = {}
  remainders: list[tuple[float, str]] = []
  for key, capacity in sorted(capacities.items()):
    exact = target_total * capacity / capacity_total
    count = min(int(exact), capacity)
    counts[key] = count
    remainders.append((exact - count, key))

  remaining = target_total - sum(counts.values())
  for _, key in sorted(remainders, reverse=True):
    if remaining <= 0:
      break
    if counts[key] < capacities[key]:
      counts[key] += 1
      remaining -= 1
  return counts


def _target_counts(
    groups: dict[str, list[tuple[int, dict[str, Any]]]],
    sample_size: int,
    rare_keep_threshold: int,
) -> dict[str, int]:
  total_rows = sum(len(rows) for rows in groups.values())
  if sample_size >= total_rows:
    return {bucket: len(rows) for bucket, rows in groups.items()}

  keep_all: dict[str, int] = {}
  large_capacities: dict[str, int] = {}
  for bucket, rows in groups.items():
    if len(rows) <= rare_keep_threshold:
      keep_all[bucket] = len(rows)
    else:
      large_capacities[bucket] = len(rows)

  kept_total = sum(keep_all.values())
  if kept_total > sample_size:
    all_capacities = {bucket: len(rows) for bucket, rows in groups.items()}
    return _largest_remainder_counts(all_capacities, sample_size)

  remaining = sample_size - kept_total
  counts = dict(keep_all)
  counts.update(_largest_remainder_counts(large_capacities, remaining))
  return counts


def _round_robin_sample(
    indexed_rows: list[tuple[int, dict[str, Any]]],
    quota: int,
    rng: random.Random,
) -> list[tuple[int, dict[str, Any]]]:
  if quota >= len(indexed_rows):
    return list(indexed_rows)
  by_episode: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
  for index, row in indexed_rows:
    by_episode[_episode_id(row, f'row_{index}')].append((index, row))

  episodes = list(by_episode)
  rng.shuffle(episodes)
  for rows in by_episode.values():
    rng.shuffle(rows)

  selected: list[tuple[int, dict[str, Any]]] = []
  while len(selected) < quota and episodes:
    next_episodes: list[str] = []
    for episode in episodes:
      rows = by_episode[episode]
      if rows:
        selected.append(rows.pop())
        if len(selected) >= quota:
          break
      if rows:
        next_episodes.append(episode)
    episodes = next_episodes
    rng.shuffle(episodes)
  return selected


def sample_rows(
    rows: list[dict[str, Any]],
    sample_size: int,
    seed: int,
    rare_keep_threshold: int,
    shuffle_output: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
  for index, row in enumerate(rows):
    groups[_action_bucket(row)].append((index, row))

  targets = _target_counts(groups, sample_size, rare_keep_threshold)
  rng = random.Random(seed)
  selected_indexed_rows: list[tuple[int, dict[str, Any]]] = []
  for bucket in sorted(groups):
    selected_indexed_rows.extend(
        _round_robin_sample(groups[bucket], targets.get(bucket, 0), rng)
    )

  if shuffle_output:
    rng.shuffle(selected_indexed_rows)
  else:
    selected_indexed_rows.sort(key=lambda item: item[0])

  sampled_rows = [row for _, row in selected_indexed_rows]
  selected_counts = Counter(_action_bucket(row) for row in sampled_rows)
  source_counts = Counter(_action_bucket(row) for row in rows)
  episode_count = len({
      _episode_id(row, f'sampled_{index}') for index, row in enumerate(sampled_rows)
  })

  summary = {
      'input_rows': len(rows),
      'sample_size': len(sampled_rows),
      'seed': seed,
      'rare_keep_threshold': rare_keep_threshold,
      'shuffle_output': shuffle_output,
      'episode_count': episode_count,
      'source_action_buckets': dict(sorted(source_counts.items())),
      'target_action_buckets': dict(sorted(targets.items())),
      'sampled_action_buckets': dict(sorted(selected_counts.items())),
      'sampled_qkv_summary': qkv_stats.summarize_qkv_rows(sampled_rows),
  }
  return sampled_rows, summary


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description='Sample a balanced subset from AndroidWorld qkv.json.'
  )
  parser.add_argument(
      '--input-json',
      default=str(REPO_ROOT / 'data' / 'out' / 'qkv.json'),
      help='Input QKV JSON file.',
  )
  parser.add_argument(
      '--output-json',
      default=str(REPO_ROOT / 'data' / 'out' / 'qkv_6000.json'),
      help='Output sampled QKV JSON file.',
  )
  parser.add_argument(
      '--summary-json',
      default=str(REPO_ROOT / 'data' / 'out' / 'qkv_6000_stats.json'),
      help='Optional output summary JSON path. Use an empty string to disable.',
  )
  parser.add_argument('--sample-size', type=int, default=6000)
  parser.add_argument('--seed', type=int, default=1607)
  parser.add_argument(
      '--rare-keep-threshold',
      type=int,
      default=200,
      help='Keep all rows for action buckets with at most this many rows.',
  )
  parser.add_argument(
      '--shuffle-output',
      action='store_true',
      help='Shuffle output rows instead of preserving source order.',
  )
  return parser


def main() -> None:
  args = build_parser().parse_args()
  if args.sample_size < 1:
    raise ValueError('--sample-size must be positive')
  if args.rare_keep_threshold < 0:
    raise ValueError('--rare-keep-threshold cannot be negative')

  rows = _read_json(Path(args.input_json))
  if not isinstance(rows, list):
    raise ValueError('input JSON must be a list of QKV rows')
  if not all(isinstance(row, dict) for row in rows):
    raise ValueError('all input rows must be JSON objects')

  sampled_rows, summary = sample_rows(
      rows,
      sample_size=args.sample_size,
      seed=args.seed,
      rare_keep_threshold=args.rare_keep_threshold,
      shuffle_output=args.shuffle_output,
  )
  _write_json(Path(args.output_json), sampled_rows)
  if args.summary_json:
    _write_json(Path(args.summary_json), summary)

  print(f'Wrote {len(sampled_rows)} rows to {args.output_json}')
  print(qkv_stats.format_summary(summary['sampled_qkv_summary']))
  print('sampled_action_buckets:')
  for bucket, count in summary['sampled_action_buckets'].items():
    print(f'  {bucket}: {count}')


if __name__ == '__main__':
  main()
