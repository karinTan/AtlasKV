# Offline AndroidWorld QKV Pipeline

This directory builds AndroidWorld-style AtlasKV data from Android Control
TFRecord shards.

The intended pipeline is:

```text
Android Control TFRecord shards
  -> qkv.json
  -> qkv_6000.json
  -> qkv_6000_deepseek_key.json
  -> key/value embedding .npy files
  -> AtlasKV training or inference
```

The important design choice for the current version:

```text
Q does not contain Current state summary.
K/key_string contains the distilled current-screen summary.
V/description contains the distilled Reason: ... and the original Action: ...
```

So the model-side runtime prompt stays close to the AndroidWorld prompt, while
the key embedding gets a shorter semantic description for retrieval and the
value embedding gets a better rationale for the supervised target action.

## 0. Work From Repo Root

Run commands from:

```bash
cd /Users/tankling/Documents/all_my_files/coding/AtlasKV
```

If imports fail, set:

```bash
export PYTHONPATH=/Users/tankling/Documents/all_my_files/coding/AtlasKV/src
```

The TFRecord conversion step needs the same environment that can import
`tensorflow` and `android_env`.

## 1. Extract QKV From TFRecords

Use this to rebuild the full local QKV file from the Android Control shards:

```bash
python3 offline_android_world_prompt/process_tfrecord.py \
  --input-glob "data/in/android_control_android_control-*" \
  --output-dir data/out \
  --output-format qkv \
  --qkv-output-json data/out/qkv.json \
  --qkv-stats-json data/out/qkv_stats.json \
  --agent t3a \
  --goal-mode episode
```

Expected outputs:

```text
data/out/qkv.json
data/out/qkv_stats.json
```

Regenerate this file after prompt-template changes. Do not reuse an older
`qkv.json` or `qkv_6000.json` that still contains `Current state summary` inside
`Q`.

On the five local shards previously checked, this produced about 26K QKV rows:

```text
total_qkv_rows: 25911
episode_count: 3825
real action steps: 20698
synthetic terminal_complete rows: 3825
synthetic keyboard_enter_after_input_text rows: 1388
```

You can re-check any QKV file with:

```bash
python3 offline_android_world_prompt/qkv_stats.py \
  --qkv-json data/out/qkv.json \
  --summary-json data/out/qkv_stats_confirm.json
```

## 2. QKV Row Format

Each generated row has:

```json
{
  "name": "aw_<episode>_<step>_<action_type>",
  "description_type": "next AndroidWorld action",
  "reason": "...",
  "description": "Reason: ...\nAction: {...}",
  "Q": "...",
  "A": "Reason: ...\nAction: {...}",
  "key_string": "For the AndroidWorld goal of ..., after ..., the current screen shows the visible UI elements, and the next action should be",
  "extended_Q": "",
  "extended_A": ""
}
```

`description` and `A` are intentionally the same. AtlasKV encodes:

```text
key embedding source:   key_string
value embedding source: description
```

## 3. Q Format

`Q` is the prompt that should match the runtime `qkv_action_v1` prompt shape.
It contains the goal, history, filtered visible UI elements, compact action
shapes, and the required answer format.

The UI filter removes Android system status-bar lines and generic container-only
lines while preserving the original AndroidWorld UI element indexes. This same
filter is used by offline Q generation, DeepSeek key distillation, and runtime
request rewriting.

It looks like:

```text
What is the next AndroidWorld action?

The current AndroidWorld user goal is: ...
History: ...
The visible UI elements are:
UI element 0: ...
UI element 1: ...

Allowed action JSON shapes:
Action: {"action_type":"status","goal_status":"complete"}
Action: {"action_type":"status","goal_status":"infeasible"}
Action: {"action_type":"answer","text":"..."}
Action: {"action_type":"click","index":0}
Action: {"action_type":"long_press","index":0}
Action: {"action_type":"input_text","text":"...","index":0}
Action: {"action_type":"keyboard_enter"}
Action: {"action_type":"navigate_home"}
Action: {"action_type":"navigate_back"}
Action: {"action_type":"scroll","direction":"up|down|left|right"}
Action: {"action_type":"scroll","direction":"up|down|left|right","index":0}
Action: {"action_type":"open_app","app_name":"..."}
Action: {"action_type":"wait"}

Please answer in exactly this format:
Reason: <one brief reason grounded in the goal, history, or visible UI elements>
Action: {"action_type": "..."}
Use concrete JSON values. Do not output UI element metadata or a second action.
```

Do not add `Current state summary` to Q. The current-screen summary is only for
the key.

## 4. Default Key Before Distillation

Before DeepSeek distillation, `build_qkv.py` writes a fallback `key_string`:

```text
For the AndroidWorld goal of {goal}, after {history}, the current screen shows the visible UI elements, and the next action should be
```

This fallback is only a placeholder good enough for inspection. For AtlasKV
training/inference, use the DeepSeek-distilled key file below.

## 5. Sample 6000 Rows Before Distillation

The full local QKV set is large and expensive to distill. Sample 6000 rows first:

```bash
python3 offline_android_world_prompt/sample_qkv.py \
  --input-json data/out/qkv.json \
  --output-json data/out/qkv_6000.json \
  --summary-json data/out/qkv_6000_stats.json \
  --sample-size 6000 \
  --seed 1607
```

Expected outputs:

```text
data/out/qkv_6000.json
data/out/qkv_6000_stats.json
```

The sampler is action-type aware. It keeps rare action buckets better than pure
random sampling and rotates within each bucket by episode.

## 6. Check The DeepSeek Prompt

Before spending API credits, print one prompt without calling the API:

```bash
python3 offline_android_world_prompt/distill_key_with_deepseek.py \
  --input-json data/out/qkv_6000.json \
  --dry-run \
  --limit 1 \
  --max-ui-chars 3000
```

The user message sent to DeepSeek has this shape:

```text
Row name: aw_...
Goal: ...
History: ...
Visible UI elements excerpt:
UI element 0: ...
UI element 1: ...
Target Action:
Action: {"action_type":"..."}

Write the AtlasKV key_string and reason.
```

No precomputed current-state summary is sent. DeepSeek must infer the compact
screen description from the UI element excerpt. The target action is sent only
so DeepSeek can write the value-side reason; `key_string` must still not leak
the answer.

Before truncating the UI excerpt, the script uses the same UI filter as `Q` so
the model sees the app UI first. Pass `--include-system-ui` only if the system
UI itself is task-relevant.

## 7. Distill Key Strings And Reasons With DeepSeek

Set the API key:

```bash
export DEEPSEEK_API_KEY="..."
```

Run a small 20-row smoke test first:

```bash
python3 offline_android_world_prompt/distill_key_with_deepseek.py \
  --input-json data/out/qkv_6000.json \
  --output-json data/out/qkv_6000_deepseek_key_20.json \
  --failed-json data/out/qkv_6000_deepseek_key_failed_20.json \
  --limit 20 \
  --max-ui-chars 8000
```

Then run the full 6000-row distillation:

```bash
python3 offline_android_world_prompt/distill_key_with_deepseek.py \
  --input-json data/out/qkv_6000.json \
  --output-json data/out/qkv_6000_deepseek_key.json \
  --failed-json data/out/qkv_6000_deepseek_key_failed.json \
  --checkpoint-every 20 \
  --max-ui-chars 8000
```

Expected outputs:

```text
data/out/qkv_6000_deepseek_key.json
data/out/qkv_6000_deepseek_key_failed.json
```

The required DeepSeek output is JSON only:

```json
{
  "key_string": "For the AndroidWorld goal of {goal}, after {history}, the current screen shows {current state}, and the next action should be",
  "reason": "One brief sentence explaining why the supplied Target Action is correct."
}
```

The script validates that `key_string`:

- starts with `For the AndroidWorld goal of `,
- contains `, after `,
- contains `, the current screen shows `,
- ends with `, and the next action should be`,
- is one line,
- does not contain `Reason:`,
- does not contain `Action:`,
- does not contain an action JSON object, and
- does not use pipe separators.

The script also validates that `reason` is one brief line and does not contain
`Reason:`, `Action:`, or a JSON object. On success it writes:

```text
reason      = DeepSeek's distilled reason
description = Reason: <distilled reason>
              Action: <original normalized target action>
A           = same as description
```

If the answer is invalid, the script retries up to 3 times. Rows that still fail
are skipped and recorded in `--failed-json`.

## 8. Generate Key/Value Embeddings

Use the distilled QKV file as the dataset path. For all-MiniLM:

```bash
python3 dataset_generation/generate_kb_embeddings_gmm.py \
  --dataset_name atlas_aw_qkv_6000 \
  --dataset_path data/out/qkv_6000_deepseek_key.json \
  --output_path data/out \
  --model_name all-MiniLM-L6-v2 \
  --generating_embeddings
```

Expected outputs:

```text
data/out/atlas_aw_qkv_6000_all-MiniLM-L6-v2_embd_key.npy
data/out/atlas_aw_qkv_6000_all-MiniLM-L6-v2_embd_value.npy
```

If you use an OpenAI-compatible embedding endpoint instead, pass the endpoint
arguments and pick the model name:

```bash
python3 dataset_generation/generate_kb_embeddings_gmm.py \
  --dataset_name atlas_aw_qkv_6000 \
  --dataset_path data/out/qkv_6000_deepseek_key.json \
  --output_path data/out \
  --model_name text-embedding-3-large \
  --endpoint_url "$OPENAI_BASE_URL" \
  --endpoint_api_key "$OPENAI_API_KEY" \
  --generating_embeddings
```

## 9. Optional Action Profile

Use this if a new TFRecord split has action types or fields you have not seen:

```bash
python3 offline_android_world_prompt/process_tfrecord.py \
  --input-glob "data/in/android_control_android_control-*" \
  --output-dir data/out \
  --output-format profile \
  --action-profile-json data/out/action_profile.json
```

Or profile an existing QKV/prompt JSON with:

```bash
python3 offline_android_world_prompt/action_profile.py \
  --prompts-json data/out/prompts.json \
  --profile-json data/out/action_profile.json
```

## 10. Useful Flags

For quick debugging:

```bash
python3 offline_android_world_prompt/process_tfrecord.py \
  --input-glob "data/in/android_control_android_control-00016-of-00020" \
  --output-dir data/out_debug \
  --output-format qkv \
  --qkv-output-json data/out_debug/qkv_debug.json \
  --qkv-stats-json data/out_debug/qkv_debug_stats.json \
  --max-records 5
```

For step-level goals instead of episode-level goals:

```bash
--goal-mode step_instruction
```

To skip invalid converted actions instead of turning them into infeasible rows:

```bash
--invalid-action-policy skip
```

To disable synthetic rows:

```bash
--no-keyboard-enter-after-input-text
--no-terminal-complete
```

## 11. Sanity Checks

After each stage, check:

```bash
python3 - <<'PY'
import json
rows = json.load(open('data/out/qkv_6000_deepseek_key.json', encoding='utf-8'))
print(len(rows))
print(rows[0]['name'])
print(rows[0]['key_string'])
print('Current state summary:' in rows[0]['Q'])
print('com.android.systemui' in rows[0]['Q'])
print(rows[0]['description'])
PY
```

The expected booleans for `Current state summary:` and `com.android.systemui`
are:

```text
False
False
```

The final QKV file used for embeddings should be:

```text
data/out/qkv_6000_deepseek_key.json
```
