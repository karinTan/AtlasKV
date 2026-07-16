# Offline Android World Prompt Builder

This folder contains a small self-contained copy of the Android World prompt/QKV
construction path needed for TFRecord data:

```text
TFRecord(GZIP)
  -> tf.train.Example
  -> accessibility_trees[i] bytes
  -> AndroidAccessibilityForest.ParseFromString(...)
  -> forest_to_ui_elements(...)
  -> T3A/M3A ui_elements text in memory
  -> AtlasKV qkv.json
```

It intentionally does not import `android_world.*`, so you can run it next to a
dataset without relying on the local repo package layout. It still needs the
same external runtime dependencies that your `data.py` already uses:
`tensorflow` and `android_env`.

## Run

From `/Users/kailing.tan/AtlasKV/offline_android_world_prompt`:

```bash
python3 ./process_tfrecord.py \
  --input-glob "/Users/kailing.tan/AtlasKV/data/in/android_control*" \
  --output-dir /Users/kailing.tan/AtlasKV/data/out/ \
  --output-format qkv \
  --qkv-output-json /Users/kailing.tan/AtlasKV/data/out/qkv.json \
  --agent t3a \
  --max-records 1
```

Main output:

```text
/Users/kailing.tan/AtlasKV/data/out/qkv.json
```

Each QKV row contains:

- `name`
- `description_type`
- `reason`
- `description`
- `Q`
- `A`
- `key_string`
- `extended_Q`
- `extended_A`

`--output-format qkv` writes the final QKV file directly and does not save
`prompts.json`. Use `--output-format both` if you also want a prompt-row
debugging file, or `--output-format prompts` for the old prompt-only output.

## Prompt Debug Output

When `prompts.json` is written, the JSON array contains objects with these
fields:

- `episode_id`
- `step_index`
- `episode_goal`
- `goal`
- `step_instruction`
- `history`
- `ui_elements_description`
- `prompt`
- `original_action`
- `target_action`

For click/long-press actions with `x`/`y`, `target_action` tries to map the
point to the best visible UI element index.

## Build QKV Data

`process_tfrecord.py --output-format qkv` is the preferred one-step path. If
you already have a `prompts.json` from a previous run, you can still convert it
directly:

```bash
python3 offline_android_world_prompt/build_qkv.py \
  --prompts-json /Users/kailing.tan/AtlasKV/data/out/prompts.json \
  --output-json /Users/kailing.tan/AtlasKV/data/out/qkv.json
```

`Q` follows the same compact AndroidWorld query structure as AtlasKV's
`qkv_action_v1` prompt strategy: overall user goal, history, visible UI
elements, allowed action shapes, and the required `Reason`/`Action` answer
format.

The converter uses deterministic reason templates. For example, `open_app`
mentions the app name, index actions mention the target UI element index, and
`input_text` mentions both the text and target index. It also:

- validates actions with the AtlasKV AndroidWorld action validator,
- requires QKV `click` and `long_press` actions to use UI indexes instead of
  raw coordinates,
- converts invalid target actions to
  `{"action_type":"status","goal_status":"infeasible"}` by default,
- adds a synthetic `keyboard_enter` row after each `input_text` row, and
- adds one synthetic `status complete` row at the end of each episode.

Use `--invalid-action-policy skip`,
`--no-keyboard-enter-after-input-text`, or `--no-terminal-complete` to disable
those defaults.

## Teacher History

If you generate history summaries with a teacher model, save them as JSON:

```json
{
  "0": [
    "Action selected: {\"action_type\": \"open_app\", \"app_name\": \"Zoho Meeting\"}. Opened Zoho Meeting.",
    "Action selected: {\"action_type\": \"wait\"}. Waited for the app to load."
  ]
}
```

Then pass:

```bash
--teacher-history-json /path/to/history.json
```

For step `k`, only summaries before `k` are included in the prompt history.
