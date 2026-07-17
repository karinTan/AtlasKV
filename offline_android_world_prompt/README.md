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
  --qkv-output-json /Users/kailing.tan/AtlasKV/data/out/android_control_seed_qkv.json \
  --qkv-stats-json /Users/kailing.tan/AtlasKV/data/out/android_control_seed_qkv_stats.json \
  --agent t3a
```

Main output:

```text
/Users/kailing.tan/AtlasKV/data/out/android_control_seed_qkv.json
/Users/kailing.tan/AtlasKV/data/out/android_control_seed_qkv_stats.json
```

## CLI Parameters

`process_tfrecord.py` exposes the following arguments:

| Parameter | Required | Default | Applies to | Purpose |
| --- | --- | --- | --- | --- |
| `--input-glob` | yes | none | all modes | Glob pattern for input Android Control TFRecord files. |
| `--output-dir` | yes | none | all modes | Output directory. Also provides default paths for `prompts.json` and `qkv.json`. |
| `--output-format` | no | `prompts` | all modes | Selects output mode: `prompts`, `qkv`, `both`, or `profile`. Use `qkv` for direct final QKV generation. |
| `--qkv-output-json` | no | `<output-dir>/qkv.json` | `qkv`, `both` | Path for final QKV JSON. |
| `--qkv-stats-json` | no | not written | `qkv`, `both` | Optional path for QKV statistics JSON. The summary is printed even when this is omitted. |
| `--action-profile-json` | no | not written | all modes | Optional path for action profile JSON. If passed, the profile is generated and saved. |
| `--action-profile-examples` | no | `3` | profile generation | Number of example rows to keep per action type and error section in the action profile. |
| `--agent` | no | `t3a` | prompt/UI text generation | UI element prompt format. `t3a` is the compact text-agent format; `m3a` uses the multimodal-agent style. |
| `--goal-mode` | no | `episode` | all modes | Chooses the goal text in prompts/Q: `episode` uses the whole episode goal; `step_instruction` uses the current step instruction. |
| `--teacher-history-json` | no | none | all modes | Optional externally generated history summaries keyed by `episode_id`. If omitted, history is built from converted prior actions. |
| `--invalid-action-policy` | no | `status_infeasible` | `qkv`, `both` | Handles actions that still fail QKV validation: `status_infeasible` converts them to `status infeasible`; `skip` drops them. |
| `--no-keyboard-enter-after-input-text` | no | disabled flag, so keyboard enter is added | `qkv`, `both` | Turns off the synthetic `keyboard_enter` row normally added after each valid `input_text` row. |
| `--no-terminal-complete` | no | disabled flag, so terminal complete is added | `qkv`, `both` | Turns off the synthetic `status complete` row normally added at the end of each episode. |
| `--max-records` | no | all records | all modes | Limits the number of TFRecord examples processed. Useful for sampling/debugging. |

For the current QKV-only generation path, the important parameters are
`--input-glob`, `--output-dir`, `--output-format qkv`, `--qkv-output-json`, and
`--qkv-stats-json`. The remaining parameters are debugging or strategy
overrides.

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
debugging file, `--output-format prompts` for the old prompt-only output, or
`--output-format profile` to only inspect raw/converted action shapes.

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
- `previous_step_instruction`

For click/long-press actions with `x`/`y`, `target_action` tries to map the
point to the best visible UI element index.

## Build QKV Data

`process_tfrecord.py --output-format qkv` is the preferred one-step path. If
you already have a `prompts.json` from a previous run, you can still convert it
directly:

```bash
python3 offline_android_world_prompt/build_qkv.py \
  --prompts-json /Users/kailing.tan/AtlasKV/data/out/prompts.json \
  --output-json /Users/kailing.tan/AtlasKV/data/out/qkv.json \
  --qkv-stats-json /Users/kailing.tan/AtlasKV/data/out/qkv_stats.json
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

`key_string` uses the template
`The next AndroidWorld action for {current_goal} by {previous_state}`.
Instruction text is normalized with deterministic morphology rules before it
is inserted into the template: imperative goals become gerunds (`Open ...` ->
`opening ...`), and the previous instruction becomes a completed-state phrase
(`Open ...` -> `having opened ...`, `clicking ...` -> `having clicked ...`).

## Action Profile

Before QKV rows are generated, `process_tfrecord.py` can print and save an
action profile so the raw action parameters are visible by action type. This is
the quick check for whether new TFRecord actions need conversion logic before
they are trusted as QKV labels.

The profile includes:

- original action type counts,
- converted action type counts,
- field names and field value types for every action type,
- a few examples per action type, and
- converted actions that would fail QKV validation.

You can also profile an existing prompt-row file directly:

```bash
python3 ./action_profile.py \
  --prompts-json /Users/kailing.tan/AtlasKV/data/out/prompts.json \
  --profile-json /Users/kailing.tan/AtlasKV/data/out/action_profile.json
```

Or profile TFRecords directly without saving prompts or QKV rows:

```bash
python3 ./process_tfrecord.py \
  --input-glob "/Users/kailing.tan/AtlasKV/data/in/android_control*" \
  --output-dir /Users/kailing.tan/AtlasKV/data/out/ \
  --output-format profile \
  --action-profile-json /Users/kailing.tan/AtlasKV/data/out/action_profile.json
```

## Action Conversion Rules

The TFRecord action is kept as `original_action`. The label used by QKV is
`target_action`. If a raw action cannot be converted to a valid QKV action, the
converter writes `target_action` as
`{"action_type":"status","goal_status":"infeasible"}` and records the cause in
`conversion_error`.

Current rules:

- `click`: use `index` if present; otherwise map `x`/`y` to the best visible UI
  element index. If no index can be found, convert to `status infeasible`.
- `long_press`: same as `click`, but the output action type remains
  `long_press`.
- `input_text`: requires `text`. Use `index` if present; otherwise map `x`/`y`
  if present; otherwise choose the best visible editable text field. If no text
  field index can be found, convert to `status infeasible`.
- `scroll`: requires `direction` in `up`, `down`, `left`, or `right`; preserves
  optional `index` when present.
- `open_app`: requires a non-empty `app_name`.
- `answer`: requires non-empty `text`.
- `status`: requires `goal_status` to be `complete` or `infeasible`.
- `keyboard_enter`, `navigate_back`, `navigate_home`, `wait`: no extra
  parameters are kept.
- unknown action types: convert to `status infeasible`.

## QKV Stats

When QKV output is enabled, `process_tfrecord.py` prints a generation summary
after writing `qkv.json`. If `--qkv-stats-json` is passed, the same summary is
also saved as structured JSON.

The summary includes:

- total QKV rows,
- source prompt-row generation counts,
- description type counts,
- episode count,
- action type distribution,
- status goal distribution,
- synthetic row counts,
- average/min/max original steps per episode, and
- average/min/max QKV rows per episode.

You can also summarize an existing QKV file directly:

```bash
python3 ./qkv_stats.py \
  --qkv-json /Users/kailing.tan/AtlasKV/data/out/qkv.json \
  --summary-json /Users/kailing.tan/AtlasKV/data/out/qkv_stats.json
```

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
