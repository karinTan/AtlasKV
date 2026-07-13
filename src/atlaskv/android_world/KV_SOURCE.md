# AndroidWorld KV embedding source

The AndroidWorld KV seed data is stored in
`src/atlaskv/inference/android_world_action_qkv_seed.json`.

For every record:

- `key_string` is the source text encoded into the precomputed key `.npy` file.
- `description` is the source text encoded into the precomputed value `.npy` file.
- `reason` stores the short, scenario-grounded reason separately for inspection.
- `A` is the training target and must be identical to `description`.

Both `description` and `A` use this canonical format:

```text
Reason: <brief reason grounded in the goal, history, or visible UI>
Action: {"action_type":"<allowed action>",...}
```

Changing `description` changes the source text for the value embeddings. Any
existing precomputed value `.npy` file must therefore be regenerated after this
JSON file changes; renaming or reusing the old file does not update its vectors.
