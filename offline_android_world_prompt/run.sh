python3 ./process_tfrecord.py \
  --input-glob "/Users/kailing.tan/AtlasKV/data/in/android_control*" \
  --output-dir /Users/kailing.tan/AtlasKV/data/out/ \
  --output-format qkv \
  --qkv-output-json /Users/kailing.tan/AtlasKV/data/out/android_control_seed_qkv.json \
  --qkv-stats-json /Users/kailing.tan/AtlasKV/data/out/android_control_seed_qkv_stats.json \
  --agent t3a \
  --goal-mode episode \
  --invalid-action-policy status_infeasible
