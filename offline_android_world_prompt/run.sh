python3 ./process_tfrecord.py \
  --input-glob "/Users/kailing.tan/AtlasKV/data/in/android_control*" \
  --output-dir /Users/kailing.tan/AtlasKV/data/out/ \
  --output-format qkv \
  --qkv-output-json /Users/kailing.tan/AtlasKV/data/out/qkv.json \
  --agent t3a \
  --max-records 1
