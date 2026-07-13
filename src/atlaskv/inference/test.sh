python ./test_android_world_server.py \
  --prompt-strategy qkv_action_v1 \
  --output-json results_qkv.json

# 1. 原文 request，完全不改
python3 ./test_android_world_server.py --prompt-strategy original --output-json results_original.json

# 2. 原文 request 基础上做输出格式增强
python3 ./test_android_world_server.py --prompt-strategy request_enhanced_v1 --output-json results_enhanced.json

# 3. 原文 request 基础上构造 AtlasKV/QKV 风格
python3 ./test_android_world_server.py --prompt-strategy qkv_action_v1 --output-json results_qkv.json

python3 ./test_android_world_server.py \
  --category pure_text \
  --prompt-strategy original \
  --output-jsonl results_original.jsonl

python3 ./test_android_world_server.py \
  --category pure_text \
  --prompt-strategy request_enhanced_v1 \
  --output-jsonl results_enhanced.jsonl

python3 ./test_android_world_server.py \
  --category pure_text \
  --prompt-strategy qkv_action_v1 \
  --output-jsonl results_qkv.jsonl

python3 ./test_android_world_server.py \
  --case ContactsAddContact_text \
  --prompt-strategy qkv_action_v1 \
  --show-prompt \
  --dry-run