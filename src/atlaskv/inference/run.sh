export HF_ENDPOINT=https://hf-mirror.com

# conda activate atlaskv

# python test_server.py \
#   --host 127.0.0.1 \
#   --port 8000 \
#   --model_name atlaskv \
#   --seed 1607 \
#   --llm_type llama3 \
#   --llm_base_dir unsloth/Meta-Llama-3.1-8B-Instruct \
#   --model_dir /root/autodl-tmp/model/AtlasKV_all-MiniLM-L6-v2_train_atlas_wiki_qkv_llama3_step_3000 \
#   --encoder_dir /root/autodl-tmp/model/AtlasKV_all-MiniLM-L6-v2_train_atlas_wiki_qkv_llama3_step_3000_encoder/encoder.pt \
#   --dataset_path /root/autodl-tmp/data/atlaskv/Prebuilt/atlas_cc_qkv.jsonl \
#   --kb_layer_frequency 3 \
#   --kb_size 10 \
#   --kb_scale_factor 10 \
#   --encoder_spec all-MiniLM-L6-v2 \
#   --precomputed_embed_keys_path /root/autodl-tmp/data/atlaskv/output_cc/atlas_cc_qkv_all-MiniLM-L6-v2_embd_key.npy \
#   --precomputed_embed_values_path /root/autodl-tmp/data/atlaskv/output_cc/atlas_cc_qkv_all-MiniLM-L6-v2_embd_value.npy \
#   --use_kg \
#   --disable_kv_injection

python test_server.py \
  --host 127.0.0.1 \
  --port 8000 \
  --model_name atlaskv \
  --seed 1607 \
  --llm_type llama3 \
  --llm_base_dir unsloth/Meta-Llama-3.1-8B-Instruct \
  --model_dir /root/autodl-tmp/model/AtlasKV_all-MiniLM-L6-v2_train_atlas_wiki_qkv_llama3_step_3000 \
  --encoder_dir /root/autodl-tmp/model/AtlasKV_all-MiniLM-L6-v2_train_atlas_wiki_qkv_llama3_step_3000_encoder/encoder.pt \
  --dataset_path /root/autodl-tmp/AtlasKV/src/atlaskv/inference/android_world_action_qkv_seed.json \
  --kb_layer_frequency 3 \
  --kb_size -1 \
  --kb_scale_factor 10 \
  --encoder_spec all-MiniLM-L6-v2 \
  --precomputed_embed_keys_path /root/autodl-tmp/AtlasKV/src/atlaskv/inference/embed/atlas_aw_qkv_all-MiniLM-L6-v2_embd_key.npy \
  --precomputed_embed_values_path /root/autodl-tmp/AtlasKV/src/atlaskv/inference/embed/atlas_aw_qkv_all-MiniLM-L6-v2_embd_value.npy \
  --use_kg