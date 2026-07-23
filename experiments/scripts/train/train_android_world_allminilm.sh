#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"

DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/out}"
MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-${REPO_ROOT}/output/android_world_action}"
TRAIN_DATASET="${TRAIN_DATASET:-qkv_6000_deepseek_key_repaired_normalized}"
HF_MODEL_SPEC="${HF_MODEL_SPEC:-unsloth/Meta-Llama-3.1-8B-Instruct}"
HF_TOKEN="${HF_TOKEN:-}"
N="${N:-5992}"
B="${B:-10}"
TOTAL_STEPS="${TOTAL_STEPS:-10001}"
LR="${LR:-1e-3}"
KB_SIZE="${KB_SIZE:-200}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-3072}"

python "${REPO_ROOT}/experiments/train.py" \
  --dataset_dir "${DATASET_DIR}" \
  --model_save_dir "${MODEL_SAVE_DIR}" \
  --seed 1607 \
  --train_dataset "${TRAIN_DATASET}" \
  --N "${N}" \
  --B "${B}" \
  --total_steps "${TOTAL_STEPS}" \
  --encoder_spec all-MiniLM-L6-v2 \
  --use_cached_embd \
  --key_embd_src key \
  --android_world_action_task \
  --outlier_num -1 \
  --max_seq_len "${MAX_SEQ_LEN}" \
  --use_lr_decay \
  --sep_query_head \
  --lr "${LR}" \
  --kb_size "${KB_SIZE}" \
  --kb_token_layer_frequency 3 \
  --hf_model_spec "${HF_MODEL_SPEC}" \
  --hf_token "${HF_TOKEN}" \
  --use_kg \
  --projector_type linear
