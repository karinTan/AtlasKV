# Copyright 2026 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HTTP server that exposes an AtlasKV model for android_world.

Runs on the GPU machine that hosts AtlasKV
(https://github.com/HKUST-KnowComp/AtlasKV). The android_world host talks to it
over the network via `infer.AtlasKvWrapper`, which POSTs to `/predict`.

This mirrors AtlasKV's own inference path (experiments/eval.py):
  - `_init_models(...)`  -> tokenizer, KBEncoder, model, kb_config   (once)
  - `answer_question(tok, mdl, Q, kb=kb_pack, kb_config=kb_cfg)`      (per call)

It must run inside the AtlasKV conda env (so `import atlaskv` works) with the
model checkpoints downloaded. Configure paths via the env vars below.

Setup (on the GPU host):
    conda activate atlaskv
    pip install fastapi "uvicorn[standard]"
    export ATLASKV_LLM_TYPE=llama3
    export ATLASKV_LLM_BASE_DIR=/path/to/Llama-3-8B          # base HF model
    export ATLASKV_MODEL_DIR=/path/to/atlaskv_checkpoint     # fine-tuned weights
    export ATLASKV_ENCODER_CKPT=/path/to/encoder.pt          # KBEncoder ckpt
    export ATLASKV_ENCODER_SPEC=OAI                          # or all-MiniLM-L6-v2
    export ATLASKV_QUERY_HEAD_PATH=/path/to/query_head.pt    # optional
    # Optional knowledge base (leave unset to run the base model with kb=None):
    #   export ATLASKV_DATASET=/path/to/kb.json
    #   export ATLASKV_KEYS=/path/to/keys.npy
    #   export ATLASKV_VALUES=/path/to/values.npy
    python server.py            # listens on 0.0.0.0:8000

Then on the android_world host:
    export ATLASKV_API_URL=http://<GPU_HOST_IP>:8000/predict
    python run.py --agent_name=t3a_atlaskv --tasks=ContactsAddContact

export ATLASKV_MODEL_DIR=/root/autodl-tmp/model/AtlasKV_all-MiniLM-L6-v2_train_atlas_wiki_qkv_llama3_step_3000
export ATLASKV_ENCODER_CKPT=/root/autodl-tmp/model/AtlasKV_all-MiniLM-L6-v2_train_atlas_wiki_qkv_llama3_step_3000_encoder/encoder.pt
export ATLASKV_LLM_BASE_DIR=NousResearch/Meta-Llama-3-8B
export ATLASKV_ENCODER_SPEC=all-MiniLM-L6-v2
export HF_ENDPOINT=https://hf-mirror.com
export ATLASKV_USE_KG=true
conda activate atlaskv
"""

import json
import os
import time
from typing import Any, List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# AtlasKV imports (require running inside its installed env).
from atlaskv.kb_encoder import KBEncoder
from atlaskv.models.kblam_config import AtlasKVConfig, KBLaMConfig
from atlaskv.models.llama3_model import (
    AtlaskvLlamaForCausalLM,
    KblamLlamaForCausalLM,
    set_llama_attention_classes,
)
from atlaskv.models.phi3_model import KBLaMPhi3ForCausalLM
from atlaskv.utils.eval_utils import _format_Q_llama, _format_Q_phi3
from atlaskv.utils.train_utils import get_kb_embd


# --- config from env -------------------------------------------------------
LLM_TYPE = os.environ.get('ATLASKV_LLM_TYPE', 'llama3')
# LLM_BASE_DIR only supplies the Llama-3 *tokenizer* (the published checkpoint
# in MODEL_DIR has no tokenizer files). It does NOT need the base weights, e.g.
# NousResearch/Meta-Llama-3-8B (ungated) works and only fetches tokenizer files.
LLM_BASE_DIR = os.environ.get('ATLASKV_LLM_BASE_DIR', 'NousResearch/Meta-Llama-3-8B')
MODEL_DIR = os.environ['ATLASKV_MODEL_DIR']  # full 17GB checkpoint
ENCODER_CKPT = os.environ['ATLASKV_ENCODER_CKPT']  # encoder.pt
# all-MiniLM-L6-v2 is a local sentence-transformer (no API key); OAI needs one.
ENCODER_SPEC = os.environ.get('ATLASKV_ENCODER_SPEC', 'all-MiniLM-L6-v2')
QUERY_HEAD_PATH = os.environ.get('ATLASKV_QUERY_HEAD_PATH', '')
KB_LAYER_FREQUENCY = int(os.environ.get('ATLASKV_KB_LAYER_FREQUENCY', '3'))
USE_KG = os.environ.get('ATLASKV_USE_KG', 'false').lower() == 'true'
OUTPUT_ATTN = os.environ.get('ATLASKV_OUTPUT_ATTN', 'false').lower() == 'true'

# Optional knowledge base. If unset, the model runs with kb=None (base behavior).
DATASET = os.environ.get('ATLASKV_DATASET', '')
KEYS = os.environ.get('ATLASKV_KEYS', '')
VALUES = os.environ.get('ATLASKV_VALUES', '')
KB_SIZE = int(os.environ.get('ATLASKV_KB_SIZE', '200'))


def load_everything():
  """Loads tokenizer, encoder, model, kb_config and (optionally) a kb_pack."""
  from transformers import AutoTokenizer

  set_llama_attention_classes(USE_KG)

  tok = AutoTokenizer.from_pretrained(
      LLM_BASE_DIR, trust_remote_code=True, padding_side='left'
  )
  tok.pad_token = '^'

  # Optional 4-bit quantization (ATLASKV_LOAD_4BIT=true) to fit a long-context
  # T3A prompt on a small GPU. Needs `pip install bitsandbytes`.
  load_kwargs = dict(
      device_map='cuda',
      torch_dtype='auto',
      trust_remote_code=True,
      attn_implementation='eager',
  )
  if os.environ.get('ATLASKV_LOAD_4BIT', 'false').lower() == 'true':
    from transformers import BitsAndBytesConfig

    load_kwargs['quantization_config'] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type='nf4',
    )

  if LLM_TYPE == 'llama3':
    ctor = AtlaskvLlamaForCausalLM if USE_KG else KblamLlamaForCausalLM
    mdl = ctor.from_pretrained(MODEL_DIR, **load_kwargs)
    if QUERY_HEAD_PATH:
      mdl.load_query_head(QUERY_HEAD_PATH)
  else:
    mdl = KBLaMPhi3ForCausalLM.from_pretrained(MODEL_DIR, **load_kwargs)

  mdl.config._attn_implementation = 'eager'
  print(f'attn_implementation = {mdl.config._attn_implementation}')

  mdl.generation_config.pad_token_id = tok.pad_token_id
  mdl.generation_config.eos_token_id = tok.eos_token_id
  mdl.eval()

  if USE_KG:
    kb_cfg = AtlasKVConfig(
        sep_query_head=True, kb_layer_frequency=KB_LAYER_FREQUENCY
    )
  else:
    kb_cfg = KBLaMConfig(
        sep_query_head=True,
        kb_layer_frequency=KB_LAYER_FREQUENCY,
        use_hierarchial_kv=False,
    )

  enc = KBEncoder(
      encoder_name=ENCODER_SPEC,
      projector_type='linear',
      # Required positional args; only used by the OAI encoder, ignored for
      # local sentence-transformers like all-MiniLM-L6-v2.
      endpoint_url=os.environ.get('ATLASKV_ENDPOINT_URL', 'unused'),
      endpoint_api_key=os.environ.get('ATLASKV_ENDPOINT_API_KEY', 'unused'),
      out_dim=mdl.config.hidden_size
      * (mdl.config.num_hidden_layers // KB_LAYER_FREQUENCY + 1),
      frozen_base_model=True,
      projector_kwargs={'mlp_depth': 1, 'mlp_hidden_dim': 512},
      device=torch.device('cuda'),
      get_oai_embd_online=ENCODER_SPEC == 'OAI',
  )
  enc.load_state_dict(torch.load(ENCODER_CKPT))

  # Build the injected knowledge pack once (or None for base behavior).
  kb_pack = None
  if DATASET and KEYS and VALUES:
    import json

    data = json.load(open(DATASET))
    keys = np.load(KEYS).astype('float32')
    vals = np.load(VALUES).astype('float32')
    idx = np.arange(min(KB_SIZE, len(data)))
    with torch.no_grad():
      kb_pack = get_kb_embd(enc, idx, precomputed_embd=(keys, vals))

  return tok, enc, mdl, kb_cfg, kb_pack


print('Loading AtlasKV (this is heavy)...')
TOK, ENC, MDL, KB_CFG, KB_PACK = load_everything()
print('AtlasKV ready.')


# --- HTTP layer ------------------------------------------------------------
app = FastAPI()


class PredictRequest(BaseModel):
  prompt: str
  temperature: float = 0.0
  max_tokens: int = 1000


class ChatMessage(BaseModel):
  role: str
  content: Any = ''


class ChatCompletionRequest(BaseModel):
  model: str = 'atlaskv'
  messages: List[ChatMessage]
  temperature: float = 0.0
  max_tokens: int = 1024
  stream: bool = False


def _format_q(prompt: str) -> str:
  return _format_Q_llama(prompt) if LLM_TYPE == 'llama3' else _format_Q_phi3(prompt)


def _message_content_to_text(content: Any) -> str:
  """Converts OpenAI text or multimodal content into AtlasKV text input."""
  if content is None:
    return ''
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    parts = []
    for item in content:
      if isinstance(item, str):
        parts.append(item)
      elif isinstance(item, dict):
        item_type = item.get('type')
        if item_type == 'text':
          parts.append(str(item.get('text', '')))
        elif item_type == 'image_url':
          parts.append('[image_url omitted by text-only AtlasKV adapter]')
        else:
          parts.append(str(item.get('text', item)))
      else:
        parts.append(str(item))
    return '\n'.join(p for p in parts if p)
  return str(content)


def _messages_to_prompt(messages: List[ChatMessage]) -> str:
  """Flattens OpenAI chat messages into a single plain-text prompt."""
  blocks = []
  for message in messages:
    role = message.role.strip().lower() or 'user'
    content = _message_content_to_text(message.content).strip()
    if not content:
      continue
    if role == 'system':
      label = 'System'
    elif role == 'assistant':
      label = 'Assistant'
    elif role == 'tool':
      label = 'Tool'
    else:
      label = 'User'
    blocks.append(f'{label}: {content}')
  blocks.append('Assistant:')
  return '\n\n'.join(blocks)


def _first_action_block(text: str) -> str:
  """Keeps only the first 'Thought: ... Action: {..}' block.

  The model hallucinates several future steps in one go, but T3A acts on a
  single action then re-observes the screen, so the later blocks are blind
  guesses. We return just the first complete Action JSON.
  """
  a = text.find('Action:')
  start = text.find('{', a) if a != -1 else -1
  if start == -1:
    return text

  try:
    _, end = json.JSONDecoder().raw_decode(text[start:])
    return text[: start + end]
  except json.JSONDecodeError:
    pass

  depth = 0
  in_string = False
  escape = False
  for i in range(start, len(text)):
    ch = text[i]
    if escape:
      escape = False
      continue
    if ch == '\\':
      escape = True
      continue
    if ch == '"':
      in_string = not in_string
      continue
    if in_string:
      continue
    if ch == '{':
      depth += 1
    elif ch == '}':
      depth -= 1
      if depth == 0:
        return text[: i + 1]
  return text


def _clean_single_step(text: str) -> str:
  """Normalizes generation to one MobileWorld-style Thought/Action step."""
  text = _first_action_block(text.strip()).strip()
  action_idx = text.find('Action:')
  json_start = text.find('{', action_idx) if action_idx != -1 else -1
  if json_start == -1:
    return text

  prefix = text[:action_idx].strip()
  if prefix.startswith('Reason:') and 'Thought:' not in prefix:
    prefix = 'Thought:' + prefix[len('Reason:') :]
  elif not prefix:
    prefix = 'Thought:'

  try:
    action, _ = json.JSONDecoder().raw_decode(text[json_start:])
    action_text = json.dumps(action, ensure_ascii=False)
    return f'{prefix}\nAction: {action_text}'.strip()
  except json.JSONDecodeError:
    return text


def _generate_text(prompt: str, temperature: float, max_tokens: int) -> str:
  # Run generation ourselves and decode ONLY the newly generated tokens, so
  # the echoed prompt (which contains the "{...}" template placeholder) never
  # reaches the client. answer_question() decodes prompt+continuation together
  # and would leak that placeholder, crashing T3A's JSON parser on `...`.
  formatted = _format_q(prompt)
  enc = TOK(formatted, return_tensors='pt', padding=True).to('cuda')
  input_len = enc['input_ids'].shape[1]
  # MobileWorld only needs one Thought+Action; cap tokens to save memory and
  # stop the model from rambling multiple hallucinated steps.
  max_new = min(max_tokens, 256)
  generation_kwargs = {}
  if temperature > 0:
    generation_kwargs['do_sample'] = True
    generation_kwargs['temperature'] = temperature
  with torch.no_grad():
    out = MDL.generate(
        input_ids=enc['input_ids'],
        attention_mask=enc['attention_mask'],
        kb_kvs=KB_PACK,
        max_new_tokens=max_new,
        tokenizer=TOK,
        output_attentions=OUTPUT_ATTN,
        kb_config=KB_CFG,
        **generation_kwargs,
    )
  new_tokens = out[0][input_len:]
  return TOK.decode(new_tokens, skip_special_tokens=True).strip()


@app.post('/predict')
def predict(req: PredictRequest):
  try:
    text = _clean_single_step(_generate_text(req.prompt, req.temperature, req.max_tokens))
    print(text)
    return {'text': text}
  except Exception as e:  # pylint: disable=broad-exception-caught
    return {'error': str(e)}


@app.post('/v1/chat/completions')
def chat_completions(req: ChatCompletionRequest):
  if req.stream:
    raise HTTPException(status_code=400, detail='stream=true is not supported')
  try:
    prompt = _messages_to_prompt(req.messages)
    text = _clean_single_step(_generate_text(prompt, req.temperature, req.max_tokens))
    created = int(time.time())
    return {
        'id': f'chatcmpl-atlaskv-{created}',
        'object': 'chat.completion',
        'created': created,
        'model': req.model,
        'choices': [
            {
                'index': 0,
                'message': {'role': 'assistant', 'content': text},
                'finish_reason': 'stop',
            }
        ],
        'usage': {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
        },
    }
  except Exception as e:  # pylint: disable=broad-exception-caught
    raise HTTPException(status_code=500, detail=str(e)) from e


@app.get('/v1/models')
def list_models():
  return {
      'object': 'list',
      'data': [
          {
              'id': 'atlaskv',
              'object': 'model',
              'created': 0,
              'owned_by': 'atlaskv',
          }
      ],
  }


@app.get('/health')
def health():
  return {'status': 'ok'}


if __name__ == '__main__':
  # 0.0.0.0 so the other host can reach it (not just localhost).
  uvicorn.run(app, host='0.0.0.0', port=8000)
