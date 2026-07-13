"""OpenAI-compatible chat completion server for AtlasKV.

The adapter intentionally keeps the external API small:

    POST /v1/chat/completions

It converts OpenAI chat messages into a plain text prompt, runs the AtlasKV
generation path, and returns an OpenAI-style response body. AndroidWorld T3A
requests are classified and handled by the dedicated ``atlaskv.android_world``
package so action selection and summarization keep their distinct contracts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from atlaskv.android_world import (
    AndroidWorldOutputError,
    PromptKind,
    openai_chat_completion_body,
    openai_error_response,
    process_t3a_output,
)
from atlaskv.kb_encoder import KBEncoder
from atlaskv.models.kblam_config import AtlasKVConfig, KBLaMConfig
from atlaskv.models.llama3_model import AtlaskvLlamaForCausalLM, KblamLlamaForCausalLM, set_llama_attention_classes
from atlaskv.models.phi3_model import KBLaMPhi3ForCausalLM
from atlaskv.utils.train_utils import get_hierarchical_kg_embd, get_kb_embd
from atlaskv.utils.eval_utils import (
    _format_Q_llama,
    _format_Q_phi3,
)

ChatRole = Literal["system", "user", "assistant", "tool"]


class ImageURL(BaseModel):
    url: str


class MessagePart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Union[str, ImageURL, Dict[str, Any]]] = None


class ChatMessage(BaseModel):
    role: ChatRole
    content: Union[str, List[MessagePart], None] = ""


class ChatCompletionRequest(BaseModel):
    model: str = "atlaskv"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0
    max_tokens: Optional[int] = Field(default=1024, alias="max_tokens")
    stream: Optional[bool] = False


class PredictionRequest(BaseModel):
    prompt: str
    model: str = "atlaskv"
    temperature: Optional[float] = 0
    max_tokens: Optional[int] = 1024


@dataclass
class AdapterConfig:
    model_name: str
    llm_type: str
    llm_base_dir: str
    model_dir: str
    encoder_dir: str
    encoder_spec: str
    query_head_path: str
    dataset_path: Optional[str]
    precomputed_embed_keys_path: Optional[str]
    precomputed_embed_values_path: Optional[str]
    precomputed_embed_root_keys_path: Optional[str]
    precomputed_embed_inter_keys_path: Optional[str]
    precomputed_embed_root_c2id_mapping_path: Optional[str]
    precomputed_embed_inter_c2id_mapping_path: Optional[str]
    precomputed_embed_root_id2c_mapping_path: Optional[str]
    precomputed_embed_inter_id2c_mapping_path: Optional[str]
    kb_size: int
    seed: int
    use_kg: bool
    use_hierarchial_kv: bool
    kb_layer_frequency: int
    kb_scale_factor: Optional[int]
    root_top_k_kb: int
    inter_top_k_kb: int
    leaf_top_k_kb: int
    encoding_batch_size: int
    device: str
    clean_single_action: bool
    include_image_placeholders: bool
    inject_kv: bool


def load_dataset_rows(dataset_path: str) -> List[Dict[str, Any]]:
    with open(dataset_path, encoding="utf-8") as f:
        first_char = ""
        while True:
            char = f.read(1)
            if not char:
                break
            if not char.isspace():
                first_char = char
                break

    if dataset_path.endswith(".jsonl") and first_char != "[":
        rows: List[Dict[str, Any]] = []
        with open(dataset_path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    preview = line[:120]
                    raise ValueError(f"Invalid JSONL row {line_no} in {dataset_path}: {preview}") from exc
                if not isinstance(item, dict):
                    raise ValueError(f"JSONL row {line_no} in {dataset_path} is not an object")
                rows.append(item)
        return rows

    with open(dataset_path, encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"JSON dataset {dataset_path} must contain a top-level list")
    return rows


class KBIndex:
    def __init__(self, encoder: KBEncoder, rows: List[Dict[str, Any]], key_path: str, value_path: str) -> None:
        self.encoder = encoder
        self.rows = rows
        self._keys = np.load(key_path).astype("float32")
        self._vals = np.load(value_path).astype("float32")
        if self._keys is not None:
            assert len(self.rows) == len(self._keys)

    def use_cache(self) -> bool:
        return self._keys is not None and self._vals is not None

    def encode(self, idx: Iterable[int]) -> Tuple[np.ndarray, np.ndarray]:
        if self.use_cache():
            return get_kb_embd(self.encoder, idx, precomputed_embd=(self._keys, self._vals))
        return get_kb_embd(self.encoder, idx, kb_dict=self.rows)

class KGIndex:
    def __init__(
        self,
        encoder: KBEncoder,
        rows: List[Dict[str, Any]],
        leaf_key_path: str,
        value_path: str,
        root_key_path: str,
        inter_key_path: str,
        root_c2id_path: str,
        inter_c2id_path: str,
        root_id2c_path: str,
        inter_id2c_path: str,
    ) -> None:
        self.encoder = encoder
        self.rows = rows
        self.leaf_keys = np.load(leaf_key_path).astype("float32")
        self.values = np.load(value_path).astype("float32")
        self.root_keys = np.load(root_key_path).astype("float32")
        self.inter_keys = np.load(inter_key_path).astype("float32")
        with open(root_c2id_path, encoding="utf-8") as f:
            self.root_c2id = json.load(f)
        with open(inter_c2id_path, encoding="utf-8") as f:
            self.inter_c2id = json.load(f)
        with open(root_id2c_path, encoding="utf-8") as f:
            self.root_id2c = json.load(f)
        with open(inter_id2c_path, encoding="utf-8") as f:
            self.inter_id2c = json.load(f)
        if len(self.rows) != len(self.leaf_keys):
            raise ValueError("Dataset rows and leaf key embeddings have different lengths")

    def use_cache(self) -> bool:
        return self.leaf_keys is not None and self.values is not None

    def encode_hier(self, idx: Iterable[int]):
        if self.use_cache():
            return get_hierarchical_kg_embd(
                self.encoder,
                idx,
                self.root_id2c,
                self.inter_id2c,
                self.root_c2id,
                self.inter_c2id,
                mode="eval",
                precomputed_embd=(self.leaf_keys, self.inter_keys, self.root_keys, self.values),
            )
        return get_hierarchical_kg_embd(
            self.encoder,
            idx,
            self.root_id2c,
            self.inter_id2c,
            self.root_c2id,
            self.inter_c2id,
            mode="eval",
            kb_dict=self.rows,
        )


class AtlasKVOpenAIAdapter:
    def __init__(self, config: AdapterConfig) -> None:
        self.config = config
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        set_llama_attention_classes(config.use_kg)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.llm_base_dir,
            trust_remote_code=True,
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.pad_token or "^"

        if config.llm_type == "llama3":
            ctor = AtlaskvLlamaForCausalLM if config.use_kg else KblamLlamaForCausalLM
            self.model = ctor.from_pretrained(
                config.model_dir,
                device_map=config.device,
                torch_dtype="auto",
                trust_remote_code=True,
            )
            if config.query_head_path:
                self.model.load_query_head(config.query_head_path)
        elif config.llm_type == "phi3":
            self.model = KBLaMPhi3ForCausalLM.from_pretrained(
                config.model_dir,
                device_map=config.device,
                torch_dtype="auto",
                trust_remote_code=True,
            )
        else:
            raise ValueError(f"Unsupported llm_type: {config.llm_type}")

        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.model.eval()

        self.kb_config = self._build_kb_config()
        self.kb_pack = self._load_kb_pack()
        self.actual_kb_size = self._kb_pack_size(self.kb_pack)

    @staticmethod
    def _kb_pack_size(kb_pack: Any) -> int:
        """Return the number of leaf KV vectors actually injected."""
        if kb_pack is None:
            return 0
        # Hierarchical packs contain (root keys, intermediate keys, leaf keys,
        # values, ...); regular packs contain (keys, values).
        vector = kb_pack[2] if len(kb_pack) >= 4 else kb_pack[0]
        return int(vector.shape[-2] if len(vector.shape) >= 3 else vector.shape[0])

    def _build_kb_config(self) -> Union[KBLaMConfig, AtlasKVConfig]:
        if self.config.use_kg:
            return AtlasKVConfig(
                sep_query_head=True,
                kb_layer_frequency=self.config.kb_layer_frequency,
                kb_scale_factor=self.config.kb_scale_factor,
                root_top_k_kb=self.config.root_top_k_kb,
                inter_top_k_kb=self.config.inter_top_k_kb,
                leaf_top_k_kb=self.config.leaf_top_k_kb,
                use_hierarchial_kv=self.config.use_hierarchial_kv,
            )
        return KBLaMConfig(
            sep_query_head=True,
            kb_layer_frequency=self.config.kb_layer_frequency,
            kb_scale_factor=self.config.kb_scale_factor,
            use_hierarchial_kv=False,
        )

    def _load_kb_pack(self):
        if not self.config.dataset_path:
            return None
        required = [self.config.encoder_dir, self.config.precomputed_embed_keys_path, self.config.precomputed_embed_values_path]
        if any(not item for item in required):
            raise ValueError("dataset_path requires encoder_dir, precomputed key embeddings, and value embeddings")

        rows = load_dataset_rows(self.config.dataset_path)

        encoder = KBEncoder(
            encoder_name=self.config.encoder_spec,
            projector_type="linear",
            endpoint_url=os.environ.get("ATLASKV_OAI_ENDPOINT_URL", "your_endpoint_url"),
            endpoint_api_key=os.environ.get("ATLASKV_OAI_ENDPOINT_API_KEY", "your_endpoint_api_key"),
            out_dim=self.model.config.hidden_size * (self.model.config.num_hidden_layers // self.config.kb_layer_frequency + 1),
            frozen_base_model=True,
            projector_kwargs={"mlp_depth": 1, "mlp_hidden_dim": 512},
            device=torch.device(self.config.device),
            get_oai_embd_online=True if self.config.encoder_spec == "OAI" else False,
            encoding_batch_size=self.config.encoding_batch_size,
        )
        encoder.load_state_dict(torch.load(self.config.encoder_dir, map_location=self.config.device))
        self.encoder = encoder.to(self.config.device)

        rng = np.random.default_rng(self.config.seed)
        kb_size = len(rows) if self.config.kb_size < 0 else min(self.config.kb_size, len(rows))
        take_idx = np.arange(len(rows)) if kb_size == len(rows) else rng.choice(len(rows), size=kb_size, replace=False)

        if self.config.use_kg and self.config.use_hierarchial_kv:
            paths = [
                self.config.precomputed_embed_keys_path,
                self.config.precomputed_embed_values_path,
                self.config.precomputed_embed_root_keys_path,
                self.config.precomputed_embed_inter_keys_path,
                self.config.precomputed_embed_root_c2id_mapping_path,
                self.config.precomputed_embed_inter_c2id_mapping_path,
                self.config.precomputed_embed_root_id2c_mapping_path,
                self.config.precomputed_embed_inter_id2c_mapping_path,
            ]
            if any(not path for path in paths):
                raise ValueError("Hierarchical KG mode requires all root/inter embedding and mapping paths")
            kg = KGIndex(encoder, rows, *(path for path in paths if path is not None))
        else:
            kg = None

        kb = KBIndex(
            encoder,
            rows,
            self.config.precomputed_embed_keys_path or "",
            self.config.precomputed_embed_values_path or "",
        )

        with torch.no_grad():
            if self.config.use_kg and self.config.use_hierarchial_kv:
                kb_pack = kg.encode_hier(take_idx)
            elif self.config.use_kg:
                kb_pack = kb.encode(take_idx)
            else:
                kb_pack = kb.encode(take_idx)

        return kb_pack

    def generate(self, prompt: str, max_new_tokens: int, temperature: Optional[float]) -> Tuple[str, Dict[str, int]]:
        formatted = self._format_prompt(prompt)
        tokenized = self.tokenizer(formatted, return_tensors="pt", padding=True).to(self.config.device)
        prompt_tokens = int(tokenized["attention_mask"].sum().item())
        kb_pack = self.kb_pack if self.config.inject_kv else None
        generation_kwargs: Dict[str, Any] = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "kb_kvs": kb_pack,
            "max_new_tokens": max_new_tokens,
            "tokenizer": self.tokenizer,
            "output_attentions": False,
            "kb_config": self.kb_config,
        }
        if temperature is not None and temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = self.model.generate(**generation_kwargs)
        if output_ids.ndim == 1:
            output_ids = output_ids.unsqueeze(0)
        generated_ids = output_ids[:, tokenized["input_ids"].shape[1] :]
        completion_tokens = int(generated_ids.shape[1])
        decoded = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return self._strip_special_tokens(decoded), usage

    def _format_prompt(self, prompt: str) -> str:
        if self.config.llm_type == "llama3":
            return _format_Q_llama(prompt)
            # return f"<|start_header_id|>user<|end_header_id|> {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        if self.config.llm_type == "phi3":
            return _format_Q_phi3(prompt)
            # return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        return prompt
    
    def _strip_special_tokens(self, decoded: str) -> str:
        replacements = [
            "<|begin_of_text|>",
            "<|eot_id|>",
            "<|start_header_id|>assistant<|end_header_id|>",
            "<|start_header_id|>user<|end_header_id|>",
            "<|end_of_text|>",
            "<|end|>",
            "<|assistant|>",
            "<|user|>",
        ]
        text = decoded
        for marker in replacements:
            text = text.replace(marker, "")
        return text.strip()


adapter: Optional[AtlasKVOpenAIAdapter] = None
app = FastAPI(title="AtlasKV OpenAI-compatible API", version="0.1.0")


def _message_content_to_text(message: ChatMessage, include_image_placeholders: bool) -> str:
    content = message.content
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    pieces: List[str] = []
    image_count = 0
    for part in content:
        if part.type == "text" and part.text:
            pieces.append(part.text)
        elif part.type == "image_url":
            image_count += 1
            if include_image_placeholders:
                pieces.append(f"[image_url omitted: screenshot {image_count}]")
    return "\n".join(pieces)


def messages_to_prompt(messages: List[ChatMessage], include_image_placeholders: bool = True) -> str:
    blocks: List[str] = []
    for message in messages:
        content = _message_content_to_text(message, include_image_placeholders)
        if content:
            blocks.append(f"{message.role.upper()}:\n{content}")
    return "\n\n".join(blocks).strip()


def keep_first_action_block(text: str) -> str:
    """Trim output to the first complete MobileWorld-style Action JSON block."""
    action_match = re.search(r"Action\s*:\s*", text)
    if not action_match:
        return text.strip()

    brace_start = text.find("{", action_match.end())
    if brace_start == -1:
        return text[: action_match.end()].strip()

    depth = 0
    in_string = False
    escaped = False
    for idx in range(brace_start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[: idx + 1].strip()
    return text.strip()


def _completion_response(model: str, content: str, usage: Dict[str, int]) -> Dict[str, Any]:
    return openai_chat_completion_body(
        model,
        content,
        usage,
        kb_size=adapter.actual_kb_size,
        kb_layer_frequency=adapter.config.kb_layer_frequency,
        kb_scale_factor=adapter.config.kb_scale_factor,
        kv_injected=adapter.config.inject_kv,
    )


def _process_generated_output(prompt: str, output: str) -> str:
    """Apply AndroidWorld handling only when the prompt matches T3A."""

    processed = process_t3a_output(prompt, output)
    if processed.prompt_kind is PromptKind.OTHER and adapter.config.clean_single_action:
        return keep_first_action_block(processed.content)
    return processed.content


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> Any:
    if adapter is None:
        raise HTTPException(status_code=503, detail="AtlasKV adapter is not initialized")
    output, _usage = adapter.generate(request.prompt, request.max_tokens or 1024, request.temperature)
    try:
        output = _process_generated_output(request.prompt, output)
    except AndroidWorldOutputError as exc:
        return openai_error_response(
            str(exc), error_type="invalid_response_error", code=exc.code, param="completion"
        )
    return {
        "text": output,
        "kb_size": adapter.actual_kb_size,
        "kb_layer_frequency": adapter.config.kb_layer_frequency,
        "kb_scale_factor": adapter.config.kb_scale_factor,
        "kv_injected": adapter.config.inject_kv,
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest) -> Any:
    if adapter is None:
        return openai_error_response(
            "AtlasKV adapter is not initialized.",
            status_code=503,
            error_type="server_error",
            code="adapter_not_initialized",
        )
    if request.stream:
        return openai_error_response(
            "stream=true is not supported by this adapter.",
            status_code=400,
            code="unsupported_streaming",
            param="stream",
        )

    prompt = messages_to_prompt(request.messages, adapter.config.include_image_placeholders)
    if not prompt:
        return openai_error_response(
            "messages produced an empty prompt.",
            status_code=400,
            code="empty_prompt",
            param="messages",
        )

    output, usage = adapter.generate(prompt, request.max_tokens or 1024, request.temperature)
    try:
        output = _process_generated_output(prompt, output)
    except AndroidWorldOutputError as exc:
        return openai_error_response(
            str(exc), error_type="invalid_response_error", code=exc.code, param="completion"
        )
    return _completion_response(request.model or adapter.config.model_name, output, usage)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an OpenAI-compatible AtlasKV adapter server")
    parser.add_argument("--host", default=os.environ.get("ATLASKV_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ATLASKV_PORT", "8000")))
    parser.add_argument("--model_name", default=os.environ.get("ATLASKV_MODEL_NAME", "atlaskv"))
    parser.add_argument("--llm_type", choices=["llama3", "phi3"], default=os.environ.get("ATLASKV_LLM_TYPE", "llama3"))
    parser.add_argument("--llm_base_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--encoder_dir", default="")
    parser.add_argument("--encoder_spec", default=os.environ.get("ATLASKV_ENCODER_SPEC", "OAI"))
    parser.add_argument("--query_head_path", default="")
    parser.add_argument("--dataset_path")
    parser.add_argument("--precomputed_embed_keys_path")
    parser.add_argument("--precomputed_embed_values_path")
    parser.add_argument("--precomputed_embed_root_keys_path")
    parser.add_argument("--precomputed_embed_inter_keys_path")
    parser.add_argument("--precomputed_embed_root_c2id_mapping_path")
    parser.add_argument("--precomputed_embed_inter_c2id_mapping_path")
    parser.add_argument("--precomputed_embed_root_id2c_mapping_path")
    parser.add_argument("--precomputed_embed_inter_id2c_mapping_path")
    parser.add_argument("--kb_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_kg", action="store_true")
    parser.add_argument("--use_hierarchial_kv", action="store_true", default=False)
    parser.add_argument("--kb_layer_frequency", type=int, default=3)
    parser.add_argument("--kb_scale_factor", type=int)
    parser.add_argument("--root_top_k_kb", type=int, default=128)
    parser.add_argument("--inter_top_k_kb", type=int, default=64)
    parser.add_argument("--leaf_top_k_kb", type=int, default=16)
    parser.add_argument("--encoding_batch_size", type=int, default=64)
    parser.add_argument("--device", default=os.environ.get("ATLASKV_DEVICE", "cuda"))
    parser.add_argument("--clean_single_action", action="store_true", default=True)
    parser.add_argument("--no_clean_single_action", action="store_false", dest="clean_single_action")
    parser.add_argument("--include_image_placeholders", action="store_true", default=True)
    parser.add_argument("--no_include_image_placeholders", action="store_false", dest="include_image_placeholders")
    parser.add_argument("--inject_kv", action="store_true", default=True)
    parser.add_argument("--disable_kv_injection", action="store_false", dest="inject_kv")
    return parser


def _config_from_args(args: argparse.Namespace) -> AdapterConfig:
    return AdapterConfig(
        model_name=args.model_name,
        llm_type=args.llm_type,
        llm_base_dir=args.llm_base_dir,
        model_dir=args.model_dir,
        encoder_dir=args.encoder_dir,
        encoder_spec=args.encoder_spec,
        query_head_path=args.query_head_path,
        dataset_path=args.dataset_path,
        precomputed_embed_keys_path=args.precomputed_embed_keys_path,
        precomputed_embed_values_path=args.precomputed_embed_values_path,
        precomputed_embed_root_keys_path=args.precomputed_embed_root_keys_path,
        precomputed_embed_inter_keys_path=args.precomputed_embed_inter_keys_path,
        precomputed_embed_root_c2id_mapping_path=args.precomputed_embed_root_c2id_mapping_path,
        precomputed_embed_inter_c2id_mapping_path=args.precomputed_embed_inter_c2id_mapping_path,
        precomputed_embed_root_id2c_mapping_path=args.precomputed_embed_root_id2c_mapping_path,
        precomputed_embed_inter_id2c_mapping_path=args.precomputed_embed_inter_id2c_mapping_path,
        kb_size=args.kb_size,
        seed=args.seed,
        use_kg=args.use_kg,
        use_hierarchial_kv=args.use_hierarchial_kv,
        kb_layer_frequency=args.kb_layer_frequency,
        kb_scale_factor=args.kb_scale_factor,
        root_top_k_kb=args.root_top_k_kb,
        inter_top_k_kb=args.inter_top_k_kb,
        leaf_top_k_kb=args.leaf_top_k_kb,
        encoding_batch_size=args.encoding_batch_size,
        device=args.device,
        clean_single_action=args.clean_single_action,
        include_image_placeholders=args.include_image_placeholders,
        inject_kv=args.inject_kv,
    )


def main() -> None:
    global adapter

    args = build_parser().parse_args()
    adapter = AtlasKVOpenAIAdapter(_config_from_args(args))

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
