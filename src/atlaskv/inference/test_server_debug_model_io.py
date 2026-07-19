"""Debug AtlasKV server that records model-side prompts and outputs.

This entrypoint intentionally keeps ``test_server.py`` clean. It reuses the
normal OpenAI-compatible server, but defaults AndroidWorld prompt rewriting to
``qkv_action_v1`` and writes the prompt that actually reaches the tokenizer.
"""

from __future__ import annotations

import contextvars
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from atlaskv.inference import test_server as base


DEFAULT_DEBUG_MODEL_IO_PATH = "atlaskv_model_io.jsonl"

_REQUEST_CONTEXT: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "atlaskv_debug_request_context",
    default={},
)
_ORIGINAL_BUILD_PARSER = base.build_parser
_ORIGINAL_CONFIG_FROM_ARGS = base._config_from_args
_ORIGINAL_REWRITE_CHAT_COMPLETION_PAYLOAD = base.rewrite_chat_completion_payload


def _payload_text_chars(payload: Dict[str, Any]) -> int:
    total = 0
    for message in payload.get("messages", []):
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            total += sum(len(part.get("text", "")) for part in content if part.get("type") == "text")
    return total


def _payload_image_count(payload: Dict[str, Any]) -> int:
    count = 0
    for message in payload.get("messages", []):
        content = message.get("content")
        if isinstance(content, list):
            count += sum(1 for part in content if part.get("type") == "image_url")
    return count


def rewrite_chat_completion_payload(
    request_payload: Dict[str, Any],
    prompt_strategy: base.PromptStrategy,
    action_max_tokens: Optional[int] = base.DEFAULT_ACTION_MAX_TOKENS,
) -> Dict[str, Any]:
    rewritten = _ORIGINAL_REWRITE_CHAT_COMPLETION_PAYLOAD(request_payload, prompt_strategy, action_max_tokens)
    _REQUEST_CONTEXT.set(
        {
            "endpoint": "/v1/chat/completions",
            "request_model": request_payload.get("model"),
            "rewritten_model": rewritten.get("model"),
            "requested_max_tokens": request_payload.get("max_tokens"),
            "rewritten_max_tokens": rewritten.get("max_tokens"),
            "request_message_count": len(request_payload.get("messages", [])),
            "rewritten_message_count": len(rewritten.get("messages", [])),
            "request_text_chars": _payload_text_chars(request_payload),
            "rewritten_text_chars": _payload_text_chars(rewritten),
            "request_image_count": _payload_image_count(request_payload),
            "rewritten_image_count": _payload_image_count(rewritten),
            "request_was_rewritten": request_payload != rewritten,
        }
    )
    return rewritten


class DebugModelIOAdapter(base.AtlasKVOpenAIAdapter):
    def _debug_model_io_path(self) -> Optional[str]:
        return getattr(self.config, "debug_model_io_path", None)

    def _write_debug_record(self, record: Dict[str, Any]) -> None:
        path = self._debug_model_io_path()
        if not path:
            return
        output_path = Path(path)
        if output_path.parent != Path("."):
            output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def generate(self, prompt: str, max_new_tokens: int, temperature: Optional[float]) -> Tuple[str, Dict[str, int]]:
        request_context = _REQUEST_CONTEXT.get({})
        _REQUEST_CONTEXT.set({})

        formatted = self._format_prompt(prompt)
        tokenized = self.tokenizer(formatted, return_tensors="pt", padding=True)
        prompt_tokens_before_generate = int(tokenized["attention_mask"].sum().item())
        base_record: Dict[str, Any] = {
            "id": uuid.uuid4().hex,
            "created": time.time(),
            "context": request_context,
            "prompt_strategy": self.config.prompt_strategy,
            "inject_kv": self.config.inject_kv,
            "actual_kb_size": self.actual_kb_size,
            "kb_layer_frequency": self.config.kb_layer_frequency,
            "kb_scale_factor": self.config.kb_scale_factor,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "llm_base_dir": self.config.llm_base_dir,
            "model_dir": self.config.model_dir,
            "encoder_spec": self.config.encoder_spec,
            "tokenizer_class": type(self.tokenizer).__name__,
            "tokenizer_name_or_path": getattr(self.tokenizer, "name_or_path", None),
            "tokenizer_model_max_length": getattr(self.tokenizer, "model_max_length", None),
            "model_max_position_embeddings": getattr(self.model.config, "max_position_embeddings", None),
            "input_ids_shape_before_generate": list(tokenized["input_ids"].shape),
            "attention_mask_shape_before_generate": list(tokenized["attention_mask"].shape),
            "prompt_tokens_before_generate": prompt_tokens_before_generate,
            "prompt_chars": len(prompt),
            "formatted_prompt_chars": len(formatted),
            "looks_like_qkv_action_prompt": "What is the next AndroidWorld action?" in prompt,
            "prompt": prompt,
            "formatted_prompt": formatted,
        }

        try:
            output, usage = super().generate(prompt, max_new_tokens, temperature)
        except Exception as exc:
            self._write_debug_record(
                {
                    **base_record,
                    "ok": False,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                }
            )
            raise

        self._write_debug_record(
            {
                **base_record,
                "ok": True,
                "usage": usage,
                "prompt_tokens_match_usage": prompt_tokens_before_generate == usage.get("prompt_tokens"),
                "raw_output": output,
            }
        )
        return output, usage


def build_parser():
    parser = _ORIGINAL_BUILD_PARSER()
    parser.description = "Run AtlasKV adapter server with model-side prompt/output JSONL debugging enabled"
    parser.set_defaults(prompt_strategy=os.environ.get("ATLASKV_PROMPT_STRATEGY", "qkv_action_v1"))
    parser.add_argument(
        "--debug_model_io_path",
        default=os.environ.get("ATLASKV_DEBUG_MODEL_IO_PATH", DEFAULT_DEBUG_MODEL_IO_PATH),
        help="JSONL path for rewritten prompts, formatted model inputs, token counts, and raw outputs.",
    )
    parser.add_argument(
        "--disable_model_io_debug",
        action="store_true",
        help="Run this debug entrypoint without writing model-side JSONL records.",
    )
    return parser


def _config_from_args(args):
    config = _ORIGINAL_CONFIG_FROM_ARGS(args)
    config.debug_model_io_path = "" if args.disable_model_io_debug else args.debug_model_io_path
    return config


def main() -> None:
    base.AtlasKVOpenAIAdapter = DebugModelIOAdapter
    base.build_parser = build_parser
    base._config_from_args = _config_from_args
    base.rewrite_chat_completion_payload = rewrite_chat_completion_payload
    base.main()


if __name__ == "__main__":
    main()
