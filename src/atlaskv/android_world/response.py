"""OpenAI-compatible HTTP response helpers for AndroidWorld failures."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

def openai_chat_completion_body(
    model: str,
    content: str,
    usage: Dict[str, int],
    **metadata: Any,
) -> Dict[str, Any]:
    """Build a non-streaming OpenAI-compatible chat completion envelope."""

    body: Dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }
    body.update(metadata)
    return body


def openai_error_body(
    message: str,
    *,
    error_type: str = "invalid_request_error",
    code: Optional[str] = None,
    param: Optional[str] = None,
) -> Dict[str, Any]:
    """Build an OpenAI-compatible error envelope."""

    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }


def openai_error_response(
    message: str,
    *,
    status_code: int = 422,
    error_type: str = "invalid_request_error",
    code: Optional[str] = None,
    param: Optional[str] = None,
) -> Any:
    """Return an OpenAI-compatible JSON error response."""

    # Keep the protocol and envelope helpers importable in lightweight tools
    # that do not load the AtlasKV HTTP server or its FastAPI dependency.
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=status_code,
        content=openai_error_body(message, error_type=error_type, code=code, param=param),
    )
