"""Compare AndroidWorld request prompts, AtlasKV model prompts, and QKV rows.

Example:
    python src/atlaskv/inference/compare_android_world_prompts.py \
      --pkl /path/ContactsAddContact_0.pkl.gz \
      --model-io /path/atlaskv_model_io.jsonl \
      --qkv /path/qkv.json \
      --step 0
"""

from __future__ import annotations

import argparse
import difflib
import gzip
import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


class Dummy:
    """Placeholder for classes missing from the current Python environment."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._args = args
        self._kwargs = kwargs

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state


_DUMMY_CLASSES: Dict[Tuple[str, str], type] = {}


def _dummy_class(module: str, name: str) -> type:
    key = (module, name)
    if key not in _DUMMY_CLASSES:
        _DUMMY_CLASSES[key] = type(name, (Dummy,), {"__module__": module})
    return _DUMMY_CLASSES[key]


class LenientUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> type:
        try:
            return super().find_class(module, name)
        except Exception:
            return _dummy_class(module, name)


def load_pickle(path: Path) -> List[Dict[str, Any]]:
    with gzip.open(path, "rb") as f:
        data = LenientUnpickler(f).load()
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise TypeError(f"Unsupported pickle root type: {type(data).__name__}")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_no}")
            rows.append(row)
    return rows


def load_qkv_rows(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON/JSONL at {path}:{line_no}") from exc
            rows.append(item)
        data = rows

    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        if any(key in data for key in ("Q", "A", "description")):
            return [data]
        rows = [value for value in data.values() if isinstance(value, dict)]
        if rows:
            return rows
    raise ValueError(f"Could not find QKV rows in {path}")


def request_body_from_response(response: Any) -> Dict[str, Any]:
    request = getattr(response, "request", None)
    body = getattr(request, "body", None)
    if body is None:
        return {}
    if isinstance(body, bytes):
        body = body.decode("utf-8", "replace")
    if not isinstance(body, str) or not body:
        return {}
    return json.loads(body)


def payload_text(payload: Dict[str, Any]) -> str:
    pieces: List[str] = []
    for message in payload.get("messages", []):
        content = message.get("content")
        if isinstance(content, str):
            if content:
                pieces.append(content)
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text" and part.get("text"):
                    pieces.append(part["text"])
    return "\n".join(pieces)


def strip_model_wrappers(text: str) -> str:
    text = text.strip()
    llama_prefix = "<|start_header_id|>user<|end_header_id|>"
    llama_suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    if text.startswith(llama_prefix):
        text = text[len(llama_prefix) :].lstrip()
    if text.endswith(llama_suffix):
        text = text[: -len(llama_suffix)].rstrip()
    return strip_role_prefix(text)


def strip_role_prefix(text: str) -> str:
    text = text.strip()
    for prefix in ("USER:\n", "user:\n"):
        if text.startswith(prefix):
            return text[len(prefix) :].lstrip()
    return text


def normalize_for_compare(text: str) -> str:
    text = strip_model_wrappers(text)
    return re.sub(r"[ \t]+", " ", text).strip()


def extract_goal(text: str) -> str:
    patterns = [
        r"The current AndroidWorld user goal is:\s*(.*?)(?:\nHistory:|\n\nHistory:)",
        (
            r"The current user goal/request is:\s*"
            r"(.*?)(?:\n\nHere is a history|\n\nThe current screenshot|\n\nHere is a list)"
        ),
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""


def extract_history(text: str) -> str:
    patterns = [
        r"History:\s*(.*?)(?:\nThe visible UI elements are:|\n\nThe visible UI elements are:)",
        r"Here is a history.*?:\s*\n(.*?)(?:\n\nThe current screenshot|\n\nHere is a list)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""


def ui_lines(text: str) -> List[str]:
    return re.findall(r"^UI element \d+: .*$", text, flags=re.MULTILINE)


def action_lines(text: str) -> List[str]:
    return re.findall(r"^Action:\s*.*$", text, flags=re.MULTILINE)


def extract_action_index(text: str) -> Optional[int]:
    match = re.search(r'"index"\s*:\s*(\d+)', text)
    return int(match.group(1)) if match else None


def words(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_+.-]+", text.lower()))


def overlap_score(query: str, candidate: str) -> float:
    q_words = words(query)
    c_words = words(candidate)
    if not q_words or not c_words:
        return 0.0
    return len(q_words & c_words) / len(q_words | c_words)


def best_qkv_matches(
    model_prompt: str,
    qkv_rows: List[Dict[str, Any]],
    limit: int,
) -> List[Tuple[float, Dict[str, Any]]]:
    scored = []
    model_goal = extract_goal(model_prompt) or model_prompt[:500]
    for row in qkv_rows:
        candidate = "\n".join(str(row.get(key, "")) for key in ("Q", "A", "description", "key_string", "name"))
        score = max(
            overlap_score(model_goal, str(row.get("Q", ""))),
            overlap_score(model_goal, str(row.get("key_string", ""))),
            overlap_score(model_goal, candidate),
        )
        scored.append((score, row))
    return sorted(scored, key=lambda item: item[0], reverse=True)[:limit]


def short(text: str, width: int = 220) -> str:
    text = text.replace("\n", "\\n")
    return text if len(text) <= width else text[:width] + "..."


def describe_prompt(label: str, text: str) -> Dict[str, Any]:
    normalized = normalize_for_compare(text)
    lines = ui_lines(normalized)
    return {
        "label": label,
        "chars": len(text),
        "normalized_chars": len(normalized),
        "goal": extract_goal(normalized),
        "history_chars": len(extract_history(normalized)),
        "ui_count": len(lines),
        "first_ui": lines[0] if lines else "",
        "last_ui": lines[-1] if lines else "",
        "action_line_count": len(action_lines(normalized)),
        "action_index": extract_action_index(normalized),
        "prefix": short(normalized[:500]),
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_diff(path: Path, left_name: str, left: str, right_name: str, right: str) -> None:
    diff = difflib.unified_diff(
        left.splitlines(),
        right.splitlines(),
        fromfile=left_name,
        tofile=right_name,
        lineterm="",
    )
    write_text(path, "\n".join(diff) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare AndroidWorld request/model/QKV prompts")
    parser.add_argument("--pkl", type=Path, required=True, help="AndroidWorld .pkl.gz episode file")
    parser.add_argument(
        "--model-io",
        type=Path,
        required=True,
        help="AtlasKV debug JSONL from test_server_debug_model_io.py",
    )
    parser.add_argument("--qkv", type=Path, required=True, help="QKV JSON, JSONL, or pasted single-row JSON")
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--model-record", type=int, help="Model IO JSONL record index; defaults to --step")
    parser.add_argument("--top-k", type=int, default=5, help="Number of closest QKV rows to print")
    parser.add_argument("--out-dir", type=Path, default=Path("prompt_compare_out"))
    args = parser.parse_args()

    trials = load_pickle(args.pkl)
    if args.trial >= len(trials):
        raise IndexError(f"--trial {args.trial} is out of range; file contains {len(trials)} trial(s)")
    trial = trials[args.trial]
    episode = trial.get("episode_data") or {}
    action_prompts = episode.get("action_prompt") or []
    raw_responses = episode.get("action_raw_response") or []
    if args.step >= len(action_prompts):
        raise IndexError(f"--step {args.step} is out of range; episode contains {len(action_prompts)} action prompt(s)")

    saved_request_prompt = action_prompts[args.step] or ""
    request_payload = request_body_from_response(raw_responses[args.step]) if args.step < len(raw_responses) else {}
    request_prompt = payload_text(request_payload) or saved_request_prompt

    model_rows = load_jsonl(args.model_io)
    model_index = args.step if args.model_record is None else args.model_record
    if model_index >= len(model_rows):
        raise IndexError(f"model record {model_index} is out of range; file contains {len(model_rows)} record(s)")
    model_record = model_rows[model_index]
    model_prompt = str(model_record.get("prompt", ""))
    formatted_prompt = str(model_record.get("formatted_prompt", ""))

    qkv_rows = load_qkv_rows(args.qkv)
    matches = best_qkv_matches(model_prompt, qkv_rows, args.top_k)
    best_qkv = matches[0][1] if matches else {}
    best_qkv_prompt = str(best_qkv.get("Q", ""))
    best_qkv_answer = str(best_qkv.get("A") or best_qkv.get("description") or "")

    normalized_model = normalize_for_compare(model_prompt)
    normalized_qkv = normalize_for_compare(best_qkv_prompt)
    normalized_request = normalize_for_compare(request_prompt)

    summary = {
        "pkl": str(args.pkl),
        "model_io": str(args.model_io),
        "qkv": str(args.qkv),
        "trial": args.trial,
        "step": args.step,
        "model_record": model_index,
        "task_template": trial.get("task_template"),
        "episode_goal": trial.get("goal"),
        "request_max_tokens": request_payload.get("max_tokens"),
        "model_context": model_record.get("context"),
        "model_usage": model_record.get("usage"),
        "model_prompt_strategy": model_record.get("prompt_strategy"),
        "model_looks_like_qkv_action_prompt": model_record.get("looks_like_qkv_action_prompt"),
        "model_raw_output": model_record.get("raw_output"),
        "request_prompt_equals_saved_action_prompt": request_prompt == saved_request_prompt,
        "model_prompt_equals_best_qkv_Q_normalized": normalized_model == normalized_qkv,
        "model_prompt_equals_request_prompt_normalized": normalized_model == normalized_request,
        "request": describe_prompt("request", request_prompt),
        "model": describe_prompt("model_prompt", model_prompt),
        "formatted_model": describe_prompt("formatted_prompt", formatted_prompt),
        "best_qkv": {
            **describe_prompt("best_qkv_Q", best_qkv_prompt),
            "name": best_qkv.get("name"),
            "description_type": best_qkv.get("description_type"),
            "key_string": best_qkv.get("key_string"),
            "answer": best_qkv_answer,
            "answer_action_index": extract_action_index(best_qkv_answer),
        },
        "top_qkv_matches": [
            {
                "score": score,
                "name": row.get("name"),
                "goal": extract_goal(str(row.get("Q", ""))),
                "key_string": row.get("key_string"),
                "action_index": extract_action_index(str(row.get("A") or row.get("description") or "")),
            }
            for score, row in matches
        ],
    }

    out_dir = args.out_dir
    write_text(out_dir / f"request_step{args.step}.txt", request_prompt)
    write_text(out_dir / f"model_prompt_record{model_index}.txt", model_prompt)
    write_text(out_dir / f"formatted_prompt_record{model_index}.txt", formatted_prompt)
    write_text(out_dir / "best_qkv_Q.txt", best_qkv_prompt)
    write_text(out_dir / "best_qkv_A.txt", best_qkv_answer)
    write_text(out_dir / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    write_diff(out_dir / "diff_request_vs_model.diff", "request_prompt", request_prompt, "model_prompt", model_prompt)
    write_diff(
        out_dir / "diff_model_vs_best_qkv.diff",
        "model_prompt_normalized",
        normalized_model,
        "best_qkv_Q_normalized",
        normalized_qkv,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"\nWrote comparison files to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
