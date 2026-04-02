"""
Shared Ollama utilities used by both synthesizer.py and sentiment.py.

Extracting these into a public module prevents sentiment.py from importing
private (_-prefixed) implementation details of synthesizer.py, which breaks
on any synthesizer refactor.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error

_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

_MODEL_PREFERENCE = [
    "meditron",
    "llama3.1:70b",
    "llama3.1:8b",
    "llama3.1",
    "llama3.2:3b",
    "llama3.2",
    "llama3",
    "mistral:7b",
    "mistral",
    "phi4",
    "phi3:medium",
    "phi3",
    "gemma2:9b",
    "gemma2",
    "qwen2.5:7b",
    "qwen2.5",
]


def ollama_available() -> bool:
    try:
        req = urllib.request.Request(
            f"{_OLLAMA_URL}/api/tags",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_best_model() -> str | None:
    """Return the best available Ollama model, or None if Ollama is not running."""
    env_model = os.getenv("OLLAMA_MODEL")
    if env_model:
        return env_model

    try:
        req = urllib.request.Request(f"{_OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            # Exact match first, then base-name prefix — preserves preference order
            for preferred in _MODEL_PREFERENCE:
                for m in data.get("models", []):
                    if m["name"] == preferred:
                        return m["name"]
                    if ":" not in preferred and m["name"].startswith(preferred + ":"):
                        return m["name"]
            if data.get("models"):
                return data["models"][0]["name"]
    except Exception:
        pass
    return None


def call_ollama(messages: list[dict], model: str) -> str:
    """POST to Ollama /api/chat. Returns the assistant message content string."""
    payload = json.dumps({
        "model":    model,
        "messages": messages,
        "format":   "json",
        "stream":   False,
        "options": {
            "temperature": 0.15,
            "num_predict": 2000,
            "top_p": 0.9,
        },
    }).encode()

    req = urllib.request.Request(
        f"{_OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode())
            return body["message"]["content"]
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama connection failed: {e}") from e
    except (KeyError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Unexpected Ollama response format: {e}") from e


def parse_llm_json(raw: str) -> dict:
    """Parse LLM response JSON, handling fences and trailing commas."""
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    brace_match = re.search(r"\{[\s\S]+\}", cleaned)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return {}
