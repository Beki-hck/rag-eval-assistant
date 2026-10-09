"""Minimal Ollama client over plain HTTP (standard library only)."""
from __future__ import annotations

import json
import os
import urllib.request

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not DEFAULT_HOST.startswith("http"):
    DEFAULT_HOST = "http://" + DEFAULT_HOST


def _post(host: str, path: str, payload: dict, timeout: float = 300) -> dict:
    req = urllib.request.Request(
        host.rstrip("/") + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def ollama_chat(model: str, messages: list[dict], host: str = DEFAULT_HOST,
                max_tokens: int = 300, fmt: str | None = None) -> str:
    payload = {"model": model, "messages": messages, "stream": False,
               "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 4096}}
    if fmt:
        payload["format"] = fmt
    return _post(host, "/api/chat", payload)["message"]["content"]


def ollama_embedder(model: str = "nomic-embed-text", host: str = DEFAULT_HOST):
    def embed(texts: list[str]) -> list[list[float]]:
        return _post(host, "/api/embed", {"model": model, "input": texts})["embeddings"]
    return embed


def ollama_available(host: str = DEFAULT_HOST) -> bool:
    try:
        with urllib.request.urlopen(host.rstrip("/") + "/api/tags", timeout=2):
            return True
    except Exception:
        return False
