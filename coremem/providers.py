"""Lightweight LLM provider factory — zero SDK dependencies (httpx only)."""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

import httpx


# ── Protocol ──────────────────────────────────────────────────────────────


class LLMProvider(Protocol):
    """Escape hatch for custom providers that don't match known prefixes."""

    async def chat(self, messages: list[dict[str, Any]]) -> ChatResponse: ...


# ── Response ──────────────────────────────────────────────────────────────


class ChatResponse:
    def __init__(self, content: str, model: str = "", usage: dict[str, int] | None = None):
        self.content = content
        self.model = model
        self.usage = usage or {}


# ── Adapters ──────────────────────────────────────────────────────────────


class _OpenAIAdapter:
    """OpenAI-compatible chat completions adapter."""

    def __init__(self, model: str, api_key: str, base_url: str):
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    async def chat(self, messages: list[dict[str, Any]]) -> ChatResponse:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        body = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.0,
        }
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
            resp = await client.post(
                f"{self._base_url}/v1/chat/completions",
                json=body, headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
        choice = data["choices"][0]
        content = choice["message"]["content"]
        return ChatResponse(
            content=content,
            model=data.get("model", self._model),
            usage=data.get("usage", {}),
        )


class _AnthropicAdapter:
    """Anthropic Messages API adapter."""

    def __init__(self, model: str, api_key: str, base_url: str):
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    async def chat(self, messages: list[dict[str, Any]]) -> ChatResponse:
        system = ""
        user_messages: list[dict[str, Any]] = []
        for m in messages:
            if m.get("role") == "system":
                system = m.get("content", "")
            else:
                user_messages.append(m)

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": user_messages,
            "temperature": 0.0,
        }
        if system:
            body["system"] = system

        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
            resp = await client.post(
                f"{self._base_url}/v1/messages",
                json=body, headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
        content = data["content"][0]["text"]
        return ChatResponse(
            content=content,
            model=data.get("model", self._model),
            usage=data.get("usage", {}),
        )


class _OllamaCloudAdapter:
    """Ollama Cloud native /api/chat adapter."""

    def __init__(self, model: str, api_key: str, base_url: str):
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    async def chat(self, messages: list[dict[str, Any]]) -> ChatResponse:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
            resp = await client.post(
                f"{self._base_url}/api/chat",
                json=body, headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
        content = data.get("message", {}).get("content", "")
        return ChatResponse(
            content=content,
            model=data.get("model", self._model),
        )


class _GeminiAdapter:
    """Gemini generateContent API adapter (OpenAI-compatible path)."""

    def __init__(self, model: str, api_key: str, base_url: str):
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    async def chat(self, messages: list[dict[str, Any]]) -> ChatResponse:
        system = ""
        contents: list[dict[str, Any]] = []
        for m in messages:
            if m.get("role") == "system":
                system = m.get("content", "")
            else:
                role = "model" if m.get("role") == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": m.get("content", "")}]})

        query = f"{self._base_url}/v1beta/models/{self._model}:generateContent"
        if self._api_key:
            query += f"?key={self._api_key}"

        body: dict[str, Any] = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
            resp = await client.post(query, json=body)
            resp.raise_for_status()
            data = resp.json()
        content = data["candidates"][0]["content"]["parts"][0]["text"]
        return ChatResponse(
            content=content,
            model=self._model,
            usage=data.get("usageMetadata", {}),
        )


# ── Known provider prefixes ───────────────────────────────────────────────

_PROVIDER_META: dict[str, dict[str, str]] = {
    "openai": {"env_key": "OPENAI_API_KEY", "base_url": "https://api.openai.com"},
    "anthropic": {"env_key": "ANTHROPIC_API_KEY", "base_url": "https://api.anthropic.com"},
    "gemini": {"env_key": "GEMINI_API_KEY", "base_url": "https://generativelanguage.googleapis.com"},
    "ollama": {"env_key": "", "base_url": "http://localhost:11434"},
    "ollama-cloud": {"env_key": "OLLAMA_API_KEY", "base_url": "https://api.ollama.com"},
}

_PROVIDER_CLASSES: dict[str, type] = {
    "anthropic": _AnthropicAdapter,
    "gemini": _GeminiAdapter,
    "ollama-cloud": _OllamaCloudAdapter,
    # openai, ollama, and unknown prefixes use _OpenAIAdapter
}


# ── Factory ────────────────────────────────────────────────────────────────


def create_provider(model_string: str) -> LLMProvider:
    """Create an LLM provider from a ``provider_prefix:model_name`` string.

    Args:
        model_string: e.g. ``"openai:gpt-4o-mini"``, ``"ollama:llama3.2"``,
            ``"anthropic:claude-sonnet-4-20250514"``.

    Returns:
        An object with ``async def chat(messages) -> ChatResponse``.
    """
    if ":" not in model_string:
        raise ValueError(
            f"Model string must be 'provider:model', got: {model_string}"
        )

    prefix, model = model_string.split(":", 1)
    meta = _PROVIDER_META.get(prefix)

    if meta is None:
        # Unknown prefix → OpenAI-compatible fallback
        api_key = os.environ.get(f"{prefix.upper()}_API_KEY", "")
        base_url = os.environ.get(f"{prefix.upper()}_BASE_URL", f"https://api.{prefix}.com")
    else:
        api_key = os.environ.get(meta["env_key"], "")
        base_url = meta["base_url"]

    cls = _PROVIDER_CLASSES.get(prefix, _OpenAIAdapter)
    return cls(model=model, api_key=api_key, base_url=base_url)
