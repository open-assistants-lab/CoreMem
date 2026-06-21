"""Lightweight LLM provider factory — zero SDK dependencies (httpx only)."""

from __future__ import annotations

import os
from typing import Any, Protocol

import httpx

# ── Protocol ──────────────────────────────────────────────────────────────


class LLMProvider(Protocol):
    """Escape hatch for custom providers that don't match known prefixes."""

    async def chat(self, messages: list[dict[str, Any]]) -> ChatResponse: ...
    async def chat_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
    ) -> ChatResponse: ...


# ── Response ──────────────────────────────────────────────────────────────


class ChatResponse:
    def __init__(
        self,
        content: str,
        model: str = "",
        usage: dict[str, int] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ):
        self.content = content
        self.model = model
        self.usage = usage or {}
        self.tool_calls = tool_calls


# ── Adapters ──────────────────────────────────────────────────────────────


class _OpenAIAdapter:
    """OpenAI-compatible chat completions adapter."""

    def __init__(self, model: str, api_key: str, base_url: str, json_mode: bool = False, tool_temp: float = 0.1):
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._json_mode = json_mode
        self._tool_temp = tool_temp

    async def chat(self, messages: list[dict[str, Any]]) -> ChatResponse:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.0,
        }
        if self._json_mode:
            body["response_format"] = {"type": "json_object"}
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

    async def chat_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
    ) -> ChatResponse:
        """OpenAI-compatible tool/function calling."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": tools[0]["function"]["name"]}},
            "temperature": self._tool_temp,
            "thinking": {"type": "disabled"},
        }
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
            resp = await client.post(
                f"{self._base_url}/v1/chat/completions",
                json=body, headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
        choice = data["choices"][0]
        tool_calls = choice["message"].get("tool_calls", [])
        if tool_calls:
            content = tool_calls[0]["function"]["arguments"]
        else:
            content = choice["message"].get("content", "")
        return ChatResponse(
            content=content,
            model=data.get("model", self._model),
            usage=data.get("usage", {}),
            tool_calls=tool_calls or None,
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

    async def chat_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
    ) -> ChatResponse:
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
        if tools:
            body["tools"] = tools

        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
            resp = await client.post(
                f"{self._base_url}/v1/messages",
                json=body, headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
        content_blocks = data.get("content", [])
        tool_calls = []
        text_content = ""
        for block in content_blocks:
            if block.get("type") == "text":
                text_content = block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": block.get("input", {}),
                    },
                })
        formatted_tool_calls = []
        for tc in tool_calls:
            formatted_tool_calls.append({
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": __import__("json").dumps(tc["function"]["arguments"]),
                },
            })
        return ChatResponse(
            content=text_content or (formatted_tool_calls[0]["function"]["arguments"] if formatted_tool_calls else text_content),
            model=data.get("model", self._model),
            usage=data.get("usage", {}),
            tool_calls=formatted_tool_calls or None,
        )


class _OllamaCloudAdapter:
    """Ollama Cloud native /api/chat adapter."""

    def __init__(self, model: str, api_key: str, base_url: str, tool_temp: float = 0.1):
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._tool_temp = tool_temp

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

    async def chat_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
    ) -> ChatResponse:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "temperature": self._tool_temp,
        }
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
            resp = await client.post(
                f"{self._base_url}/api/chat",
                json=body, headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
        content = data.get("message", {}).get("content", "")
        tool_calls_data = data.get("message", {}).get("tool_calls", [])
        formatted_tool_calls = None
        if tool_calls_data:
            formatted_tool_calls = []
            for tc in tool_calls_data:
                args = tc.get("function", {}).get("arguments", "{}")
                if isinstance(args, dict):
                    import json as _json
                    args = _json.dumps(args)
                formatted_tool_calls.append({
                    "id": tc.get("id", f"tc_{len(formatted_tool_calls)}"),
                    "type": "function",
                    "function": {
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": args,
                    },
                })
        return ChatResponse(
            content=content,
            model=data.get("model", self._model),
            tool_calls=formatted_tool_calls,
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

    async def chat_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
    ) -> ChatResponse:
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
        if tools:
            gemini_tools = [{"functionDeclarations": [
                {"name": t["function"]["name"], "description": t["function"].get("description", "")}
                for t in tools
            ]}]
            body["tools"] = gemini_tools

        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
            resp = await client.post(query, json=body)
            resp.raise_for_status()
            data = resp.json()
        candidates = data.get("candidates", [])
        if candidates:
            candidate = candidates[0]
            content_block = candidate.get("content", {}).get("parts", [])
            tool_calls = []
            text_content = ""
            for part in content_block:
                if "functionCall" in part:
                    fc = part["functionCall"]
                    tool_calls.append({
                        "id": f"gc_{len(tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": fc.get("name", ""),
                            "arguments": __import__("json").dumps(fc.get("args", {})),
                        },
                    })
                else:
                    text_content = part.get("text", "")
            return ChatResponse(
                content=text_content or (tool_calls[0]["function"]["arguments"] if tool_calls else text_content),
                model=self._model,
                usage=data.get("usageMetadata", {}),
                tool_calls=tool_calls or None,
            )
        return ChatResponse(content="", model=self._model, tool_calls=None)


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


def create_provider(model_string: str, **kwargs: Any) -> LLMProvider:
    """Create an LLM provider from a ``provider_prefix:model_name`` string.

    Args:
        model_string: e.g. ``"openai:gpt-4o-mini"``, ``"ollama:llama3.2"``,
            ``"anthropic:claude-sonnet-4-20250514"``.
        **kwargs: passed to the provider adapter constructor (e.g. ``tool_temp=0.1``).

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
    return cls(model=model, api_key=api_key, base_url=base_url, **kwargs)
