"""Utility for calling local LLM models via OpenAI-compatible API."""

from __future__ import annotations

import os
from typing import Any

LOCAL_API_BASE = os.getenv("LOCAL_API_BASE", "http://172.16.59.4:4000/v1")

# Available models
MODEL_MINIMAX = "default"       # MiniMax-M2.5
MODEL_GEMMA3  = "gemma3-27b"    # Gemma3-27B (vllm)

DEFAULT_MODEL = os.getenv("LOCAL_MODEL", MODEL_GEMMA3)


class LocalModelClient:

    def __init__(
        self,
        connection: str | None = None,
        model_id: str | None = None,
    ) -> None:
        from openai import BadRequestError, OpenAI

        self._BadRequestError = BadRequestError
        self._connection = connection or LOCAL_API_BASE
        self._model = model_id or DEFAULT_MODEL

        if not self._connection:
            raise ValueError(
                "Local model URL is required. Set LOCAL_API_BASE env var "
                "or pass connection= to LocalModelClient()."
            )

        self._client = OpenAI(
            base_url=self._connection,
            api_key="EMPTY",
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = "auto",
        **extra: Any,
    ):
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        kwargs.update(extra)

        try:
            return self._client.chat.completions.create(**kwargs)
        except self._BadRequestError as exc:
            raise RuntimeError(f"Model request failed: {exc}") from exc

    def chat_with_files(
        self,
        file_paths: list[str] | str,
        system: str | None = None,
        **extra: Any,
    ):
        """Send one or multiple files as user content (no added prompt)."""
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        parts = []
        for path in file_paths:
            with open(path, encoding="utf-8") as f:
                parts.append(f"<file name='{path}'>\n{f.read()}\n</file>")

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": "\n\n".join(parts)})

        return self.chat(messages=messages, **extra)


if __name__ == "__main__":
    import sys

    client = LocalModelClient()

    if len(sys.argv) > 1:
        # Usage: python call_models.py archivo.xml [otros.c ...]
        resp = client.chat_with_files(file_paths=sys.argv[1:])
    else:
        resp = client.chat(messages=[{"role": "user", "content": "Hola, ¿qué tal?"}])

    print(resp.choices[0].message.content)
