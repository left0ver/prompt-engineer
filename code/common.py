"""Shared helpers for prompt-engineering experiments."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _required_env(name: str) -> str:
    """Return a required environment variable with a useful error message."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"缺少环境变量 {name}，请在项目根目录的 .env 中配置它。")
    return value


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    """Create one reusable OpenAI-compatible client per process."""
    base_url = os.getenv("LLM_BASE_URL") or None
    return OpenAI(
        api_key=_required_env("LLM_API_KEY"),
        base_url=base_url,
        timeout=60.0,
        max_retries=2,
    )


def call_model(
    prompt: str,
    *,
    system_prompt: str | None = None,
    temperature: float | None = None,
) -> str:
    """Call the configured chat model and return its text response.

    Set ``temperature`` only when a prompting technique requires stochastic
    sampling, such as self-consistency. ``None`` preserves provider defaults.
    """
    if not prompt.strip():
        raise ValueError("prompt 不能为空。")

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    request: dict[str, Any] = {
        "model": _required_env("LLM_MODEL"),
        "messages": messages,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        # "extra_body": {"enable_thinking": False},
    }
    if temperature is not None:
        request["temperature"] = temperature

    response = _get_client().chat.completions.create(**request)
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("模型返回了空响应。")
    return content.strip()
