"""OpenAI and Anthropic streaming, a Claude preset, and local fallback."""

from __future__ import annotations

import os
from collections.abc import Iterator


def _setting(name: str, default: str = "") -> str:
    """Read environment variables first, then Streamlit deployment secrets."""
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        import streamlit as st
        value = st.secrets.get(name, "")
    except (FileNotFoundError, KeyError):
        value = ""
    return str(value).strip() or default


PROVIDER = _setting("LLM_PROVIDER", "groq_primary").lower()
if PROVIDER not in {"openai", "anthropic", "claude", "local", "groq_primary", "groq_secondary"}:
    PROVIDER = "openai"


def is_configured(provider: str) -> bool:
    """Return whether the selected provider has a credential in the environment."""
    provider = provider.lower()
    if provider in {"groq_primary", "groq_secondary"}:
        return bool(_setting(provider.upper() + "_API_KEY"))
    if provider == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    if provider in {"anthropic", "claude"}:
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    return provider == "local"


def is_available(provider: str | None = None) -> bool:
    """Local echo is always available; remote providers need credentials."""
    return is_configured(provider or PROVIDER)


def provider_status(provider: str) -> str:
    if provider in {"groq_primary", "groq_secondary"}:
        label = provider.replace("_", " ").title()
        return f"{label} is configured." if is_configured(provider) else f"{label} is not configured; set {provider.upper()}_API_KEY."
    if provider == "local":
        return "Local prompt echo — no model call is made."
    if is_configured(provider):
        return f"{'OpenAI' if provider == 'openai' else provider.title()} is configured."
    return f"{'OpenAI' if provider == 'openai' else provider.title()} is not configured; sending uses the local prompt echo."


def _local_fallback(system_prompt: str, user_prompt: str, reason: str) -> Iterator[str]:
    yield f"[Local fallback — {reason}; no LLM call was made.]\n\n"
    yield f"System prompt:\n{system_prompt}\n\nUser prompt:\n{user_prompt}"


def _bounded_history(history):
    """Keep recent complete user/assistant turns within a 2000-byte budget."""
    selected = []
    used = 0
    items = history or []
    for index in range(len(items) - 2, -1, -2):
        pair = items[index:index + 2]
        if [item.get("role") for item in pair] != ["user", "assistant"]:
            break
        size = sum(len(str(item.get("content", "")).encode("utf-8")) for item in pair)
        if used + size > 2000:
            break
        selected[0:0] = pair
        used += size
    return selected


def stream_response(
    system_prompt: str,
    user_prompt: str,
    *,
    provider: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> Iterator[str]:
    """Stream response text; provider errors are raised for the UI to report."""
    selected = (provider or PROVIDER).strip().lower()
    if selected == "local":
        yield from _local_fallback(system_prompt, user_prompt, "local provider selected")
        return

    if selected in {"groq_primary", "groq_secondary"}:
        key_name = selected.upper() + "_API_KEY"
        key = _setting(key_name)
        if not key:
            yield from _local_fallback(system_prompt, user_prompt, key_name + " is missing")
            return
        from openai import OpenAI, RateLimitError, AuthenticationError
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend({"role": item["role"], "content": str(item["content"])}
                        for item in _bounded_history(history)
                        if item.get("role") in {"user", "assistant"} and item.get("content"))
        messages.append({"role": "user", "content": user_prompt})
        model = _setting(selected.upper() + "_MODEL", _setting("GROQ_MODEL", "qwen/qwen3.8-27b"))
        try:
            with OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1", timeout=90, max_retries=0) as client:
                stream = client.chat.completions.create(model=model, messages=messages,
                    max_completion_tokens=4096, stream=True)
                try:
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                finally:
                    stream.close()
        except RateLimitError as error:
            raise RuntimeError("Groq rate limit reached. Wait before retrying or reduce the prompt size.") from error
        except AuthenticationError as error:
            raise RuntimeError(f"Groq rejected the selected credential. Check {key_name}.") from error
        return

    if selected == "openai":
        if not is_configured("openai"):
            yield from _local_fallback(system_prompt, user_prompt, "OPENAI_API_KEY is missing")
            return
        from openai import OpenAI

        client = OpenAI()
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(
            {"role": item["role"], "content": str(item["content"])}
            for item in _bounded_history(history)
            if item.get("role") in {"user", "assistant"} and item.get("content")
        )
        messages.append({"role": "user", "content": user_prompt})
        stream = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            max_completion_tokens=8192,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            text = chunk.choices[0].delta.content
            if text:
                yield text
        return

    if selected in {"anthropic", "claude"}:
        if not is_configured("anthropic"):
            yield from _local_fallback(
                system_prompt,
                user_prompt,
                "ANTHROPIC_API_KEY is missing",
            )
            return
        from anthropic import Anthropic

        client = Anthropic()
        messages = [
            {"role": item["role"], "content": str(item["content"])}
            for item in _bounded_history(history)
            if item.get("role") in {"user", "assistant"} and item.get("content")
        ]
        messages.append({"role": "user", "content": user_prompt})
        with client.messages.stream(
            model=(os.getenv("CLAUDE_MODEL") if selected == "claude" else None)
                  or os.getenv("ANTHROPIC_MODEL") or "claude-sonnet-5-5",
            max_tokens=8192,
            system=system_prompt,
            messages=messages,
        ) as stream:
            yield from stream.text_stream
        return

    raise ValueError(f"Unknown LLM provider: {selected}")


__all__ = ["PROVIDER", "is_available", "is_configured", "provider_status", "stream_response"]
