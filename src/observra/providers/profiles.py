"""Provider-specific trace extraction profiles.

Profiles contain only provider request/response parsing. Gateway routing,
span lifecycle, context propagation, and transport I/O remain in transport.py.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from observra.providers.transport import (
    MAX_ATTR_TEXT_LEN,
    ExtractModelName,
    ExtractText,
    ExtractUsage,
)


def _json_object(body_text: str) -> Mapping[str, Any]:
    try:
        value = json.loads(body_text)
        return value if isinstance(value, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _truncate(value: str) -> str | None:
    return value[:MAX_ATTR_TEXT_LEN] if value else None


def _content_text(value: Any) -> str | None:
    """Extract human-readable text from common provider content shapes."""
    if isinstance(value, str):
        return _truncate(value)
    if isinstance(value, list):
        parts = [_content_text(item) for item in value]
        return _truncate("\n".join(part for part in parts if part))
    if isinstance(value, dict):
        for key in (
            "text",
            "content",
            "parts",
            "message",
            "messages",
            "contents",
            "prompt",
            "response",
        ):
            text = _content_text(value.get(key))
            if text:
                return text
    return None


def _model(body_text: str) -> str | None:
    model = _json_object(body_text).get("model")
    return model if isinstance(model, str) else None


def _openai_input(body_text: str) -> str | None:
    return _content_text(_json_object(body_text).get("messages"))


def _openai_output(body_text: str) -> str | None:
    data = _json_object(body_text)
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return _truncate(body_text)
    message = choices[0].get("message") or choices[0].get("delta") or {}
    text = _content_text(message.get("content") if isinstance(message, dict) else message)
    if text:
        return text
    if isinstance(message, dict):
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls and isinstance(tool_calls[0], dict):
            function = tool_calls[0].get("function") or {}
            name = function.get("name") if isinstance(function, dict) else None
            if isinstance(name, str):
                return f"[tool_call: {name}(...)]"
    return _truncate(body_text)


def _openai_usage(body_text: str) -> tuple[int | None, int | None]:
    usage = _json_object(body_text).get("usage")
    if not isinstance(usage, dict):
        return None, None
    return usage.get("prompt_tokens"), usage.get("completion_tokens")


def _anthropic_input(body_text: str) -> str | None:
    data = _json_object(body_text)
    return _content_text(data.get("system")) or _content_text(data.get("messages"))


def _anthropic_output(body_text: str) -> str | None:
    content = _json_object(body_text).get("content")
    text = _content_text(content)
    if text:
        return text
    if isinstance(content, list) and content and isinstance(content[0], dict):
        name = content[0].get("name")
        if isinstance(name, str):
            return f"[tool_use: {name}(...)]"
    return _truncate(body_text)


def _anthropic_usage(body_text: str) -> tuple[int | None, int | None]:
    usage = _json_object(body_text).get("usage")
    if not isinstance(usage, dict):
        return None, None
    return usage.get("input_tokens"), usage.get("output_tokens")


def _ollama_input(body_text: str) -> str | None:
    data = _json_object(body_text)
    return _content_text(data.get("messages")) or _content_text(data.get("prompt"))


def _ollama_output(body_text: str) -> str | None:
    data = _json_object(body_text)
    return (
        _content_text(data.get("message"))
        or _content_text(data.get("response"))
        or _truncate(body_text)
    )


def _ollama_usage(body_text: str) -> tuple[int | None, int | None]:
    data = _json_object(body_text)
    return data.get("prompt_eval_count"), data.get("eval_count")


def _gemini_input(body_text: str) -> str | None:
    return _content_text(_json_object(body_text).get("contents"))


def _gemini_output(body_text: str) -> str | None:
    data = _json_object(body_text)
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        return _truncate(body_text)
    content = candidates[0].get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    text = _content_text(parts)
    if text:
        return text
    if isinstance(parts, list):
        for part in parts:
            function_call = part.get("functionCall") if isinstance(part, dict) else None
            if isinstance(function_call, dict) and isinstance(function_call.get("name"), str):
                return f"[function_call: {function_call['name']}(...)]"
    return _truncate(body_text)


def _gemini_usage(body_text: str) -> tuple[int | None, int | None]:
    usage = _json_object(body_text).get("usageMetadata")
    if not isinstance(usage, dict):
        return None, None
    return usage.get("promptTokenCount"), usage.get("candidatesTokenCount")


@dataclass(frozen=True)
class ProviderTraceProfile:
    """Provider identity, gateway route, and field extractors for one API shape."""

    provider_name: str
    gateway_route: str
    span_name: str
    extract_model_name: ExtractModelName
    extract_input_text: ExtractText
    extract_output_text: ExtractText
    extract_usage: ExtractUsage

    def transport_kwargs(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "gateway_route": self.gateway_route,
            "span_name": self.span_name,
            "extract_model_name": self.extract_model_name,
            "extract_input_text": self.extract_input_text,
            "extract_output_text": self.extract_output_text,
            "extract_usage": self.extract_usage,
        }


GEMINI_PROFILE = ProviderTraceProfile(
    "google",
    "gemini",
    "gemini.generate_content",
    _model,
    _gemini_input,
    _gemini_output,
    _gemini_usage,
)
OPENAI_PROFILE = ProviderTraceProfile(
    "openai",
    "openai",
    "openai.generate",
    _model,
    _openai_input,
    _openai_output,
    _openai_usage,
)
ANTHROPIC_PROFILE = ProviderTraceProfile(
    "anthropic",
    "anthropic",
    "anthropic.generate",
    _model,
    _anthropic_input,
    _anthropic_output,
    _anthropic_usage,
)
GROQ_PROFILE = ProviderTraceProfile(
    "groq",
    "groq",
    "groq.generate",
    _model,
    _openai_input,
    _openai_output,
    _openai_usage,
)
TOGETHER_PROFILE = ProviderTraceProfile(
    "together", "together", "together.generate", _model, _openai_input, _openai_output, _openai_usage
)
FIREWORKS_PROFILE = ProviderTraceProfile(
    "fireworks", "fireworks", "fireworks.generate", _model, _openai_input, _openai_output, _openai_usage
)
DEEPSEEK_PROFILE = ProviderTraceProfile(
    "deepseek", "deepseek", "deepseek.generate", _model, _openai_input, _openai_output, _openai_usage
)
XAI_PROFILE = ProviderTraceProfile(
    "xai", "xai", "xai.generate", _model, _openai_input, _openai_output, _openai_usage
)
MISTRAL_PROFILE = ProviderTraceProfile(
    "mistral", "mistral", "mistral.generate", _model, _openai_input, _openai_output, _openai_usage
)
COHERE_PROFILE = ProviderTraceProfile(
    "cohere", "cohere", "cohere.generate", _model, _openai_input, _openai_output, _openai_usage
)
HUGGINGFACE_PROFILE = ProviderTraceProfile(
    "huggingface",
    "huggingface",
    "huggingface.generate",
    _model,
    _openai_input,
    _openai_output,
    _openai_usage,
)
OPENROUTER_PROFILE = ProviderTraceProfile(
    "openrouter",
    "openrouter",
    "openrouter.generate",
    _model,
    _openai_input,
    _openai_output,
    _openai_usage,
)
OLLAMA_PROFILE = ProviderTraceProfile(
    "ollama",
    "ollama",
    "ollama.generate",
    _model,
    _ollama_input,
    _ollama_output,
    _ollama_usage,
)

PROVIDER_TRACE_PROFILES = {
    "gemini": GEMINI_PROFILE,
    "openai": OPENAI_PROFILE,
    "anthropic": ANTHROPIC_PROFILE,
    "groq": GROQ_PROFILE,
    "together": TOGETHER_PROFILE,
    "fireworks": FIREWORKS_PROFILE,
    "deepseek": DEEPSEEK_PROFILE,
    "xai": XAI_PROFILE,
    "mistral": MISTRAL_PROFILE,
    "cohere": COHERE_PROFILE,
    "huggingface": HUGGINGFACE_PROFILE,
    "openrouter": OPENROUTER_PROFILE,
    "ollama": OLLAMA_PROFILE,
}


def get_trace_profile(name: str) -> ProviderTraceProfile:
    """Return trace profile for a known provider routing key."""
    return PROVIDER_TRACE_PROFILES[name]
