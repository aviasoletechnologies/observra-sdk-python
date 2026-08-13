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
    if not value:
        return None
    if len(value) <= MAX_ATTR_TEXT_LEN:
        return value
    marker = "\n[truncated]"
    return f"{value[: MAX_ATTR_TEXT_LEN - len(marker)]}{marker}"


_BINARY_VALUE_KEYS = frozenset(
    {
        "inlineData",
        "inline_data",
        "image_url",
        "signature",
        "thoughtSignature",
        "thought_signature",
    }
)


def _content_text(value: Any) -> str | None:
    """Collect all textual content from a model-visible provider payload section.

    Tool arguments/results, system instructions, and provider-specific content blocks
    can be nested differently across APIs. A first-match walk silently loses sibling
    messages and function responses, so recurse through every non-binary value.
    """
    parts: list[str] = []

    def collect(item: Any) -> None:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, list):
            for child in item:
                collect(child)
        elif isinstance(item, Mapping):
            for key, child in item.items():
                if key not in _BINARY_VALUE_KEYS:
                    collect(child)

    collect(value)
    return _truncate("\n".join(parts))


def _sections_text(data: Mapping[str, Any], *keys: str) -> str | None:
    """Collect text from every model-visible top-level payload section in order."""
    return _content_text([data[key] for key in keys if key in data])


def _model(body_text: str) -> str | None:
    model = _json_object(body_text).get("model")
    return model if isinstance(model, str) else None


def _openai_input(body_text: str) -> str | None:
    data = _json_object(body_text)
    return _sections_text(data, "instructions", "messages", "input", "prompt", "tools")


def _openai_output(body_text: str) -> str | None:
    data = _json_object(body_text)
    return _sections_text(data, "choices", "output", "message", "response", "text") or _truncate(
        body_text
    )


def _openai_usage(body_text: str) -> tuple[int | None, int | None]:
    usage = _json_object(body_text).get("usage")
    if not isinstance(usage, dict):
        return None, None
    return usage.get("prompt_tokens"), usage.get("completion_tokens")


def _anthropic_input(body_text: str) -> str | None:
    data = _json_object(body_text)
    return _sections_text(data, "system", "messages", "tools")


def _anthropic_output(body_text: str) -> str | None:
    data = _json_object(body_text)
    return _sections_text(data, "content") or _truncate(body_text)


def _anthropic_usage(body_text: str) -> tuple[int | None, int | None]:
    usage = _json_object(body_text).get("usage")
    if not isinstance(usage, dict):
        return None, None
    return usage.get("input_tokens"), usage.get("output_tokens")


def _ollama_input(body_text: str) -> str | None:
    data = _json_object(body_text)
    return _sections_text(data, "system", "messages", "prompt", "tools")


def _ollama_output(body_text: str) -> str | None:
    data = _json_object(body_text)
    return _sections_text(data, "message", "response") or _truncate(body_text)


def _ollama_usage(body_text: str) -> tuple[int | None, int | None]:
    data = _json_object(body_text)
    return data.get("prompt_eval_count"), data.get("eval_count")


def _gemini_input(body_text: str) -> str | None:
    data = _json_object(body_text)
    return _sections_text(data, "systemInstruction", "contents", "tools")


def _gemini_output(body_text: str) -> str | None:
    data = _json_object(body_text)
    return _sections_text(data, "candidates") or _truncate(body_text)


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
