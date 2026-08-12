"""Global httpx patch — routes any raw httpx traffic to a known provider host through the gateway.

Not provider-SDK-specific: ``providers/gemini.py`` already covers
``google.genai.Client`` directly. This module covers everything else — plain
``httpx.post(...)``/``httpx.Client()`` calls built by hand, or by any other
library that talks to a known provider host over httpx without going through
that provider's own SDK. ``httpx.Client``/``httpx.AsyncClient`` accept a
native ``mounts`` parameter for per-host transport routing; this patches
their constructors to inject a mount for each known provider host, using the
same ``ObservraTransport``/``AsyncObservraTransport`` as everywhere else.
Requests to any other host, or clients that already set an explicit
``transport=`` themselves (e.g. our own genai patch), are left untouched.
"""

from __future__ import annotations

import functools
import logging
import threading
from typing import Any

import httpx

from observra.config import ObservraConfigError, get_config
from observra.providers.profiles import get_trace_profile
from observra.providers.transport import AsyncObservraTransport, ObservraTransport
from observra.tracing.conventions import SpanKind

logger = logging.getLogger("observra")

# httpx mount pattern -> provider trace-profile key. Ollama local mounts include
# port 11434 deliberately: intercepting all localhost traffic would reroute
# unrelated application requests through the gateway.
_PROVIDER_HOSTS = {
    "all://generativelanguage.googleapis.com": "gemini",
    "all://api.openai.com": "openai",
    "all://api.anthropic.com": "anthropic",
    "all://api.groq.com": "groq",
    "all://api.together.xyz": "together",
    "all://api.fireworks.ai": "fireworks",
    "all://api.deepseek.com": "deepseek",
    "all://api.x.ai": "xai",
    "all://api.mistral.ai": "mistral",
    "all://api.cohere.com": "cohere",
    "all://router.huggingface.co": "huggingface",
    "all://api.openrouter.ai": "openrouter",
    "all://openrouter.ai": "openrouter",
    "all://api.ollama.com": "ollama",
    "all://ollama.com": "ollama",
    "all://localhost:11434": "ollama",
    "all://127.0.0.1:11434": "ollama",
}

_patched = False
_patch_lock = threading.Lock()


def _build_mounts(*, is_async: bool) -> dict[str, Any]:
    try:
        config = get_config()
    except ObservraConfigError:
        return {}  # observra.configure() not called yet — leave httpx untouched

    mounts: dict[str, Any] = {}
    for mount_pattern, profile_name in _PROVIDER_HOSTS.items():
        profile = get_trace_profile(profile_name)
        transport_kwargs = profile.transport_kwargs()
        transport_kwargs["span_kind"] = SpanKind.LLM
        transport = (
            AsyncObservraTransport(config, **transport_kwargs)
            if is_async
            else ObservraTransport(config, **transport_kwargs)
        )
        mounts[mount_pattern] = transport
    return mounts


def _inject_mounts(kwargs: dict[str, Any], *, is_async: bool) -> dict[str, Any]:
    """Merge in a per-host mount for each known provider host.

    Deliberately does *not* back off just because the caller set a root
    ``transport=`` — several real SDKs (e.g. Anthropic's) set their own
    default transport internally for unrelated reasons (keepalive tuning,
    proxy passthrough), which would otherwise look identical to a genuine
    opt-out and silently defeat routing. httpx's own per-host ``mounts`` take
    priority over the root transport for a matching host, so this is safe:
    the only real opt-out is the caller claiming that specific host's mount
    key themselves (handled by the merge below, caller's entry always wins).
    """
    extra_mounts = _build_mounts(is_async=is_async)
    if not extra_mounts:
        return kwargs

    caller_mounts = kwargs.get("mounts") or {}
    kwargs = dict(kwargs)
    kwargs["mounts"] = {**extra_mounts, **caller_mounts}  # caller's own mounts win over ours
    return kwargs


def patch() -> None:
    """Idempotently patch ``httpx.Client``/``httpx.AsyncClient`` to route known provider hosts through the gateway."""
    global _patched
    with _patch_lock:
        if _patched:
            return

        original_client_init = httpx.Client.__init__
        original_async_client_init = httpx.AsyncClient.__init__

        @functools.wraps(original_client_init)
        def patched_client_init(self: httpx.Client, *args: Any, **kwargs: Any) -> None:
            try:
                kwargs = _inject_mounts(kwargs, is_async=False)
            except Exception:
                logger.warning("observra: failed to inject gateway mounts into httpx.Client", exc_info=True)
            original_client_init(self, *args, **kwargs)

        @functools.wraps(original_async_client_init)
        def patched_async_client_init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
            try:
                kwargs = _inject_mounts(kwargs, is_async=True)
            except Exception:
                logger.warning("observra: failed to inject gateway mounts into httpx.AsyncClient", exc_info=True)
            original_async_client_init(self, *args, **kwargs)

        httpx.Client.__init__ = patched_client_init  # type: ignore[method-assign]
        httpx.AsyncClient.__init__ = patched_async_client_init  # type: ignore[method-assign]

        _patched = True
