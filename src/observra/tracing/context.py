"""W3C Trace Context helpers.

OTel's own ``context`` module already tracks the "current active span" via
``contextvars``, propagating correctly across normal calls, async/await, and
threads. This module only adds the thin conveniences the rest of the SDK
needs on top of that — it does not reimplement propagation.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from contextvars import ContextVar, Token
from typing import Any

from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

_propagator = TraceContextTextMapPropagator()
_framework_llm_span: ContextVar[Any | None] = ContextVar(
    "observra_framework_llm_span",
    default=None,
)


def activate_framework_llm_span(span: Any) -> Token[Any]:
    """Mark an active framework LLM span for provider transport enrichment."""
    return _framework_llm_span.set(span)


def deactivate_framework_llm_span(token: Token[Any]) -> None:
    """Restore the prior framework LLM marker after its callback run ends."""
    _framework_llm_span.reset(token)


def active_framework_llm_span() -> Any | None:
    """Return active framework LLM span only when it still matches OTel context."""
    span = _framework_llm_span.get()
    if span is None:
        return None
    try:
        if span.get_span_context() == trace.get_current_span().get_span_context():
            return span
    except Exception:  # noqa: BLE001
        return None
    return None


def current_traceparent() -> str | None:
    """Return the ``traceparent`` header string for the currently active span, if any."""
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if not ctx.is_valid:
        return None

    carrier: dict[str, str] = {}
    _propagator.inject(carrier)
    return carrier.get("traceparent")


def inject_traceparent(headers: MutableMapping[str, str]) -> MutableMapping[str, str]:
    """Inject the active span's ``traceparent`` into an outbound header dict, in place."""
    _propagator.inject(headers)
    return headers


def has_active_span() -> bool:
    return trace.get_current_span().get_span_context().is_valid
