"""Private per-config TracerProvider — never the global OTel provider.

Architecture requirement #2: a host application may already run its own OTel
stack. This module never calls ``opentelemetry.trace.set_tracer_provider()``;
each ``ObservraTracer`` owns a private ``TracerProvider`` instance reached
only through ``self._provider.get_tracer(...)``.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import sys
import threading
from collections.abc import Iterator
from typing import Any

from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from observra.config import ObservraConfig
from observra.tracing.exporter import GatewayExporter

logger = logging.getLogger("observra")

# Bounded memory under a slow/unreachable gateway (requirement #3) — the
# BatchSpanProcessor drops oldest spans once these caps are hit rather than
# growing without limit.
_MAX_QUEUE_SIZE = 2048
_MAX_EXPORT_BATCH_SIZE = 512


class ObservraTracer:
    """Wraps a private ``TracerProvider`` + gateway exporter for one config."""

    def __init__(self, config: ObservraConfig) -> None:
        resource = Resource.create({"service.name": "observra"})
        self._provider = TracerProvider(resource=resource)
        self._exporter = GatewayExporter(config.gateway_url, config.gateway_key)
        processor = BatchSpanProcessor(
            self._exporter,
            max_queue_size=_MAX_QUEUE_SIZE,
            max_export_batch_size=_MAX_EXPORT_BATCH_SIZE,
        )
        self._provider.add_span_processor(processor)
        self._tracer = self._provider.get_tracer("observra")
        atexit.register(self._shutdown)

    def _shutdown(self) -> None:
        try:
            self._provider.shutdown()
        except Exception:
            logger.warning("observra: error shutting down tracer provider", exc_info=True)

    @contextlib.contextmanager
    def start_span(
        self,
        name: str,
        kind: str,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[trace_api.Span | None]:
        """Open a span as the active context, isolated from the user's real call.

        Span setup/teardown failures are swallowed and logged (requirement
        #1 — telemetry must never break the caller's request). Exceptions
        raised by the caller's code *inside* the ``with`` block are recorded
        on the span if one exists, then always re-raised unchanged.
        """
        attrs = dict(attributes or {})
        attrs["observra.span_kind"] = kind

        span_cm = None
        span: trace_api.Span | None = None
        try:
            span_cm = self._tracer.start_as_current_span(
                name,
                kind=trace_api.SpanKind.CLIENT,
                attributes=attrs,
                record_exception=True,
                set_status_on_exception=True,
            )
            span = span_cm.__enter__()
        except Exception:
            logger.warning("observra: failed to start span %r", name, exc_info=True)

        try:
            yield span
        except BaseException:
            if span_cm is not None:
                try:
                    span_cm.__exit__(*sys.exc_info())
                except Exception:
                    logger.warning("observra: failed to close span %r", name, exc_info=True)
            raise
        else:
            if span_cm is not None:
                try:
                    span_cm.__exit__(None, None, None)
                except Exception:
                    logger.warning("observra: failed to close span %r", name, exc_info=True)


    def start_detached_span(
        self,
        name: str,
        kind: str,
        attributes: dict[str, Any] | None = None,
    ) -> trace_api.Span | None:
        """Start a span whose end isn't tied to a ``with`` block.

        For callback-driven integrations (Step 6 framework instrumentation)
        where start/end happen in separate calls correlated by a run id,
        rather than a single Python scope. Caller is responsible for calling
        ``span.end()`` and for attaching/detaching it as the active context
        if children must nest under it.
        """
        attrs = dict(attributes or {})
        attrs["observra.span_kind"] = kind
        try:
            return self._tracer.start_span(
                name,
                kind=trace_api.SpanKind.INTERNAL,
                attributes=attrs,
                record_exception=True,
                set_status_on_exception=True,
            )
        except Exception:
            logger.warning("observra: failed to start detached span %r", name, exc_info=True)
            return None


def safe_set_attributes(span: trace_api.Span | None, attributes: dict[str, Any]) -> None:
    """Best-effort attribute set — never lets a bad attribute value break the call."""
    if span is None:
        return
    try:
        cleaned = {k: v for k, v in attributes.items() if v is not None}
        span.set_attributes(cleaned)
    except Exception:
        logger.warning("observra: failed to set span attributes", exc_info=True)


def safe_add_event(span: trace_api.Span | None, name: str, attributes: dict[str, Any]) -> None:
    if span is None:
        return
    try:
        span.add_event(name, attributes=attributes)
    except Exception:
        logger.warning("observra: failed to add span event %r", name, exc_info=True)


def safe_end_span(span: trace_api.Span | None, error: BaseException | None = None) -> None:
    if span is None:
        return
    try:
        if error is not None:
            span.record_exception(error)
            span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(error)))
        span.end()
    except Exception:
        logger.warning("observra: failed to end span", exc_info=True)


_tracers_by_config_id: dict[int, ObservraTracer] = {}
_registry_lock = threading.Lock()


def get_tracer(config: ObservraConfig) -> ObservraTracer:
    """Return (creating + caching if needed) the tracer for a given config instance."""
    key = id(config)
    with _registry_lock:
        tracer = _tracers_by_config_id.get(key)
        if tracer is None:
            tracer = ObservraTracer(config)
            _tracers_by_config_id[key] = tracer
        return tracer
